"""Pipeline do cenário de fisioterapia: pose → ângulos → regras → Findings + artefatos.

Saídas em outputs/video/<paciente>/:
- clip_<video>_<nn>.mp4      clipes anotados de cada desvio
- frames/<video>/...         quadros anotados (desvios e amostras)
- annotated_<video>.mp4      vídeo inteiro anotado (opcional)
- runs/<video>.json          resumo da execução (lido pelo relatório e pelas métricas)
- report.html                relatório automático (re-gerado a cada execução)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from contracts.finding import Finding, ModelInfo
from modules.video.annotate import RED, ascii_text, draw_pose, put_label, put_lines
from modules.video.config import VideoConfig
from modules.video.physio import JOINT_PT, PhysioEvent, PhysioResult, PoseFrame, analyze_pose_frames
from modules.video.pose import PoseEstimator
from modules.video.timebase import MediaContext, artifact_ref
from modules.video.videoio import Mp4Writer, VideoInfo, iter_frames, probe, save_jpeg, stride_for

log = logging.getLogger(__name__)

EVENT_PT = {"amplitude_incompleta": "amplitude incompleta", "postura_fora_padrao": "postura fora da faixa"}


@dataclass
class EventArtifacts:
    clip: Path | None = None
    frame: Path | None = None


@dataclass
class RenderPlan:
    events: list[PhysioEvent]
    artifacts: list[EventArtifacts] = field(default_factory=list)
    samples: list[Path] = field(default_factory=list)
    annotated_video: Path | None = None


def _processing_stats(info: VideoInfo, n_processed: int, infer_s: float, wall_s: float, stride: int) -> dict:
    return {
        "stride": stride,
        "processed_frames": n_processed,
        "inference_seconds": round(infer_s, 3),
        "wall_seconds": round(wall_s, 3),
        # FPS de inferência: quadros analisados por segundo gasto no modelo.
        "inference_fps": round(n_processed / infer_s, 2) if infer_s > 0 else None,
        # FPS de ponta a ponta: quadros do vídeo cobertos por segundo de processamento.
        "end_to_end_fps": round(info.frame_count / wall_s, 2) if wall_s > 0 else None,
    }


def estimate_poses(path: Path, info: VideoInfo, estimator: PoseEstimator, process_fps: float):
    stride = stride_for(info, process_fps)
    frames: list[PoseFrame] = []
    infer_s = 0.0
    t0 = time.perf_counter()
    for idx, frame in iter_frames(path, stride):
        ti = time.perf_counter()
        det = estimator.predict(frame)
        infer_s += time.perf_counter() - ti
        if det is None:
            frames.append(PoseFrame(idx, idx / info.fps))
        else:
            frames.append(PoseFrame(idx, idx / info.fps, det.keypoints, det.conf, det.box))
    wall = time.perf_counter() - t0
    return frames, _processing_stats(info, len(frames), infer_s, wall, stride)


def _overlay_lines(row: dict | None, joint_col: str, joint_pt: str, rep: int | None) -> list[str]:
    lines = []
    if row is not None:
        ang = row.get(joint_col)
        trunk = row.get("trunk_angle")
        if ang is not None and np.isfinite(ang):
            lines.append(ascii_text(f"{joint_pt}: {ang:.0f}°"))
        if trunk is not None and np.isfinite(trunk):
            lines.append(ascii_text(f"tronco: {trunk:.0f}°"))
    if rep is not None:
        lines.append(f"repeticao {rep}")
    return lines


def render(
    ctx: MediaContext,
    info: VideoInfo,
    frames: list[PoseFrame],
    result: PhysioResult,
    cfg: VideoConfig,
    patient_dir: Path,
) -> RenderPlan:
    """Uma passada sequencial pelo vídeo gera clipes, quadros e o vídeo anotado."""
    out = cfg.output
    stem = ctx.stem
    plan = RenderPlan(events=result.events)
    joint = cfg.physio.primary_angle.joint
    joint_col = f"{joint}_{result.side}"
    min_conf = cfg.models.min_keypoint_conf

    pose_by_idx = {f.frame_idx: f for f in frames}
    rows = {int(r["frame_idx"]): r for r in result.series.to_dict("records")} if not result.series.empty else {}
    processed = sorted(pose_by_idx)

    pad = out.clip_padding_s
    windows = []
    for k, ev in enumerate(result.events, start=1):
        a = EventArtifacts(
            clip=patient_dir / f"clip_{stem}_{k:02d}.mp4",
            frame=patient_dir / "frames" / stem / f"desvio_{k:02d}_{ev.type}.jpg",
        )
        plan.artifacts.append(a)
        windows.append((max(0.0, ev.t_start - pad), ev.t_end + pad))

    with_pose = [f.frame_idx for f in frames if f.keypoints is not None]
    n_samples = min(out.sample_frames, len(with_pose))
    sample_idx = (
        {with_pose[int(i)] for i in np.linspace(0, len(with_pose) - 1, n_samples)} if n_samples else set()
    )
    ref_frames: dict[int, list[int]] = {}
    for k, ev in enumerate(result.events):
        ref_frames.setdefault(ev.frame_ref, []).append(k)
    rep_at = [(r.t_start, r.t_end, r.number) for r in result.reps]

    writers: dict[int, Mp4Writer] = {}
    full: Mp4Writer | None = None
    if out.annotated_video:
        plan.annotated_video = patient_dir / f"annotated_{stem}.mp4"
        full = Mp4Writer(plan.annotated_video, info.width, info.height, info.fps)

    last_pose: PoseFrame | None = None
    p = 0
    try:
        for idx, frame in iter_frames(ctx.input, 1):
            t = idx / info.fps
            while p < len(processed) and processed[p] <= idx:
                last_pose = pose_by_idx[processed[p]]
                p += 1
            row = rows.get(last_pose.frame_idx) if last_pose is not None else None
            rep = next((n for a, b, n in rep_at if a <= t <= b), None)
            active = [k for k, ev in enumerate(result.events) if ev.t_start <= t <= ev.t_end]

            canvas = frame.copy()
            if last_pose is not None:
                color = RED if active else (80, 200, 80)
                draw_pose(canvas, last_pose.keypoints, last_pose.conf, min_conf, color)
            put_lines(canvas, _overlay_lines(row, joint_col, JOINT_PT[joint], rep))
            for j, k in enumerate(active):
                label = ascii_text(f"DESVIO: {EVENT_PT[result.events[k].type]} (rep {result.events[k].rep})")
                put_label(canvas, label, (8, canvas.shape[0] - 12 - 40 * j), RED)

            if full is not None:
                full.write(canvas)
            for k, (a, b) in enumerate(windows):
                if a <= t <= b:
                    if k not in writers:
                        writers[k] = Mp4Writer(plan.artifacts[k].clip, info.width, info.height, info.fps)
                    writers[k].write(canvas)
                elif k in writers and t > b:
                    writers.pop(k).close()
            for k in ref_frames.get(idx, []):
                save_jpeg(plan.artifacts[k].frame, canvas, out.jpeg_quality)
            if idx in sample_idx:
                plan.samples.append(
                    save_jpeg(patient_dir / "frames" / stem / f"amostra_{idx:06d}.jpg", canvas, out.jpeg_quality)
                )
    finally:
        for w in writers.values():
            w.close()
        if full is not None:
            full.close()
    return plan


def events_to_findings(
    ctx: MediaContext, result: PhysioResult, plan: RenderPlan, estimator: PoseEstimator
) -> list[Finding]:
    findings = []
    for ev, art in zip(result.events, plan.artifacts):
        artifacts = [artifact_ref(p) for p in (art.clip, art.frame) if p is not None and p.exists()]
        findings.append(
            Finding(
                patient_id=ctx.patient_id,
                source="video",
                type=ev.type,
                timestamp_start=ctx.at(ev.t_start),
                timestamp_end=ctx.at(ev.t_end),
                score=ev.score,
                severity=ev.severity,
                description=ev.description,
                evidence={
                    "metric": ev.metric,
                    "value": ev.value,
                    "expected": list(ev.expected),
                    "deviation": ev.deviation,
                    "rep": ev.rep,
                    "side": result.side,
                    "video": artifact_ref(ctx.input),
                    "media_start_s": round(ev.t_start, 3),
                    "media_end_s": round(ev.t_end, 3),
                },
                model=ModelInfo(name=estimator.name, version=estimator.version),
                artifacts=artifacts,
            )
        )
    return findings


def _series_for_report(result: PhysioResult, joint_col: str, max_points: int = 600) -> dict[str, list]:
    s = result.series
    if s.empty:
        return {"t": [], "primary": [], "trunk": []}
    step = max(1, len(s) // max_points)
    sub = s.iloc[::step]

    def clean(col: str) -> list:
        if col not in sub:
            return [None] * len(sub)
        return [None if not np.isfinite(v) else round(float(v), 1) for v in sub[col]]

    return {"t": [round(float(v), 3) for v in sub["t"]], "primary": clean(joint_col), "trunk": clean("trunk_angle")}


def run_physio(
    ctx: MediaContext, cfg: VideoConfig, estimator: PoseEstimator, outputs_root: Path
) -> tuple[list[Finding], dict[str, Any]]:
    """Processa um vídeo de fisioterapia e grava artefatos. Devolve (Findings, resumo)."""
    info = probe(ctx.input)
    patient_dir = outputs_root / "video" / ctx.patient_id
    patient_dir.mkdir(parents=True, exist_ok=True)

    frames, stats = estimate_poses(ctx.input, info, estimator, cfg.physio.process_fps)
    result = analyze_pose_frames(frames, cfg.physio, cfg.models.min_keypoint_conf)
    plan = render(ctx, info, frames, result, cfg, patient_dir)
    findings = events_to_findings(ctx, result, plan, estimator)

    joint_col = f"{cfg.physio.primary_angle.joint}_{result.side}"
    pose_rate = float(np.mean([f.keypoints is not None for f in frames])) if frames else 0.0
    summary: dict[str, Any] = {
        "scenario": "physio",
        "patient_id": ctx.patient_id,
        "video": artifact_ref(ctx.input),
        "video_stem": ctx.stem,
        "artifact": ctx.artifact,
        "demo_offset": ctx.demo_offset,
        "demo_start": ctx.at(0).isoformat(),
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "video_info": {
            "fps": round(info.fps, 3), "frames": info.frame_count, "width": info.width,
            "height": info.height, "duration_s": round(info.duration_s, 2),
        },
        "model": {"name": estimator.name, "version": estimator.version},
        "processing": stats,
        "exercise": cfg.physio.exercise,
        "primary_angle": {**cfg.physio.primary_angle.model_dump(), "side_used": result.side},
        "thresholds": cfg.physio.rules.model_dump(mode="json"),
        "frames_with_pose": round(pose_rate, 3),
        "repetitions": [
            {"rep": r.number, "start_s": round(r.t_start, 2), "end_s": round(r.t_end, 2),
             "peak_angle": round(r.peak_angle, 1), "depth": round(r.depth, 1)}
            for r in result.reps
        ],
        "events": [
            {
                **ev.as_dict(),
                "finding_id": str(f.finding_id),
                "timestamp_start": f.timestamp_start.isoformat(),
                "timestamp_end": f.timestamp_end.isoformat(),
                "clip": artifact_ref(a.clip) if a.clip and a.clip.exists() else None,
                "frame": artifact_ref(a.frame) if a.frame and a.frame.exists() else None,
            }
            for ev, f, a in zip(result.events, findings, plan.artifacts)
        ],
        "samples": [artifact_ref(p) for p in plan.samples],
        "annotated_video": artifact_ref(plan.annotated_video) if plan.annotated_video else None,
        "series": _series_for_report(result, joint_col),
    }
    write_run_summary(patient_dir, ctx.stem, "physio", summary)
    return findings, summary


def write_run_summary(patient_dir: Path, stem: str, scenario: str, summary: dict) -> Path:
    path = patient_dir / "runs" / f"{scenario}_{stem}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return path

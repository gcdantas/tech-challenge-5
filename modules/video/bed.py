"""Cenário do leito: YOLOv8 para pessoa e leito, borda do leito como zona crítica.

Gera outputs/video/<paciente>/movement.parquet (~5 Hz) para o I3, com as colunas
do plano: patient_id, timestamp, centroid_x, centroid_y, torso_angle,
velocidade, in_bed. NÃO emite Findings: saída do leito e imobilidade são
sinalizadas só pelo módulo de anomalias (I3).

Unidades (a confirmar com o I3):
- centroid_x, centroid_y: centro da caixa da pessoa, normalizado 0–1 pelo quadro;
- torso_angle: inclinação ombros→quadris em relação à vertical, graus (0 em pé, 90 deitado);
- velocidade: deslocamento do centroide em unidades normalizadas por segundo (suavizado);
- in_bed: fração da caixa da pessoa dentro da caixa do leito >= in_bed_min_overlap.
Sem pessoa visível: centroide, tronco e velocidade NaN; in_bed mantém o último
estado por até missing_hold_s (oclusão) e depois vale false.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from modules.video.annotate import BLUE, GREEN, ORANGE, RED, ascii_text, draw_box, draw_edge_zone, draw_pose, put_lines
from modules.video.config import VideoConfig
from modules.video.geometry import inclination_from_vertical, moving_average
from modules.video.metrics import labels_for
from modules.video.pipeline import _processing_stats, write_run_summary
from modules.video.pose import ObjectDetector, PoseEstimator
from modules.video.timebase import MediaContext, artifact_ref
from modules.video.videoio import Mp4Writer, iter_frames, probe, save_jpeg, stride_for

log = logging.getLogger(__name__)

MOVEMENT_COLUMNS = ["patient_id", "timestamp", "centroid_x", "centroid_y", "torso_angle", "velocidade", "in_bed"]


@dataclass
class BedSample:
    frame_idx: int
    t: float
    person_box: np.ndarray | None
    keypoints: np.ndarray | None
    conf: np.ndarray | None
    bed_box: np.ndarray | None


# --------------------------------------------------------------------------- geometria


def box_area(b) -> float:
    return max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))


def overlap_fraction(person, bed) -> float:
    """Fração da área da pessoa que está dentro do leito."""
    ix1, iy1 = max(person[0], bed[0]), max(person[1], bed[1])
    ix2, iy2 = min(person[2], bed[2]), min(person[3], bed[3])
    a = box_area(person)
    return box_area((ix1, iy1, ix2, iy2)) / a if a > 0 else 0.0


def inner_box(bed, margin: float) -> np.ndarray:
    x1, y1, x2, y2 = (float(v) for v in bed)
    dx, dy = (x2 - x1) * margin, (y2 - y1) * margin
    return np.array([x1 + dx, y1 + dy, x2 - dx, y2 - dy])


def in_edge_zone(point, bed, inner) -> bool:
    x, y = point
    inside_outer = bed[0] <= x <= bed[2] and bed[1] <= y <= bed[3]
    inside_inner = inner[0] < x < inner[2] and inner[1] < y < inner[3]
    return inside_outer and not inside_inner


def median_box(boxes: list[np.ndarray]) -> np.ndarray | None:
    return np.median(np.stack(boxes), axis=0) if boxes else None


def torso_angle(kp: np.ndarray | None, conf: np.ndarray | None, min_conf: float) -> float:
    if kp is None or conf is None:
        return float("nan")
    sh = [kp[i] for i in (5, 6) if conf[i] >= min_conf]
    hp = [kp[i] for i in (11, 12) if conf[i] >= min_conf]
    if not sh or not hp:
        return float("nan")
    return inclination_from_vertical(np.mean(hp, axis=0), np.mean(sh, axis=0))


# --------------------------------------------------------------------------- série


def build_movement(
    samples: list[BedSample], bed_box: np.ndarray, width: int, height: int, ctx: MediaContext, cfg: VideoConfig
) -> pd.DataFrame:
    bc = cfg.bed
    t = np.array([s.t for s in samples], float)
    cx = np.full(len(samples), np.nan)
    cy = np.full(len(samples), np.nan)
    torso = np.full(len(samples), np.nan)
    in_bed = np.zeros(len(samples), bool)

    last_state, last_seen = False, -np.inf
    for i, s in enumerate(samples):
        if s.person_box is not None:
            x1, y1, x2, y2 = s.person_box
            cx[i] = ((x1 + x2) / 2) / width
            cy[i] = ((y1 + y2) / 2) / height
            torso[i] = torso_angle(s.keypoints, s.conf, cfg.models.min_keypoint_conf)
            last_state = overlap_fraction(s.person_box, bed_box) >= bc.in_bed_min_overlap
            last_seen = s.t
            in_bed[i] = last_state
        else:
            in_bed[i] = last_state if (s.t - last_seen) <= bc.missing_hold_s else False

    vel = np.full(len(samples), np.nan)
    if len(samples) > 1:
        dt = np.diff(t)
        step = np.hypot(np.diff(cx), np.diff(cy)) / np.where(dt > 0, dt, np.nan)
        vel[1:] = step
        vel[0] = 0.0 if np.isfinite(cx[0]) else np.nan
        period = float(np.median(dt)) if len(dt) else 1.0
        vel = moving_average(vel, max(1, int(round(bc.velocity_smoothing_s / period))))

    return pd.DataFrame(
        {
            "patient_id": ctx.patient_id,
            "timestamp": pd.DatetimeIndex([ctx.at(float(x)) for x in t]),
            "centroid_x": np.round(cx, 4),
            "centroid_y": np.round(cy, 4),
            "torso_angle": np.round(torso, 2),
            "velocidade": np.round(vel, 4),
            "in_bed": in_bed,
        },
        columns=MOVEMENT_COLUMNS,
    )


def write_parquet(df: pd.DataFrame, path: Path, meta: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    md = dict(table.schema.metadata or {})
    md.update({k.encode(): (v if isinstance(v, str) else json.dumps(v)).encode() for k, v in meta.items()})
    pq.write_table(table.replace_schema_metadata(md), path)
    return path


# --------------------------------------------------------------------------- execução


def _pick(dets, label):
    cands = [d for d in dets if d.label == label]
    return max(cands, key=lambda d: box_area(d.box) * d.score) if cands else None


def collect_samples(ctx, info, cfg, pose: PoseEstimator, detector: ObjectDetector):
    stride = stride_for(info, cfg.bed.sample_hz)
    samples: list[BedSample] = []
    infer_s, t0 = 0.0, time.perf_counter()
    for idx, frame in iter_frames(ctx.input, stride):
        ti = time.perf_counter()
        dets = detector.detect(frame)
        p = pose.predict(frame)
        infer_s += time.perf_counter() - ti
        bed = _pick(dets, cfg.bed.bed_class)
        person = _pick(dets, cfg.bed.person_class)
        box = person.box if person is not None else (p.box if p is not None else None)
        samples.append(
            BedSample(
                idx, idx / info.fps, box,
                p.keypoints if p is not None else None, p.conf if p is not None else None,
                bed.box if bed is not None else None,
            )
        )
    stats = _processing_stats(info, len(samples), infer_s, time.perf_counter() - t0, stride)
    return samples, stride, stats


def resolve_bed_box(samples: list[BedSample], width: int, height: int, cfg: VideoConfig) -> tuple[np.ndarray, str]:
    if cfg.bed.bed_box_override is not None:
        x1, y1, x2, y2 = cfg.bed.bed_box_override
        return np.array([x1 * width, y1 * height, x2 * width, y2 * height]), "caixa definida em config/video.yaml"
    boxes = [s.bed_box for s in samples if s.bed_box is not None]
    if not boxes:
        raise ValueError(
            "nenhum leito detectado pelo YOLOv8 neste vídeo; defina bed.bed_box_override "
            "([x1, y1, x2, y2] normalizados) em config/video.yaml"
        )
    return median_box(boxes), f"detectado pelo YOLOv8 (mediana de {len(boxes)} de {len(samples)} quadros)"


def _frames_to_save(df: pd.DataFrame, samples: list[BedSample], n_max: int) -> set[int]:
    changes = np.flatnonzero(np.diff(df["in_bed"].to_numpy().astype(int)) != 0) + 1
    picks = {samples[i].frame_idx for i in changes[: n_max // 2]}
    rest = max(1, n_max - len(picks))
    for i in np.linspace(0, len(samples) - 1, min(rest, len(samples))):
        picks.add(samples[int(i)].frame_idx)
    return picks


def annotate_and_save(ctx, info, stride, samples, df, bed_box, cfg, patient_dir) -> tuple[list[Path], Path | None]:
    inner = inner_box(bed_box, cfg.bed.edge_margin)
    by_idx = {s.frame_idx: (s, row) for s, row in zip(samples, df.to_dict("records"))}
    save = _frames_to_save(df, samples, cfg.bed.max_frames_saved)
    saved: list[Path] = []
    writer = None
    video_path = None
    if cfg.output.annotated_video:
        video_path = patient_dir / f"annotated_{ctx.stem}.mp4"
        writer = Mp4Writer(video_path, info.width, info.height, info.fps / stride)
    try:
        for idx, frame in iter_frames(ctx.input, stride):
            if idx not in by_idx:
                continue
            s, row = by_idx[idx]
            canvas = frame.copy()
            draw_edge_zone(canvas, bed_box, inner, ORANGE)
            draw_box(canvas, bed_box, "leito", BLUE)
            state = "sem pessoa visivel"
            if s.person_box is not None:
                center = ((s.person_box[0] + s.person_box[2]) / 2, (s.person_box[1] + s.person_box[3]) / 2)
                edge = in_edge_zone(center, bed_box, inner)
                state = "no leito" if row["in_bed"] else "fora do leito"
                if edge:
                    state += " (zona da borda)"
                color = GREEN if row["in_bed"] and not edge else (ORANGE if row["in_bed"] else RED)
                draw_box(canvas, s.person_box, "pessoa", color)
                draw_pose(canvas, s.keypoints, s.conf, cfg.models.min_keypoint_conf, color)
            lines = [ascii_text(f"t = {s.t:.1f} s | {state}")]
            if np.isfinite(row["torso_angle"]):
                lines.append(ascii_text(f"tronco: {row['torso_angle']:.0f}°"))
            put_lines(canvas, lines)
            if writer is not None:
                writer.write(canvas)
            if idx in save:
                saved.append(save_jpeg(patient_dir / "frames" / ctx.stem / f"leito_{idx:06d}.jpg", canvas,
                                       cfg.output.jpeg_quality))
    finally:
        if writer is not None:
            writer.close()
    return saved, video_path


def run_bed(
    ctx: MediaContext, cfg: VideoConfig, pose: PoseEstimator, detector: ObjectDetector, outputs_root: Path
) -> dict[str, Any]:
    info = probe(ctx.input)
    patient_dir = outputs_root / "video" / ctx.patient_id
    samples, stride, stats = collect_samples(ctx, info, cfg, pose, detector)
    if not samples:
        raise ValueError(f"nenhum quadro lido de {ctx.input}")
    bed_box, bed_source = resolve_bed_box(samples, info.width, info.height, cfg)
    df = build_movement(samples, bed_box, info.width, info.height, ctx, cfg)
    effective_hz = info.fps / stride

    parquet = write_parquet(
        df,
        patient_dir / "movement.parquet",
        {
            "mock": "false",
            "generator": "modules/video/bed.py",
            "video": artifact_ref(ctx.input),
            "demo_offset": str(ctx.demo_offset),
            "sample_hz": f"{effective_hz:.3f}",
            "bed_box_px": [round(float(v), 1) for v in bed_box],
            "units": {
                "centroid": "normalizado 0-1 pelo quadro",
                "torso_angle": "graus em relação à vertical",
                "velocidade": "unidades normalizadas por segundo",
            },
            "models": [detector.name, pose.name],
        },
    )

    labels = labels_for(ctx.stem)
    labels_path = None
    if labels is not None and labels.bed_exits:
        labels_path = patient_dir / "bed_exit_labels.json"
        labels_path.write_text(
            json.dumps(
                {
                    "patient_id": ctx.patient_id,
                    "video": artifact_ref(ctx.input),
                    "bed_exits": [{"media_s": s, "timestamp": ctx.at(s).isoformat()} for s in labels.bed_exits],
                },
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )

    frames, video_path = annotate_and_save(ctx, info, stride, samples, df, bed_box, cfg, patient_dir)
    summary = {
        "scenario": "bed",
        "patient_id": ctx.patient_id,
        "video": artifact_ref(ctx.input),
        "video_stem": ctx.stem,
        "artifact": ctx.artifact,
        "demo_offset": ctx.demo_offset,
        "demo_start": ctx.at(0).isoformat(),
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "video_info": {"fps": round(info.fps, 3), "frames": info.frame_count, "width": info.width,
                       "height": info.height, "duration_s": round(info.duration_s, 2)},
        "models": [f"{detector.name} (pessoa e leito)", f"{pose.name} (tronco)"],
        "processing": stats,
        "sample_hz": round(effective_hz, 2),
        "bed_source": bed_source,
        "bed_box_px": [round(float(v), 1) for v in bed_box],
        "samples_count": len(df),
        "person_rate": round(float(np.mean([s.person_box is not None for s in samples])), 3),
        "in_bed_rate": round(float(df["in_bed"].mean()), 3),
        "movement_parquet": artifact_ref(parquet),
        "bed_exit_labels": artifact_ref(labels_path) if labels_path else None,
        "frames": [artifact_ref(p) for p in frames],
        "annotated_video": artifact_ref(video_path) if video_path else None,
    }
    write_run_summary(patient_dir, ctx.stem, "bed", summary)
    return summary

"""Relatório automático por paciente (outputs/video/<paciente>/report.html).

Reúne os resumos em runs/*.json (fisioterapia, leito e OpenPose) e, se houver,
metrics.json. Cada vídeo processado vira uma seção.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from contracts.patients import REPO_ROOT
from modules.video.physio import JOINT_PT

log = logging.getLogger(__name__)
TEMPLATES = Path(__file__).resolve().parent / "templates"
EVENT_PT = {"amplitude_incompleta": "Amplitude incompleta", "postura_fora_padrao": "Postura fora da faixa"}
SIDE_PT = {"left": "esquerdo", "right": "direito"}


def _rel(path: str | None, base: Path) -> str | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.exists():
        return None
    return Path(os.path.relpath(p, base)).as_posix()


def _hms(iso: str) -> str:
    return datetime.fromisoformat(iso).strftime("%H:%M:%S")


def _fmt_full(iso: str) -> str:
    return datetime.fromisoformat(iso).strftime("%d/%m/%Y %H:%M:%S (UTC%z)")


def angle_chart_svg(run: dict[str, Any], width: int = 860, height: int = 240) -> str | None:
    s = run.get("series") or {}
    ts = s.get("t") or []
    if len(ts) < 2:
        return None
    pad_l, pad_r, pad_t, pad_b = 40, 10, 10, 28
    t0, t1 = ts[0], ts[-1] if ts[-1] > ts[0] else ts[0] + 1
    w, h = width - pad_l - pad_r, height - pad_t - pad_b

    def x(t: float) -> float:
        return pad_l + (t - t0) / (t1 - t0) * w

    def y(a: float) -> float:
        return pad_t + (1 - a / 180.0) * h

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Ângulos ao longo do vídeo">']
    for ev in run.get("events", []):
        parts.append(
            f'<rect x="{x(ev["t_start"]):.1f}" y="{pad_t}" width="{max(1.0, x(ev["t_end"]) - x(ev["t_start"])):.1f}" '
            f'height="{h}" fill="#c0392b" opacity="0.18"/>'
        )
    for a in (0, 45, 90, 135, 180):
        parts.append(f'<line x1="{pad_l}" x2="{width - pad_r}" y1="{y(a):.1f}" y2="{y(a):.1f}" stroke="#d9dee4"/>')
        parts.append(f'<text x="{pad_l - 6}" y="{y(a) + 4:.1f}" font-size="11" text-anchor="end" fill="#5b6673">{a}°</text>')
    nticks = 6
    for i in range(nticks + 1):
        tv = t0 + (t1 - t0) * i / nticks
        parts.append(f'<text x="{x(tv):.1f}" y="{height - 8}" font-size="11" text-anchor="middle" fill="#5b6673">{tv:.0f} s</text>')

    th = run.get("thresholds", {})
    direction = run.get("primary_angle", {}).get("direction", "decrease")
    amp = th.get("amplitude_incompleta", {}).get("expected")
    post = th.get("postura_fora_padrao", {}).get("expected")
    if amp:
        lim = amp[1] if direction == "decrease" else amp[0]
        parts.append(f'<line x1="{pad_l}" x2="{width - pad_r}" y1="{y(lim):.1f}" y2="{y(lim):.1f}" stroke="#2e86c1" stroke-dasharray="5 4"/>')
    if post:
        parts.append(f'<line x1="{pad_l}" x2="{width - pad_r}" y1="{y(post[1]):.1f}" y2="{y(post[1]):.1f}" stroke="#d68910" stroke-dasharray="5 4"/>')

    for key, color in (("primary", "#2e86c1"), ("trunk", "#d68910")):
        vals = s.get(key) or []
        segs, cur = [], []
        for t, v in zip(ts, vals):
            if v is None:
                if cur:
                    segs.append(cur)
                cur = []
            else:
                cur.append(f"{x(t):.1f},{y(v):.1f}")
        if cur:
            segs.append(cur)
        for seg in segs:
            parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{" ".join(seg)}"/>')
    parts.append("</svg>")
    return "".join(parts)


def _physio_ctx(run: dict, base: Path) -> dict:
    run = dict(run)
    joint = run.get("primary_angle", {}).get("joint", "knee")
    run["joint_pt"] = f"ângulo do {JOINT_PT.get(joint, joint)}"
    run["side_pt"] = SIDE_PT.get(run.get("primary_angle", {}).get("side_used"), "—")
    run["demo_start_fmt"] = _fmt_full(run["demo_start"])
    events = []
    for ev in run.get("events", []):
        ev = dict(ev)
        ev["type_pt"] = EVENT_PT.get(ev["type"], ev["type"])
        ev["start_fmt"] = _hms(ev["timestamp_start"])
        ev["end_fmt"] = _hms(ev["timestamp_end"])
        ev["frame_rel"] = _rel(ev.get("frame"), base)
        ev["clip_rel"] = _rel(ev.get("clip"), base)
        events.append(ev)
    run["events"] = events
    run["samples_rel"] = [r for r in (_rel(p, base) for p in run.get("samples", [])) if r]
    run["annotated_rel"] = _rel(run.get("annotated_video"), base)
    run["chart"] = angle_chart_svg(run)
    return run


def _bed_ctx(run: dict, base: Path) -> dict:
    run = dict(run)
    run["demo_start_fmt"] = _fmt_full(run["demo_start"])
    run["frames_rel"] = [r for r in (_rel(p, base) for p in run.get("frames", [])) if r]
    return run


def _openpose_ctx(run: dict, base: Path) -> dict:
    run = dict(run)
    frames = []
    for fr in run.get("frames", []):
        rel = _rel(fr.get("image"), base)
        if rel:
            frames.append({**fr, "rel": rel})
    run["frames"] = frames
    return run


def _fmt_ratio(v: float | None) -> str:
    return "—" if v is None else f"{v:.2f}"


def render_report(patient_dir: Path, patient_id: str) -> Path:
    runs_dir = patient_dir / "runs"
    physio, bed, openpose, skipped = [], [], [], []
    builders = {"physio": (_physio_ctx, physio), "bed": (_bed_ctx, bed), "openpose": (_openpose_ctx, openpose)}
    for p in sorted(runs_dir.glob("*.json")) if runs_dir.exists() else []:
        try:
            run = json.loads(p.read_text(encoding="utf-8"))
            build, target = builders[run.get("scenario")]
            target.append(build(run, patient_dir))
        except (KeyError, TypeError, ValueError) as exc:
            log.warning("resumo de execução ignorado no relatório: %s (%r)", p, exc)
            skipped.append(p.name)

    metrics = None
    mpath = patient_dir / "metrics.json"
    if mpath.exists():
        metrics = json.loads(mpath.read_text(encoding="utf-8"))
        for r in metrics.get("rows", []):
            for k in ("precision", "recall", "f1"):
                r[f"{k}_fmt"] = _fmt_ratio(r.get(k))

    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(["html", "j2"]))
    html = env.get_template("report.html.j2").render(
        patient_id=patient_id,
        generated_at=datetime.now().astimezone().strftime("%d/%m/%Y %H:%M"),
        physio=physio,
        bed=bed,
        openpose=openpose,
        metrics=metrics,
        skipped=skipped,
    )
    out = patient_dir / "report.html"
    out.write_text(html, encoding="utf-8")
    return out

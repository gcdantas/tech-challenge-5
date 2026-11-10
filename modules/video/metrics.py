"""Métricas do módulo de vídeo: precisão, recall e F1 por evento contra rótulos manuais; FPS.

Rótulos: um YAML por vídeo em data/labels/<nome_do_video>.yaml (modelo em
data/labels/video.example.yaml), com tempos em segundos da mídia.

    python -m modules.video.metrics --patient P001

Lê outputs/video/<paciente>/runs/*.json e grava metrics.json e metrics.md na
mesma pasta (o relatório HTML passa a mostrar a tabela).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from contracts.patients import REPO_ROOT

DEFAULT_LABELS_DIR = REPO_ROOT / "data" / "labels"
EVENT_TYPES = ("amplitude_incompleta", "postura_fora_padrao")
DISCLAIMER = (
    "As métricas foram calculadas sobre uma amostra pequena, roteirizada e destinada à validação "
    "técnica do protótipo; não representam validação clínica nem desempenho esperado em ambiente "
    "hospitalar real."
)


class LabelEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str
    start: float = Field(ge=0)
    end: float = Field(ge=0)

    @model_validator(mode="after")
    def _ordem(self) -> LabelEvent:
        if self.end < self.start:
            raise ValueError("end anterior a start")
        return self


class VideoLabels(BaseModel):
    model_config = ConfigDict(extra="forbid")

    video: str | None = None
    events: list[LabelEvent] = Field(default_factory=list)
    bed_exits: list[float] = Field(default_factory=list)


def load_labels(path: str | Path) -> VideoLabels:
    with Path(path).open(encoding="utf-8") as fh:
        return VideoLabels.model_validate(yaml.safe_load(fh) or {})


def labels_for(stem: str, labels_dir: Path = DEFAULT_LABELS_DIR) -> VideoLabels | None:
    for ext in (".yaml", ".yml"):
        p = labels_dir / f"{stem}{ext}"
        if p.exists():
            return load_labels(p)
    return None


@dataclass(frozen=True)
class Interval:
    type: str
    start: float
    end: float


def temporal_iou(a: Interval, b: Interval) -> float:
    inter = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    union = max(a.end, b.end) - min(a.start, b.start)
    if union <= 0:  # dois instantes iguais
        return 1.0 if a.start == b.start else 0.0
    return inter / union


def match_events(pred: list[Interval], gt: list[Interval], iou_threshold: float) -> tuple[int, int, int]:
    """Casamento guloso 1-para-1 por IoU temporal, só entre eventos do mesmo tipo.

    Devolve (verdadeiros positivos, falsos positivos, falsos negativos).
    """
    pairs = []
    for i, p in enumerate(pred):
        for j, g in enumerate(gt):
            if p.type == g.type:
                iou = temporal_iou(p, g)
                if iou >= iou_threshold:
                    pairs.append((iou, i, j))
    pairs.sort(reverse=True)
    used_p, used_g = set(), set()
    for _, i, j in pairs:
        if i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
    tp = len(used_p)
    return tp, len(pred) - tp, len(gt) - tp


def prf(tp: int, fp: int, fn: int) -> tuple[float | None, float | None, float | None]:
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    if precision is None or recall is None or (precision + recall) == 0:
        f1 = None if precision is None or recall is None else 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def _row(video: str, type_: str, tp: int, fp: int, fn: int) -> dict:
    p, r, f = prf(tp, fp, fn)
    rnd = lambda v: None if v is None else round(v, 3)  # noqa: E731
    return {"video": video, "type": type_, "tp": tp, "fp": fp, "fn": fn,
            "precision": rnd(p), "recall": rnd(r), "f1": rnd(f)}


def evaluate_events(
    video: str, pred: list[Interval], gt: list[Interval], iou_threshold: float, types=EVENT_TYPES
) -> list[dict]:
    rows = []
    total = [0, 0, 0]
    for t in types:
        tp, fp, fn = match_events([p for p in pred if p.type == t], [g for g in gt if g.type == t], iou_threshold)
        rows.append(_row(video, t, tp, fp, fn))
        total = [total[0] + tp, total[1] + fp, total[2] + fn]
    rows.append(_row(video, "todos", *total))
    return rows


def evaluate_patient(patient_dir: Path, labels_dir: Path = DEFAULT_LABELS_DIR, iou_threshold: float = 0.3) -> dict:
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((patient_dir / "runs").glob("*.json"))]
    rows, fps, missing = [], [], []
    agg = {t: [0, 0, 0] for t in EVENT_TYPES}
    for run in runs:
        proc = run.get("processing", {})
        fps.append({
            "video": run.get("video_stem"), "scenario": run.get("scenario"),
            "inference_fps": proc.get("inference_fps"), "end_to_end_fps": proc.get("end_to_end_fps"),
        })
        if run.get("scenario") != "physio":
            continue
        labels = labels_for(run["video_stem"], labels_dir)
        if labels is None:
            missing.append(run["video_stem"])
            continue
        pred = [Interval(e["type"], e["t_start"], e["t_end"]) for e in run.get("events", [])]
        gt = [Interval(e.type, e.start, e.end) for e in labels.events]
        video_rows = evaluate_events(run["video_stem"], pred, gt, iou_threshold)
        rows.extend(video_rows)
        for r in video_rows:
            if r["type"] in agg:
                agg[r["type"]][0] += r["tp"]
                agg[r["type"]][1] += r["fp"]
                agg[r["type"]][2] += r["fn"]
    if len({r["video"] for r in rows}) > 1:
        total = [0, 0, 0]
        for t, (tp, fp, fn) in agg.items():
            rows.append(_row("(todos os vídeos)", t, tp, fp, fn))
            total = [total[0] + tp, total[1] + fp, total[2] + fn]
        rows.append(_row("(todos os vídeos)", "todos", *total))
    return {
        "iou_threshold": iou_threshold,
        "rows": rows,
        "fps": fps,
        "videos_without_labels": missing,
        "disclaimer": DISCLAIMER,
    }


def to_markdown(result: dict) -> str:
    def f(v):
        return "—" if v is None else f"{v:.2f}"

    lines = [
        "# Métricas do módulo de vídeo", "", f"> {DISCLAIMER}", "",
        f"IoU temporal mínima para casar evento previsto e rótulo: {result['iou_threshold']}", "",
        "| Vídeo | Evento | VP | FP | FN | Precisão | Recall | F1 |", "|---|---|---|---|---|---|---|---|",
    ]
    for r in result["rows"]:
        lines.append(
            f"| {r['video']} | {r['type']} | {r['tp']} | {r['fp']} | {r['fn']} | "
            f"{f(r['precision'])} | {f(r['recall'])} | {f(r['f1'])} |"
        )
    if not result["rows"]:
        lines.append("| — | sem rótulos | | | | | | |")
    lines += ["", "| Vídeo | Cenário | FPS de inferência | FPS ponta a ponta |", "|---|---|---|---|"]
    for r in result["fps"]:
        lines.append(f"| {r['video']} | {r['scenario']} | {r['inference_fps'] or '—'} | {r['end_to_end_fps'] or '—'} |")
    if result["videos_without_labels"]:
        lines += ["", "Vídeos sem rótulos: " + ", ".join(result["videos_without_labels"])]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    from modules.video.config import load_video_config
    from modules.video.report import render_report

    ap = argparse.ArgumentParser(description="Métricas do módulo de vídeo contra rótulos manuais.")
    ap.add_argument("--patient", required=True)
    ap.add_argument("--outputs", default=str(REPO_ROOT / "outputs"))
    ap.add_argument("--labels", default=str(DEFAULT_LABELS_DIR))
    ap.add_argument("--iou", type=float, default=None, help="padrão: metrics.iou_threshold do config/video.yaml")
    args = ap.parse_args(argv)

    iou = args.iou if args.iou is not None else load_video_config().metrics.iou_threshold
    patient_dir = Path(args.outputs) / "video" / args.patient
    result = evaluate_patient(patient_dir, Path(args.labels), iou)
    (patient_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    md = to_markdown(result)
    (patient_dir / "metrics.md").write_text(md, encoding="utf-8")
    render_report(patient_dir, args.patient)
    print(md)


if __name__ == "__main__":
    main()

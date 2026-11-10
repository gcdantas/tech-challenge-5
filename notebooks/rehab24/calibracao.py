"""Calibra os limiares de perfil das regras de fisioterapia com o REHAB24-6.

    python -m notebooks.rehab24.calibracao

Pré-requisito: o módulo já rodou nos 6 agachamentos (outputs/rehab24-6/.../runs);
os picos e a série do tronco vêm desses resumos e não dependem dos limiares.

Método
- Só repetições em que a pessoa está DE PERFIL para a câmera 18 (vista da demo).
- Para cada repetição anotada: o pico do joelho da repetição detectada que casa
  com ela (IoU temporal >= 0,3) e a maior inclinação do tronco mantida por
  ``min_duration_s`` dentro do intervalo anotado. Sem repetição detectada
  correspondente, conta como não sinalizada.
- Regra combinada (como o módulo): sinalizada se pico > A (amplitude incompleta)
  ou tronco > T por >= min_duration_s (postura).
- Busca em grade de A e T maximizando o F1 da classe "repetição incorreta".
  Empate: o par mais conservador (A e T maiores).

Relatórios:
1. separação por pessoa: calibra em PM_022, PM_043, PM_126; testa em PM_105, PM_113, PM_118;
2. validação deixando uma pessoa de fora (estabilidade do limiar escolhido);
3. limiares atuais do config/video.yaml × limiares antes da calibração (100°, 45°).

Saídas: outputs/calibracao/limiares_perfil.{json,md}.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from modules.video.config import load_video_config

ROOT = Path(__file__).resolve().parents[2]
SEG = ROOT / "data" / "video" / "rehab24-6" / "Segmentation.csv"
RUNS = ROOT / "outputs" / "rehab24-6" / "video" / "P001" / "runs"
OUT = ROOT / "outputs" / "calibracao"
FPS = 30.0
CALIB = ["PM_022", "PM_043", "PM_126"]
TEST = ["PM_105", "PM_113", "PM_118"]
GRID_A = list(range(80, 155, 5)) + [math.inf]   # limite do pico do joelho (graus); inf = regra desligada
GRID_T = list(range(15, 75, 5)) + [math.inf]    # limite do tronco (graus); inf = regra desligada
BEFORE = (100.0, 45.0)                          # limiares antes da calibração


def iou(a, b) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def sustained_max(t: np.ndarray, v: np.ndarray, start: float, end: float, min_dur: float) -> float:
    """Maior valor que a série mantém (mínimo da janela) por >= min_dur dentro de [start, end]."""
    m = (t >= start) & (t <= end) & np.isfinite(v)
    tt, vv = t[m], v[m]
    best = -math.inf
    for i in range(len(tt)):
        j = np.searchsorted(tt, tt[i] + min_dur)
        if j < len(tt):
            best = max(best, float(vv[i: j + 1].min()))
    return best


def rep_features(vid: str, min_dur: float) -> list[dict]:
    seg = pd.read_csv(SEG, sep=";")
    run = json.loads((RUNS / f"physio_{vid}-Camera18-30fps-transposed.json").read_text(encoding="utf-8"))
    t = np.array(run["series"]["t"], float)
    trunk = np.array([np.nan if x is None else x for x in run["series"]["trunk"]], float)
    det = run["repetitions"]
    rows = []
    for r in seg[seg.video_id == vid].sort_values("first_frame").itertuples():
        if r.cam17_orientation != "front":  # a câmera 18 vê de perfil quando a pessoa está de frente para a 17
            continue
        span = (r.first_frame / FPS, r.last_frame / FPS)
        match = max(det, key=lambda d: iou((d["start_s"], d["end_s"]), span), default=None)
        matched = match is not None and iou((match["start_s"], match["end_s"]), span) >= 0.3
        rows.append({
            "video": vid, "rep": int(r.repetition_number), "incorrect": int(r.correctness) == 0,
            "peak": match["peak_angle"] if matched else math.nan,
            "trunk": sustained_max(t, trunk, *span, min_dur) if matched else math.nan,
        })
    return rows


def score(rows: list[dict], a: float, t: float) -> dict:
    c = {"vp": 0, "fp": 0, "fn": 0, "vn": 0}
    for r in rows:
        flagged = (np.isfinite(r["peak"]) and r["peak"] > a) or (np.isfinite(r["trunk"]) and r["trunk"] > t)
        c["vp" if flagged and r["incorrect"] else "fp" if flagged else "fn" if r["incorrect"] else "vn"] += 1
    p = c["vp"] / (c["vp"] + c["fp"]) if c["vp"] + c["fp"] else None
    rc = c["vp"] / (c["vp"] + c["fn"]) if c["vp"] + c["fn"] else None
    f = 2 * p * rc / (p + rc) if p and rc else 0.0
    return {**c, "precision": p, "recall": rc, "f1": f}


def grid_search(rows: list[dict]) -> tuple[float, float, dict]:
    best = None
    for a in GRID_A:
        for t in GRID_T:
            s = score(rows, a, t)
            key = (round(s["f1"], 6), a, t)  # empate: mais conservador
            if best is None or key > best[0]:
                best = (key, a, t, s)
    assert best is not None
    return best[1], best[2], best[3]


def fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float) and math.isinf(v):
        return "desligada"
    if isinstance(v, float):
        return f"{v:.2f}".replace(".", ",")
    return str(v)


def row_md(label: str, sc: dict) -> str:
    return (f"| {label} | {sc['vp']} | {sc['fp']} | {sc['fn']} | {sc['vn']} | "
            f"{fmt(sc['precision'])} | {fmt(sc['recall'])} | {fmt(sc['f1'])} |")


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Calibra os limiares de perfil com o REHAB24-6.")
    ap.add_argument("--min-duration", type=float, default=None, help="padrão: o do config/video.yaml")
    args = ap.parse_args(argv)

    cfg = load_video_config()
    rules = cfg.physio.rules
    min_dur = args.min_duration or rules.postura_fora_padrao.min_duration_s
    current = (float(rules.amplitude_incompleta.expected[1]), float(rules.postura_fora_padrao.expected[1]))
    vids = CALIB + TEST
    rows = {v: rep_features(v, min_dur) for v in vids}
    calib = [r for v in CALIB for r in rows[v]]
    test = [r for v in TEST for r in rows[v]]
    every = calib + test

    a_split, t_split, _ = grid_search(calib)
    lopo = []
    for held in vids:
        a, t, _ = grid_search([r for v in vids if v != held for r in rows[v]])
        lopo.append({"held_out": held, "A": a, "T": t, **score(rows[held], a, t)})

    res = {
        "min_duration_s": min_dur,
        "split": {"calib": CALIB, "test": TEST, "chosen": [a_split, t_split],
                  "test_score": score(test, a_split, t_split)},
        "leave_one_out": lopo,
        "config": {"A": current[0], "T": current[1],
                   "calib": score(calib, *current), "test": score(test, *current), "all": score(every, *current)},
        "before": {"A": BEFORE[0], "T": BEFORE[1],
                   "calib": score(calib, *BEFORE), "test": score(test, *BEFORE), "all": score(every, *BEFORE)},
        "peaks_correct": sorted(round(r["peak"]) for r in every if not r["incorrect"] and np.isfinite(r["peak"])),
        "peaks_incorrect": sorted(round(r["peak"]) for r in every if r["incorrect"] and np.isfinite(r["peak"])),
        "trunk_correct": sorted(round(r["trunk"]) for r in every if not r["incorrect"] and np.isfinite(r["trunk"])),
        "trunk_incorrect": sorted(round(r["trunk"]) for r in every if r["incorrect"] and np.isfinite(r["trunk"])),
    }

    hdr = ["| Conjunto e limiares | VP | FP | FN | VN | Precisão | Recall | F1 |", "|---|---|---|---|---|---|---|---|"]
    lines = [
        f"Repetições de perfil: {len(calib)} na calibração ({sum(r['incorrect'] for r in calib)} incorretas), "
        f"{len(test)} no teste ({sum(r['incorrect'] for r in test)} incorretas). "
        f"Postura: {fmt(float(min_dur))} s fora da faixa.",
        "",
        f"1. Separação por pessoa: a grade escolhe joelho {fmt(float(a_split))}, tronco {fmt(float(t_split))}° "
        f"(F1 no teste {fmt(res['split']['test_score']['f1'])}).",
        "",
        "2. Deixando uma pessoa de fora:",
        "",
        "| Pessoa de fora | Joelho | Tronco | VP | FP | FN | VN |",
        "|---|---|---|---|---|---|---|",
    ]
    for x in lopo:
        lines.append(f"| {x['held_out']} | {fmt(float(x['A']))} | {fmt(float(x['T']))} | {x['vp']} | {x['fp']} | {x['fn']} | {x['vn']} |")
    lines.append("")
    lines.append(f"Tronco mais escolhido: {Counter(x['T'] for x in lopo).most_common(1)[0][0]}°.")
    lines += ["", f"3. Limiares do config ({fmt(current[0])}°, {fmt(current[1])}°) × antes da calibração (100°, 45°):", ""] + hdr
    for name, key in (("calibração", "calib"), ("teste", "test"), ("todos", "all")):
        lines.append(row_md(f"{name}, antes", res["before"][key]))
        lines.append(row_md(f"{name}, config", res["config"][key]))
    lines += ["", f"Pico do joelho (°), corretas: {res['peaks_correct']}",
              f"Pico do joelho (°), incorretas: {res['peaks_incorrect']}",
              f"Tronco mantido (°), corretas: {res['trunk_correct']}",
              f"Tronco mantido (°), incorretas: {res['trunk_incorrect']}"]
    md = "\n".join(lines) + "\n"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "limiares_perfil.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (OUT / "limiares_perfil.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())

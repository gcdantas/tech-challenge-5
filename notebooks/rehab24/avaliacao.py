"""Compara as execuções do módulo de vídeo com as anotações do REHAB24-6.

    python -m notebooks.rehab24.avaliacao [--runs outputs/rehab24-6/video/P001/runs]

O REHAB24-6 anota cada repetição (quadro inicial e final, a 30 FPS) e se ela foi
feita corretamente segundo um fisioterapeuta. Não anota o TIPO do erro, então a
avaliação é por repetição:

- contagem: repetição detectada × anotada, casadas 1 para 1 por IoU temporal >= 0,3;
- sinalização: uma repetição anotada conta como "sinalizada" se algum desvio do
  módulo cair nela (contido nela, com 0,25 s de tolerância, ou IoU >= 0,3).
  Mede precisão, recall e F1 da classe "repetição incorreta".

A vista da câmera vem da coluna cam17_orientation: com a pessoa de frente para a
câmera 17, a câmera 18 a vê de perfil; em meio-perfil, as duas veem meio-perfil.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SEG = ROOT / "data" / "video" / "rehab24-6" / "Segmentation.csv"
RUNS = ROOT / "outputs" / "rehab24-6" / "video" / "P001" / "runs"
FPS = 30.0
VIEW = {
    ("Camera18", "front"): "perfil", ("Camera18", "half-profile"): "meio-perfil",
    ("Camera18", "profile"): "frente", ("Camera17", "front"): "frente",
    ("Camera17", "half-profile"): "meio-perfil", ("Camera17", "profile"): "perfil",
}


def iou(a, b) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def hits(ev, rep) -> bool:
    return (rep[0] - 0.25 <= ev[0] and ev[1] <= rep[1] + 0.25) or iou(ev, rep) >= 0.3


def match_count(det, gt, thr=0.3) -> int:
    pairs = sorted(((iou(a, b), i, j) for i, a in enumerate(det) for j, b in enumerate(gt)), reverse=True)
    ui, uj = set(), set()
    for v, i, j in pairs:
        if v >= thr and i not in ui and j not in uj:
            ui.add(i)
            uj.add(j)
    return len(ui)


def prf(c: dict) -> tuple[str, str, str]:
    p = c["vp"] / (c["vp"] + c["fp"]) if c["vp"] + c["fp"] else None
    r = c["vp"] / (c["vp"] + c["fn"]) if c["vp"] + c["fn"] else None
    f = 2 * p * r / (p + r) if p and r else (0.0 if p is not None and r is not None else None)
    fmt = lambda v: "—" if v is None else f"{v:.2f}".replace(".", ",")  # noqa: E731
    return fmt(p), fmt(r), fmt(f)


def evaluate(runs_dir: Path) -> dict:
    seg = pd.read_csv(SEG, sep=";")
    videos, by_view, total = [], defaultdict(lambda: defaultdict(int)), defaultdict(int)
    perf, openpose = [], []
    for path in sorted(runs_dir.glob("*.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        stem = run["video_stem"]
        vid, cam = stem.split("-")[0], stem.split("-")[1]
        if run["scenario"] == "openpose":
            openpose.append({"video": vid, "frames": run["processing"]["processed_frames"],
                             "fps": run["mean_fps"], "missing": run["missing_keypoint_ratio"],
                             "failed": run["failed_frames"], "crop": bool(run.get("person_crop"))})
            continue
        if run["scenario"] != "physio":
            continue
        g = seg[seg.video_id == vid].sort_values("repetition_number")
        gt = [((r.first_frame / FPS, r.last_frame / FPS), int(r.correctness), VIEW[(cam, r.cam17_orientation)])
              for r in g.itertuples()]
        det = [(r["start_s"], r["end_s"]) for r in run["repetitions"]]
        events = [(e["t_start"], e["t_end"]) for e in run["events"]]
        c = defaultdict(int)
        for span, correct, view in gt:
            flagged = any(hits(ev, span) for ev in events)
            key = ("vp" if not correct else "fp") if flagged else ("fn" if not correct else "vn")
            c[key] += 1
            total_key = run["exercise"]
            by_view[(total_key, view)][key] += 1
            total[(total_key, key)] += 1
        types = defaultdict(int)
        for e in run["events"]:
            types[e["type"]] += 1
        videos.append({
            "video": vid, "exercise": run["exercise"], "cam": cam,
            "views": sorted({v for _, _, v in gt}),
            "duration": run["video_info"]["duration_s"],
            "reps_det": len(det), "reps_gt": len(gt), "reps_match": match_count(det, [s for s, _, _ in gt]),
            "incorrect": sum(1 for _, k, _ in gt if not k), "events": dict(types), **c,
        })
        proc = run["processing"]
        perf.append({"video": vid, "inference_fps": proc["inference_fps"], "end_to_end_fps": proc["end_to_end_fps"],
                     "pose": run["frames_with_pose"]})
    return {"videos": videos, "by_view": by_view, "total": total, "perf": perf, "openpose": openpose}


def to_markdown(res: dict) -> str:
    out = ["| Vídeo | Exercício | Vista | Duração | Repetições (detectadas / anotadas / casadas) | Incorretas anotadas | Desvios sinalizados |",
           "|---|---|---|---|---|---|---|"]
    for v in res["videos"]:
        ev = ", ".join(f"{k} {n}" for k, n in sorted(v["events"].items())) or "nenhum"
        out.append(f"| {v['video']} | {v['exercise']} | {' e '.join(v['views'])} | {v['duration']:.0f} s | "
                   f"{v['reps_det']} / {v['reps_gt']} / {v['reps_match']} | {v['incorrect']} | {ev} |")
    out += ["", "| Exercício | Vista | VP | FP | FN | VN | Precisão | Recall | F1 |", "|---|---|---|---|---|---|---|---|---|"]
    exercises = sorted({ex for ex, _ in res["by_view"]})
    for ex in exercises:
        for (e, view), c in sorted(res["by_view"].items()):
            if e == ex:
                out.append(f"| {ex} | {view} | {c['vp']} | {c['fp']} | {c['fn']} | {c['vn']} | " + " | ".join(prf(c)) + " |")
        tot = {k: res["total"][(ex, k)] for k in ("vp", "fp", "fn", "vn")}
        out.append(f"| {ex} | **todas** | {tot['vp']} | {tot['fp']} | {tot['fn']} | {tot['vn']} | " + " | ".join(prf(tot)) + " |")
    out += ["", "| Vídeo | Quadros com pose | FPS de inferência | FPS ponta a ponta |", "|---|---|---|---|"]
    for p in res["perf"]:
        out.append(f"| {p['video']} | {p['pose']:.0%} | {p['inference_fps']} | {p['end_to_end_fps']} |")
    for o in res["openpose"]:
        out.append(f"\nOpenPose em {o['video']}: {o['frames']} quadros, {o['fps']} FPS, "
                   f"{o['missing']:.0%} de keypoints ausentes, {o['failed']} quadro(s) sem pose, recorte da pessoa: {o['crop']}.")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", default=str(RUNS))
    args = ap.parse_args(argv)
    if not SEG.exists():
        print("erro: rode antes: python -m data.download_rehab24", file=sys.stderr)
        return 2
    print(to_markdown(evaluate(Path(args.runs))))
    return 0


if __name__ == "__main__":
    sys.exit(main())

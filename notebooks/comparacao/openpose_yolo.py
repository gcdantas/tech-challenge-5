"""Comparação OpenPose × YOLOv8-pose nos mesmos quadros, com o gabarito do REHAB24-6.

    python -m data.download_rehab24 --gabarito
    python -m notebooks.comparacao.openpose_yolo [--videos PM_105 PM_113 PM_118]

Para cada vídeo de agachamento (câmera 18):
- ``--spread`` quadros espalhados pelo trecho das repetições anotadas;
- ``--static`` quadros SEGUIDOS do trecho em que o gabarito 3D mostra a pessoa
  mais parada (para medir a variação de um ponto estático).

Critérios do plano: FPS médio, proporção de keypoints ausentes, variação quadro a
quadro de um ponto estático e falhas completas de pose. Com o gabarito da
captura de movimento, também: erro em pixels por articulação e erro do ângulo do
joelho em relação ao ângulo real em 3D.

Condições: os dois modelos rodam em CPU, nos mesmos quadros; o OpenPose recebe o
recorte da pessoa dado pelo YOLOv8 (como no módulo) e seu tempo conta só a
inferência; cada modelo roda uma vez antes da medição (aquecimento).
O gabarito 2D é a projeção da captura de movimento com câmera estimada pelos
autores do dataset; tem erro próprio, então serve para comparar os modelos
entre si, não como precisão absoluta.

Saídas em outputs/comparacao/: openpose_yolo.json, openpose_yolo.md e um quadro
por vídeo com os dois esqueletos e o gabarito.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from modules.video.annotate import ORANGE, draw_pose, put_lines
from modules.video.config import load_video_config
from modules.video.geometry import SKELETON, angle_at

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "video" / "rehab24-6"
OUT = ROOT / "outputs" / "comparacao"
FPS_DATASET = 30.0

# Keypoint COCO-17 -> articulação do esqueleto do REHAB24-6 (joints_names.txt).
# Conferido no PM_105: cada par é o de menor distância entre YOLOv8 e gabarito.
COCO_TO_GT = {
    5: 7, 6: 12,     # ombros       <- LeftArm, RightArm
    7: 8, 8: 13,     # cotovelos    <- LeftForeArm, RightForeArm
    9: 9, 10: 14,    # punhos       <- LeftHand, RightHand
    11: 16, 12: 21,  # quadris      <- LeftUpLeg, RightUpLeg
    13: 17, 14: 22,  # joelhos      <- LeftLeg, RightLeg
    15: 18, 16: 23,  # tornozelos   <- LeftFoot, RightFoot
}
JOINT_PT = {5: "ombro E", 6: "ombro D", 7: "cotovelo E", 8: "cotovelo D", 9: "punho E", 10: "punho D",
            11: "quadril E", 12: "quadril D", 13: "joelho E", 14: "joelho D", 15: "tornozelo E", 16: "tornozelo D"}
KNEES = {"esquerdo": (11, 13, 15), "direito": (12, 14, 16)}  # quadril, joelho, tornozelo (COCO-17)
STATIC_POINTS = [11, 12, 13, 14, 15, 16]                     # quadris, joelhos, tornozelos
VIEW = {"front": "perfil", "half-profile": "meio-perfil", "profile": "frente"}  # vista na câmera 18


# --------------------------------------------------------------------------- funções puras


def angle_3d(a: np.ndarray, vertex: np.ndarray, b: np.ndarray) -> float:
    v1, v2 = a - vertex, b - vertex
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return math.degrees(math.acos(float(np.clip(np.dot(v1, v2) / n, -1, 1)))) if n > 0 else math.nan


def gt_knee_angle_3d(j3: np.ndarray, side: str) -> float:
    """Ângulo real do joelho (graus) a partir das articulações 3D de um quadro."""
    hip, knee, ankle = (COCO_TO_GT[i] for i in KNEES[side])
    return angle_3d(j3[hip, :3], j3[knee, :3], j3[ankle, :3])


def static_window(j3_all: np.ndarray, n: int, lo: int = 0, hi: int | None = None) -> int:
    """Início da janela de ``n`` quadros seguidos com menor movimento médio no gabarito 3D."""
    hi = len(j3_all) if hi is None else hi
    speed = np.linalg.norm(np.diff(j3_all[lo:hi, :, :3], axis=0), axis=2).mean(axis=1)  # m/quadro
    if len(speed) < n:
        return lo
    csum = np.concatenate([[0.0], np.cumsum(speed)])
    window = csum[n - 1:] - csum[: len(csum) - n + 1]
    return lo + int(np.argmin(window))


def frame_to_frame(points: np.ndarray) -> float:
    """Deslocamento médio (px) entre quadros seguidos; ``points`` é (quadros, pontos, 2) com NaN."""
    d = np.linalg.norm(np.diff(points, axis=0), axis=2)
    return float(np.nanmean(d)) if np.isfinite(d).any() else math.nan


def view_of(frame_idx: int, reps: pd.DataFrame) -> str:
    """Vista da pessoa na câmera 18 no quadro: a da repetição anotada mais próxima."""
    mid = (reps["first_frame"] + reps["last_frame"]) / 2
    row = reps.iloc[int(np.argmin(np.abs(mid - frame_idx)))]
    return VIEW[row["cam17_orientation"]]


# --------------------------------------------------------------------------- execução


def read_frames(path: Path, indices: list[int]) -> dict[int, np.ndarray]:
    wanted, out = set(indices), {}
    cap = cv2.VideoCapture(str(path))
    idx = 0
    try:
        while len(out) < len(wanted) and cap.grab():
            if idx in wanted:
                ok, frame = cap.retrieve()
                if ok:
                    out[idx] = frame
            idx += 1
    finally:
        cap.release()
    return out


def run_models(frames: dict[int, np.ndarray], yolo, openpose, kp_conf: float) -> dict[str, dict[int, dict]]:
    res: dict[str, dict[int, dict]] = {"yolo": {}, "openpose": {}}
    for idx, frame in sorted(frames.items()):
        t0 = time.perf_counter()
        det = yolo.predict(frame)
        t_yolo = time.perf_counter() - t0
        if det is None:
            kp, conf = np.full((17, 2), np.nan), np.zeros(17)
        else:
            kp = det.keypoints.astype(float).copy()
            conf = det.conf.astype(float)
            kp[conf < kp_conf] = np.nan
        res["yolo"][idx] = {"kp": kp, "time": t_yolo}

        op = openpose.predict(frame, det.box if det is not None else None)
        kp_op, _ = op.to_coco17()
        res["openpose"][idx] = {"kp": kp_op.astype(float), "time": op.inference_s}
    return res


def evaluate_video(vid: str, args, yolo, openpose, kp_conf: float, seg: pd.DataFrame) -> dict:
    video = DATA / f"{vid}-Camera18-30fps-transposed.mp4"
    j2 = np.load(DATA / "joints2d" / f"{vid}-c18-30fps.npy", allow_pickle=False)
    j3 = np.load(DATA / "joints3d" / f"{vid}-30fps.npy", allow_pickle=False)
    reps = seg[seg.video_id == vid].sort_values("first_frame")
    first, last = int(reps.first_frame.min()), int(reps.last_frame.max())

    spread = sorted({int(round(i)) for i in np.linspace(first, last, args.spread)})
    s0 = static_window(j3, args.static)
    static = list(range(s0, s0 + args.static))
    frames = read_frames(video, spread + static)
    res = run_models(frames, yolo, openpose, kp_conf)

    out: dict = {"video": vid, "spread_frames": len(spread), "static_frames": static[0:1] + static[-1:],
                 "static_gt_motion_mm": round(float(np.linalg.norm(np.diff(j3[static, :, :3], axis=0), axis=2).mean()) * 1000, 2),
                 "models": {}}
    gt_static = j2[static][:, [COCO_TO_GT[i] for i in STATIC_POINTS]]
    out["gt_static_jitter_px"] = round(frame_to_frame(gt_static), 2)

    for model, per in res.items():
        idxs = sorted(per)
        kps = np.stack([per[i]["kp"] for i in idxs])            # (n, 17, 2)
        valid = np.isfinite(kps[..., 0])
        times = [per[i]["time"] for i in idxs]
        m = {
            "fps": round(len(idxs) / sum(times), 2),
            "missing_ratio": round(1 - valid.sum() / valid.size, 4),
            "failed_frames": int((valid.sum(axis=1) == 0).sum()),
            "frames": len(idxs),
        }
        # ponto estático
        st = np.stack([per[i]["kp"][STATIC_POINTS] for i in static if i in per])
        m["static_jitter_px"] = round(frame_to_frame(st), 2)

        # erro em pixels contra o gabarito 2D (quadros espalhados)
        errs, per_joint, rel = [], {k: [] for k in COCO_TO_GT}, []
        knee = {"perfil": [], "meio-perfil": []}
        knee_gt2d = {"perfil": [], "meio-perfil": []}
        for i in spread:
            if i not in per:
                continue
            g = j2[i]
            torso = np.linalg.norm((g[7] + g[12]) / 2 - (g[16] + g[21]) / 2)
            for c, gi in COCO_TO_GT.items():
                p = per[i]["kp"][c]
                if np.all(np.isfinite(p)) and np.all(np.isfinite(g[gi])):
                    e = float(np.linalg.norm(p - g[gi]))
                    errs.append(e)
                    per_joint[c].append(e)
                    rel.append(e / torso if torso > 0 else math.nan)
            view = view_of(i, reps)
            for side, (h, k, a) in KNEES.items():
                true = gt_knee_angle_3d(j3[i], side)
                p = per[i]["kp"]
                if np.all(np.isfinite(p[[h, k, a]])):
                    knee[view].append(abs(angle_at(p[h], p[k], p[a]) - true))
                gh, gk, ga = (j2[i, COCO_TO_GT[x]] for x in (h, k, a))
                knee_gt2d[view].append(abs(angle_at(gh, gk, ga) - true))
        m["gt_error_px_mean"] = round(float(np.mean(errs)), 1) if errs else None
        m["gt_error_px_median"] = round(float(np.median(errs)), 1) if errs else None
        m["gt_error_torso_pct"] = round(float(np.nanmean(rel)) * 100, 1) if rel else None
        m["gt_error_by_joint_px"] = {JOINT_PT[c]: round(float(np.mean(v)), 1) for c, v in per_joint.items() if v}
        m["knee_angle_error_deg"] = {v: (round(float(np.mean(x)), 1) if x else None) for v, x in knee.items()}
        m["knee_angle_samples"] = {v: len(x) for v, x in knee.items()}
        out["models"][model] = m
    out["gt2d_knee_angle_error_deg"] = {v: (round(float(np.mean(x)), 1) if x else None) for v, x in knee_gt2d.items()}

    # quadro ilustrativo: o mais fundo do agachamento entre os espalhados
    deepest = min((i for i in spread if i in frames), key=lambda i: gt_knee_angle_3d(j3[i], "esquerdo"))
    canvas = frames[deepest].copy()
    ones = np.ones(17)
    for kp, color in ((res["yolo"][deepest]["kp"], (80, 200, 80)), (res["openpose"][deepest]["kp"], ORANGE)):
        draw_pose(canvas, np.nan_to_num(kp, nan=-1.0), np.where(np.isfinite(kp[:, 0]), ones, 0), 0.5, color, SKELETON)
    for gi in COCO_TO_GT.values():
        x, y = j2[deepest, gi]
        cv2.circle(canvas, (int(x), int(y)), 9, (0, 255, 255), -1)
    put_lines(canvas, [f"{vid} quadro {deepest}", "verde: YOLOv8-pose", "laranja: OpenPose", "amarelo: gabarito"])
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / f"{vid}_comparacao.jpg"), cv2.resize(canvas, None, fx=0.5, fy=0.5))
    return out


def summarize(results: list[dict]) -> dict:
    total: dict = {}
    for model in ("yolo", "openpose"):
        ms = [r["models"][model] for r in results]
        frames = sum(m["frames"] for m in ms)
        w = lambda key: sum(m[key] * m["frames"] for m in ms) / frames  # noqa: E731
        knee = {}
        for v in ("perfil", "meio-perfil"):
            n = sum(m["knee_angle_samples"][v] for m in ms)
            knee[v] = round(sum((m["knee_angle_error_deg"][v] or 0) * m["knee_angle_samples"][v] for m in ms) / n, 1) if n else None
        total[model] = {
            "fps": round(frames / sum(m["frames"] / m["fps"] for m in ms), 2),
            "missing_ratio": round(w("missing_ratio"), 4),
            "failed_frames": sum(m["failed_frames"] for m in ms),
            "static_jitter_px": round(float(np.nanmean([m["static_jitter_px"] for m in ms])), 2),
            "gt_error_px_mean": round(float(np.mean([m["gt_error_px_mean"] for m in ms])), 1),
            "gt_error_torso_pct": round(float(np.mean([m["gt_error_torso_pct"] for m in ms])), 1),
            "knee_angle_error_deg": knee,
            "frames": frames,
        }
    return total


def to_markdown(results: list[dict], total: dict) -> str:
    y, o = total["yolo"], total["openpose"]
    pct = lambda v: f"{v * 100:.1f}%".replace(".", ",")  # noqa: E731
    fmt = lambda v: "—" if v is None else f"{v}".replace(".", ",")  # noqa: E731
    gt_knee = {v: np.mean([r["gt2d_knee_angle_error_deg"][v] for r in results if r["gt2d_knee_angle_error_deg"][v] is not None])
               for v in ("perfil", "meio-perfil")}
    lines = [
        f"Vídeos: {', '.join(r['video'] for r in results)} (REHAB24-6, câmera 18); {y['frames']} quadros por modelo.",
        "",
        "| Critério | OpenPose (OpenCV DNN, CPU) | YOLOv8n-pose (CPU) |",
        "|---|---|---|",
        f"| FPS médio (só inferência) | {fmt(o['fps'])} | {fmt(y['fps'])} |",
        f"| Keypoints ausentes (17 por quadro) | {pct(o['missing_ratio'])} | {pct(y['missing_ratio'])} |",
        f"| Variação de ponto estático (px/quadro; gabarito: {fmt(round(float(np.mean([r['gt_static_jitter_px'] for r in results])), 2))}) | {fmt(o['static_jitter_px'])} | {fmt(y['static_jitter_px'])} |",
        f"| Falhas completas de pose | {o['failed_frames']} | {y['failed_frames']} |",
        f"| Erro médio contra o gabarito 2D (px) | {fmt(o['gt_error_px_mean'])} | {fmt(y['gt_error_px_mean'])} |",
        f"| Erro médio contra o gabarito 2D (% do tronco) | {fmt(o['gt_error_torso_pct'])}% | {fmt(y['gt_error_torso_pct'])}% |",
        f"| Erro do ângulo do joelho × 3D real, perfil (°) | {fmt(o['knee_angle_error_deg']['perfil'])} | {fmt(y['knee_angle_error_deg']['perfil'])} |",
        f"| Erro do ângulo do joelho × 3D real, meio-perfil (°) | {fmt(o['knee_angle_error_deg']['meio-perfil'])} | {fmt(y['knee_angle_error_deg']['meio-perfil'])} |",
        "",
        f"Referência: o próprio gabarito 2D, comparado com o ângulo 3D, erra {fmt(round(gt_knee['perfil'], 1))}° de perfil e "
        f"{fmt(round(gt_knee['meio-perfil'], 1))}° em meio-perfil (efeito da projeção em 2D).",
        "",
        "Erro médio por articulação (px):",
        "",
        "| Articulação | OpenPose | YOLOv8n-pose |",
        "|---|---|---|",
    ]
    joints = list(JOINT_PT.values())
    for j in joints:
        ov = np.mean([r["models"]["openpose"]["gt_error_by_joint_px"].get(j, np.nan) for r in results])
        yv = np.mean([r["models"]["yolo"]["gt_error_by_joint_px"].get(j, np.nan) for r in results])
        lines.append(f"| {j} | {fmt(round(float(ov), 1))} | {fmt(round(float(yv), 1))} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Comparação OpenPose × YOLOv8-pose com gabarito (REHAB24-6).")
    ap.add_argument("--videos", nargs="+", default=["PM_105", "PM_113", "PM_118"])
    ap.add_argument("--spread", type=int, default=30, help="quadros espalhados por vídeo")
    ap.add_argument("--static", type=int, default=30, help="quadros seguidos no trecho parado")
    args = ap.parse_args(argv)

    for vid in args.videos:
        for p in (DATA / f"{vid}-Camera18-30fps-transposed.mp4", DATA / "joints2d" / f"{vid}-c18-30fps.npy",
                  DATA / "joints3d" / f"{vid}-30fps.npy"):
            if not p.exists():
                print(f"erro: falta {p}. Rode: python -m data.download_rehab24 --gabarito", file=sys.stderr)
                return 2

    from modules.video.openpose import OpenPoseCOCO
    from modules.video.pose import YoloPoseEstimator

    cfg = load_video_config()
    yolo = YoloPoseEstimator(cfg.resolve(cfg.models.pose), cfg.models.device, cfg.models.imgsz, cfg.models.conf)
    openpose = OpenPoseCOCO.from_config(cfg)
    warm = np.zeros((480, 640, 3), np.uint8)
    yolo.predict(warm)
    openpose.predict(warm)

    seg = pd.read_csv(DATA / "Segmentation.csv", sep=";")
    results = []
    for vid in args.videos:
        print(f"{vid}...", flush=True)
        results.append(evaluate_video(vid, args, yolo, openpose, cfg.models.min_keypoint_conf, seg))
    total = summarize(results)
    md = to_markdown(results, total)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "openpose_yolo.json").write_text(json.dumps({"videos": results, "total": total}, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    (OUT / "openpose_yolo.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"Saídas em {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

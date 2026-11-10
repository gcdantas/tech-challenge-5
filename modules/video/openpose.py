"""OpenPose (modelo COCO, Caffe) pelo módulo DNN do OpenCV.

Requisito obrigatório do enunciado: rodar o OpenPose em pelo menos um vídeo de
fisioterapia e guardar os quadros com o esqueleto anotado.

    python -m modules.video.openpose --input data/video/rehab24-6/PM_105-Camera18-30fps-transposed.mp4 --patient P001 [--max-frames 20]

Abordagem de pessoa única (máximo de cada mapa de calor), adequada à sessão de
fisioterapia com um paciente em cena. O build do OpenCV do PyPI roda o DNN em
CPU; por isso o OpenPose processa só ``max_frames`` quadros espalhados pelo
vídeo. A comparação completa OpenPose × YOLOv8-pose é desejável (Integrante 1);
os keypoints ficam salvos em runs/openpose_<video>.json para ela.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from contracts.patients import DEFAULT_PATIENTS_PATH, load_patients
from modules.video.annotate import ORANGE, draw_pose, put_lines
from modules.video.config import DEFAULT_VIDEO_CONFIG, OpenPoseCfg, VideoConfig, load_video_config
from modules.video.timebase import MediaContext, artifact_ref, resolve_context
from modules.video.videoio import probe, save_jpeg

log = logging.getLogger(__name__)

# Ordem dos 18 keypoints do modelo COCO do OpenPose.
OPENPOSE_KEYPOINTS = [
    "nose", "neck", "right_shoulder", "right_elbow", "right_wrist", "left_shoulder",
    "left_elbow", "left_wrist", "right_hip", "right_knee", "right_ankle", "left_hip",
    "left_knee", "left_ankle", "right_eye", "left_eye", "right_ear", "left_ear",
]
OPENPOSE_PAIRS = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10),
    (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17),
]
# Índice OpenPose-18 de cada keypoint COCO-17 (mesma ordem do YOLOv8-pose).
TO_COCO17 = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]


@dataclass
class OpenPoseResult:
    keypoints: np.ndarray  # (18, 2) pixels (NaN quando ausente)
    conf: np.ndarray  # (18,) probabilidade do mapa de calor (0 quando ausente)
    inference_s: float

    @property
    def found(self) -> int:
        return int(np.isfinite(self.keypoints[:, 0]).sum())

    def to_coco17(self) -> tuple[np.ndarray, np.ndarray]:
        return self.keypoints[TO_COCO17], self.conf[TO_COCO17]


class OpenPoseCOCO:
    name = "openpose-coco"
    version = "pose_iter_440000"

    def __init__(self, prototxt: str | Path, caffemodel: str | Path, input_size: int = 368, threshold: float = 0.1):
        for p in (prototxt, caffemodel):
            if not Path(p).exists():
                raise FileNotFoundError(f"pesos do OpenPose não encontrados: {p}. Rode: python -m data.download_weights --only openpose")
        self.net = cv2.dnn.readNetFromCaffe(str(prototxt), str(caffemodel))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.input_size = input_size
        self.threshold = threshold
        self.backend = "OpenCV DNN (CPU)"

    @classmethod
    def from_config(cls, cfg: VideoConfig) -> OpenPoseCOCO:
        op: OpenPoseCfg = cfg.openpose
        return cls(cfg.resolve(op.prototxt), cfg.resolve(op.caffemodel), op.input_size, op.heatmap_threshold)

    def predict(self, frame_bgr: np.ndarray, box: np.ndarray | None = None, margin: float = 0.15) -> OpenPoseResult:
        """Pose de uma pessoa. Com ``box`` (x1 y1 x2 y2), roda só no recorte da pessoa
        (com margem) e devolve os pontos nas coordenadas do quadro inteiro."""
        if box is None:
            return self._predict(frame_bgr)
        H, W = frame_bgr.shape[:2]
        x1, y1, x2, y2 = (float(v) for v in box)
        mx, my = (x2 - x1) * margin, (y2 - y1) * margin
        cx1, cy1 = int(max(0, x1 - mx)), int(max(0, y1 - my))
        cx2, cy2 = int(min(W, x2 + mx)), int(min(H, y2 + my))
        if cx2 - cx1 < 16 or cy2 - cy1 < 16:
            return self._predict(frame_bgr)
        res = self._predict(frame_bgr[cy1:cy2, cx1:cx2])
        res.keypoints = res.keypoints + np.array([cx1, cy1])
        return res

    def _predict(self, frame_bgr: np.ndarray) -> OpenPoseResult:
        h, w = frame_bgr.shape[:2]
        in_h = self.input_size
        in_w = max(8, int(round(in_h * w / h / 8)) * 8)
        blob = cv2.dnn.blobFromImage(frame_bgr, 1.0 / 255, (in_w, in_h), (0, 0, 0), swapRB=False, crop=False)
        t0 = time.perf_counter()
        self.net.setInput(blob)
        out = self.net.forward()
        dt = time.perf_counter() - t0
        oh, ow = out.shape[2], out.shape[3]
        kps = np.full((18, 2), np.nan)
        conf = np.zeros(18)
        for i in range(18):
            heat = out[0, i]
            _, prob, _, (px, py) = cv2.minMaxLoc(heat)
            if prob > self.threshold:
                kps[i] = (w * px / ow, h * py / oh)
                conf[i] = float(prob)
        return OpenPoseResult(kps, conf, dt)


def annotate_openpose(frame: np.ndarray, res: OpenPoseResult, t: float) -> np.ndarray:
    canvas = frame.copy()
    kp = np.nan_to_num(res.keypoints, nan=-1.0)
    conf = np.where(np.isfinite(res.keypoints[:, 0]), 1.0, 0.0)
    draw_pose(canvas, kp, conf, min_conf=0.5, color=ORANGE, skeleton=OPENPOSE_PAIRS)
    put_lines(canvas, [f"OpenPose COCO (OpenCV DNN)", f"t = {t:.1f} s | {res.found}/18 keypoints"])
    return canvas


def _default_person_finder(cfg: VideoConfig):
    """Pessoa principal pelo YOLOv8-pose (mesmo critério do pipeline), se os pesos existirem."""
    weights = cfg.resolve(cfg.models.pose)
    if not weights.exists():
        log.warning("sem %s: OpenPose roda no quadro inteiro (pode misturar pessoas)", weights)
        return None
    from modules.video.pose import YoloPoseEstimator

    m = cfg.models
    return YoloPoseEstimator(weights, m.device, m.imgsz, m.conf)


def run_openpose(
    ctx: MediaContext,
    cfg: VideoConfig,
    outputs_root: Path,
    max_frames: int | None = None,
    model: OpenPoseCOCO | None = None,
    crop_person: bool = True,
) -> dict:
    """Roda o OpenPose em ``max_frames`` quadros espalhados pelo vídeo.

    Com ``crop_person`` (padrão), o YOLOv8 localiza a pessoa principal e o
    OpenPose roda só no recorte dela, evitando misturar paciente e fisioterapeuta.
    """
    model = model or OpenPoseCOCO.from_config(cfg)
    finder = _default_person_finder(cfg) if crop_person else None
    info = probe(ctx.input)
    n = max(1, min(max_frames or cfg.openpose.max_frames, info.frame_count or 1))
    wanted = sorted({int(round(i)) for i in np.linspace(0, max(0, info.frame_count - 1), n)})
    patient_dir = outputs_root / "video" / ctx.patient_id
    frames_dir = patient_dir / "frames" / ctx.stem

    frames = []
    cap = cv2.VideoCapture(str(ctx.input))
    try:
        idx, k = 0, 0
        while k < len(wanted):
            ok = cap.grab()
            if not ok:
                break
            if idx == wanted[k]:
                ok, frame = cap.retrieve()
                k += 1
                if ok:
                    t = idx / info.fps
                    person = finder.predict(frame) if finder is not None else None
                    res = model.predict(frame, person.box if person is not None else None)
                    img = save_jpeg(frames_dir / f"openpose_{idx:06d}.jpg", annotate_openpose(frame, res, t),
                                    cfg.output.jpeg_quality)
                    frames.append({
                        "frame_idx": idx,
                        "t": round(t, 3),
                        "timestamp": ctx.at(t).isoformat(),
                        "image": artifact_ref(img),
                        "keypoints_found": res.found,
                        "person_crop": person is not None,
                        "inference_s": round(res.inference_s, 4),
                        "keypoints": [
                            None if not np.isfinite(x) else [round(float(x), 1), round(float(y), 1), round(float(c), 3)]
                            for (x, y), c in zip(res.keypoints, res.conf)
                        ],
                    })
            idx += 1
    finally:
        cap.release()

    total_inf = sum(f["inference_s"] for f in frames)
    summary = {
        "scenario": "openpose",
        "patient_id": ctx.patient_id,
        "video": artifact_ref(ctx.input),
        "video_stem": ctx.stem,
        "artifact": ctx.artifact,
        "demo_offset": ctx.demo_offset,
        "demo_start": ctx.at(0).isoformat(),
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "model": {"name": model.name, "version": model.version, "input_size": model.input_size},
        "backend": model.backend,
        "person_crop": finder.name if finder is not None else None,
        "keypoint_names": OPENPOSE_KEYPOINTS,
        "processing": {
            "processed_frames": len(frames),
            "inference_seconds": round(total_inf, 3),
            "inference_fps": round(len(frames) / total_inf, 2) if total_inf > 0 else None,
            "end_to_end_fps": None,
        },
        "mean_fps": round(len(frames) / total_inf, 2) if total_inf > 0 else None,
        "missing_keypoint_ratio": round(1 - sum(f["keypoints_found"] for f in frames) / (18 * len(frames)), 3)
        if frames else 1.0,
        "failed_frames": sum(1 for f in frames if f["keypoints_found"] == 0),
        "frames": frames,
    }
    from modules.video.pipeline import write_run_summary

    write_run_summary(patient_dir, ctx.stem, "openpose", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    from modules.video import default_outputs_root
    from modules.video.report import render_report

    ap = argparse.ArgumentParser(description="OpenPose (OpenCV DNN) num vídeo de fisioterapia.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--patient", required=True)
    ap.add_argument("--max-frames", type=int, default=None, help="padrão: openpose.max_frames do config")
    ap.add_argument("--config", default=str(DEFAULT_VIDEO_CONFIG))
    ap.add_argument("--patients", default=str(DEFAULT_PATIENTS_PATH))
    ap.add_argument("--outputs", default=None)
    ap.add_argument("--no-crop", action="store_true", help="roda no quadro inteiro, sem recortar a pessoa principal")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not Path(args.input).exists():
        print(f"erro: vídeo não encontrado: {args.input}", file=sys.stderr)
        return 2
    cfg = load_video_config(args.config)
    root = Path(args.outputs) if args.outputs else default_outputs_root()
    try:
        ctx = resolve_context(args.input, args.patient, "physio", load_patients(args.patients))
        summary = run_openpose(ctx, cfg, root, args.max_frames, crop_person=not args.no_crop)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 2
    render_report(root / "video" / args.patient, args.patient)
    print(
        f"OpenPose: {summary['processing']['processed_frames']} quadros, {summary['mean_fps']} FPS, "
        f"{summary['missing_keypoint_ratio']:.0%} keypoints ausentes, {summary['failed_frames']} quadro(s) sem pose"
    )
    print(f"Quadros: {root / 'video' / args.patient / 'frames' / ctx.stem}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

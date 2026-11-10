"""Estimadores de pose e detectores de objetos (YOLOv8 via ultralytics).

Os pesos ficam em weights/ e são baixados por ``python -m data.download_weights``;
nunca entram no Git.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass
class PoseDetection:
    keypoints: np.ndarray  # (17, 2) pixels
    conf: np.ndarray  # (17,)
    box: np.ndarray  # (4,) x1 y1 x2 y2 pixels
    score: float


@dataclass
class Detection:
    label: str
    box: np.ndarray  # (4,) x1 y1 x2 y2 pixels
    score: float


class PoseEstimator(Protocol):
    name: str
    version: str

    def predict(self, frame_bgr: np.ndarray) -> PoseDetection | None:
        """Pose da pessoa principal do quadro, ou None."""


class ObjectDetector(Protocol):
    name: str
    version: str

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]: ...


def _require_weights(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"pesos não encontrados: {path}. Rode: python -m data.download_weights"
        )
    return path


def _ultralytics_version() -> str:
    import ultralytics

    return ultralytics.__version__


def main_person(dets: list[PoseDetection]) -> PoseDetection | None:
    """Pessoa principal: maior área de caixa ponderada pela confiança."""
    if not dets:
        return None

    def weight(d: PoseDetection) -> float:
        x1, y1, x2, y2 = d.box
        return max(0.0, (x2 - x1)) * max(0.0, (y2 - y1)) * d.score

    return max(dets, key=weight)


class YoloPoseEstimator:
    def __init__(self, weights: str | Path, device=None, imgsz: int = 640, conf: float = 0.25) -> None:
        from ultralytics import YOLO

        self.weights = _require_weights(Path(weights))
        self.model = YOLO(str(self.weights))
        self.device, self.imgsz, self.conf = device, imgsz, conf
        self.name = self.weights.stem
        self.version = _ultralytics_version()

    def predict_all(self, frame_bgr: np.ndarray) -> list[PoseDetection]:
        res = self.model.predict(
            frame_bgr, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False
        )[0]
        if res.keypoints is None or res.boxes is None or len(res.boxes) == 0:
            return []
        kxy = res.keypoints.xy.cpu().numpy()
        kconf = res.keypoints.conf
        kconf = kconf.cpu().numpy() if kconf is not None else np.ones(kxy.shape[:2])
        boxes = res.boxes.xyxy.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy()
        return [
            PoseDetection(keypoints=kxy[i], conf=kconf[i], box=boxes[i], score=float(scores[i]))
            for i in range(len(boxes))
        ]

    def predict(self, frame_bgr: np.ndarray) -> PoseDetection | None:
        return main_person(self.predict_all(frame_bgr))


class YoloDetector:
    def __init__(
        self, weights: str | Path, classes: list[str], device=None, imgsz: int = 640, conf: float = 0.25
    ) -> None:
        from ultralytics import YOLO

        self.weights = _require_weights(Path(weights))
        self.model = YOLO(str(self.weights))
        names = self.model.names
        lookup = {v: k for k, v in names.items()}
        missing = [c for c in classes if c not in lookup]
        if missing:
            raise ValueError(f"classes inexistentes no modelo {self.weights.name}: {missing}")
        self.class_ids = [lookup[c] for c in classes]
        self.names = names
        self.device, self.imgsz, self.conf = device, imgsz, conf
        self.name = self.weights.stem
        self.version = _ultralytics_version()

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        res = self.model.predict(
            frame_bgr, imgsz=self.imgsz, conf=self.conf, device=self.device,
            classes=self.class_ids, verbose=False,
        )[0]
        if res.boxes is None:
            return []
        boxes = res.boxes.xyxy.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        scores = res.boxes.conf.cpu().numpy()
        return [
            Detection(label=self.names[int(c)], box=b, score=float(s)) for b, c, s in zip(boxes, cls, scores)
        ]

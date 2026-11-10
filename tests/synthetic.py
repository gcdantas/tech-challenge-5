"""Poses e vídeos SINTÉTICOS para testes (nunca usados como dado da demo)."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from modules.video.geometry import KP
from modules.video.physio import PoseFrame
from modules.video.pose import PoseDetection


def squat_keypoints(knee_angle: float, trunk_angle: float, scale: float = 1.0, origin=(150.0, 220.0)):
    """Esqueleto de perfil com ângulo de joelho e inclinação de tronco dados (graus)."""
    ox, oy = origin
    s = scale
    ankle = np.array([ox, oy])
    knee = ankle + np.array([0.0, -50.0 * s])
    th = math.radians(knee_angle)
    hip = knee + 50.0 * s * np.array([math.sin(th), math.cos(th)])
    ph = math.radians(trunk_angle)
    shoulder = hip + 60.0 * s * np.array([math.sin(ph), -math.cos(ph)])
    elbow = shoulder + np.array([15.0 * s, 20.0 * s])
    wrist = elbow + np.array([15.0 * s, 0.0])
    nose = shoulder + np.array([5.0 * s, -20.0 * s])
    kp = np.zeros((17, 2))
    kp[KP["nose"]] = kp[KP["left_eye"]] = kp[KP["right_eye"]] = nose
    kp[KP["left_ear"]] = kp[KP["right_ear"]] = nose
    for side in ("left", "right"):
        kp[KP[f"{side}_shoulder"]] = shoulder
        kp[KP[f"{side}_elbow"]] = elbow
        kp[KP[f"{side}_wrist"]] = wrist
        kp[KP[f"{side}_hip"]] = hip
        kp[KP[f"{side}_knee"]] = knee
        kp[KP[f"{side}_ankle"]] = ankle
    return kp, np.full(17, 0.9)


# (profundidade mínima do joelho, inclinação máxima do tronco) por repetição.
# Rep 3: amplitude incompleta (para em 130°). Rep 6: tronco a 62° (postura fora da faixa).
SQUAT_REPS = [(90, 25), (88, 25), (130, 25), (92, 25), (90, 25), (90, 62)]
REST_KNEE, REST_TRUNK = 172.0, 5.0
REP_S, PAUSE_S = 2.0, 1.0


def squat_angles(t: float) -> tuple[float, float]:
    """Ângulos do joelho e do tronco no instante t (s); repouso fora das repetições."""
    period = REP_S + PAUSE_S
    k = int(t // period)
    if k >= len(SQUAT_REPS):
        return REST_KNEE, REST_TRUNK
    phase = (t - k * period) / REP_S
    if phase >= 1.0:
        return REST_KNEE, REST_TRUNK
    w = math.sin(math.pi * phase)  # 0 → 1 → 0
    knee_min, trunk_max = SQUAT_REPS[k]
    return REST_KNEE - (REST_KNEE - knee_min) * w, REST_TRUNK + (trunk_max - REST_TRUNK) * w


def session_duration() -> float:
    return len(SQUAT_REPS) * (REP_S + PAUSE_S) + 1.0


def squat_pose_frames(fps: float = 15.0) -> list[PoseFrame]:
    frames = []
    n = int(session_duration() * fps)
    for i in range(n):
        t = i / fps
        knee, trunk = squat_angles(t)
        kp, conf = squat_keypoints(knee, trunk)
        frames.append(PoseFrame(i, t, kp, conf, np.array([100, 80, 220, 230.0])))
    return frames


class FakePoseEstimator:
    """Devolve a pose sintética do agachamento conforme a ordem das chamadas."""

    name = "fake-pose"
    version = "test"

    def __init__(self, video_fps: float, stride: int) -> None:
        self.dt = stride / video_fps
        self.calls = 0

    def predict(self, frame_bgr):
        t = self.calls * self.dt
        self.calls += 1
        knee, trunk = squat_angles(t)
        kp, conf = squat_keypoints(knee, trunk)
        return PoseDetection(keypoints=kp, conf=conf, box=np.array([100, 80, 220, 230.0]), score=0.9)


def write_synthetic_video(path: Path, seconds: float, fps: float = 15.0, size=(320, 240)) -> Path:
    """MP4 sintético (fundo cinza com o quadro numerado), só para testar E/S."""
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(int(seconds * fps)):
        frame = np.full((h, w, 3), 90, np.uint8)
        cv2.putText(frame, f"SINTETICO {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(frame)
    writer.release()
    return path

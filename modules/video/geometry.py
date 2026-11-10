"""Geometria da pose: ângulos articulares e tratamento das séries.

Keypoints no formato COCO-17 (o mesmo do YOLOv8-pose), em pixels.
"""

from __future__ import annotations

import math

import numpy as np

KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
KP = {name: i for i, name in enumerate(KEYPOINTS)}

# Ligações do esqueleto COCO-17 para desenho.
SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4),
]

# Articulação -> (ponto A, vértice, ponto B) por lado.
JOINTS = {
    "knee": ("hip", "knee", "ankle"),
    "hip": ("shoulder", "hip", "knee"),
    "elbow": ("shoulder", "elbow", "wrist"),
    "shoulder": ("hip", "shoulder", "elbow"),
}
SIDES = ("left", "right")


def angle_at(a: np.ndarray, vertex: np.ndarray, b: np.ndarray) -> float:
    """Ângulo ABC em graus (0–180) no vértice. NaN se algum vetor for nulo."""
    v1 = np.asarray(a, float) - np.asarray(vertex, float)
    v2 = np.asarray(b, float) - np.asarray(vertex, float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0 or not (np.isfinite(n1) and np.isfinite(n2)):
        return math.nan
    cos = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return math.degrees(math.acos(cos))


def inclination_from_vertical(bottom: np.ndarray, top: np.ndarray) -> float:
    """Inclinação (graus) do vetor bottom→top em relação à vertical para cima.

    0 = em pé/ereto, 90 = horizontal (deitado). Coordenadas de imagem (y para baixo).
    """
    dx = float(top[0] - bottom[0])
    dy = float(bottom[1] - top[1])  # positivo quando top está acima
    if dx == 0 and dy == 0:
        return math.nan
    return math.degrees(math.atan2(abs(dx), dy))


def _point(kp: np.ndarray, conf: np.ndarray, name: str, min_conf: float) -> np.ndarray | None:
    i = KP[name]
    if conf[i] < min_conf or not np.all(np.isfinite(kp[i])):
        return None
    return kp[i]


def _mid(p: np.ndarray | None, q: np.ndarray | None) -> np.ndarray | None:
    if p is not None and q is not None:
        return (p + q) / 2.0
    return p if p is not None else q


def pose_metrics(kp: np.ndarray, conf: np.ndarray, min_conf: float) -> dict[str, float]:
    """Métricas de um quadro: ângulos por lado, tronco e confiança média por lado."""
    out: dict[str, float] = {}
    for joint, (a, v, b) in JOINTS.items():
        for side in SIDES:
            pa = _point(kp, conf, f"{side}_{a}", min_conf)
            pv = _point(kp, conf, f"{side}_{v}", min_conf)
            pb = _point(kp, conf, f"{side}_{b}", min_conf)
            ok = pa is not None and pv is not None and pb is not None
            out[f"{joint}_{side}"] = angle_at(pa, pv, pb) if ok else math.nan
    shoulders = _mid(_point(kp, conf, "left_shoulder", min_conf), _point(kp, conf, "right_shoulder", min_conf))
    hips = _mid(_point(kp, conf, "left_hip", min_conf), _point(kp, conf, "right_hip", min_conf))
    out["trunk_angle"] = (
        inclination_from_vertical(hips, shoulders) if shoulders is not None and hips is not None else math.nan
    )
    for side in SIDES:
        idx = [KP[f"{side}_{n}"] for n in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle")]
        out[f"conf_{side}"] = float(np.mean(conf[idx]))
    return out


def interpolate_gaps(values: np.ndarray, max_gap: int) -> np.ndarray:
    """Interpola linearmente lacunas (NaN) de até ``max_gap`` amostras internas."""
    v = np.asarray(values, float).copy()
    isnan = np.isnan(v)
    if not isnan.any() or isnan.all():
        return v
    idx = np.arange(len(v))
    i = 0
    while i < len(v):
        if not isnan[i]:
            i += 1
            continue
        j = i
        while j < len(v) and isnan[j]:
            j += 1
        if i > 0 and j < len(v) and (j - i) <= max_gap:
            v[i:j] = np.interp(idx[i:j], [i - 1, j], [v[i - 1], v[j]])
        i = j
    return v


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Média móvel centrada que ignora NaN (NaN só onde a janela inteira é NaN)."""
    v = np.asarray(values, float)
    if window <= 1 or len(v) == 0:
        return v.copy()
    half = window // 2
    out = np.full(len(v), math.nan)
    for i in range(len(v)):
        seg = v[max(0, i - half): i + half + 1]
        seg = seg[np.isfinite(seg)]
        if len(seg) and np.isfinite(v[i]):
            out[i] = float(seg.mean())
    return out

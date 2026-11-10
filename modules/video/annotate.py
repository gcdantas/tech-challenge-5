"""Desenho de esqueleto, caixas, zona crítica e rótulos sobre os quadros."""

from __future__ import annotations

import cv2
import numpy as np

from modules.video.geometry import SKELETON

GREEN = (80, 200, 80)
ORANGE = (0, 150, 255)
RED = (40, 40, 220)
BLUE = (220, 140, 40)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def draw_pose(
    frame: np.ndarray,
    keypoints: np.ndarray | None,
    conf: np.ndarray | None,
    min_conf: float = 0.3,
    color: tuple[int, int, int] = GREEN,
    skeleton: list[tuple[int, int]] = SKELETON,
) -> np.ndarray:
    if keypoints is None or conf is None:
        return frame
    th = max(2, frame.shape[0] // 300)
    for a, b in skeleton:
        if a < len(conf) and b < len(conf) and conf[a] >= min_conf and conf[b] >= min_conf:
            pa = tuple(int(v) for v in keypoints[a])
            pb = tuple(int(v) for v in keypoints[b])
            cv2.line(frame, pa, pb, color, th, cv2.LINE_AA)
    for i, (x, y) in enumerate(keypoints):
        if conf[i] >= min_conf:
            cv2.circle(frame, (int(x), int(y)), th + 2, WHITE, -1, cv2.LINE_AA)
            cv2.circle(frame, (int(x), int(y)), th + 2, color, 1, cv2.LINE_AA)
    return frame


def draw_box(frame: np.ndarray, box, label: str, color=BLUE) -> np.ndarray:
    x1, y1, x2, y2 = (int(v) for v in box)
    th = max(2, frame.shape[0] // 360)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, th, cv2.LINE_AA)
    put_label(frame, label, (x1, max(0, y1 - 4)), color)
    return frame


def draw_edge_zone(frame: np.ndarray, outer, inner, color=ORANGE, alpha: float = 0.3) -> np.ndarray:
    """Pinta a faixa entre a caixa externa (leito) e a interna (zona da borda)."""
    overlay = frame.copy()
    ox1, oy1, ox2, oy2 = (int(v) for v in outer)
    ix1, iy1, ix2, iy2 = (int(v) for v in inner)
    mask = np.zeros(frame.shape[:2], np.uint8)
    cv2.rectangle(mask, (ox1, oy1), (ox2, oy2), 255, -1)
    cv2.rectangle(mask, (ix1, iy1), (ix2, iy2), 0, -1)
    overlay[mask > 0] = color
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)
    return frame


def put_label(frame: np.ndarray, text: str, org: tuple[int, int], color=BLACK, scale: float | None = None) -> None:
    scale = scale or max(0.5, frame.shape[0] / 900)
    th = max(1, int(scale * 2))
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
    x, y = org
    y = max(h + base, y)
    cv2.rectangle(frame, (x, y - h - base), (x + w + 4, y + base // 2), color, -1)
    cv2.putText(frame, text, (x + 2, y - base // 2), cv2.FONT_HERSHEY_SIMPLEX, scale, WHITE, th, cv2.LINE_AA)


def put_lines(frame: np.ndarray, lines: list[str], color=BLACK) -> np.ndarray:
    """Bloco de texto no canto superior esquerdo (OpenCV não desenha acentos: use ASCII)."""
    scale = max(0.5, frame.shape[0] / 900)
    step = int(32 * scale) + 8
    for i, line in enumerate(lines):
        put_label(frame, line, (8, 8 + step * (i + 1)), color, scale)
    return frame


def ascii_text(text: str) -> str:
    """Remove acentos para o cv2.putText (fonte Hershey só tem ASCII)."""
    import unicodedata

    norm = unicodedata.normalize("NFKD", text)
    return "".join(c for c in norm if not unicodedata.combining(c)).replace("°", " graus")

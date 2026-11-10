"""Leitura de vídeo (OpenCV) e escrita de MP4 H.264 (ffmpeg do imageio-ffmpeg).

H.264/yuv420p toca direto no navegador, no relatório HTML e no Streamlit.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0


def probe(path: str | Path) -> VideoInfo:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"vídeo não encontrado: {path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"não foi possível abrir o vídeo: {path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        info = VideoInfo(
            path=path,
            fps=fps,
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        cap.release()
    return info


def stride_for(info: VideoInfo, target_fps: float) -> int:
    """Passo entre quadros analisados para chegar perto de ``target_fps``."""
    if target_fps <= 0:
        return 1
    return max(1, int(round(info.fps / target_fps)))


def iter_frames(path: str | Path, stride: int = 1) -> Iterator[tuple[int, np.ndarray]]:
    """Itera (índice do quadro, quadro BGR) a cada ``stride`` quadros."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"não foi possível abrir o vídeo: {path}")
    idx = 0
    try:
        while True:
            ok = cap.grab()
            if not ok:
                break
            if idx % stride == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                yield idx, frame
            idx += 1
    finally:
        cap.release()


def read_frame(path: str | Path, frame_idx: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


class Mp4Writer:
    """Escreve quadros BGR num MP4 H.264. Dimensões ímpares são ajustadas para pares."""

    def __init__(self, path: str | Path, width: int, height: int, fps: float) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.size = (width - width % 2, height - height % 2)
        self._gen = imageio_ffmpeg.write_frames(
            str(self.path), self.size, fps=fps, codec="libx264", pix_fmt_in="rgb24",
            macro_block_size=2, output_params=["-movflags", "+faststart"],
        )
        self._gen.send(None)
        self.frames = 0

    def write(self, frame_bgr: np.ndarray) -> None:
        h, w = frame_bgr.shape[:2]
        if (w, h) != self.size:
            frame_bgr = frame_bgr[: self.size[1], : self.size[0]]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self._gen.send(np.ascontiguousarray(rgb))
        self.frames += 1

    def close(self) -> None:
        self._gen.close()

    def __enter__(self) -> Mp4Writer:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def save_jpeg(path: str | Path, frame_bgr: np.ndarray, quality: int = 88) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError(f"falha ao codificar JPEG: {path}")
    path.write_bytes(buf.tobytes())  # imwrite falha com caminhos não ASCII no Windows
    return path

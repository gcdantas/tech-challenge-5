"""Baixa alguns vídeos do dataset público REHAB24-6 para testar o módulo de vídeo.

    python -m data.download_rehab24             # os 4 vídeos usados no capítulo 3
    python -m data.download_rehab24 --list Ex6  # lista os vídeos de um exercício

REHAB24-6 (Zenodo, doi:10.5281/zenodo.13305826), licença CC BY-NC 4.0: uso
acadêmico/não comercial, citando
  Černek, A., Sedmidubsky, J., Budikova, P.: REHAB24-6: Physical Therapy Dataset
  for Analyzing Pose Estimation Methods. SISAP 2024.

Os vídeos ficam num único videos.zip de 2,5 GB. Este script lê só o índice do
zip e os vídeos escolhidos por requisições HTTP Range, sem baixar o resto.
Tudo vai para data/video/rehab24-6/ (fora do Git).
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "video" / "rehab24-6"
RECORD = "https://zenodo.org/api/records/13305826/files"
VIDEOS_ZIP = f"{RECORD}/videos.zip/content"
ANNOTATIONS = ["Segmentation.csv", "Segmentation.txt"]

# Vídeos usados no capítulo 3. Câmera 18 mostra o agachamento de perfil quando a
# pessoa está de frente para a câmera 17; câmera 17 mostra a abdução de frente.
DEFAULT_VIDEOS = [
    "Ex6/PM_022-Camera18-30fps-transposed.mp4",  # agachamento, quase todas corretas
    "Ex6/PM_043-Camera18-30fps-transposed.mp4",  # agachamento, metade incorretas
    "Ex6/PM_105-Camera18-30fps-transposed.mp4",  # agachamento, metade incorretas
    "Ex1/PM_001-Camera17-30fps.mp4",             # abdução de braço
]
UA = {"User-Agent": "tech-challenge-video-tests/1.0"}


class HttpRangeFile(io.RawIOBase):
    """Arquivo remoto somente leitura, com seek, via HTTP Range."""

    def __init__(self, url: str):
        req = urllib.request.Request(url, method="HEAD", headers=UA)
        with urllib.request.urlopen(req, timeout=60) as r:
            self.url = r.geturl()
            self.size = int(r.headers["Content-Length"])
        self.pos = 0
        self.fetched = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def read(self, n=-1):
        if self.pos >= self.size:
            return b""
        end = self.size - 1 if n is None or n < 0 else min(self.size, self.pos + n) - 1
        req = urllib.request.Request(self.url, headers={**UA, "Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req, timeout=300) as r:
            if r.status != 206:
                raise OSError(f"o servidor não aceitou leitura parcial (HTTP {r.status})")
            data = r.read()
        self.pos += len(data)
        self.fetched += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def _open_zip() -> tuple[zipfile.ZipFile, HttpRangeFile]:
    raw = HttpRangeFile(VIDEOS_ZIP)
    return zipfile.ZipFile(io.BufferedReader(raw, buffer_size=4 << 20)), raw


def get_annotations() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    for name in ANNOTATIONS:
        dest = DEST / name
        if dest.exists():
            print(f"ok (já existe): {dest}")
            continue
        req = urllib.request.Request(f"{RECORD}/{name}/content", headers=UA)
        with urllib.request.urlopen(req, timeout=120) as r:
            dest.write_bytes(r.read())
        print(f"ok: {dest}")


def get_videos(members: list[str]) -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    pending = [m for m in members if not (DEST / Path(m).name).exists()]
    for m in members:
        if m not in pending:
            print(f"ok (já existe): {DEST / Path(m).name}")
    if not pending:
        return
    zf, raw = _open_zip()
    names = set(zf.namelist())
    for m in pending:
        if m not in names:
            raise SystemExit(f"erro: {m} não está no videos.zip (use --list)")
        target = DEST / Path(m).name  # nome achatado: nunca escreve fora de DEST
        tmp = target.with_suffix(".part")
        with zf.open(m) as src, tmp.open("wb") as out:
            while chunk := src.read(4 << 20):
                out.write(chunk)
        tmp.replace(target)
        print(f"ok: {target} ({target.stat().st_size / 2**20:.1f} MiB)")
    print(f"lidos {raw.fetched / 2**20:.1f} MiB de {raw.size / 2**20:.1f} MiB do videos.zip")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Baixa vídeos selecionados do REHAB24-6 (CC BY-NC 4.0).")
    ap.add_argument("--list", metavar="FILTRO", nargs="?", const="", help="lista os vídeos do zip")
    ap.add_argument("videos", nargs="*", help="membros do zip (padrão: os 4 do capítulo 3)")
    args = ap.parse_args(argv)
    if args.list is not None:
        zf, _ = _open_zip()
        for info in zf.infolist():
            if args.list in info.filename and not info.is_dir():
                print(f"{info.file_size / 2**20:8.1f} MiB  {info.filename}")
        return 0
    get_annotations()
    get_videos(args.videos or DEFAULT_VIDEOS)
    return 0


if __name__ == "__main__":
    sys.exit(main())

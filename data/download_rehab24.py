"""Baixa vídeos do dataset público REHAB24-6 (fisioterapia) para data/video/rehab24-6/.

    python -m data.download_rehab24             # os 7 vídeos usados no projeto
    python -m data.download_rehab24 --list Ex6  # lista os vídeos de um exercício
    python -m data.download_rehab24 Ex6/PM_008-Camera18-30fps-transposed.mp4

REHAB24-6 (Zenodo, doi:10.5281/zenodo.13305826), licença CC BY-NC 4.0: uso
acadêmico/não comercial, citando
  Černek, A., Sedmidubsky, J., Budikova, P.: REHAB24-6: Physical Therapy Dataset
  for Analyzing Pose Estimation Methods. SISAP 2024.

Os vídeos ficam num único videos.zip de 2,5 GB; só os escolhidos são lidos.
Tudo fica fora do Git.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from data.remote_zip import download, extract_members, list_members

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "video" / "rehab24-6"
RECORD = "https://zenodo.org/api/records/13305826/files"
VIDEOS_ZIP = f"{RECORD}/videos.zip/content"
ANNOTATIONS = ["Segmentation.csv", "Segmentation.txt"]

# Câmera 18 mostra o agachamento de perfil quando a pessoa está de frente para a
# câmera 17 (primeira metade de cada vídeo) e em meio-perfil depois que ela gira.
# A câmera 17 mostra a abdução de frente.
DEFAULT_VIDEOS = [
    "Ex6/PM_105-Camera18-30fps-transposed.mp4",  # agachamento, pessoa 6: vídeo de fisioterapia do P001
    "Ex6/PM_022-Camera18-30fps-transposed.mp4",  # agachamento, pessoa 2
    "Ex6/PM_043-Camera18-30fps-transposed.mp4",  # agachamento, pessoa 5
    "Ex6/PM_113-Camera18-30fps-transposed.mp4",  # agachamento, pessoa 8
    "Ex6/PM_118-Camera18-30fps-transposed.mp4",  # agachamento, pessoa 9
    "Ex6/PM_126-Camera18-30fps-transposed.mp4",  # agachamento, pessoa 7
    "Ex1/PM_001-Camera17-30fps.mp4",             # abdução do braço direito, pessoa 1
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Baixa vídeos do REHAB24-6 (CC BY-NC 4.0).")
    ap.add_argument("--list", metavar="FILTRO", nargs="?", const="", help="lista os vídeos do zip")
    ap.add_argument("videos", nargs="*", help="membros do zip (padrão: os 7 usados no projeto)")
    args = ap.parse_args(argv)
    if args.list is not None:
        for name, size in list_members(VIDEOS_ZIP, args.list):
            print(f"{size / 2**20:8.1f} MiB  {name}")
        return 0
    for name in ANNOTATIONS:
        download(f"{RECORD}/{name}/content", DEST / name)
    extract_members(VIDEOS_ZIP, args.videos or DEFAULT_VIDEOS, DEST)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Baixa o vídeo do cenário do leito, do dataset público Charades, para data/video/charades/.

    python -m data.download_charades          # anotações e o vídeo SSP8C
    python -m data.download_charades XXXXX    # outro vídeo, pelo id do Charades

Charades (Allen Institute for AI; Sigurdsson et al., "Hollywood in Homes", ECCV
2016). Licença de uso não comercial: pesquisa acadêmica; não permite
redistribuir nem publicar os dados modificados; em publicação acadêmica,
só trechos curtos ou quadros quando necessário. Por isso os vídeos ficam fora
do Git e cada integrante baixa a própria cópia.

SSP8C (quarto, 30 s): a pessoa está deitada na cama e se levanta. Anotação do
Charades: "Lying on a bed" 0–11,5 s; "Someone is standing up from somewhere"
13,7–19,1 s.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

from data.remote_zip import download, extract_members

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "video" / "charades"
BUCKET = "https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades"
VIDEOS_ZIP = f"{BUCKET}/Charades_v1_480.zip"  # vídeos em 480p, 16 GB; só os pedidos são lidos
ANNOTATIONS_ZIP = f"{BUCKET}/Charades.zip"
ANNOTATION_FILES = ["license.txt", "README.txt", "Charades_v1_classes.txt", "Charades_v1_train.csv", "Charades_v1_test.csv"]
DEFAULT_VIDEOS = ["SSP8C"]


def get_annotations() -> None:
    zpath = download(ANNOTATIONS_ZIP, DEST / "Charades.zip")
    out = DEST / "anotacoes"
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath) as zf:
        for name in ANNOTATION_FILES:
            target = out / name
            if not target.exists():
                target.write_bytes(zf.read(f"Charades/{name}"))
    print(f"ok: anotações e licença em {out}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Baixa vídeos do Charades (licença não comercial do AI2).")
    ap.add_argument("ids", nargs="*", help="ids dos vídeos (padrão: SSP8C)")
    args = ap.parse_args(argv)
    get_annotations()
    extract_members(VIDEOS_ZIP, [f"Charades_v1_480/{i}.mp4" for i in (args.ids or DEFAULT_VIDEOS)], DEST)
    return 0


if __name__ == "__main__":
    sys.exit(main())

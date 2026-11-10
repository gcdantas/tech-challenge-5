"""Baixa os pesos dos modelos de vídeo para weights/ (fora do Git).

    python -m data.download_weights            # tudo
    python -m data.download_weights --only yolo
    python -m data.download_weights --only openpose

- YOLOv8n-pose e YOLOv8n: releases oficiais da Ultralytics (ultralytics/assets).
- OpenPose COCO (Caffe): prototxt do repositório oficial da CMU no GitHub; o
  caffemodel é tentado primeiro no servidor da CMU e, se ele não responder, no
  espelho do Hugging Face (camenduru/openpose). O SHA-256 é conferido nos dois casos.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights"

YOLO_WEIGHTS = ["yolov8n-pose.pt", "yolov8n.pt"]

OPENPOSE = [
    {
        "file": "openpose/pose_deploy_linevec.prototxt",
        "urls": [
            "https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/coco/pose_deploy_linevec.prototxt",
            "https://huggingface.co/camenduru/openpose/resolve/main/models/pose/coco/pose_deploy_linevec.prototxt",
        ],
        "sha256": "17051b87f709aa094e09c5da7b78e9016a1f37b2b452ed1f190fe74cce70b1ad",
        "size": 46372,
    },
    {
        "file": "openpose/pose_iter_440000.caffemodel",
        "urls": [
            "http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel",
            "https://huggingface.co/camenduru/openpose/resolve/main/models/pose/coco/pose_iter_440000.caffemodel",
        ],
        # SHA-256 publicado pelo Git LFS do espelho no Hugging Face.
        "sha256": "b4cf475576abd7b15d5316f1ee65eb492b5c9f5865e70a2e7882ed31fb682549",
        "size": 209274056,
    },
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ok(path: Path, item: dict) -> bool:
    if not path.exists():
        return False
    if item.get("size") and path.stat().st_size != item["size"]:
        return False
    return not item.get("sha256") or sha256(path) == item["sha256"]


def _download(url: str, dest: Path, timeout: float) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "tech-challenge-weights/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, tmp.open("wb") as fh:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 2**20:7.1f} / {total / 2**20:.1f} MiB", end="", flush=True)
    print()
    tmp.replace(dest)


def get_openpose() -> bool:
    ok_all = True
    for item in OPENPOSE:
        dest = WEIGHTS / item["file"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        if _ok(dest, item):
            print(f"ok (já existe): {dest}")
            continue
        for url in item["urls"]:
            print(f"baixando {url}")
            try:
                _download(url, dest, timeout=30)
            except Exception as exc:  # noqa: BLE001 - tenta o próximo espelho
                print(f"  falhou: {exc}")
                continue
            if _ok(dest, item):
                print(f"ok: {dest} (sha256 {sha256(dest)[:16]}…)")
                break
            print("  arquivo não confere (tamanho ou SHA-256); tentando o próximo espelho")
            dest.unlink(missing_ok=True)
        else:
            print(f"ERRO: não foi possível obter {item['file']}", file=sys.stderr)
            ok_all = False
    return ok_all


def get_yolo() -> bool:
    from ultralytics.utils.downloads import attempt_download_asset

    WEIGHTS.mkdir(parents=True, exist_ok=True)
    ok_all = True
    for name in YOLO_WEIGHTS:
        dest = WEIGHTS / name
        if dest.exists():
            print(f"ok (já existe): {dest}")
            continue
        try:
            got = Path(attempt_download_asset(dest))
        except Exception as exc:  # noqa: BLE001
            print(f"ERRO: {name}: {exc}", file=sys.stderr)
            ok_all = False
            continue
        if got.resolve() != dest.resolve() and got.exists():
            got.replace(dest)
        print(f"ok: {dest}")
    return ok_all


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Baixa os pesos dos modelos de vídeo para weights/.")
    ap.add_argument("--only", choices=["yolo", "openpose"], default=None)
    args = ap.parse_args(argv)
    ok = True
    if args.only in (None, "yolo"):
        ok &= get_yolo()
    if args.only in (None, "openpose"):
        ok &= get_openpose()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

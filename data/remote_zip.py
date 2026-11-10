"""Leitura de membros de um .zip remoto por HTTP Range, sem baixar o arquivo inteiro.

Usado pelos scripts de download de datasets públicos (REHAB24-6, Charades), cujos
vídeos ficam em zips de vários GB. Só o índice do zip e os vídeos pedidos são lidos.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

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


def open_remote_zip(url: str) -> tuple[zipfile.ZipFile, HttpRangeFile]:
    raw = HttpRangeFile(url)
    return zipfile.ZipFile(io.BufferedReader(raw, buffer_size=4 << 20)), raw


def list_members(url: str, contains: str = "") -> list[tuple[str, int]]:
    zf, _ = open_remote_zip(url)
    return [(i.filename, i.file_size) for i in zf.infolist() if contains in i.filename and not i.is_dir()]


def extract_members(url: str, members: list[str], dest: Path) -> None:
    """Extrai ``members`` do zip remoto para ``dest`` (nomes achatados; pula os que já existem)."""
    dest.mkdir(parents=True, exist_ok=True)
    pending = [m for m in members if not (dest / Path(m).name).exists()]
    for m in members:
        if m not in pending:
            print(f"ok (já existe): {dest / Path(m).name}")
    if not pending:
        return
    zf, raw = open_remote_zip(url)
    names = set(zf.namelist())
    for m in pending:
        if m not in names:
            raise SystemExit(f"erro: {m} não está em {url}")
        target = dest / Path(m).name  # nome achatado: nunca escreve fora de dest
        tmp = target.with_suffix(target.suffix + ".part")
        with zf.open(m) as src, tmp.open("wb") as out:
            while chunk := src.read(4 << 20):
                out.write(chunk)
        tmp.replace(target)
        print(f"ok: {target} ({target.stat().st_size / 2**20:.1f} MiB)")
    print(f"lidos {raw.fetched / 2**20:.1f} MiB de {raw.size / 2**20:.1f} MiB do zip remoto")


def download(url: str, dest: Path, timeout: float = 120) -> Path:
    if dest.exists():
        print(f"ok (já existe): {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        dest.write_bytes(r.read())
    print(f"ok: {dest}")
    return dest

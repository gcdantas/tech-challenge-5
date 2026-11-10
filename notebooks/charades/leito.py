"""Compara a série do leito (movement.parquet) com a anotação do Charades.

    python -m notebooks.charades.leito [--video SSP8C] [--parquet outputs/video/P001/movement.parquet]

Mostra as transições de ``in_bed`` (em segundos do vídeo), a inclinação do tronco
no início e no fim e as ações do Charades ligadas a cama e a levantar-se.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ANN = ROOT / "data" / "video" / "charades" / "anotacoes"
BED_ACTIONS = {"c133": "acordando na cama", "c134": "deitado na cama", "c135": "sentado na cama",
               "c154": "levantando-se"}


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--video", default="SSP8C")
    ap.add_argument("--parquet", default=str(ROOT / "outputs" / "video" / "P001" / "movement.parquet"))
    args = ap.parse_args(argv)

    df = pd.read_parquet(args.parquet)
    t = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()
    changes = [(round(float(t.iloc[i]), 1), bool(df["in_bed"].iloc[i]))
               for i in range(1, len(df)) if df["in_bed"].iloc[i] != df["in_bed"].iloc[i - 1]]
    print(f"amostras: {len(df)} | no leito: {df['in_bed'].mean():.0%}")
    print("transições de in_bed (s, novo valor):", changes)
    print(f"tronco: início {df['torso_angle'].iloc[:15].median():.0f}° | fim {df['torso_angle'].iloc[-15:].median():.0f}°")

    ann = pd.concat([pd.read_csv(ANN / "Charades_v1_train.csv"), pd.read_csv(ANN / "Charades_v1_test.csv")])
    row = ann.set_index("id").loc[args.video]
    for a in str(row["actions"]).split(";"):
        code, start, end = a.split()
        if code in BED_ACTIONS:
            print(f"Charades {code} ({BED_ACTIONS[code]}): {float(start):.1f}–{float(end):.1f} s")
    import pyarrow.parquet as pq

    raw = pq.read_schema(args.parquet).metadata or {}
    meta = {k.decode(): v.decode() for k, v in raw.items() if k.decode() in ("video", "bed_box_px", "mock")}
    print("metadados do parquet:", json.dumps(meta, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

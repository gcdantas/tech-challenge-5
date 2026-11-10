"""CLI do módulo de vídeo.

    python -m modules.video.run --input data/video/P001_fisio.mp4 --patient P001 \
        --out outputs/video/P001/findings.jsonl

Aceita vários --input do mesmo paciente; os Findings de todos os vídeos de
fisioterapia vão para o mesmo --out. Vídeos do leito não geram Findings: se
nenhum vídeo de fisioterapia for processado, o --out não é tocado (para não
apagar Findings de uma execução anterior).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from contracts.finding import write_findings
from contracts.patients import DEFAULT_PATIENTS_PATH, load_patients
from modules.video import analyze, default_outputs_root
from modules.video.config import DEFAULT_VIDEO_CONFIG, load_video_config
from modules.video.timebase import resolve_context


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Análise de vídeo (I1): fisioterapia e leito.")
    ap.add_argument("--input", nargs="+", required=True, help="vídeo(s) .mp4")
    ap.add_argument("--patient", required=True, help="patient_id, ex.: P001")
    ap.add_argument("--out", default=None, help="findings.jsonl (padrão: outputs/video/<paciente>/findings.jsonl)")
    ap.add_argument("--scenario", choices=["physio", "bed"], default=None,
                    help="obrigatório só para vídeo fora do patients.yaml")
    ap.add_argument("--config", default=str(DEFAULT_VIDEO_CONFIG))
    ap.add_argument("--patients", default=str(DEFAULT_PATIENTS_PATH))
    ap.add_argument("--outputs", default=None, help="raiz das saídas (padrão: outputs/)")
    ap.add_argument("--no-annotated-video", action="store_true", help="não grava o vídeo inteiro anotado")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_video_config(args.config)
    if args.no_annotated_video:
        cfg.output.annotated_video = False
    pf = load_patients(args.patients)
    root = Path(args.outputs) if args.outputs else default_outputs_root()
    out = Path(args.out) if args.out else root / "video" / args.patient / "findings.jsonl"

    # Valida todos os vídeos antes de processar qualquer um.
    contexts = []
    for path in args.input:
        if not Path(path).exists():
            print(f"erro: vídeo não encontrado: {path}", file=sys.stderr)
            return 2
        try:
            contexts.append(resolve_context(path, args.patient, args.scenario, pf))
        except (KeyError, ValueError) as exc:
            print(f"erro: {exc}", file=sys.stderr)
            return 2

    pose = None
    findings, physio_runs = [], 0
    for ctx in contexts:
        t0 = time.perf_counter()
        if pose is None:
            from modules.video.pose import YoloPoseEstimator

            m = cfg.models
            try:
                pose = YoloPoseEstimator(cfg.resolve(m.pose), m.device, m.imgsz, m.conf)
            except FileNotFoundError as exc:
                print(f"erro: {exc}", file=sys.stderr)
                return 2
        try:
            result = analyze(ctx.input, args.patient, scenario=ctx.scenario, config=cfg,
                             outputs_root=root, patients=pf, pose_estimator=pose)
        except (FileNotFoundError, ValueError) as exc:
            print(f"erro ({ctx.input}): {exc}", file=sys.stderr)
            return 2
        dt = time.perf_counter() - t0
        if ctx.scenario == "physio":
            physio_runs += 1
            findings.extend(result)
            print(f"[fisioterapia] {ctx.input}: {len(result)} desvio(s) em {dt:.1f} s")
        else:
            print(f"[leito] {ctx.input}: movement.parquet gerado em {dt:.1f} s (sem Findings)")

    if physio_runs:
        write_findings(out, findings)
        print(f"Findings: {out} ({len(findings)})")
    else:
        print("Nenhum vídeo de fisioterapia: findings.jsonl não foi alterado.")
    print(f"Relatório: {root / 'video' / args.patient / 'report.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

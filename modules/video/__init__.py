"""Análise de vídeo (Integrante 1).

Interface do contrato: ``analyze(input, patient_id) -> list[Finding]``.

- Fisioterapia: YOLOv8-pose, ângulos articulares e amplitude por repetição;
  emite Findings ``source: video`` (amplitude_incompleta, postura_fora_padrao),
  clipes e quadros anotados e o relatório HTML.
- Leito: YOLOv8 para pessoa e leito, zona da borda como área crítica; gera
  movement.parquet e quadros anotados e NÃO emite Findings (saída do leito e
  imobilidade são do I3).

O cenário vem do ``kind`` do artefato em data/patients.yaml; para um vídeo fora
do patients.yaml, informe ``scenario``.
"""

from __future__ import annotations

import os
from pathlib import Path

from contracts.finding import Finding
from contracts.patients import REPO_ROOT, PatientsFile


def default_outputs_root() -> Path:
    env = os.environ.get("TC_OUTPUTS_DIR")
    return Path(env) if env else REPO_ROOT / "outputs"


def analyze(
    input: str | Path,  # noqa: A002 - nome definido no contrato do plano
    patient_id: str,
    *,
    scenario: str | None = None,
    config=None,
    outputs_root: str | Path | None = None,
    patients: PatientsFile | None = None,
    pose_estimator=None,
    detector=None,
) -> list[Finding]:
    """Analisa um vídeo e devolve os Findings (lista vazia no cenário do leito).

    Os artefatos (clipes, quadros, relatório, movement.parquet) são gravados em
    ``<outputs_root>/video/<patient_id>/``. ``pose_estimator`` e ``detector``
    permitem injetar modelos (usado nos testes); por padrão são os YOLOv8 do
    config/video.yaml.
    """
    from modules.video.config import load_video_config
    from modules.video.report import render_report
    from modules.video.timebase import resolve_context

    cfg = config or load_video_config()
    root = Path(outputs_root) if outputs_root else default_outputs_root()
    ctx = resolve_context(input, patient_id, scenario, patients)
    m = cfg.models

    if pose_estimator is None:
        from modules.video.pose import YoloPoseEstimator

        pose_estimator = YoloPoseEstimator(cfg.resolve(m.pose), m.device, m.imgsz, m.conf)

    if ctx.scenario == "physio":
        from modules.video.pipeline import run_physio

        findings, _ = run_physio(ctx, cfg, pose_estimator, root)
    else:
        from modules.video.bed import run_bed

        if detector is None:
            from modules.video.pose import YoloDetector

            detector = YoloDetector(
                cfg.resolve(m.detect), [cfg.bed.bed_class, cfg.bed.person_class], m.device, m.imgsz, m.conf
            )
        run_bed(ctx, cfg, pose_estimator, detector, root)
        findings = []

    render_report(root / "video" / patient_id, patient_id)
    return findings


__all__ = ["analyze"]

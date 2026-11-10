"""CLI python -m modules.video.run.

Os testes com modelo real rodam o YOLOv8-pose num vídeo SINTÉTICO (sem pessoa):
validam E/S e o contrato, não a qualidade da pose. São pulados se os pesos
não tiverem sido baixados (python -m data.download_weights --only yolo).
"""

import subprocess
import sys
from pathlib import Path

import pytest

from contracts import read_findings
from contracts.patients import REPO_ROOT
from modules.video.run import main
from tests.synthetic import write_synthetic_video

POSE_WEIGHTS = REPO_ROOT / "weights" / "yolov8n-pose.pt"
needs_weights = pytest.mark.skipif(not POSE_WEIGHTS.exists(), reason="pesos YOLOv8 não baixados")


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    return write_synthetic_video(tmp_path_factory.mktemp("cli") / "sintetico_cli.mp4", 3.0)


def test_video_inexistente_retorna_2(tmp_path):
    assert main(["--input", str(tmp_path / "nao_existe.mp4"), "--patient", "P001"]) == 2


def test_video_fora_do_patients_sem_cenario_retorna_2(video, tmp_path):
    assert main(["--input", str(video), "--patient", "P001", "--outputs", str(tmp_path)]) == 2


def test_paciente_inexistente_retorna_2(video, tmp_path):
    rc = main(["--input", str(video), "--patient", "P999", "--scenario", "physio", "--outputs", str(tmp_path)])
    assert rc == 2


@needs_weights
def test_cli_fisioterapia_com_yolo_real(video, tmp_path):
    out = tmp_path / "findings.jsonl"
    rc = main([
        "--input", str(video), "--patient", "P001", "--scenario", "physio",
        "--outputs", str(tmp_path), "--out", str(out), "--no-annotated-video",
    ])
    assert rc == 0
    assert out.exists()
    assert read_findings(out) == []  # vídeo sintético sem pessoa: nenhum desvio
    assert (tmp_path / "video" / "P001" / "report.html").exists()
    assert (tmp_path / "video" / "P001" / "runs" / "physio_sintetico_cli.json").exists()


@needs_weights
def test_cli_como_modulo(video, tmp_path):
    """Executa exatamente como no plano: python -m modules.video.run ..."""
    out = tmp_path / "findings.jsonl"
    proc = subprocess.run(
        [sys.executable, "-m", "modules.video.run", "--input", str(video), "--patient", "P001",
         "--scenario", "physio", "--outputs", str(tmp_path), "--out", str(out), "--no-annotated-video"],
        cwd=REPO_ROOT, capture_output=True, text=True, encoding="utf-8", timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    assert Path(out).exists()

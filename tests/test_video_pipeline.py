"""Pipeline de fisioterapia de ponta a ponta num vídeo SINTÉTICO, com pose injetada.

Exercita leitura do vídeo, regras, demo_offset, clipes, quadros, relatório e
Findings, sem depender de modelo nem de vídeos reais.
"""

import json
from datetime import timedelta
from pathlib import Path

import pytest

from contracts import read_findings, write_findings
from contracts.patients import PatientsFile
from modules.video import analyze
from modules.video.config import load_video_config
from modules.video.videoio import probe, stride_for
from tests.synthetic import FakePoseEstimator, session_duration, write_synthetic_video

FPS = 15.0


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    return write_synthetic_video(tmp_path_factory.mktemp("video") / "sintetico_fisio.mp4", session_duration(), FPS)


def _patients(video, offset: float) -> PatientsFile:
    return PatientsFile.model_validate(
        {
            "demo_reference": "2026-11-03T10:00:00-03:00",
            "patients": {
                "P001": {"artifacts": {"video_fisio": {"kind": "video_physio", "path": str(video), "demo_offset": offset}}}
            },
        }
    )


def _run(video, out, offset=600.0, **cfg_over):
    cfg = load_video_config()
    for k, v in cfg_over.items():
        setattr(cfg.output, k, v)
    info = probe(video)
    est = FakePoseEstimator(info.fps, stride_for(info, cfg.physio.process_fps))
    return analyze(video, "P001", config=cfg, outputs_root=out, patients=_patients(video, offset), pose_estimator=est)


def test_findings_validos_com_demo_offset_e_artefatos(video, tmp_path):
    findings = _run(video, tmp_path)
    assert [(f.type, f.evidence["rep"]) for f in findings] == [
        ("amplitude_incompleta", 3), ("postura_fora_padrao", 6),
    ]
    # ida e volta pelo contrato
    write_findings(tmp_path / "f.jsonl", findings)
    assert read_findings(tmp_path / "f.jsonl") == findings

    ref = _patients(video, 600).demo_reference
    for f in findings:
        assert f.source == "video" and f.patient_id == "P001"
        assert f.timestamp_start.utcoffset() == timedelta(hours=-3)
        media_start = f.evidence["media_start_s"]
        expected = ref + timedelta(seconds=600 + media_start)
        assert abs(f.timestamp_start - expected) < timedelta(milliseconds=1)
        assert len(f.artifacts) == 2
        for a in f.artifacts:
            assert Path(a).exists(), a  # fora do repositório o caminho é absoluto

    pdir = tmp_path / "video" / "P001"
    clips = sorted(pdir.glob("clip_*.mp4"))
    assert len(clips) == 2
    clip_info = probe(clips[0])
    assert clip_info.frame_count > 0
    assert (pdir / "annotated_sintetico_fisio.mp4").exists()
    assert (pdir / "report.html").exists()
    run = json.loads((pdir / "runs" / "physio_sintetico_fisio.json").read_text(encoding="utf-8"))
    assert len(run["repetitions"]) == 6
    assert run["processing"]["inference_fps"] > 0
    assert {e["finding_id"] for e in run["events"]} == {str(f.finding_id) for f in findings}


def test_clipe_cobre_o_desvio_com_folga(video, tmp_path):
    findings = _run(video, tmp_path, annotated_video=False)
    f = findings[0]
    clip = next(a for a in f.artifacts if a.endswith(".mp4"))
    info = probe(Path(clip))
    span = f.evidence["media_end_s"] - f.evidence["media_start_s"]
    assert info.duration_s == pytest.approx(span + 2 * 1.0, abs=0.3)
    assert not (tmp_path / "video" / "P001" / "annotated_sintetico_fisio.mp4").exists()


def test_relatorio_usa_linguagem_obrigatoria(video, tmp_path):
    _run(video, tmp_path, annotated_video=False)
    html = (tmp_path / "video" / "P001" / "report.html").read_text(encoding="utf-8")
    assert "O sistema sinaliza risco e apoia a equipe; a decisão é clínica." in html
    assert "Paciente sintético P001" in html
    assert "Amplitude incompleta" in html and "Postura fora da faixa" in html
    assert "10:10:" in html  # horários já na linha do tempo da demo (offset 600 s)
    assert "<svg" in html
    lowered = html.lower()
    assert "diagnostica" not in lowered and "detecta disartria" not in lowered

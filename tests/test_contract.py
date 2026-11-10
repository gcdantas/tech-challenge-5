"""Testes do contrato Finding (contracts/finding.py)."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from contracts import DEMO_TZ, SOURCES, Finding, read_findings, write_findings

FIXTURES = Path(__file__).resolve().parents[1] / "contracts" / "fixtures" / "findings_example.jsonl"


def _base(**overrides):
    data = {
        "patient_id": "P001",
        "source": "video",
        "type": "amplitude_incompleta",
        "timestamp_start": "2026-11-03T10:15:00-03:00",
        "timestamp_end": "2026-11-03T10:15:12-03:00",
        "score": 0.5,
        "severity": "medium",
        "description": "teste",
        "evidence": {},
        "model": {"name": "m", "version": "1"},
        "artifacts": [],
    }
    data.update(overrides)
    return data


def test_fixtures_validam_e_cobrem_todas_as_origens():
    findings = read_findings(FIXTURES)
    assert {f.source for f in findings} == set(SOURCES)


def test_campos_do_plano_exatamente():
    expected = {
        "finding_id", "patient_id", "source", "type", "timestamp_start", "timestamp_end",
        "score", "severity", "description", "evidence", "model", "artifacts",
    }
    assert set(Finding.model_fields) == expected


@pytest.mark.parametrize("score", [-0.01, 1.01])
def test_score_fora_de_0_1_rejeitado(score):
    with pytest.raises(ValidationError):
        Finding.model_validate(_base(score=score))


def test_source_invalida_rejeitada():
    with pytest.raises(ValidationError):
        Finding.model_validate(_base(source="imagem"))


def test_severity_invalida_rejeitada():
    with pytest.raises(ValidationError):
        Finding.model_validate(_base(severity="grave"))


@pytest.mark.parametrize(
    "ts", ["2026-11-03T10:15:00", "2026-11-03T13:15:00Z", "2026-11-03T10:15:00-02:00"]
)
def test_timestamp_sem_fuso_ou_fuso_errado_rejeitado(ts):
    with pytest.raises(ValidationError):
        Finding.model_validate(_base(timestamp_start=ts))


def test_fim_antes_do_inicio_rejeitado():
    with pytest.raises(ValidationError):
        Finding.model_validate(_base(timestamp_end="2026-11-03T10:14:00-03:00"))


def test_campo_extra_rejeitado():
    with pytest.raises(ValidationError):
        Finding.model_validate(_base(alerta=True))


def test_finding_id_gerado_e_uuid4():
    f = Finding.model_validate(_base())
    assert f.finding_id.version == 4


def test_ida_e_volta_jsonl(tmp_path):
    original = Finding.model_validate(_base())
    out = write_findings(tmp_path / "x" / "findings.jsonl", [original])
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert '"timestamp_start": "2026-11-03T10:15:00-03:00"' in lines[0]
    (lido,) = read_findings(out)
    assert lido == original


def test_linha_invalida_indica_arquivo_e_linha(tmp_path):
    p = tmp_path / "findings.jsonl"
    p.write_text('\n{"patient_id": "P001"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"findings\.jsonl:2"):
        read_findings(p)


def test_demo_tz_e_menos_3():
    assert datetime(2026, 11, 3, tzinfo=DEMO_TZ).utcoffset() == timedelta(hours=-3)
    assert DEMO_TZ != timezone.utc

"""Todos os mocks do M1 validam contra o contrato e seguem os caminhos do plano."""

import json
from datetime import timedelta

import pandas as pd
import pyarrow.parquet as pq

from contracts import SOURCES, read_findings
from scripts.make_mocks import MOVEMENT_COLUMNS, make_all

EXPECTED_FINDINGS = {
    "video/P001/findings.jsonl": {("video", "amplitude_incompleta"), ("video", "postura_fora_padrao")},
    "audio/P001/findings.jsonl": {("audio", "termo_critico"), ("audio", "indicador_vocal_alterado")},
    "text/P001/findings.jsonl": {("text", "termo_critico")},
    "anomaly/P001/findings.jsonl": {
        ("vitals", "dessaturacao"),
        ("prescription", "dose_duplicada"),
        ("movement", "saida_do_leito"),
    },
}


def test_todos_os_findings_jsonl_validam_e_estao_nos_caminhos_do_plano(mocks_dir):
    found = sorted(p.relative_to(mocks_dir).as_posix() for p in mocks_dir.rglob("findings.jsonl"))
    assert found == sorted(EXPECTED_FINDINGS)
    for rel, expected in EXPECTED_FINDINGS.items():
        findings = read_findings(mocks_dir / rel)  # valida com Pydantic
        assert {(f.source, f.type) for f in findings} == expected


def test_uma_origem_de_cada_tipo_incluindo_text(mocks_dir):
    sources = {f.source for p in mocks_dir.rglob("findings.jsonl") for f in read_findings(p)}
    assert sources == set(SOURCES)


def test_mocks_marcados_como_ficticios(mocks_dir):
    for p in mocks_dir.rglob("findings.jsonl"):
        for f in read_findings(p):
            assert f.description.startswith("[MOCK]")
            assert f.model.name.startswith("mock-")
            assert f.evidence.get("mock") is True
    assert pq.read_schema(mocks_dir / "video/P001/movement.parquet").metadata[b"mock"] == b"true"
    transcript = json.loads((mocks_dir / "audio/P001/transcript.json").read_text(encoding="utf-8"))
    assert transcript["mock"] is True


def test_movement_parquet_colunas_frequencia_e_saida_do_leito(mocks_dir):
    df = pd.read_parquet(mocks_dir / "video/P001/movement.parquet")
    assert list(df.columns) == MOVEMENT_COLUMNS
    assert df["in_bed"].dtype == bool
    assert (df["patient_id"] == "P001").all()
    ts = df["timestamp"]
    assert ts.dt.tz is not None
    assert ts.iloc[0].utcoffset() == timedelta(hours=-3)
    step = ts.diff().dropna().dt.total_seconds()
    assert abs(step.median() - 0.2) < 1e-6  # ~5 Hz
    # parado no leito, depois saindo (uma única transição True -> False)
    assert df["in_bed"].iloc[0] and not df["in_bed"].iloc[-1]
    assert (df["in_bed"].astype(int).diff().fillna(0) != 0).sum() == 1


def test_saida_do_leito_do_i3_coincide_com_o_parquet(mocks_dir):
    df = pd.read_parquet(mocks_dir / "video/P001/movement.parquet")
    first_out = df.loc[~df["in_bed"], "timestamp"].iloc[0]
    (saida,) = [f for f in read_findings(mocks_dir / "anomaly/P001/findings.jsonl") if f.type == "saida_do_leito"]
    assert saida.timestamp_start == first_out.to_pydatetime()


def test_transcript_tem_termo_critico_com_timestamps(mocks_dir):
    tr = json.loads((mocks_dir / "audio/P001/transcript.json").read_text(encoding="utf-8"))
    terms = [t["term"] for s in tr["segments"] for t in s["critical_terms"]]
    assert "falta de ar" in terms
    assert all(s["timestamp_start"].endswith("-03:00") for s in tr["segments"])


def test_mocks_sao_deterministicos(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    make_all(a)
    make_all(b)
    for p in a.rglob("*.jsonl"):
        assert p.read_bytes() == (b / p.relative_to(a)).read_bytes()
    pd.testing.assert_frame_equal(
        pd.read_parquet(a / "video/P001/movement.parquet"), pd.read_parquet(b / "video/P001/movement.parquet")
    )


def test_skip_preserva_saidas_reais_e_ids_dos_outros(tmp_path):
    real = tmp_path / "video" / "P001" / "findings.jsonl"
    real.parent.mkdir(parents=True)
    real.write_text("", encoding="utf-8")  # simula saída real do I1 (sem desvios)
    written = make_all(tmp_path, skip=["i1"])
    assert "video_findings" not in written and "movement" not in written
    assert real.read_text(encoding="utf-8") == ""
    assert not (tmp_path / "video" / "P001" / "movement.parquet").exists()

    full = tmp_path / "full"
    make_all(full)
    for rel in ("audio/P001/findings.jsonl", "anomaly/P001/findings.jsonl"):
        assert (tmp_path / rel).read_bytes() == (full / rel).read_bytes()

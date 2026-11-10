"""Teste de fumaça do cenário P001 com mocks: a fusão emite o alerta crítico."""

from datetime import timedelta

from app.fusion import fuse, load_fusion_config, read_alerts, write_alerts
from app.loaders import load_findings
from contracts import read_findings


def _critical(alerts):
    return [a for a in alerts if a.level == "critico"]


def test_p001_alerta_critico_pela_regra_do_plano(mocks_dir, tmp_path):
    loaded = load_findings(mocks_dir, "P001")
    assert not loaded.errors
    alerts = fuse(loaded.findings, load_fusion_config())

    (crit,) = _critical(alerts)
    assert crit.patient_id == "P001"
    assert crit.rule == "respiratorio_vocal_spo2"
    by_id = {str(f.finding_id): f for f in loaded.findings}
    used = {(by_id[i].source, by_id[i].type) for i in crit.finding_ids}
    assert ("audio", "indicador_vocal_alterado") in used
    assert ("vitals", "dessaturacao") in used
    assert used & {("audio", "termo_critico"), ("text", "termo_critico")}
    starts = [by_id[i].timestamp_start for i in crit.finding_ids]
    assert max(starts) - min(starts) <= timedelta(minutes=30)

    # Findings 1 e 2 do cenário (vídeo e saída do leito), isolados, viram alertas de atenção.
    isolated = {a.rule: a.level for a in alerts if a.rule.startswith("isolado:")}
    assert isolated["isolado:video:amplitude_incompleta"] == "atencao"
    assert isolated["isolado:movement:saida_do_leito"] == "atencao"

    # alerts.jsonl mostra o mesmo que a fusão
    path = write_alerts(tmp_path / "alerts.jsonl", alerts)
    assert read_alerts(path) == alerts
    assert read_alerts(path)[0].email_sent_at is None


def test_fusao_deterministica(mocks_dir, tmp_path):
    cfg = load_fusion_config()
    a = write_alerts(tmp_path / "a.jsonl", fuse(load_findings(mocks_dir).findings, cfg))
    b = write_alerts(tmp_path / "b.jsonl", fuse(load_findings(mocks_dir).findings, cfg))
    assert a.read_bytes() == b.read_bytes()


def test_sem_alinhamento_temporal_a_regra_nao_dispara(mocks_dir):
    """Controle negativo: com a SpO2 2 h depois, fora da janela, não há alerta crítico."""
    findings = load_findings(mocks_dir, "P001").findings
    shifted = []
    for f in findings:
        if f.type == "dessaturacao":
            f = f.model_copy(
                update={
                    "timestamp_start": f.timestamp_start + timedelta(hours=2),
                    "timestamp_end": f.timestamp_end + timedelta(hours=2),
                }
            )
        shifted.append(f)
    assert not _critical(fuse(shifted, load_fusion_config()))


def test_regra_aceita_termo_respiratorio_so_da_nota(mocks_dir):
    findings = [
        f
        for f in load_findings(mocks_dir, "P001").findings
        if not (f.source == "audio" and f.type == "termo_critico")
    ]
    (crit,) = _critical(fuse(findings, load_fusion_config()))
    text_ids = {str(f.finding_id) for f in read_findings(mocks_dir / "text/P001/findings.jsonl")}
    assert text_ids & set(crit.finding_ids)

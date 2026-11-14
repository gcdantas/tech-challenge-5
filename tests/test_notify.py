"""E-mail do alerta crítico: envio único, contingência .eml e segredos fora dos arquivos."""

import smtplib
from email import message_from_bytes, policy

import pytest

from app.fusion import fuse, load_fusion_config, read_alerts, write_alerts
from app.loaders import load_findings
from app.notify import (
    NOT_CONFIGURED,
    AlertsConfig,
    SmtpSettings,
    load_alerts_config,
    notify,
    read_deliveries,
)

SETTINGS = SmtpSettings(
    host="smtp.exemplo.test",
    port=587,
    user="alertas@exemplo.test",
    password="segredo-de-teste",
    sender="alertas@exemplo.test",
    recipients=["plantao@exemplo.test", "enfermagem@exemplo.test"],
)


@pytest.fixture
def cenario(mocks_dir):
    findings = load_findings(mocks_dir, "P001").findings
    return findings, fuse(findings, load_fusion_config())


class Caixa:
    """Substitui o servidor SMTP: guarda as mensagens em vez de enviar."""

    def __init__(self, falha: Exception | None = None):
        self.mensagens = []
        self.falha = falha

    def __call__(self, msg, settings, timeout):
        if self.falha:
            raise self.falha
        self.mensagens.append(msg)


def test_config_padrao_envia_so_o_critico():
    cfg = load_alerts_config()
    assert cfg.email.enabled
    assert cfg.email.levels == ["critico"]


def test_smtp_envia_um_email_por_alerta_critico(cenario, tmp_path):
    findings, alerts = cenario
    caixa = Caixa()
    out = notify(alerts, findings, tmp_path, settings=SETTINGS, sender=caixa)

    (msg,) = caixa.mensagens
    assert msg["To"] == "plantao@exemplo.test, enfermagem@exemplo.test"
    assert msg["Subject"].startswith("[DEMONSTRAÇÃO] Alerta CRÍTICO - P001")
    corpo = msg.get_content()
    assert "respiratorio_vocal_spo2" in corpo
    assert "dessaturacao" in corpo and "indicador_vocal_alterado" in corpo
    assert "a decisão é clínica" in corpo

    (crit,) = [a for a in out if a.level == "critico"]
    assert crit.email_sent_at is not None
    assert all(a.email_sent_at is None for a in out if a.level != "critico")

    # alerts.jsonl guarda o horário do envio
    path = write_alerts(tmp_path / "alerts.jsonl", out)
    assert [a.email_sent_at for a in read_alerts(path) if a.level == "critico"] == [crit.email_sent_at]


def test_nao_reenvia_quando_a_fusao_roda_de_novo(cenario, tmp_path):
    findings, alerts = cenario
    caixa = Caixa()
    primeiro = notify(alerts, findings, tmp_path, settings=SETTINGS, sender=caixa)
    segundo = notify(alerts, findings, tmp_path, settings=SETTINGS, sender=caixa)
    assert len(caixa.mensagens) == 1
    assert primeiro == segundo


def test_sem_smtp_grava_eml_e_nao_marca_como_enviado(cenario, tmp_path):
    findings, alerts = cenario
    out = notify(alerts, findings, tmp_path, settings=None)

    (crit,) = [a for a in out if a.level == "critico"]
    assert crit.email_sent_at is None
    delivery = read_deliveries(tmp_path)[crit.alert_id]
    assert delivery.channel == "eml" and delivery.error == NOT_CONFIGURED
    eml = message_from_bytes((tmp_path / delivery.eml_path).read_bytes(), policy=policy.default)
    assert eml["X-Alert-Id"] == crit.alert_id
    assert "respiratorio_vocal_spo2" in eml.get_content()


def test_eml_por_falta_de_configuracao_e_enviado_quando_o_smtp_aparece(cenario, tmp_path):
    findings, alerts = cenario
    notify(alerts, findings, tmp_path, settings=None)
    caixa = Caixa()
    out = notify(alerts, findings, tmp_path, settings=SETTINGS, sender=caixa)
    assert len(caixa.mensagens) == 1
    assert [a.email_sent_at is not None for a in out if a.level == "critico"] == [True]


def test_falha_do_smtp_cai_no_eml_e_nao_insiste(cenario, tmp_path):
    findings, alerts = cenario
    caixa = Caixa(falha=smtplib.SMTPAuthenticationError(535, b"credenciais recusadas"))
    out = notify(alerts, findings, tmp_path, settings=SETTINGS, sender=caixa)

    (crit,) = [a for a in out if a.level == "critico"]
    assert crit.email_sent_at is None
    delivery = read_deliveries(tmp_path)[crit.alert_id]
    assert delivery.channel == "eml" and "SMTPAuthenticationError" in delivery.error
    assert (tmp_path / delivery.eml_path).exists()

    # nova rodada com o servidor funcionando: não tenta sozinho depois de uma falha real
    ok = Caixa()
    notify(alerts, findings, tmp_path, settings=SETTINGS, sender=ok)
    assert ok.mensagens == []


def test_senha_nao_aparece_em_arquivo_nenhum(cenario, tmp_path):
    findings, alerts = cenario
    notify(alerts, findings, tmp_path, settings=SETTINGS, sender=Caixa(falha=OSError("sem rede")))
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert SETTINGS.password.encode() not in path.read_bytes()
    assert SETTINGS.password not in repr(SETTINGS)


def test_desligado_no_yaml_nao_faz_nada(cenario, tmp_path):
    findings, alerts = cenario
    cfg = AlertsConfig.model_validate({"email": {"enabled": False}})
    caixa = Caixa()
    assert notify(alerts, findings, tmp_path, cfg=cfg, settings=SETTINGS, sender=caixa) == alerts
    assert caixa.mensagens == [] and not list(tmp_path.iterdir())


def test_settings_do_ambiente():
    assert SmtpSettings.from_env({}) is None
    assert SmtpSettings.from_env({"SMTP_HOST": "smtp.exemplo.test", "SMTP_USER": "a@exemplo.test"}) is None
    s = SmtpSettings.from_env(
        {
            "SMTP_HOST": "smtp.exemplo.test",
            "SMTP_PORT": "",
            "SMTP_USER": "a@exemplo.test",
            "SMTP_PASSWORD": "x",
            "ALERT_EMAIL_TO": "b@exemplo.test; c@exemplo.test",
        }
    )
    assert s.port == 587 and s.sender == "a@exemplo.test"
    assert s.recipients == ["b@exemplo.test", "c@exemplo.test"]

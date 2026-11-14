"""Notificação por e-mail do alerta crítico (I4; plano v5, correção 3).

Fluxo final do alerta à equipe médica: depois da fusão, cada alerta cujo nível
está em ``config/alerts.yaml`` gera um e-mail. Três garantias:

1. **Um envio por alerta.** O ``alert_id`` é determinístico, e cada entrega fica
   registrada em ``outputs/alerts_delivery.jsonl``. O Streamlit roda a fusão a
   cada interação; sem esse registro, a equipe receberia o mesmo e-mail várias
   vezes.
2. **Contingência.** Sem SMTP configurado, ou se o envio falhar, a mensagem é
   gravada como ``.eml`` em ``outputs/alerts/eml/``. O alerta nunca se perde, e
   o arquivo abre em qualquer cliente de e-mail.
3. **Segredos fora do Git.** Servidor, usuário, senha, remetente e
   destinatários vêm só do ``.env``.

``email_sent_at`` só é preenchido quando o servidor SMTP aceitou a mensagem.
"""

from __future__ import annotations

import os
import smtplib
import ssl
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from email.utils import format_datetime, make_msgid
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from app.fusion import Alert, AlertLevel
from contracts.finding import Finding
from contracts.patients import REPO_ROOT

DEFAULT_ALERTS_CONFIG = REPO_ROOT / "config" / "alerts.yaml"
DELIVERY_LOG = "alerts_delivery.jsonl"
BRT = timezone(timedelta(hours=-3))

LEVEL_LABEL = {"critico": "CRÍTICO", "urgente": "URGENTE", "atencao": "ATENÇÃO"}
NOT_CONFIGURED = "SMTP não configurado no .env"


class EmailConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    levels: list[AlertLevel] = Field(default_factory=lambda: ["critico"])
    subject_prefix: str = "[DEMONSTRAÇÃO]"
    eml_dir: str = "alerts/eml"
    timeout_seconds: float = 15


class AlertsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: EmailConfig = Field(default_factory=EmailConfig)


def load_alerts_config(path: str | Path = DEFAULT_ALERTS_CONFIG) -> AlertsConfig:
    with Path(path).open(encoding="utf-8") as fh:
        return AlertsConfig.model_validate(yaml.safe_load(fh) or {})


class SmtpSettings(BaseModel):
    """Credenciais lidas do ambiente. Nunca são gravadas em arquivo nem em log."""

    model_config = ConfigDict(extra="forbid")

    host: str
    port: int = 587
    user: str = ""
    password: str = Field(default="", repr=False)
    sender: str
    recipients: list[str]

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> SmtpSettings | None:
        """Devolve None se faltar servidor, remetente ou destinatário."""
        env = os.environ if env is None else env
        host = env.get("SMTP_HOST", "").strip()
        user = env.get("SMTP_USER", "").strip()
        sender = env.get("ALERT_EMAIL_FROM", "").strip() or user
        recipients = [r.strip() for r in env.get("ALERT_EMAIL_TO", "").replace(";", ",").split(",") if r.strip()]
        if not (host and sender and recipients):
            return None
        return cls(
            host=host,
            port=int(env.get("SMTP_PORT", "").strip() or 587),
            user=user,
            password=env.get("SMTP_PASSWORD", ""),
            sender=sender,
            recipients=recipients,
        )


class Delivery(BaseModel):
    """Linha de alerts_delivery.jsonl: o que aconteceu com o e-mail de um alerta."""

    model_config = ConfigDict(extra="forbid")

    alert_id: str
    channel: Literal["smtp", "eml"]
    recorded_at: datetime
    sent_at: datetime | None = None
    recipients: list[str] = Field(default_factory=list)
    eml_path: str | None = None
    error: str | None = None


def read_deliveries(out_dir: Path) -> dict[str, Delivery]:
    """Última entrega registrada de cada alerta."""
    path = out_dir / DELIVERY_LOG
    if not path.exists():
        return {}
    latest: dict[str, Delivery] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                d = Delivery.model_validate_json(line)
                latest[d.alert_id] = d
    return latest


def _append_delivery(out_dir: Path, delivery: Delivery) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / DELIVERY_LOG).open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(delivery.model_dump_json() + "\n")


def build_email(
    alert: Alert,
    findings: Iterable[Finding],
    sender: str,
    recipients: list[str],
    subject_prefix: str = "",
    now: datetime | None = None,
) -> EmailMessage:
    """Monta a mensagem: nível, paciente, horário, regra e as evidências da fusão."""
    by_id = {str(f.finding_id): f for f in findings}
    evidence = [by_id[i] for i in alert.finding_ids if i in by_id]
    when = alert.created_at.strftime("%d/%m/%Y %H:%M:%S")

    lines = [
        f"ALERTA {LEVEL_LABEL[alert.level]} - paciente {alert.patient_id}",
        f"Emitido em: {when}",
        f"Regra: {alert.rule}",
        "",
        alert.description.strip(),
        "",
        f"Evidências ({len(alert.finding_ids)}):",
    ]
    for f in evidence:
        lines.append(
            f"  - {f.timestamp_start.strftime('%H:%M:%S')} | {f.source} | {f.type} | "
            f"severidade {f.severity} | score {f.score:.2f}"
        )
        lines.append(f"    {f.description}")
    missing = len(alert.finding_ids) - len(evidence)
    if missing:
        lines.append(f"  - {missing} evidência(s) não encontrada(s) em outputs/")
    lines += [
        "",
        f"Janela analisada: {alert.window_start.strftime('%H:%M:%S')} a {alert.window_end.strftime('%H:%M:%S')}",
        f"Identificador do alerta: {alert.alert_id}",
        "",
        "Protótipo de demonstração técnica com paciente sintético. O sistema sinaliza",
        "risco e apoia a equipe; a decisão é clínica.",
    ]

    msg = EmailMessage()
    subject = f"Alerta {LEVEL_LABEL[alert.level]} - {alert.patient_id} - {alert.created_at.strftime('%H:%M')}"
    msg["Subject"] = f"{subject_prefix} {subject}".strip()
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg["Date"] = format_datetime(now or datetime.now(BRT))
    msg["Message-ID"] = make_msgid(idstring=alert.alert_id, domain="tech-challenge.local")
    msg["X-Alert-Id"] = alert.alert_id
    msg.set_content("\n".join(lines))
    return msg


def send_smtp(msg: EmailMessage, settings: SmtpSettings, timeout: float = 15) -> None:
    """Envia pela porta 465 (TLS direto) ou por STARTTLS nas demais."""
    context = ssl.create_default_context()
    if settings.port == 465:
        server: smtplib.SMTP = smtplib.SMTP_SSL(settings.host, settings.port, timeout=timeout, context=context)
    else:
        server = smtplib.SMTP(settings.host, settings.port, timeout=timeout)
    with server:
        if settings.port != 465:
            server.starttls(context=context)
        if settings.user:
            server.login(settings.user, settings.password)
        server.send_message(msg)


def _write_eml(msg: EmailMessage, out_dir: Path, eml_dir: str, alert_id: str) -> Path:
    folder = out_dir / eml_dir
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{alert_id}.eml"
    path.write_bytes(bytes(msg))
    return path


def _should_try(previous: Delivery | None, settings: SmtpSettings | None) -> bool:
    if previous is None:
        return True
    if previous.channel == "smtp":
        return False
    # Havia só o .eml porque o SMTP não estava configurado: tenta agora que está.
    # Depois de uma falha real de envio não insiste sozinho (cada recarga da tela
    # esperaria o timeout); para tentar de novo, apague a linha do registro.
    return settings is not None and previous.error == NOT_CONFIGURED


def notify(
    alerts: list[Alert],
    findings: Iterable[Finding],
    out_dir: Path,
    cfg: AlertsConfig | None = None,
    settings: SmtpSettings | None = None,
    sender: Callable[[EmailMessage, SmtpSettings, float], None] = send_smtp,
    now: Callable[[], datetime] = lambda: datetime.now(BRT),
) -> list[Alert]:
    """Envia o e-mail dos alertas ainda não entregues e devolve os alertas com
    ``email_sent_at`` preenchido a partir do registro de entregas."""
    cfg = cfg or load_alerts_config()
    if not cfg.email.enabled:
        return alerts
    findings = list(findings)
    deliveries = read_deliveries(out_dir)

    for alert in alerts:
        if alert.level not in cfg.email.levels:
            continue
        if not _should_try(deliveries.get(alert.alert_id), settings):
            continue
        moment = now()
        msg = build_email(
            alert,
            findings,
            sender=settings.sender if settings else "alertas@tech-challenge.local",
            recipients=settings.recipients if settings else ["equipe-medica@tech-challenge.local"],
            subject_prefix=cfg.email.subject_prefix,
            now=moment,
        )
        error = NOT_CONFIGURED
        if settings is not None:
            try:
                sender(msg, settings, cfg.email.timeout_seconds)
                error = None
            except (OSError, smtplib.SMTPException) as exc:
                error = f"{type(exc).__name__}: {exc}"
        if error is None:
            delivery = Delivery(
                alert_id=alert.alert_id,
                channel="smtp",
                recorded_at=moment,
                sent_at=moment,
                recipients=settings.recipients,
            )
        else:
            eml = _write_eml(msg, out_dir, cfg.email.eml_dir, alert.alert_id)
            delivery = Delivery(
                alert_id=alert.alert_id,
                channel="eml",
                recorded_at=moment,
                eml_path=eml.relative_to(out_dir).as_posix(),
                error=error,
            )
        _append_delivery(out_dir, delivery)
        deliveries[alert.alert_id] = delivery

    return [
        a.model_copy(update={"email_sent_at": deliveries[a.alert_id].sent_at})
        if a.alert_id in deliveries and deliveries[a.alert_id].sent_at
        else a
        for a in alerts
    ]

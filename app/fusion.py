"""Fusão tardia mínima e alertas (versão provisória do M1; dono: I4).

Fusão tardia: cada modalidade é processada separadamente e convertida em
evidências estruturadas (Findings), combinadas por paciente e proximidade
temporal. Regras explícitas têm precedência; Findings fora de regras viram
alertas isolados pela severidade.

Ainda NÃO implementado (fica para o I4): combinação ponderada dos scores,
e-mail no alerta crítico, confirmação de leitura, processamento incremental.
"""

from __future__ import annotations

import itertools
import json
import uuid
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from contracts.finding import Finding
from contracts.patients import REPO_ROOT

DEFAULT_FUSION_CONFIG = REPO_ROOT / "config" / "fusion.yaml"
AlertLevel = Literal["atencao", "urgente", "critico"]
LEVEL_ORDER = {"atencao": 0, "urgente": 1, "critico": 2}

_ALERT_NS = uuid.UUID("6f1c2a52-0d55-4c55-9f1e-6b8b3c9c0a10")


class Matcher(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    type: str
    respiratory_term: bool = False


class Rule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    level: AlertLevel
    description: str = ""
    conditions: list[list[Matcher]]


class FusionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    window_minutes: float = 30
    cooldown_minutes: float = 30
    respiratory_terms: list[str] = Field(default_factory=list)
    severity_to_level: dict[str, AlertLevel]
    rules: list[Rule] = Field(default_factory=list)


class Alert(BaseModel):
    """Linha de alerts.jsonl (formato provisório)."""

    model_config = ConfigDict(extra="forbid")

    alert_id: str
    patient_id: str
    level: AlertLevel
    rule: str
    description: str
    finding_ids: list[str]
    window_start: datetime
    window_end: datetime
    created_at: datetime
    email_sent_at: datetime | None = None

    def to_json_line(self) -> str:
        return json.dumps(self.model_dump(mode="json"), ensure_ascii=False)


def load_fusion_config(path: str | Path = DEFAULT_FUSION_CONFIG) -> FusionConfig:
    with Path(path).open(encoding="utf-8") as fh:
        return FusionConfig.model_validate(yaml.safe_load(fh))


def _matches(f: Finding, m: Matcher, cfg: FusionConfig) -> bool:
    if f.source != m.source or f.type != m.type:
        return False
    if m.respiratory_term:
        term = str(f.evidence.get("term", "")).lower()
        return any(t.lower() in term for t in cfg.respiratory_terms)
    return True


def _alert_id(patient_id: str, rule: str, finding_ids: Iterable[str]) -> str:
    key = "|".join([patient_id, rule, *sorted(finding_ids)])
    return str(uuid.uuid5(_ALERT_NS, key))


def _rule_alerts(patient_id: str, findings: list[Finding], rule: Rule, cfg: FusionConfig) -> list[Alert]:
    candidates = [[f for f in findings if any(_matches(f, m, cfg) for m in alts)] for alts in rule.conditions]
    if not all(candidates):
        return []

    window = timedelta(minutes=cfg.window_minutes)
    combos = []
    for combo in itertools.product(*candidates):
        if len({f.finding_id for f in combo}) < len(combo):
            continue
        starts = [f.timestamp_start for f in combo]
        if max(starts) - min(starts) <= window:
            # O alerta nasce quando chega o último Finding necessário.
            combos.append((max(f.timestamp_end for f in combo), min(starts), combo))
    combos.sort(key=lambda c: (c[0], c[1]))

    alerts: list[Alert] = []
    last_fired: datetime | None = None
    cooldown = timedelta(minutes=cfg.cooldown_minutes)
    for created, wstart, combo in combos:
        if last_fired is not None and created - last_fired < cooldown:
            continue
        ids = [str(f.finding_id) for f in combo]
        alerts.append(
            Alert(
                alert_id=_alert_id(patient_id, rule.name, ids),
                patient_id=patient_id,
                level=rule.level,
                rule=rule.name,
                description=rule.description,
                finding_ids=ids,
                window_start=wstart,
                window_end=created,
                created_at=created,
            )
        )
        last_fired = created
    return alerts


def fuse(findings: Iterable[Finding], cfg: FusionConfig) -> list[Alert]:
    """Aplica as regras por paciente e devolve os alertas, ordenados por horário."""
    by_patient: dict[str, list[Finding]] = defaultdict(list)
    for f in findings:
        by_patient[f.patient_id].append(f)

    alerts: list[Alert] = []
    for patient_id, items in sorted(by_patient.items()):
        items.sort(key=lambda f: f.timestamp_start)
        used: set[str] = set()
        for rule in sorted(cfg.rules, key=lambda r: -LEVEL_ORDER[r.level]):
            for alert in _rule_alerts(patient_id, items, rule, cfg):
                alerts.append(alert)
                used.update(alert.finding_ids)
        for f in items:
            fid = str(f.finding_id)
            if fid in used:
                continue
            alerts.append(
                Alert(
                    alert_id=_alert_id(patient_id, f"isolado:{f.source}:{f.type}", [fid]),
                    patient_id=patient_id,
                    level=cfg.severity_to_level[f.severity],
                    rule=f"isolado:{f.source}:{f.type}",
                    description=f.description,
                    finding_ids=[fid],
                    window_start=f.timestamp_start,
                    window_end=f.timestamp_end,
                    created_at=f.timestamp_end,
                )
            )
    alerts.sort(key=lambda a: (a.created_at, -LEVEL_ORDER[a.level], a.alert_id))
    return alerts


def write_alerts(path: str | Path, alerts: Iterable[Alert]) -> Path:
    """Grava alerts.jsonl. Nesta versão mínima o arquivo é reescrito a cada fusão
    (resultado determinístico); o I4 decide o modo incremental."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for a in alerts:
            fh.write(a.to_json_line() + "\n")
    return path


def read_alerts(path: str | Path) -> list[Alert]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [Alert.model_validate_json(line) for line in fh if line.strip()]


def alerts_summary(alerts: Iterable[Alert]) -> dict[str, Any]:
    counts: dict[str, int] = defaultdict(int)
    for a in alerts:
        counts[a.level] += 1
    return dict(counts)

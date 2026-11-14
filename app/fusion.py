"""Fusão tardia e alertas (dono: I4).

Fusão tardia: cada modalidade é processada separadamente e convertida em
evidências estruturadas (Findings), combinadas por paciente e proximidade
temporal, em três etapas por paciente:

1. **Regras explícitas** (``rules`` em ``config/fusion.yaml``), com precedência.
2. **Combinação ponderada dos scores** (``weighted``): Findings de fontes
   diferentes na mesma janela viram um único alerta, com score
   ``soma(peso_fonte * maior_score_da_fonte) / soma(pesos das fontes presentes)``.
   É uma heurística de priorização do protótipo, não probabilidade clínica.
3. **Findings isolados**: o que sobra vira alerta pela severidade.

Deduplicação e cooldown por paciente e tipo de alerta (a regra, ``ponderado``
ou ``isolado:<fonte>:<tipo>``): dentro do cooldown, um novo alerta do mesmo
tipo não é emitido; seus Findings entram em ``suppressed_finding_ids`` do
alerta anterior. Se o nível subir (escalonamento), o alerta sai mesmo assim.

Ainda NÃO implementado: confirmação de leitura, processamento incremental.
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
from pydantic import BaseModel, ConfigDict, Field, field_validator

from contracts.finding import Finding, Source
from contracts.patients import REPO_ROOT

DEFAULT_FUSION_CONFIG = REPO_ROOT / "config" / "fusion.yaml"
AlertLevel = Literal["atencao", "urgente", "critico"]
LEVEL_ORDER = {"atencao": 0, "urgente": 1, "critico": 2}
WEIGHTED_RULE = "ponderado"

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


class WeightedConfig(BaseModel):
    """Combinação ponderada dos scores (heurística de priorização)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    # Peso de cada fonte; fonte sem peso não entra na combinação.
    source_weights: dict[Source, float] = Field(default_factory=dict)
    # Mínimo de fontes diferentes na janela para combinar (1 fonte = isolado).
    min_sources: int = Field(default=2, ge=2)
    # Score combinado mínimo de cada nível. Nível ausente não é alcançado só
    # pelo score (por padrão, a combinação nunca gera "critico" sozinha).
    thresholds: dict[AlertLevel, float] = Field(default_factory=dict)

    @field_validator("source_weights")
    @classmethod
    def _pesos_positivos(cls, value: dict[str, float]) -> dict[str, float]:
        bad = [k for k, w in value.items() if w <= 0]
        if bad:
            raise ValueError(f"pesos devem ser > 0: {bad}")
        return value

    @field_validator("thresholds")
    @classmethod
    def _limiares_0_1(cls, value: dict[str, float]) -> dict[str, float]:
        bad = [k for k, t in value.items() if not 0.0 <= t <= 1.0]
        if bad:
            raise ValueError(f"limiares devem estar em 0–1: {bad}")
        return value

    def level_for(self, score: float) -> AlertLevel | None:
        """Maior nível cujo limiar o score atinge; None se não atinge nenhum."""
        reached = [lvl for lvl, t in self.thresholds.items() if score >= t]
        return max(reached, key=LEVEL_ORDER.__getitem__, default=None)


class FusionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    window_minutes: float = 30
    # Cooldown por paciente e tipo de alerta (regra, "ponderado" ou isolado:<fonte>:<tipo>).
    cooldown_minutes: float = 30
    respiratory_terms: list[str] = Field(default_factory=list)
    severity_to_level: dict[str, AlertLevel]
    rules: list[Rule] = Field(default_factory=list)
    weighted: WeightedConfig = Field(default_factory=lambda: WeightedConfig(enabled=False))


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
    # Score combinado (ponderado por fonte) dos Findings do alerta, para priorizar.
    score: float | None = None
    # Findings do mesmo tipo que chegaram dentro do cooldown e não geraram alerta novo.
    suppressed_finding_ids: list[str] = Field(default_factory=list)

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


def weighted_score(findings: Iterable[Finding], cfg: FusionConfig) -> float:
    """Média ponderada, por fonte, do maior score de cada fonte.

    Fonte sem peso em ``weighted.source_weights`` conta com peso 1. Com uma
    única fonte, o resultado é o maior score dela.
    """
    best: dict[str, float] = {}
    for f in findings:
        best[f.source] = max(best.get(f.source, 0.0), f.score)
    if not best:
        return 0.0
    weights = {src: cfg.weighted.source_weights.get(src, 1.0) for src in best}
    total = sum(weights[src] * best[src] for src in best) / sum(weights.values())
    return round(total, 3)


class _Dedup:
    """Deduplicação e cooldown por paciente e tipo de alerta.

    Um alerta é emitido se não houver anterior do mesmo tipo, se o cooldown já
    passou ou se o nível subiu. Caso contrário, seus Findings entram em
    ``suppressed_finding_ids`` do último alerta emitido daquele tipo.
    """

    def __init__(self, cooldown: timedelta) -> None:
        self.cooldown = cooldown
        self.last: dict[str, Alert] = {}
        self.emitted: list[Alert] = []

    def offer(self, alert: Alert) -> Alert:
        prev = self.last.get(alert.rule)
        if (
            prev is None
            or alert.created_at - prev.created_at >= self.cooldown
            or LEVEL_ORDER[alert.level] > LEVEL_ORDER[prev.level]
        ):
            self.last[alert.rule] = alert
            self.emitted.append(alert)
            return alert
        known = set(prev.finding_ids) | set(prev.suppressed_finding_ids)
        prev.suppressed_finding_ids.extend(i for i in alert.finding_ids if i not in known)
        return prev


def _make_alert(
    patient_id: str,
    rule: str,
    level: AlertLevel,
    description: str,
    group: list[Finding],
    cfg: FusionConfig,
) -> Alert:
    """Alerta nascido quando chega o último Finding do grupo."""
    ids = [str(f.finding_id) for f in group]
    created = max(f.timestamp_end for f in group)
    return Alert(
        alert_id=_alert_id(patient_id, rule, ids),
        patient_id=patient_id,
        level=level,
        rule=rule,
        description=description,
        finding_ids=ids,
        window_start=min(f.timestamp_start for f in group),
        window_end=created,
        created_at=created,
        score=weighted_score(group, cfg),
    )


def _rule_combos(findings: list[Finding], rule: Rule, cfg: FusionConfig) -> list[list[Finding]]:
    """Combinações de Findings que satisfazem a regra na mesma janela, em ordem de chegada."""
    candidates = [[f for f in findings if any(_matches(f, m, cfg) for m in alts)] for alts in rule.conditions]
    if not all(candidates):
        return []

    window = timedelta(minutes=cfg.window_minutes)
    combos = []
    # Os candidatos já vêm em ordem de início; o índice desempata pela ordem de chegada.
    for idx, combo in enumerate(itertools.product(*candidates)):
        if len({f.finding_id for f in combo}) < len(combo):
            continue
        starts = [f.timestamp_start for f in combo]
        if max(starts) - min(starts) <= window:
            created = max(f.timestamp_end for f in combo)
            combos.append((created, min(starts), idx, list(combo)))
    combos.sort(key=lambda c: c[:3])
    return [c[3] for c in combos]


def _severity_level(group: Iterable[Finding], cfg: FusionConfig) -> AlertLevel:
    return max((cfg.severity_to_level[f.severity] for f in group), key=LEVEL_ORDER.__getitem__)


def _weighted_groups(findings: list[Finding], cfg: FusionConfig) -> list[tuple[list[Finding], float, AlertLevel]]:
    """Agrupa Findings de fontes diferentes na mesma janela (gulosa, em ordem de início).

    Cada grupo começa num Finding âncora e reúne os Findings ainda livres que
    começam até ``window_minutes`` depois dele. O grupo vale se tiver ao menos
    ``min_sources`` fontes e o score combinado atingir algum limiar; o nível é o
    maior entre o do score e o da severidade de cada Finding (a combinação
    nunca rebaixa um Finding).
    """
    wcfg = cfg.weighted
    window = timedelta(minutes=cfg.window_minutes)
    eligible = [f for f in findings if f.source in wcfg.source_weights]
    taken: set[str] = set()
    groups = []
    for anchor in eligible:
        if str(anchor.finding_id) in taken:
            continue
        group = [
            f
            for f in eligible
            if str(f.finding_id) not in taken
            and anchor.timestamp_start <= f.timestamp_start <= anchor.timestamp_start + window
        ]
        if len({f.source for f in group}) < wcfg.min_sources:
            continue
        score = weighted_score(group, cfg)
        by_score = wcfg.level_for(score)
        if by_score is None:
            continue
        level = max(by_score, _severity_level(group, cfg), key=LEVEL_ORDER.__getitem__)
        groups.append((group, score, level))
        taken.update(str(f.finding_id) for f in group)
    return groups


def _weighted_description(group: list[Finding], score: float, cfg: FusionConfig) -> str:
    best: dict[str, Finding] = {}
    for f in group:
        if f.source not in best or f.score > best[f.source].score:
            best[f.source] = f
    parts = [
        f"{src} {f.score:.2f} (peso {cfg.weighted.source_weights.get(src, 1.0):g}, {f.type})"
        for src, f in sorted(best.items())
    ]
    return (
        f"Combinação ponderada de {len(best)} fontes na janela de {cfg.window_minutes:g} min: "
        f"score {score:.2f} = " + "; ".join(parts) + ". Heurística de priorização do protótipo, "
        "não é probabilidade clínica calibrada."
    )


def fuse(findings: Iterable[Finding], cfg: FusionConfig) -> list[Alert]:
    """Aplica regras, combinação ponderada e isolados por paciente, com
    deduplicação e cooldown por paciente e tipo. Devolve os alertas por horário."""
    by_patient: dict[str, list[Finding]] = defaultdict(list)
    for f in findings:
        by_patient[f.patient_id].append(f)

    alerts: list[Alert] = []
    cooldown = timedelta(minutes=cfg.cooldown_minutes)
    for patient_id, items in sorted(by_patient.items()):
        items.sort(key=lambda f: (f.timestamp_start, str(f.finding_id)))
        dedup = _Dedup(cooldown)
        used: set[str] = set()

        def consume(alert: Alert) -> None:
            kept = dedup.offer(alert)
            used.update(alert.finding_ids)
            used.update(kept.suppressed_finding_ids)

        # 1. Regras explícitas, das mais graves para as menos graves.
        for rule in sorted(cfg.rules, key=lambda r: -LEVEL_ORDER[r.level]):
            for combo in _rule_combos(items, rule, cfg):
                consume(_make_alert(patient_id, rule.name, rule.level, rule.description, combo, cfg))

        # 2. Combinação ponderada dos scores nos Findings que sobraram.
        if cfg.weighted.enabled:
            free = [f for f in items if str(f.finding_id) not in used]
            for group, score, level in _weighted_groups(free, cfg):
                desc = _weighted_description(group, score, cfg)
                consume(_make_alert(patient_id, WEIGHTED_RULE, level, desc, group, cfg))

        # 3. O que sobrou vira alerta isolado pela severidade.
        for f in items:
            if str(f.finding_id) in used:
                continue
            level = cfg.severity_to_level[f.severity]
            consume(_make_alert(patient_id, f"isolado:{f.source}:{f.type}", level, f.description, [f], cfg))

        alerts.extend(dedup.emitted)
    alerts.sort(key=lambda a: (a.created_at, -LEVEL_ORDER[a.level], a.alert_id))
    return alerts


def write_alerts(path: str | Path, alerts: Iterable[Alert]) -> Path:
    """Grava alerts.jsonl. O arquivo é reescrito a cada fusão (resultado
    determinístico); o modo incremental ainda não existe."""
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

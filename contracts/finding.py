"""Contrato comum: o Finding (plano v5, seção "Contrato comum").

Finding é a evidência que I1, I2 ou I3 produzem; alerta é o que só o I4 emite,
depois da fusão. Toda saída de módulo que vira alerta é uma lista de Findings
em JSON Lines (um por linha), validada por este modelo.

Regras do plano refletidas aqui:
- schema enxuto: campos extras são rejeitados; o que for específico de um
  módulo vai dentro de ``evidence``;
- ``score`` normalizado em 0–1;
- horários ISO 8601 com fuso -03:00.

Quem mudar o schema abre PR e marca os outros três integrantes.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import UUID4, BaseModel, ConfigDict, Field, field_validator, model_validator

#: Fuso da demonstração (horário de Brasília, sem horário de verão).
DEMO_TZ = timezone(timedelta(hours=-3))

Source = Literal["video", "audio", "text", "vitals", "prescription", "movement"]
Severity = Literal["low", "medium", "high", "critical"]

SOURCES: tuple[str, ...] = Source.__args__
SEVERITIES: tuple[str, ...] = Severity.__args__


class ModelInfo(BaseModel):
    """Modelo (ou regra) que produziu o Finding."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)


class Finding(BaseModel):
    """Evidência estruturada produzida por um módulo de análise."""

    model_config = ConfigDict(extra="forbid")

    finding_id: UUID4 = Field(default_factory=uuid4)
    patient_id: str = Field(min_length=1)
    source: Source
    type: str = Field(min_length=1)
    timestamp_start: datetime
    timestamp_end: datetime
    score: float = Field(ge=0.0, le=1.0)
    severity: Severity
    description: str = Field(min_length=1)
    evidence: dict[str, Any] = Field(default_factory=dict)
    model: ModelInfo
    artifacts: list[str] = Field(default_factory=list)

    @field_validator("timestamp_start", "timestamp_end")
    @classmethod
    def _fuso_demo(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != timedelta(hours=-3):
            raise ValueError("timestamp deve ser ISO 8601 com fuso -03:00")
        return value

    @model_validator(mode="after")
    def _ordem_temporal(self) -> Finding:
        if self.timestamp_end < self.timestamp_start:
            raise ValueError("timestamp_end anterior a timestamp_start")
        return self

    def to_json_line(self) -> str:
        """Serializa em uma linha JSON (formato do findings.jsonl)."""
        return json.dumps(self.model_dump(mode="json"), ensure_ascii=False)


def to_demo_tz(value: datetime) -> datetime:
    """Converte um datetime com fuso para -03:00 (sem alterar o instante)."""
    if value.tzinfo is None:
        raise ValueError("datetime sem fuso; não é possível converter com segurança")
    return value.astimezone(DEMO_TZ)


def read_findings(path: str | Path) -> list[Finding]:
    """Lê e valida um findings.jsonl. Linhas em branco são ignoradas.

    Levanta ``ValueError`` indicando arquivo e linha quando uma linha é inválida.
    """
    path = Path(path)
    findings: list[Finding] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                findings.append(Finding.model_validate_json(line))
            except ValueError as exc:  # pydantic.ValidationError herda de ValueError
                raise ValueError(f"{path}:{lineno}: Finding inválido: {exc}") from exc
    return findings


def write_findings(path: str | Path, findings: Iterable[Finding]) -> Path:
    """Grava Findings em JSON Lines (UTF-8, um por linha), criando as pastas."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for finding in findings:
            fh.write(finding.to_json_line() + "\n")
    return path

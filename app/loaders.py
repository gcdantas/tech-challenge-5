"""Leitura das saídas dos módulos em outputs/ (I4 só consome arquivos e a interface analyze)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from contracts.finding import Finding, read_findings
from contracts.patients import DEFAULT_PATIENTS_PATH, REPO_ROOT

# Pastas de cada módulo em outputs/ (plano v5, tabelas de saída).
MODULE_DIRS = ("video", "audio", "text", "anomaly")


def outputs_dir() -> Path:
    """Pasta de saídas; pode ser trocada com TC_OUTPUTS_DIR (usado nos testes)."""
    env = os.environ.get("TC_OUTPUTS_DIR")
    return Path(env) if env else REPO_ROOT / "outputs"


def patients_path() -> Path:
    env = os.environ.get("TC_PATIENTS_PATH")
    return Path(env) if env else DEFAULT_PATIENTS_PATH


@dataclass
class LoadResult:
    findings: list[Finding] = field(default_factory=list)
    files: list[Path] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def load_findings(out_dir: Path, patient_id: str | None = None) -> LoadResult:
    """Lê outputs/<modulo>/<paciente>/findings.jsonl de todos os módulos."""
    result = LoadResult()
    pid = patient_id or "*"
    for module in MODULE_DIRS:
        for path in sorted(out_dir.glob(f"{module}/{pid}/findings.jsonl")):
            try:
                result.findings.extend(read_findings(path))
                result.files.append(path)
            except (ValueError, OSError) as exc:
                result.errors.append(str(exc))
    result.findings.sort(key=lambda f: (f.timestamp_start, f.source))
    return result


def findings_dataframe(findings: list[Finding]) -> pd.DataFrame:
    rows = []
    for f in findings:
        rows.append(
            {
                "inicio": f.timestamp_start,
                "fim": f.timestamp_end,
                "origem": f.source,
                "tipo": f.type,
                "severidade": f.severity,
                "score": f.score,
                "descricao": f.description,
                "modelo": f"{f.model.name} {f.model.version}",
                "mock": bool(f.evidence.get("mock")),
                "finding_id": str(f.finding_id),
            }
        )
    return pd.DataFrame(rows)


def load_movement(out_dir: Path, patient_id: str) -> pd.DataFrame | None:
    path = out_dir / "video" / patient_id / "movement.parquet"
    return pd.read_parquet(path) if path.exists() else None


def load_transcript(out_dir: Path, patient_id: str) -> dict[str, Any] | None:
    path = out_dir / "audio" / patient_id / "transcript.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

"""Leitura de data/patients.yaml e alinhamento temporal pelo demo_offset.

instante_demo = demo_reference + demo_offset do artefato + tempo interno da mídia
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATIENTS_PATH = REPO_ROOT / "data" / "patients.yaml"

ArtifactKind = Literal["video_physio", "video_bed", "audio", "note", "vitals", "prescriptions"]


class Artifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: ArtifactKind
    path: str | None = None
    demo_offset: float = Field(description="Deslocamento em segundos em relação a demo_reference")


class Patient(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = ""
    artifacts: dict[str, Artifact] = Field(default_factory=dict)

    def artifacts_of(self, kind: str) -> dict[str, Artifact]:
        return {name: a for name, a in self.artifacts.items() if a.kind == kind}


class PatientsFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    demo_reference: datetime
    patients: dict[str, Patient]

    @field_validator("demo_reference")
    @classmethod
    def _fuso(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != timedelta(hours=-3):
            raise ValueError("demo_reference deve ter fuso -03:00")
        return value

    def patient(self, patient_id: str) -> Patient:
        try:
            return self.patients[patient_id]
        except KeyError:
            raise KeyError(f"paciente {patient_id!r} não está em patients.yaml") from None

    def artifact(self, patient_id: str, name: str) -> Artifact:
        patient = self.patient(patient_id)
        try:
            return patient.artifacts[name]
        except KeyError:
            raise KeyError(f"artefato {name!r} não existe para {patient_id}") from None

    def find_artifact_by_path(
        self, patient_id: str, media_path: str | Path, root: Path = REPO_ROOT
    ) -> tuple[str, Artifact] | None:
        """Procura o artefato do paciente cujo ``path`` aponta para ``media_path``."""
        target = _resolve(media_path, root)
        for name, art in self.patient(patient_id).artifacts.items():
            if art.path and _resolve(art.path, root) == target:
                return name, art
        return None

    def demo_time(self, offset_seconds: float, media_seconds: float = 0.0) -> datetime:
        """Instante na linha do tempo da demo (fuso -03:00)."""
        return self.demo_reference + timedelta(seconds=offset_seconds + media_seconds)

    def artifact_time(self, patient_id: str, name: str, media_seconds: float = 0.0) -> datetime:
        return self.demo_time(self.artifact(patient_id, name).demo_offset, media_seconds)


def _resolve(path: str | Path, root: Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = root / p
    return p.resolve()


def load_patients(path: str | Path = DEFAULT_PATIENTS_PATH) -> PatientsFile:
    with Path(path).open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return PatientsFile.model_validate(raw)

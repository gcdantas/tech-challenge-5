"""Contexto de uma mídia: paciente, cenário e alinhamento temporal (demo_offset)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from contracts.patients import REPO_ROOT, PatientsFile, load_patients

log = logging.getLogger(__name__)

Scenario = Literal["physio", "bed"]
KIND_TO_SCENARIO: dict[str, Scenario] = {"video_physio": "physio", "video_bed": "bed"}


@dataclass
class MediaContext:
    input: Path
    patient_id: str
    scenario: Scenario
    demo_offset: float
    artifact: str | None
    patients: PatientsFile

    @property
    def stem(self) -> str:
        return self.input.stem

    def at(self, media_seconds: float) -> datetime:
        """Instante na linha do tempo da demo (fuso -03:00)."""
        return self.patients.demo_time(self.demo_offset, media_seconds)


def resolve_context(
    input_path: str | Path,
    patient_id: str,
    scenario: Scenario | None = None,
    patients: PatientsFile | None = None,
) -> MediaContext:
    """Descobre cenário e demo_offset pelo patients.yaml.

    Se o vídeo não estiver no patients.yaml, ``scenario`` é obrigatório e o
    demo_offset vale 0 (início no demo_reference), com aviso no log.
    """
    pf = patients or load_patients()
    pf.patient(patient_id)  # valida o paciente
    path = Path(input_path)
    found = pf.find_artifact_by_path(patient_id, path)
    if found is not None:
        name, art = found
        if art.kind not in KIND_TO_SCENARIO:
            raise ValueError(f"artefato {name!r} de {patient_id} é {art.kind!r}, não é vídeo")
        yaml_scenario = KIND_TO_SCENARIO[art.kind]
        if scenario is not None and scenario != yaml_scenario:
            raise ValueError(
                f"cenário {scenario!r} diverge do patients.yaml ({name!r} é {yaml_scenario!r})"
            )
        return MediaContext(path, patient_id, yaml_scenario, art.demo_offset, name, pf)

    if scenario is None:
        raise ValueError(
            f"{path} não está em patients.yaml para {patient_id}; informe o cenário (--scenario physio|bed)"
        )
    log.warning("%s fora do patients.yaml: usando demo_offset = 0", path)
    return MediaContext(path, patient_id, scenario, 0.0, None, pf)


def artifact_ref(path: str | Path) -> str:
    """Caminho para o campo ``artifacts``: relativo à raiz do repositório quando possível."""
    p = Path(path).resolve()
    try:
        return p.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return p.as_posix()

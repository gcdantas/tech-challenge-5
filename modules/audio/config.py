"""Configuração validada e caminhos independentes do diretório de execução."""
from pathlib import Path
import re

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from contracts.patients import REPO_ROOT


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class SpeechConfig(Strict):
    language: str = "pt-BR"
    timeout_seconds: float = Field(default=600, gt=0)


class LanguageConfig(Strict):
    language: str = "pt"
    model_version: str = "latest"
    api_version: str = "2023-04-01"
    negative_threshold: float = Field(default=0.9, ge=0, le=1)
    min_sentence_chars: int = Field(default=8, ge=1)
    entity_threshold: float = Field(default=0.7, ge=0, le=1)


class SeverityConfig(Strict):
    medium: float = Field(default=0.5, ge=0, le=1)
    high: float = Field(default=0.85, ge=0, le=1)
    critical: float = Field(default=0.98, ge=0, le=1)

    @model_validator(mode="after")
    def ordered(self):
        if not self.medium < self.high < self.critical:
            raise ValueError("limiares de severidade devem ser crescentes")
        return self

    def level(self, score: float) -> str:
        for name in ("critical", "high", "medium"):
            if score >= getattr(self, name):
                return name
        return "low"


class Term(Strict):
    term: str = Field(min_length=1)
    pattern: str
    score: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def regex(self):
        try:
            re.compile(self.pattern)
        except re.error as exc:
            raise ValueError("regex de termo inválida") from exc
        return self


class VocalConfig(Strict):
    enabled: bool = True
    baseline_seconds: float = Field(default=30, gt=0)
    window_seconds: float = Field(default=5, ge=1)
    min_baseline_windows: int = Field(default=5, ge=2)
    min_voiced_seconds: float = Field(default=1, gt=0)
    min_features: int = Field(default=2, ge=1, le=5)
    z_threshold: float = Field(default=2.5, gt=0)
    score_z_cap: float = Field(default=5, gt=0)
    std_epsilon: float = Field(default=1e-6, gt=0)
    pitch_floor: float = Field(default=75, gt=0)
    pitch_ceiling: float = Field(default=500, gt=0)
    silence_dbfs: float = Field(default=-40, lt=0)
    frame_seconds: float = Field(default=0.02, gt=0)
    min_pause_seconds: float = Field(default=0.2, gt=0)

    @model_validator(mode="after")
    def ranges(self):
        if self.pitch_floor >= self.pitch_ceiling:
            raise ValueError("pitch_floor deve ser menor que pitch_ceiling")
        if self.score_z_cap < self.z_threshold:
            raise ValueError("score_z_cap deve cobrir z_threshold")
        if self.frame_seconds > self.window_seconds:
            raise ValueError("frame_seconds deve caber na janela")
        return self


class AudioConfig(Strict):
    speech: SpeechConfig = Field(default_factory=SpeechConfig)
    language: LanguageConfig = Field(default_factory=LanguageConfig)
    severity: SeverityConfig = Field(default_factory=SeverityConfig)
    terms: list[Term]
    vocal: VocalConfig = Field(default_factory=VocalConfig)


def load_config(path: str | Path | None = None) -> AudioConfig:
    path = Path(path) if path else REPO_ROOT / "config/audio.yaml"
    return AudioConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))

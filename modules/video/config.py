"""Leitura de config/video.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from contracts.patients import REPO_ROOT

DEFAULT_VIDEO_CONFIG = REPO_ROOT / "config" / "video.yaml"

Severity = Literal["low", "medium", "high", "critical"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelsCfg(_Strict):
    pose: str
    detect: str
    device: str | int | None = None
    imgsz: int = 640
    conf: float = 0.25
    min_keypoint_conf: float = 0.3


class OpenPoseCfg(_Strict):
    prototxt: str
    caffemodel: str
    input_size: int = 368
    heatmap_threshold: float = 0.1
    max_frames: int = 20


class PrimaryAngleCfg(_Strict):
    joint: Literal["knee", "hip", "elbow", "shoulder"] = "knee"
    side: Literal["left", "right", "auto"] = "auto"
    direction: Literal["decrease", "increase"] = "decrease"


class RepetitionCfg(_Strict):
    rest_percentile: float = 90
    start_delta: float = 15
    min_depth: float = 25
    min_duration_s: float = 0.8


class SeverityStep(_Strict):
    min_deviation: float
    severity: Severity


class AmplitudeRuleCfg(_Strict):
    expected: tuple[float, float]
    score_full_deviation: float = 60
    severity: list[SeverityStep]


class PostureRuleCfg(_Strict):
    metric: str = "trunk_angle"
    expected: tuple[float, float]
    min_duration_s: float = 0.5
    score_full_deviation: float = 40
    severity: list[SeverityStep]


class PhysioRulesCfg(_Strict):
    amplitude_incompleta: AmplitudeRuleCfg
    postura_fora_padrao: PostureRuleCfg


class PhysioCfg(_Strict):
    exercise: str = "exercicio"
    process_fps: float = 15
    smoothing_window_s: float = 0.3
    max_gap_s: float = 0.5
    primary_angle: PrimaryAngleCfg = Field(default_factory=PrimaryAngleCfg)
    repetition: RepetitionCfg = Field(default_factory=RepetitionCfg)
    rules: PhysioRulesCfg


class BedCfg(_Strict):
    sample_hz: float = 5
    bed_class: str = "bed"
    person_class: str = "person"
    bed_box_override: tuple[float, float, float, float] | None = None
    edge_margin: float = 0.12
    in_bed_min_overlap: float = 0.5
    velocity_smoothing_s: float = 0.6
    missing_hold_s: float = 1.0
    max_frames_saved: int = 8


class OutputCfg(_Strict):
    clip_padding_s: float = 1.0
    annotated_video: bool = True
    sample_frames: int = 4
    jpeg_quality: int = 88


class MetricsCfg(_Strict):
    iou_threshold: float = 0.3


class VideoConfig(_Strict):
    models: ModelsCfg
    openpose: OpenPoseCfg
    physio: PhysioCfg
    bed: BedCfg = Field(default_factory=BedCfg)
    output: OutputCfg = Field(default_factory=OutputCfg)
    metrics: MetricsCfg = Field(default_factory=MetricsCfg)

    def resolve(self, path: str) -> Path:
        """Caminhos do YAML são relativos à raiz do repositório."""
        p = Path(path)
        return p if p.is_absolute() else REPO_ROOT / p


def load_video_config(path: str | Path = DEFAULT_VIDEO_CONFIG) -> VideoConfig:
    with Path(path).open(encoding="utf-8") as fh:
        return VideoConfig.model_validate(yaml.safe_load(fh))


def severity_for(deviation: float, steps: list[SeverityStep]) -> Severity:
    chosen: Severity = steps[0].severity
    for step in sorted(steps, key=lambda s: s.min_deviation):
        if deviation >= step.min_deviation:
            chosen = step.severity
    return chosen

"""Cenário de fisioterapia: séries de ângulos, repetições e as duas regras de desvio.

Esta camada não depende de modelo: recebe keypoints por quadro e devolve
eventos. Assim as regras são testáveis com séries sintéticas.

Eventos (source: video, só fisioterapia):
- amplitude_incompleta: o pico do ângulo principal na repetição não chegou à faixa esperada;
- postura_fora_padrao: a métrica de postura (ex.: inclinação do tronco) ficou fora da
  faixa esperada por pelo menos ``min_duration_s`` dentro da repetição.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from modules.video.config import PhysioCfg, severity_for
from modules.video.geometry import interpolate_gaps, moving_average, pose_metrics

JOINT_PT = {"knee": "joelho", "hip": "quadril", "elbow": "cotovelo", "shoulder": "ombro"}
METRIC_PT = {"trunk_angle": "inclinação do tronco"}


@dataclass
class PoseFrame:
    """Pose da pessoa principal num quadro analisado (None = sem pessoa)."""

    frame_idx: int
    t: float  # segundos desde o início da mídia
    keypoints: np.ndarray | None = None  # (17, 2) pixels
    conf: np.ndarray | None = None  # (17,)
    box: np.ndarray | None = None  # (4,) x1 y1 x2 y2


@dataclass
class Repetition:
    number: int
    i_start: int
    i_end: int
    i_peak: int
    t_start: float
    t_end: float
    t_peak: float
    peak_angle: float
    rest_angle: float
    depth: float


@dataclass
class PhysioEvent:
    type: str
    rep: int
    t_start: float
    t_end: float
    t_ref: float  # instante do quadro representativo (para o frame anotado)
    frame_ref: int
    metric: str
    value: float
    expected: tuple[float, float]
    deviation: float
    score: float
    severity: str
    description: str

    def as_dict(self) -> dict:
        d = asdict(self)
        d["expected"] = list(self.expected)
        return d


def build_series(frames: list[PoseFrame], cfg: PhysioCfg, min_kp_conf: float) -> pd.DataFrame:
    """Uma linha por quadro analisado, com ângulos interpolados e suavizados."""
    rows = []
    for f in frames:
        row = {"frame_idx": f.frame_idx, "t": f.t, "has_pose": f.keypoints is not None}
        if f.keypoints is not None and f.conf is not None:
            row.update(pose_metrics(f.keypoints, f.conf, min_kp_conf))
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if len(df) > 1:
        dt = float(np.median(np.diff(df["t"].to_numpy())))
    else:
        dt = 1.0 / cfg.process_fps
    max_gap = max(1, int(round(cfg.max_gap_s / dt)))
    window = max(1, int(round(cfg.smoothing_window_s / dt)))
    metric_cols = [c for c in df.columns if c not in ("frame_idx", "t", "has_pose") and not c.startswith("conf_")]
    for col in metric_cols:
        raw = df[col].to_numpy(dtype=float)
        df[f"{col}_raw"] = raw
        df[col] = moving_average(interpolate_gaps(raw, max_gap), window)
    return df


def choose_side(series: pd.DataFrame, cfg: PhysioCfg) -> str:
    if cfg.primary_angle.side != "auto":
        return cfg.primary_angle.side
    joint = cfg.primary_angle.joint
    best, best_score = "left", -1.0
    for side in ("left", "right"):
        col = f"{joint}_{side}"
        if col not in series:
            continue
        valid = float(series[col].notna().mean())
        conf = float(series.get(f"conf_{side}", pd.Series([0.0])).mean(skipna=True) or 0.0)
        score = valid + 0.5 * conf
        if score > best_score:
            best, best_score = side, score
    return best


def detect_repetitions(t: np.ndarray, angle: np.ndarray, cfg: PhysioCfg) -> list[Repetition]:
    """Repetições por histerese em torno do ângulo de repouso estimado."""
    rc = cfg.repetition
    finite = angle[np.isfinite(angle)]
    if len(finite) < 3:
        return []
    decrease = cfg.primary_angle.direction == "decrease"
    rest = float(np.percentile(finite, rc.rest_percentile if decrease else 100 - rc.rest_percentile))
    excursion = (rest - angle) if decrease else (angle - rest)

    reps: list[Repetition] = []
    i, n = 0, len(angle)
    while i < n:
        if not (np.isfinite(excursion[i]) and excursion[i] > rc.start_delta):
            i += 1
            continue
        j = i
        # Lacunas longas (NaN que sobrou após a interpolação) encerram a repetição.
        while j < n and np.isfinite(excursion[j]) and excursion[j] > rc.start_delta:
            j += 1
        seg = excursion[i:j]
        i_peak = i + int(np.nanargmax(seg))
        depth = float(excursion[i_peak])
        end = min(j, n - 1)
        if depth >= rc.min_depth and (t[end] - t[i]) >= rc.min_duration_s:
            reps.append(
                Repetition(
                    number=len(reps) + 1, i_start=i, i_end=end, i_peak=i_peak,
                    t_start=float(t[i]), t_end=float(t[end]), t_peak=float(t[i_peak]),
                    peak_angle=float(angle[i_peak]), rest_angle=rest, depth=depth,
                )
            )
        i = j
    return reps


def _score(deviation: float, full: float) -> float:
    return round(float(min(1.0, max(0.0, deviation / full))), 3) if full > 0 else 1.0


def _fmt(v: float) -> str:
    return f"{v:.0f}°"


def _metric_column(series: pd.DataFrame, metric: str, side: str) -> str:
    if metric in series:
        return metric
    if f"{metric}_{side}" in series:
        return f"{metric}_{side}"
    raise KeyError(f"métrica de postura {metric!r} não existe na série")


def evaluate_rules(series: pd.DataFrame, reps: list[Repetition], side: str, cfg: PhysioCfg) -> list[PhysioEvent]:
    events: list[PhysioEvent] = []
    joint = cfg.primary_angle.joint
    decrease = cfg.primary_angle.direction == "decrease"
    amp = cfg.rules.amplitude_incompleta
    post = cfg.rules.postura_fora_padrao
    t = series["t"].to_numpy()
    frame_idx = series["frame_idx"].to_numpy()
    post_col = _metric_column(series, post.metric, side)
    post_vals = series[post_col].to_numpy(dtype=float)
    lo, hi = post.expected
    post_name = METRIC_PT.get(post.metric, post.metric)

    for rep in reps:
        # --- amplitude incompleta ---------------------------------------
        if decrease:
            deviation = rep.peak_angle - amp.expected[1]
            limit = f"até {_fmt(amp.expected[1])}"
        else:
            deviation = amp.expected[0] - rep.peak_angle
            limit = f"até {_fmt(amp.expected[0])}"
        if deviation > 0:
            events.append(
                PhysioEvent(
                    type="amplitude_incompleta", rep=rep.number,
                    t_start=rep.t_start, t_end=rep.t_end, t_ref=rep.t_peak,
                    frame_ref=int(frame_idx[rep.i_peak]),
                    metric=f"{joint}_angle", value=round(rep.peak_angle, 1), expected=tuple(amp.expected),
                    deviation=round(deviation, 1), score=_score(deviation, amp.score_full_deviation),
                    severity=severity_for(deviation, amp.severity),
                    description=(
                        f"Amplitude incompleta na repetição {rep.number} ({cfg.exercise}): ângulo do "
                        f"{JOINT_PT[joint]} chegou a {_fmt(rep.peak_angle)}, esperado {limit}"
                    ),
                )
            )

        # --- postura fora da faixa -------------------------------------
        seg = slice(rep.i_start, rep.i_end + 1)
        vals = post_vals[seg]
        outside = np.isfinite(vals) & ((vals < lo) | (vals > hi))
        best = _longest_run(outside)
        if best is None:
            continue
        a, b = best
        i0, i1 = rep.i_start + a, rep.i_start + b
        duration = float(t[i1] - t[i0]) if i1 > i0 else 0.0
        if duration < post.min_duration_s:
            continue
        run = post_vals[i0: i1 + 1]
        excess = np.where(run > hi, run - hi, lo - run)
        k = int(np.nanargmax(excess))
        deviation = float(excess[k])
        value = float(run[k])
        events.append(
            PhysioEvent(
                type="postura_fora_padrao", rep=rep.number,
                t_start=float(t[i0]), t_end=float(t[i1]), t_ref=float(t[i0 + k]),
                frame_ref=int(frame_idx[i0 + k]),
                metric=post.metric, value=round(value, 1), expected=tuple(post.expected),
                deviation=round(deviation, 1), score=_score(deviation, post.score_full_deviation),
                severity=severity_for(deviation, post.severity),
                description=(
                    f"{str(post_name).capitalize()} de {_fmt(value)} por "
                    f"{f'{duration:.1f}'.replace('.', ',')} s na repetição "
                    f"{rep.number} ({cfg.exercise}), fora da faixa esperada ({_fmt(lo)}–{_fmt(hi)})"
                ),
            )
        )
    events.sort(key=lambda e: (e.t_start, e.type))
    return events


def _longest_run(mask: np.ndarray) -> tuple[int, int] | None:
    best, start = None, None
    for i, m in enumerate(list(mask) + [False]):
        if m and start is None:
            start = i
        elif not m and start is not None:
            if best is None or (i - 1 - start) > (best[1] - best[0]):
                best = (start, i - 1)
            start = None
    return best


@dataclass
class PhysioResult:
    series: pd.DataFrame
    side: str
    reps: list[Repetition]
    events: list[PhysioEvent]


def analyze_pose_frames(frames: list[PoseFrame], cfg: PhysioCfg, min_kp_conf: float) -> PhysioResult:
    series = build_series(frames, cfg, min_kp_conf)
    if series.empty or not series["has_pose"].any():
        return PhysioResult(series=series, side="left", reps=[], events=[])
    side = choose_side(series, cfg)
    col = f"{cfg.primary_angle.joint}_{side}"
    angle = series[col].to_numpy(dtype=float) if col in series else np.full(len(series), math.nan)
    reps = detect_repetitions(series["t"].to_numpy(), angle, cfg)
    events = evaluate_rules(series, reps, side, cfg) if reps else []
    return PhysioResult(series=series, side=side, reps=reps, events=events)

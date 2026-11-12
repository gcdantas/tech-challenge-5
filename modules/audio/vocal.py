"""Medidas acústicas locais e desvio intra-locutor; não é um diagnóstico."""
import json
import math
from pathlib import Path
import wave

import numpy as np

from .config import VocalConfig
from .speech import wav_info

FEATURES = ("taxa_fala", "pausas", "pitch", "jitter", "shimmer")


def read_samples(path: str | Path) -> np.ndarray:
    wav_info(path)
    with wave.open(str(path), "rb") as f:
        return np.frombuffer(f.readframes(f.getnframes()), dtype="<i2").astype(float) / 32768


def _finite(value):
    return float(value) if math.isfinite(value) else None


def extract_window(samples: np.ndarray, words: list[dict] | None, start: float,
                   config: VocalConfig, sample_rate: int = 16000) -> dict:
    """Taxa: palavras/min por timestamps; pausas: fração de silêncio sustentado.

    Pitch em Hz, jitter/shimmer locais em fração (não porcentagem).
    Não há estimativa artificial da taxa de fala quando faltam palavras.
    """
    import parselmouth
    from parselmouth.praat import call

    duration = len(samples) / sample_rate
    result = {name: None for name in FEATURES}
    if duration < config.window_seconds - 1e-6:
        return result
    frame = max(1, round(config.frame_seconds * sample_rate))
    frames = samples[:len(samples) // frame * frame].reshape(-1, frame)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    silent = rms < 10 ** (config.silence_dbfs / 20)
    edges = np.diff(np.r_[False, silent, False].astype(int))
    runs = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    pause_frames = sum(n for n in runs if n * frame / sample_rate >= config.min_pause_seconds)
    result["pausas"] = float(pause_frames * frame / sample_rate / duration)
    sound = parselmouth.Sound(samples, sampling_frequency=sample_rate)
    pitch = sound.to_pitch(time_step=config.frame_seconds, pitch_floor=config.pitch_floor,
                           pitch_ceiling=config.pitch_ceiling)
    frequencies = pitch.selected_array["frequency"]
    voiced = frequencies[frequencies > 0]
    if len(voiced) * config.frame_seconds < config.min_voiced_seconds:
        # Silêncio e ruído sem periodicidade não constituem alteração vocal.
        return {name: None for name in FEATURES}
    if words is not None:
        count = sum(start <= (w["start"] + w["end"]) / 2 < start + duration for w in words)
        result["taxa_fala"] = count * 60 / duration
    result["pitch"] = float(np.median(voiced))
    try:
        points = call(sound, "To PointProcess (periodic, cc)", config.pitch_floor, config.pitch_ceiling)
        floor, ceiling = 0.8 / config.pitch_ceiling, 1.25 / config.pitch_floor
        result["jitter"] = _finite(call(points, "Get jitter (local)", 0, 0, floor, ceiling, 1.3))
        result["shimmer"] = _finite(call([sound, points], "Get shimmer (local)",
                                          0, 0, floor, ceiling, 1.3, 1.6))
    except parselmouth.PraatError:
        # Ausência de pulsos suficientes não vira zero nem Finding.
        pass
    return result


def windows(samples: np.ndarray, words: list[dict] | None, start: float, end: float,
            config: VocalConfig) -> list[dict]:
    step = round(config.window_seconds * 16000)
    first, stop = round(start * 16000), min(round(end * 16000), len(samples))
    result = []
    for i in range(first, stop - step + 1, step):
        a, b = i / 16000, (i + step) / 16000
        result.append({"start": a, "end": b,
                       "features": extract_window(samples[i:i + step], words, a, config)})
    return result


def compare(features: dict, baseline: list[dict], config: VocalConfig) -> dict:
    zscores, stats, skipped = {}, {}, {}
    for name in FEATURES:
        values = [row[name] for row in baseline if row.get(name) is not None and math.isfinite(row[name])]
        current = features.get(name)
        if len(values) < config.min_baseline_windows:
            skipped[name] = "amostras_insuficientes"
            continue
        mean, std = float(np.mean(values)), float(np.std(values, ddof=1))
        stats[name] = {"mean": mean, "std": std, "n": len(values)}
        if current is None or not math.isfinite(current):
            skipped[name] = "medida_ausente"
        elif std <= config.std_epsilon:
            skipped[name] = "desvio_padrao_zero_ou_insuficiente"
        else:
            zscores[name] = (float(current) - mean) / std
    changed = [name for name, z in zscores.items() if abs(z) >= config.z_threshold]
    # Pelo menos N medidas acima do limiar. Score é uma heurística, não probabilidade.
    ordered = sorted((abs(zscores[name]) for name in changed), reverse=True)
    score = min(1.0, ordered[config.min_features - 1] / config.score_z_cap) if len(changed) >= config.min_features else 0.0
    return {"zscores": zscores, "baseline_stats": stats, "skipped": skipped,
            "changed_features": changed, "altered": len(changed) >= config.min_features, "score": score}


def speaker_metadata(path: Path, patient_id: str) -> dict | None:
    sidecar = path.with_suffix(path.suffix + ".json")
    if not sidecar.exists():
        return None
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    if (metadata.get("patient_id") != patient_id or not metadata.get("speaker_id")
            or metadata.get("single_speaker") is not True):
        raise ValueError("Metadados vocais devem confirmar patient_id, speaker_id e single_speaker=true")
    return metadata


def analyze_vocal(path: Path, segments: list[dict], patient_id: str, config: VocalConfig,
                  baseline_path: Path | None = None, baseline_segments: list[dict] | None = None) -> dict:
    if not config.enabled:
        return {"status": "disabled", "windows": []}
    metadata = speaker_metadata(path, patient_id)
    if metadata is None:
        if baseline_path:
            raise ValueError("Comparação vocal exige metadados de locutor nos dois WAVs")
        return {"status": "not_evaluated", "reason": "missing_speaker_metadata", "windows": []}
    samples = read_samples(path)
    # Ausência de palavra em qualquer segmento torna a taxa indisponível.
    words = [w for s in segments for w in s["words"]] if segments and all(s.get("words") for s in segments) else None
    baseline_description = "primeiros 30 s da consulta" if config.baseline_seconds == 30 else f"primeiros {config.baseline_seconds:g} s"
    if baseline_path:
        if baseline_path.resolve() == path.resolve():
            raise ValueError("Baseline separado não pode ser o próprio arquivo da consulta")
        other = speaker_metadata(baseline_path, patient_id)
        if not other or other["speaker_id"] != metadata["speaker_id"] or other.get("condition") != "neutral":
            raise ValueError("Baseline deve ser neutro e do mesmo locutor, confirmado no .wav.json")
        base_samples = read_samples(baseline_path)
        base_words = ([w for s in baseline_segments for w in s["words"]]
                      if baseline_segments and all(s.get("words") for s in baseline_segments) else None)
        base = windows(base_samples, base_words, 0, len(base_samples) / 16000, config)
        analysis_start = 0
        baseline_description = str(baseline_path)
    else:
        if metadata.get("neutral_prefix_seconds", 0) < config.baseline_seconds:
            return {"status": "not_evaluated", "reason": "neutral_prefix_not_confirmed", "windows": []}
        base = windows(samples, words, 0, config.baseline_seconds, config)
        analysis_start = config.baseline_seconds
    current = windows(samples, words, analysis_start, len(samples) / 16000, config)
    comparisons = [{**row, **compare(row["features"], [b["features"] for b in base], config)} for row in current]
    usable = any(len(row["zscores"]) >= config.min_features for row in comparisons)
    return {"status": "evaluated" if usable else "insufficient_data", "speaker_id": metadata["speaker_id"],
            "baseline": baseline_description, "baseline_windows": base, "windows": comparisons,
            "units": {"taxa_fala": "words/min", "pausas": "fraction", "pitch": "Hz",
                      "jitter": "fraction", "shimmer": "fraction"}}

"""Regras da fisioterapia sobre poses sintéticas (sem modelo)."""

import numpy as np
import pytest

from modules.video.config import load_video_config, severity_for
from modules.video.physio import PoseFrame, analyze_pose_frames, detect_repetitions
from tests.synthetic import SQUAT_REPS, squat_keypoints, squat_pose_frames


@pytest.fixture()
def cfg():
    return load_video_config()


def test_conta_as_repeticoes(cfg):
    res = analyze_pose_frames(squat_pose_frames(), cfg.physio, 0.3)
    assert len(res.reps) == len(SQUAT_REPS)
    peaks = [round(r.peak_angle) for r in res.reps]
    for got, (expected, _) in zip(peaks, SQUAT_REPS):
        assert got == pytest.approx(expected, abs=4)  # suavização reduz um pouco o pico


def test_detecta_os_dois_eventos_nas_repeticoes_certas(cfg):
    res = analyze_pose_frames(squat_pose_frames(), cfg.physio, 0.3)
    got = [(e.type, e.rep) for e in res.events]
    assert got == [("amplitude_incompleta", 3), ("postura_fora_padrao", 6)]

    amp, post = res.events
    assert amp.metric == "knee_angle"
    assert amp.value > cfg.physio.rules.amplitude_incompleta.expected[1]
    assert amp.severity == "medium"  # ~30° de desvio
    assert 0 < amp.score <= 1
    assert post.metric == "trunk_angle"
    assert post.value > 45
    assert post.t_end - post.t_start >= cfg.physio.rules.postura_fora_padrao.min_duration_s
    assert 0 < post.score <= 1
    assert "repetição 3" in amp.description


def test_sem_desvios_quando_tudo_esta_na_faixa(cfg):
    frames = []
    for i in range(15 * 12):
        t = i / 15
        w = max(0.0, np.sin(np.pi * (t % 3) / 2)) if (t % 3) < 2 else 0.0
        kp, conf = squat_keypoints(172 - 82 * w, 5 + 20 * w)
        frames.append(PoseFrame(i, t, kp, conf))
    res = analyze_pose_frames(frames, cfg.physio, 0.3)
    assert len(res.reps) == 4
    assert res.events == []


def test_postura_breve_nao_dispara(cfg):
    """Tronco fora da faixa por menos que min_duration_s não vira evento."""
    cfg.physio.rules.postura_fora_padrao.min_duration_s = 5.0
    res = analyze_pose_frames(squat_pose_frames(), cfg.physio, 0.3)
    assert [e.type for e in res.events] == ["amplitude_incompleta"]


def test_sem_pose_nao_quebra(cfg):
    frames = [PoseFrame(i, i / 15) for i in range(30)]
    res = analyze_pose_frames(frames, cfg.physio, 0.3)
    assert res.reps == [] and res.events == []


def test_direcao_increase(cfg):
    """Movimento que aumenta o ângulo (ex.: abdução de ombro), com limiares invertidos."""
    cfg.physio.primary_angle.direction = "increase"
    cfg.physio.rules.amplitude_incompleta.expected = (150, 180)
    t = np.arange(0, 12, 1 / 15)
    peaks = [160, 120, 165]  # repetição 2 incompleta
    angle = np.full(len(t), 20.0)
    for k, p in enumerate(peaks):
        m = (t >= k * 4) & (t < k * 4 + 2)
        angle[m] = 20 + (p - 20) * np.sin(np.pi * (t[m] - k * 4) / 2)
    reps = detect_repetitions(t, angle, cfg.physio)
    assert [round(r.peak_angle) for r in reps] == peaks


def test_severity_for():
    steps = load_video_config().physio.rules.amplitude_incompleta.severity
    assert severity_for(5, steps) == "low"
    assert severity_for(20, steps) == "medium"
    assert severity_for(50, steps) == "high"

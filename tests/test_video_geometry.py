import math

import numpy as np
import pytest

from modules.video.geometry import (
    angle_at,
    inclination_from_vertical,
    interpolate_gaps,
    moving_average,
    pose_metrics,
)
from tests.synthetic import squat_keypoints


def test_angle_at():
    assert angle_at([1, 0], [0, 0], [0, 1]) == pytest.approx(90)
    assert angle_at([-1, 0], [0, 0], [1, 0]) == pytest.approx(180)
    assert angle_at([1, 0], [0, 0], [1, 0]) == pytest.approx(0)
    assert math.isnan(angle_at([0, 0], [0, 0], [1, 0]))


def test_inclinacao_em_relacao_a_vertical():
    assert inclination_from_vertical(np.array([0, 10]), np.array([0, 0])) == pytest.approx(0)
    assert inclination_from_vertical(np.array([0, 0]), np.array([10, 0])) == pytest.approx(90)
    assert inclination_from_vertical(np.array([0, 10]), np.array([10, 0])) == pytest.approx(45)
    assert inclination_from_vertical(np.array([0, 10]), np.array([-10, 0])) == pytest.approx(45)


@pytest.mark.parametrize("knee,trunk", [(170, 5), (90, 30), (130, 60)])
def test_pose_metrics_recupera_angulos_do_esqueleto(knee, trunk):
    kp, conf = squat_keypoints(knee, trunk)
    m = pose_metrics(kp, conf, 0.3)
    assert m["knee_left"] == pytest.approx(knee, abs=1e-6)
    assert m["knee_right"] == pytest.approx(knee, abs=1e-6)
    assert m["trunk_angle"] == pytest.approx(trunk, abs=1e-6)


def test_keypoint_com_baixa_confianca_vira_nan():
    kp, conf = squat_keypoints(90, 10)
    conf[13] = 0.1  # joelho esquerdo
    m = pose_metrics(kp, conf, 0.3)
    assert math.isnan(m["knee_left"])
    assert not math.isnan(m["knee_right"])


def test_interpolate_gaps_so_preenche_lacunas_curtas():
    v = np.array([0, np.nan, 2, np.nan, np.nan, np.nan, 6, np.nan])
    out = interpolate_gaps(v, max_gap=2)
    assert out[1] == pytest.approx(1)
    assert np.isnan(out[3:6]).all()  # lacuna de 3 > 2
    assert np.isnan(out[7])  # borda não é extrapolada


def test_moving_average_ignora_nan():
    out = moving_average(np.array([1, 2, np.nan, 4, 5]), 3)
    assert out[1] == pytest.approx(1.5)
    assert np.isnan(out[2])
    assert out[3] == pytest.approx(4.5)

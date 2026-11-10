"""Funções puras da comparação OpenPose × YOLOv8-pose (notebooks/comparacao)."""

import math

import numpy as np
import pandas as pd
import pytest

from notebooks.comparacao.openpose_yolo import (
    COCO_TO_GT,
    angle_3d,
    frame_to_frame,
    gt_knee_angle_3d,
    static_window,
    view_of,
)


def test_angle_3d():
    assert angle_3d(np.array([1, 0, 0.0]), np.zeros(3), np.array([0, 0, 1.0])) == pytest.approx(90)
    assert angle_3d(np.array([0, 1, 0.0]), np.zeros(3), np.array([0, -1, 0.0])) == pytest.approx(180)
    assert math.isnan(angle_3d(np.zeros(3), np.zeros(3), np.array([1, 0, 0.0])))


def test_angulo_do_joelho_usa_quadril_joelho_tornozelo_do_gabarito():
    j3 = np.zeros((26, 4))
    j3[COCO_TO_GT[11], :3] = [0, 1.0, 0]   # quadril esquerdo
    j3[COCO_TO_GT[13], :3] = [0, 0.5, 0]   # joelho esquerdo
    j3[COCO_TO_GT[15], :3] = [0.5, 0.5, 0]  # tornozelo esquerdo: perna dobrada a 90°
    assert gt_knee_angle_3d(j3, "esquerdo") == pytest.approx(90)


def test_janela_estatica_encontra_o_trecho_parado():
    rng = np.random.default_rng(0)
    j3 = np.cumsum(rng.normal(0, 0.01, (200, 26, 4)), axis=0)  # movimento contínuo
    j3[120:160] = j3[120]                                       # 40 quadros parados
    start = static_window(j3, 30)
    assert 120 <= start <= 130


def test_variacao_quadro_a_quadro_ignora_ausentes():
    pts = np.array([[[0, 0]], [[3, 4]], [[np.nan, np.nan]], [[3, 4]]], float)
    assert frame_to_frame(pts) == pytest.approx(5.0)  # só o par (0,0)->(3,4) é válido


def test_vista_vem_da_repeticao_mais_proxima():
    reps = pd.DataFrame({"first_frame": [0, 300], "last_frame": [100, 400],
                         "cam17_orientation": ["front", "half-profile"]})
    assert view_of(50, reps) == "perfil"
    assert view_of(390, reps) == "meio-perfil"

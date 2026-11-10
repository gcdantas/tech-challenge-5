"""Rótulos desenhados nos quadros: cabem na largura e não se sobrepõem."""

import numpy as np
import pytest

from modules.video.annotate import put_label, put_labels_bottom, put_lines

LONG = "DESVIO: amplitude incompleta (rep 15)"


@pytest.mark.parametrize("w,h", [(1080, 1920), (1920, 1080), (320, 240)])
def test_rotulos_do_rodape_nao_se_sobrepoem_e_cabem(w, h):
    frame = np.zeros((h, w, 3), np.uint8)
    a, b = put_labels_bottom(frame, ["DESVIO: postura fora da faixa (rep 15)", LONG])
    assert a[2] <= w and b[2] <= w
    assert b[3] <= a[1]  # o segundo fica inteiro acima do primeiro
    assert a[3] <= h


@pytest.mark.parametrize("w", [400, 120])
def test_texto_longo_encolhe_ou_corta_para_caber(w):
    frame = np.zeros((1920, w, 3), np.uint8)
    _, _, x2, _ = put_label(frame, LONG * 2, (8, 100))
    assert x2 <= w - 8


def test_bloco_superior_sem_sobreposicao():
    frame = np.full((1920, 1080, 3), 128, np.uint8)
    put_lines(frame, ["joelho: 142 graus", "tronco: 54 graus", "repeticao 15"])
    # coluna 8: primeira coluna das faixas pretas (o texto, antisserrilhado, começa perto de x = 10)
    band = (frame[:, 8, :] == 0).all(axis=1).astype(int)
    starts = np.flatnonzero(np.diff(band) == 1)
    assert len(starts) == 3  # três faixas separadas por fundo cinza

"""OpenPose via OpenCV DNN. Pulado se os pesos não foram baixados.

Usa a imagem de exemplo que vem DENTRO do pacote ultralytics (ultralytics/assets,
nada é baixado) só para conferir que o modelo encontra keypoints de pessoas reais.
"""

import json

import cv2
import numpy as np
import pytest

from contracts.patients import REPO_ROOT
from modules.video.config import load_video_config
from modules.video.openpose import TO_COCO17, OpenPoseCOCO, main
from tests.synthetic import write_synthetic_video

CFG = load_video_config()
WEIGHTS_OK = CFG.resolve(CFG.openpose.caffemodel).exists() and CFG.resolve(CFG.openpose.prototxt).exists()
needs_openpose = pytest.mark.skipif(not WEIGHTS_OK, reason="pesos do OpenPose não baixados")


def _asset(name):
    from ultralytics.utils import ASSETS

    return ASSETS / name


def test_mapeamento_para_coco17_e_permutacao():
    assert sorted(TO_COCO17) == sorted(set(TO_COCO17))
    assert len(TO_COCO17) == 17 and 1 not in TO_COCO17  # "neck" não existe no COCO-17


@needs_openpose
def test_openpose_encontra_keypoints_em_foto_com_pessoa():
    img = cv2.imread(str(_asset("zidane.jpg")))
    res = OpenPoseCOCO.from_config(CFG).predict(img)
    assert res.found >= 8
    kp17, conf17 = res.to_coco17()
    assert kp17.shape == (17, 2) and conf17.shape == (17,)


@needs_openpose
def test_cli_openpose_salva_quadros_anotados(tmp_path):
    video = write_synthetic_video(tmp_path / "sintetico_op.mp4", 2.0)
    rc = main(["--input", str(video), "--patient", "P001", "--max-frames", "3", "--outputs", str(tmp_path)])
    assert rc == 0
    pdir = tmp_path / "video" / "P001"
    imgs = sorted((pdir / "frames" / "sintetico_op").glob("openpose_*.jpg"))
    assert len(imgs) == 3
    run = json.loads((pdir / "runs" / "openpose_sintetico_op.json").read_text(encoding="utf-8"))
    assert run["processing"]["processed_frames"] == 3
    assert run["mean_fps"] > 0
    assert "OpenPose" in (pdir / "report.html").read_text(encoding="utf-8")


@needs_openpose
@pytest.mark.skipif(not CFG.resolve(CFG.models.pose).exists(), reason="pesos YOLOv8 não baixados")
def test_recorte_da_pessoa_principal_mantem_pontos_dentro_dela():
    from modules.video.pose import YoloPoseEstimator

    img = cv2.imread(str(_asset("zidane.jpg")))
    person = YoloPoseEstimator(CFG.resolve(CFG.models.pose)).predict(img)
    assert person is not None
    res = OpenPoseCOCO.from_config(CFG).predict(img, person.box)
    assert res.found >= 8
    x1, y1, x2, y2 = person.box
    w, h = x2 - x1, y2 - y1
    pts = res.keypoints[np.isfinite(res.keypoints[:, 0])]
    assert (pts[:, 0] >= x1 - 0.2 * w).all() and (pts[:, 0] <= x2 + 0.2 * w).all()
    assert (pts[:, 1] >= y1 - 0.2 * h).all() and (pts[:, 1] <= y2 + 0.2 * h).all()

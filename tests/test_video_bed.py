"""Cenário do leito em vídeo SINTÉTICO, com detector e pose injetados."""

import json
from datetime import timedelta

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from contracts import read_findings, write_findings
from contracts.patients import REPO_ROOT, PatientsFile
from modules.video import analyze
from modules.video.bed import MOVEMENT_COLUMNS, in_edge_zone, inner_box, overlap_fraction
from modules.video.config import load_video_config
from modules.video.pose import Detection, PoseDetection
from modules.video.run import main
from tests.synthetic import squat_keypoints, write_synthetic_video

FPS, SECONDS, EXIT_AT = 15.0, 6.0, 3.0
BED = np.array([40, 100, 200, 200.0])


class FakeDetector:
    """Leito fixo; a pessoa está deitada no leito e, a partir de EXIT_AT, sai para a direita."""

    name, version = "fake-detect", "test"

    def __init__(self, stride: int):
        self.dt, self.calls = stride / FPS, 0

    def person_box(self, t):
        if t < EXIT_AT:
            return np.array([60, 120, 180, 180.0])
        shift = min(1.0, (t - EXIT_AT) / 1.0) * 140
        return np.array([60 + shift, 60, 120 + shift, 200.0])

    def detect(self, frame):
        t = self.calls * self.dt
        self.calls += 1
        return [Detection("bed", BED, 0.8), Detection("person", self.person_box(t), 0.9)]


class FakePose:
    name, version = "fake-pose", "test"

    def __init__(self, stride: int):
        self.dt, self.calls = stride / FPS, 0

    def predict(self, frame):
        t = self.calls * self.dt
        self.calls += 1
        trunk = 85.0 if t < EXIT_AT else 5.0
        kp, conf = squat_keypoints(170, trunk)
        return PoseDetection(kp, conf, np.array([60, 60, 180, 200.0]), 0.9)


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    return write_synthetic_video(tmp_path_factory.mktemp("bed") / "sintetico_leito.mp4", SECONDS, FPS)


def _pf(video, offset=600.0):
    return PatientsFile.model_validate({
        "demo_reference": "2026-11-03T10:00:00-03:00",
        "patients": {"P001": {"artifacts": {"video_leito": {"kind": "video_bed", "path": str(video), "demo_offset": offset}}}},
    })


def test_geometria_do_leito():
    assert overlap_fraction([0, 0, 10, 10], [0, 0, 10, 10]) == 1
    assert overlap_fraction([0, 0, 10, 10], [5, 0, 20, 10]) == pytest.approx(0.5)
    inner = inner_box([0, 0, 100, 100], 0.1)
    assert list(inner) == [10, 10, 90, 90]
    assert in_edge_zone((5, 50), [0, 0, 100, 100], inner)
    assert not in_edge_zone((50, 50), [0, 0, 100, 100], inner)
    assert not in_edge_zone((150, 50), [0, 0, 100, 100], inner)


def test_movement_parquet_do_leito(video, tmp_path):
    cfg = load_video_config()
    stride = round(FPS / cfg.bed.sample_hz)
    findings = analyze(video, "P001", config=cfg, outputs_root=tmp_path, patients=_pf(video),
                       pose_estimator=FakePose(stride), detector=FakeDetector(stride))
    assert findings == []  # o leito não emite Findings (são do I3)

    pdir = tmp_path / "video" / "P001"
    df = pd.read_parquet(pdir / "movement.parquet")
    assert list(df.columns) == MOVEMENT_COLUMNS
    assert df["in_bed"].dtype == bool
    step = df["timestamp"].diff().dropna().dt.total_seconds().median()
    assert step == pytest.approx(1 / 5, abs=0.01)  # ~5 Hz
    assert df["timestamp"].iloc[0] == pd.Timestamp("2026-11-03T10:10:00-03:00")  # demo_offset 600 s
    assert df["in_bed"].iloc[0] and not df["in_bed"].iloc[-1]
    first_out = df.loc[~df["in_bed"], "timestamp"].iloc[0]
    assert timedelta(seconds=600 + EXIT_AT) <= first_out - pd.Timestamp("2026-11-03T10:00:00-03:00") <= timedelta(seconds=600 + EXIT_AT + 1)
    assert df.loc[df["in_bed"], "torso_angle"].median() > 60  # deitado
    assert df.loc[~df["in_bed"], "torso_angle"].median() < 20  # em pé
    assert df["velocidade"].max() > 0

    meta = pq.read_schema(pdir / "movement.parquet").metadata
    assert meta[b"mock"] == b"false"
    run = json.loads((pdir / "runs" / "bed_sintetico_leito.json").read_text(encoding="utf-8"))
    assert run["frames"] and all((REPO_ROOT / f).exists() or __import__("pathlib").Path(f).exists() for f in run["frames"])
    html = (pdir / "report.html").read_text(encoding="utf-8")
    assert "Leito" in html and "não sinaliza eventos" in html


def test_sem_leito_detectado_pede_override(video, tmp_path):
    class NoBed(FakeDetector):
        def detect(self, frame):
            return [d for d in super().detect(frame) if d.label != "bed"]

    cfg = load_video_config()
    stride = round(FPS / cfg.bed.sample_hz)
    with pytest.raises(ValueError, match="bed_box_override"):
        analyze(video, "P001", config=cfg, outputs_root=tmp_path, patients=_pf(video),
                pose_estimator=FakePose(stride), detector=NoBed(stride))

    cfg.bed.bed_box_override = (0.1, 0.4, 0.6, 0.85)
    analyze(video, "P001", config=cfg, outputs_root=tmp_path, patients=_pf(video),
            pose_estimator=FakePose(stride), detector=NoBed(stride))
    assert (tmp_path / "video" / "P001" / "movement.parquet").exists()


WEIGHTS = load_video_config()


@pytest.mark.skipif(
    not (WEIGHTS.resolve(WEIGHTS.models.pose).exists() and WEIGHTS.resolve(WEIGHTS.models.detect).exists()),
    reason="pesos YOLOv8 não baixados",
)
def test_cli_leito_nao_sobrescreve_findings(video, tmp_path):
    """Com o YOLOv8 real num vídeo sem leito, a CLI falha com mensagem clara e não toca no findings.jsonl."""
    out = tmp_path / "findings.jsonl"
    write_findings(out, read_findings(REPO_ROOT / "contracts" / "fixtures" / "findings_example.jsonl")[:1])
    before = out.read_bytes()
    rc = main(["--input", str(video), "--patient", "P001", "--scenario", "bed",
               "--outputs", str(tmp_path), "--out", str(out)])
    assert rc == 2
    assert out.read_bytes() == before

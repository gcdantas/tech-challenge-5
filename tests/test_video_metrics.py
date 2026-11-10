import json

import pytest

from modules.video.metrics import (
    Interval,
    evaluate_events,
    evaluate_patient,
    labels_for,
    load_labels,
    main,
    match_events,
    prf,
    temporal_iou,
    to_markdown,
)
from contracts.patients import REPO_ROOT


def test_temporal_iou():
    assert temporal_iou(Interval("a", 0, 10), Interval("a", 5, 15)) == pytest.approx(5 / 15)
    assert temporal_iou(Interval("a", 0, 1), Interval("a", 2, 3)) == 0
    assert temporal_iou(Interval("a", 3, 3), Interval("a", 3, 3)) == 1


def test_match_um_para_um_e_por_tipo():
    gt = [Interval("amp", 0, 10), Interval("amp", 20, 30), Interval("post", 40, 45)]
    pred = [
        Interval("amp", 1, 9),      # VP
        Interval("amp", 2, 8),      # duplicado do mesmo rótulo -> FP
        Interval("post", 20, 30),   # tipo errado -> FP
        Interval("post", 40, 46),   # VP
    ]
    tp, fp, fn = match_events(pred, gt, 0.3)
    assert (tp, fp, fn) == (2, 2, 1)


def test_prf_casos_limite():
    assert prf(0, 0, 0) == (None, None, None)
    assert prf(0, 2, 0) == (0.0, None, None)
    assert prf(0, 1, 1) == (0.0, 0.0, 0.0)
    p, r, f = prf(3, 1, 2)
    assert (p, r) == (0.75, 0.6)
    assert f == pytest.approx(2 * 0.75 * 0.6 / 1.35)


def test_evaluate_events_tem_linha_por_tipo_e_total():
    rows = evaluate_events("v", [Interval("amplitude_incompleta", 0, 5)], [Interval("amplitude_incompleta", 0, 5)], 0.3)
    by_type = {r["type"]: r for r in rows}
    assert by_type["amplitude_incompleta"]["f1"] == 1.0
    assert by_type["postura_fora_padrao"]["precision"] is None
    assert by_type["todos"]["tp"] == 1


def test_modelo_de_rotulos_valida():
    labels = load_labels(REPO_ROOT / "data" / "labels" / "video.example.yaml")
    assert {e.type for e in labels.events} == {"amplitude_incompleta", "postura_fora_padrao"}


def _write_run(pdir, stem, events, scenario="physio"):
    (pdir / "runs").mkdir(parents=True, exist_ok=True)
    run = {
        "scenario": scenario, "video_stem": stem,
        "processing": {"inference_fps": 30.0, "end_to_end_fps": 20.0},
        "events": [{"type": t, "t_start": a, "t_end": b} for t, a, b in events],
    }
    (pdir / "runs" / f"{scenario}_{stem}.json").write_text(json.dumps(run), encoding="utf-8")


def test_evaluate_patient_e_cli(tmp_path):
    pdir = tmp_path / "video" / "P001"
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "v1.yaml").write_text(
        "events:\n  - {type: amplitude_incompleta, start: 10, end: 15}\n"
        "  - {type: postura_fora_padrao, start: 30, end: 33}\n",
        encoding="utf-8",
    )
    _write_run(pdir, "v1", [("amplitude_incompleta", 10.5, 15.2), ("amplitude_incompleta", 50, 52)])
    _write_run(pdir, "v2", [])  # sem rótulo
    _write_run(pdir, "leito", [], scenario="bed")

    res = evaluate_patient(pdir, labels, 0.3)
    total = next(r for r in res["rows"] if r["video"] == "v1" and r["type"] == "todos")
    assert (total["tp"], total["fp"], total["fn"]) == (1, 1, 1)
    assert res["videos_without_labels"] == ["v2"]
    assert {r["scenario"] for r in res["fps"]} == {"physio", "bed"}
    assert "validação clínica" in to_markdown(res)

    assert labels_for("v1", labels) is not None
    main(["--patient", "P001", "--outputs", str(tmp_path), "--labels", str(labels)])
    assert (pdir / "metrics.json").exists() and (pdir / "metrics.md").exists()
    html = (pdir / "report.html").read_text(encoding="utf-8")
    assert "Métricas" in html and "amostra pequena" in html

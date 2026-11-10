"""Testes de data/patients.yaml e do alinhamento temporal."""

from datetime import timedelta

from contracts.patients import REPO_ROOT, load_patients


def test_patients_yaml_carrega_com_demo_offset():
    pf = load_patients()
    p001 = pf.patient("P001")
    assert p001.artifacts, "P001 sem artefatos"
    for art in p001.artifacts.values():
        assert isinstance(art.demo_offset, float)
    assert pf.demo_reference.utcoffset() == timedelta(hours=-3)


def test_p001_tem_artefatos_de_todas_as_modalidades():
    kinds = {a.kind for a in load_patients().patient("P001").artifacts.values()}
    assert {"video_physio", "video_bed", "audio", "note", "vitals", "prescriptions"} <= kinds


def test_demo_time_soma_offset_e_tempo_da_midia():
    pf = load_patients()
    t = pf.artifact_time("P001", "video_leito", media_seconds=40)
    assert t == pf.demo_reference + timedelta(seconds=640)
    assert t.utcoffset() == timedelta(hours=-3)


def test_busca_artefato_pelo_caminho_relativo_ou_absoluto():
    pf = load_patients()
    fisio = pf.artifact("P001", "video_fisio").path
    rel = pf.find_artifact_by_path("P001", fisio)
    absol = pf.find_artifact_by_path("P001", REPO_ROOT / fisio)
    assert rel is not None and rel[0] == "video_fisio"
    assert absol == rel
    assert pf.find_artifact_by_path("P001", "data/video/outro.mp4") is None

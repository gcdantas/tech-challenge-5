"""O Streamlit inicia, mostra uma linha do tempo com todas as origens e o alerta crítico."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def test_streamlit_inicia_com_mocks(mocks_dir, monkeypatch):
    monkeypatch.setenv("TC_OUTPUTS_DIR", str(mocks_dir))
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, at.exception
    assert any("Crítico" in e.value for e in at.error)
    assert (mocks_dir / "alerts.jsonl").exists()
    # a tabela da linha do tempo traz Findings das 6 origens
    tables = [d.value for d in at.dataframe if "origem" in d.value.columns]
    assert len(tables) == 1
    assert set(tables[0]["origem"]) == {"video", "audio", "text", "vitals", "prescription", "movement"}


def test_streamlit_inicia_sem_saidas(tmp_path, monkeypatch):
    monkeypatch.setenv("TC_OUTPUTS_DIR", str(tmp_path))
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, at.exception

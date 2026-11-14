from pathlib import Path

import pytest

from scripts.make_mocks import make_all


@pytest.fixture(autouse=True)
def _sem_smtp_real(monkeypatch):
    """Nenhum teste envia e-mail de verdade, mesmo com um .env preenchido na máquina.
    Variável vazia já definida não é sobrescrita pelo load_dotenv(override=False)."""
    for name in ("SMTP_HOST", "ALERT_EMAIL_TO"):
        monkeypatch.setenv(name, "")


@pytest.fixture(scope="session")
def mocks_dir(tmp_path_factory) -> Path:
    """Gera todos os mocks numa pasta temporária (não toca em outputs/)."""
    out = tmp_path_factory.mktemp("outputs")
    make_all(out)
    return out

from pathlib import Path

import pytest

from scripts.make_mocks import make_all


@pytest.fixture(scope="session")
def mocks_dir(tmp_path_factory) -> Path:
    """Gera todos os mocks numa pasta temporária (não toca em outputs/)."""
    out = tmp_path_factory.mktemp("outputs")
    make_all(out)
    return out

import pytest


@pytest.fixture
def tmp_vault(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "materias").mkdir()
    (vault / "templates").mkdir()
    (vault / ".meta").mkdir()
    (vault / ".meta" / "sessions.json").write_text("[]")
    (vault / "profile.md").write_text("# Perfil\n\nEstilo: bullet points")
    return vault


@pytest.fixture
def tmp_db(tmp_path):
    return tmp_path / "test.db"


@pytest.fixture
def tmp_chroma(tmp_path):
    return tmp_path / "chroma"

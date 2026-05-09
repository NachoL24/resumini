from pathlib import Path
from unittest.mock import MagicMock, patch

from resumini.agent.query_node import run_query
from resumini.vault.manager import VaultManager


def test_query_returns_answer():
    vault = MagicMock()
    vault.read_profile.return_value = "# Perfil\nEstilo: bullets"
    chroma = MagicMock()
    chroma.search.return_value = [
        {
            "id": "civil_u1",
            "content": "La prescripcion extintiva libera al deudor tras el plazo",
            "metadata": {"materia": "civil", "file": "u1.md"},
        },
        {
            "id": "penal_u1",
            "content": "La prescripcion de la accion penal extingue la persecucion",
            "metadata": {"materia": "penal", "file": "u1.md"},
        },
    ]
    with patch("resumini.agent.query_node.get_llm") as mock_llm:
        mock_llm.return_value.invoke.return_value = MagicMock(
            content="La prescripcion funciona distinto en civil y penal..."
        )
        result = run_query("que es la prescripcion?", vault, chroma)
    assert "prescripcion" in result.lower()


def test_query_resolves_wikilinks(tmp_vault):
    vault = VaultManager(tmp_vault)
    vault.write_note("civil", "prescripcion.md", "# Prescripcion civil\n\nLa prescripcion extintiva libera al deudor")
    chroma = MagicMock()
    chroma.search.return_value = [
        {"id": "civil_prescripcion", "content": "prescripcion extintiva", "metadata": {"materia": "civil", "file": "prescripcion.md"}}
    ]
    with patch("resumini.agent.query_node.get_llm") as mock_llm:
        mock_llm.return_value.invoke.return_value = MagicMock(content="La prescripcion en derecho civil libera al deudor...")
        result = run_query("que dice [[prescripcion]] en civil?", vault, chroma)
    assert "prescripcion" in result.lower()

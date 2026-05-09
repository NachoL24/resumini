from unittest.mock import MagicMock, patch

from resumini.agent.query_node import run_query


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

from unittest.mock import MagicMock, patch

from resumini.agent.edit_node import run_edit
from resumini.vault.manager import VaultManager


def test_edit_updates_note(tmp_vault):
    vault = VaultManager(tmp_vault)
    vault.write_note("civil", "resumen.md", "# Resumen\n\nContenido original")
    chroma = MagicMock()
    with patch("resumini.agent.edit_node.get_llm") as mock_llm:
        mock_llm.return_value.invoke.return_value = MagicMock(
            content="# Resumen\n\nContenido original\n\n## Agregado\n\nNuevo contenido"
        )
        result = run_edit("civil", "resumen.md", "agrega una seccion sobre prescripcion", vault, chroma)
    assert "Agregado" in result
    assert "Agregado" in vault.read_note("civil", "resumen.md")

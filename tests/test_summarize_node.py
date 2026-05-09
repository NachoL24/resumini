from unittest.mock import MagicMock, patch

from resumini.agent.summarize_node import run_summarize
from resumini.vault.manager import VaultManager
from resumini.vault.obsidian import parse_note


def test_summarize_produces_output(tmp_vault):
    vault = VaultManager(tmp_vault)
    vault.write_note(
        "civil",
        "u1_raw.md",
        "# Unidad 1\n\nLa prescripcion extintiva es un instituto que libera al deudor. Requiere plazo y reclamacion.",
    )
    chroma = MagicMock()
    chroma.search.return_value = [
        {
            "id": "civil_u1_raw",
            "content": "prescripcion extintiva",
            "metadata": {"materia": "civil", "file": "u1_raw.md"},
        }
    ]
    with patch("resumini.agent.summarize_node.get_llm") as mock_llm:
        mock_llm.return_value.invoke.return_value = MagicMock(
            content="# Resumen U1\n\n- Prescripcion extintiva libera al deudor\n- Requiere plazo y reclamacion"
        )
        result = run_summarize("civil", "u1_raw.md", "u1_resumen.md", vault, chroma)
        assert "Prescripcion" in result
        note = vault.read_note("civil", "u1_resumen.md")
        assert "Prescripcion" in note


def test_summarize_produces_obsidian_format(tmp_vault):
    vault = VaultManager(tmp_vault)
    vault.write_note(
        "civil",
        "u1_raw.md",
        "# Unidad 1\n\nLa prescripcion extintiva es un instituto que libera al deudor. Requiere plazo y reclamacion.",
    )
    chroma = MagicMock()
    chroma.search.return_value = [
        {
            "id": "civil_u1_raw",
            "content": "prescripcion extintiva",
            "metadata": {"materia": "civil", "file": "u1_raw.md"},
        }
    ]
    with patch("resumini.agent.summarize_node.get_llm") as mock_llm:
        mock_llm.return_value.invoke.return_value = MagicMock(
            content='---\nmateria: civil\ntype: summary\nsource: u1_raw.md\ndate: "2026-05-09"\ntags:\n  - civil\n---\n\n# Resumen U1\n\n> [!def] Prescripcion extintiva\n> Instituto que libera al deudor\n\n- Requiere plazo y reclamacion ^prescripcion'
        )
        result = run_summarize("civil", "u1_raw.md", "u1_resumen.md", vault, chroma)
        note = parse_note(result)
        assert note.frontmatter.get("type") == "summary"
        assert note.frontmatter.get("materia") == "civil"
        assert len(note.callouts) >= 1

from unittest.mock import MagicMock, patch

from resumini.vault.manager import VaultManager
from resumini.db.sqlite import SQLiteStore
from resumini.agent.graph import AgentState, build_graph
from resumini.agent.router import Intent
from resumini.vault.obsidian import parse_note


def test_full_pipeline_summarize(tmp_vault, tmp_db, tmp_chroma):
    vault = VaultManager(tmp_vault)
    vault.write_note(
        "civil",
        "u1_raw.md",
        "# Unidad 1\n\nLa prescripcion extintiva es un instituto que libera al deudor.",
    )

    sqlite = SQLiteStore(str(tmp_db))
    sqlite.initialize()

    chroma = MagicMock()

    with (
        patch("resumini.agent.router.get_llm") as mock_router_llm,
        patch("resumini.agent.summarize_node.get_llm") as mock_summ_llm,
    ):
        mock_router_llm.return_value.invoke.return_value = MagicMock(content="summarize")
        mock_summ_llm.return_value.invoke.return_value = MagicMock(
            content="# Resumen U1\n\n- **Prescripcion extintiva**: libera al deudor\n- Requiere plazo"
        )

        graph = build_graph(vault, chroma, sqlite)
        state = AgentState(
            message="haceme un resumen",
            intent=None,
            materia="civil",
            source_file="u1_raw.md",
            output_file="u1_resumen.md",
            pdf_path=None,
            edit_instruction=None,
            output=None,
            valid=False,
            errors=[],
            metadata={},
        )
        result = graph.invoke(state)

        assert result["valid"] is True
        assert result["output"] is not None


def test_full_pipeline_produces_obsidian_format(tmp_path):
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    (vault_dir / "materias").mkdir()
    (vault_dir / "templates").mkdir()
    (vault_dir / ".meta").mkdir()
    (vault_dir / ".meta" / "sessions.json").write_text("[]")
    (vault_dir / "profile.md").write_text("---\ntype: profile\n---\n\n# Perfil\n\nEstilo: bullets")

    vault = VaultManager(vault_dir)
    sqlite = SQLiteStore(str(tmp_path / "test.db"))
    sqlite.initialize()
    chroma = MagicMock()

    with (
        patch("resumini.agent.graph.classify_intent", return_value=Intent.SUMMARIZE),
        patch("resumini.agent.summarize_node.get_llm") as mock_llm,
    ):
        mock_llm.return_value.invoke.return_value = MagicMock(
            content="---\nmateria: civil\ntype: summary\n---\n\n# Resumen\n\n> [!def] Test\nDefinition here"
        )

        graph = build_graph(vault, chroma, sqlite)
        state = AgentState(
            message="resumen",
            intent=Intent.SUMMARIZE,
            materia="civil",
            source_file="u1.md",
            output_file="u1_resumen.md",
            pdf_path=None,
            edit_instruction=None,
            output=None,
            valid=False,
            errors=[],
            metadata={},
        )
        vault.write_note("civil", "u1.md", "# U1\n\nContent")
        result = graph.invoke(state)
        output = result.get("output", "")
        note = parse_note(output)
        assert note.frontmatter.get("type") == "summary"

    sqlite.close()

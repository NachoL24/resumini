from unittest.mock import MagicMock, patch

from resumini.vault.manager import VaultManager
from resumini.db.sqlite import SQLiteStore
from resumini.agent.graph import AgentState, build_graph


def test_full_pipeline_summarize(tmp_vault, tmp_db, tmp_chroma):
    vault = VaultManager(tmp_vault)
    vault.write_note("civil", "u1_raw.md", "# Unidad 1\n\nLa prescripcion extintiva es un instituto que libera al deudor.")

    sqlite = SQLiteStore(str(tmp_db))
    sqlite.initialize()

    chroma = MagicMock()

    with patch("resumini.agent.router.get_llm") as mock_router_llm, \
         patch("resumini.agent.summarize_node.get_llm") as mock_summ_llm:
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

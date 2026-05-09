from unittest.mock import MagicMock

from resumini.agent.graph import AgentState, build_graph


def test_agent_state_has_required_fields():
    state = AgentState(
        message="haceme un resumen",
        materia="civil",
        intent=None,
        output=None,
        source_file=None,
        output_file=None,
        pdf_path=None,
        edit_instruction=None,
        valid=False,
        errors=[],
        metadata={},
    )
    assert state["message"] == "haceme un resumen"


def test_build_graph_returns_compiled():
    vault = MagicMock()
    chroma = MagicMock()
    sqlite = MagicMock()
    graph = build_graph(vault, chroma, sqlite)
    assert graph is not None

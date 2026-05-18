"""Tests for system prompts."""

from resumini.agent.prompts import build_system_prompt


def test_build_system_prompt_includes_materia_and_profile():
    result = build_system_prompt(materia="civil", profile="Estilo: bullet points")
    assert "civil" in result
    assert "bullet points" in result


def test_build_system_prompt_handles_no_active_materia():
    result = build_system_prompt(materia=None, profile="x")
    lowered = result.lower()
    assert "ninguna" in lowered or "preguntale" in lowered


def test_build_system_prompt_forbids_emojis_explicitly():
    result = build_system_prompt(materia="x", profile="y").lower()
    assert "emoji" in result and "no" in result


def test_build_system_prompt_lists_all_tools_by_name():
    result = build_system_prompt(materia="x", profile="y")
    for tool_name in [
        "ingest_pdf",
        "summarize",
        "search_vault",
        "edit_note",
        "list_materias",
        "list_notes",
        "read_note",
    ]:
        assert tool_name in result


def test_build_system_prompt_instructs_to_ask_before_summarize():
    result = build_system_prompt(materia="x", profile="y").lower()
    assert "pregunt" in result and "summarize" in result

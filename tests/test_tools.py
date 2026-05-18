"""Tests for agent tools."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import fitz

from resumini.agent.tools import make_readonly_tools, make_task_tools, make_tools
from resumini.vault.manager import VaultManager


def _get_tool(tools, name):
    for t in tools:
        if t.name == name:
            return t
    raise AssertionError(f"Tool {name!r} not in {[t.name for t in tools]}")


def _write_minimal_pdf(path: Path, text: str = "Test"):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()


def test_make_task_tools_returns_four_named_tools(tmp_vault):
    vault = VaultManager(tmp_vault)
    chroma = MagicMock()
    tools = make_task_tools(vault, chroma)
    names = {t.name for t in tools}
    assert names == {"ingest_pdf", "summarize", "search_vault", "edit_note"}


def test_search_vault_returns_formatted_fragments(tmp_vault):
    vault = VaultManager(tmp_vault)
    chroma = MagicMock()
    chroma.search.return_value = [
        {
            "id": "civil_u1",
            "content": "La prescripcion extintiva libera al deudor",
            "metadata": {"materia": "civil", "file": "u1.md"},
        },
        {
            "id": "penal_u1",
            "content": "La prescripcion de la accion penal extingue la persecucion",
            "metadata": {"materia": "penal", "file": "u1.md"},
        },
    ]
    tools = make_task_tools(vault, chroma)
    search = _get_tool(tools, "search_vault")
    result = search.invoke({"question": "que es la prescripcion?"})
    assert "civil/u1.md" in result
    assert "penal/u1.md" in result
    assert "prescripcion" in result.lower()


def test_search_vault_empty_results_returns_friendly_message(tmp_vault):
    vault = VaultManager(tmp_vault)
    chroma = MagicMock()
    chroma.search.return_value = []
    tools = make_task_tools(vault, chroma)
    search = _get_tool(tools, "search_vault")
    result = search.invoke({"question": "nada existe"})
    assert "no se encontraron" in result.lower()


def test_search_vault_resolves_wikilinks_in_question(tmp_vault):
    vault = VaultManager(tmp_vault)
    vault.write_note(
        "civil", "prescripcion.md", "# Prescripcion\n\nLibera al deudor tras plazo"
    )
    chroma = MagicMock()
    chroma.search.return_value = []
    tools = make_task_tools(vault, chroma)
    search = _get_tool(tools, "search_vault")
    result = search.invoke({"question": "que dice [[prescripcion]]?"})
    assert "Libera al deudor" in result


def test_search_vault_does_not_call_llm(tmp_vault):
    vault = VaultManager(tmp_vault)
    chroma = MagicMock()
    chroma.search.return_value = []
    tools = make_task_tools(vault, chroma)
    search = _get_tool(tools, "search_vault")
    with patch("resumini.agent.query_node.get_llm") as mock_llm:
        search.invoke({"question": "algo"})
        mock_llm.assert_not_called()


def test_ingest_pdf_returns_success_and_writes_vault(tmp_vault, tmp_path):
    pdf = tmp_path / "doc.pdf"
    _write_minimal_pdf(pdf, "Hola mundo PDF en civil")
    vault = VaultManager(tmp_vault)
    chroma = MagicMock()
    tools = make_task_tools(vault, chroma)
    ingest = _get_tool(tools, "ingest_pdf")
    result = ingest.invoke(
        {"pdf_path": str(pdf), "materia": "civil", "filename": "u1.md"}
    )
    assert "civil" in result and "u1.md" in result
    assert "Hola mundo PDF" in vault.read_note("civil", "u1.md")


def test_ingest_pdf_missing_file_returns_error_string(tmp_vault):
    vault = VaultManager(tmp_vault)
    chroma = MagicMock()
    tools = make_task_tools(vault, chroma)
    ingest = _get_tool(tools, "ingest_pdf")
    result = ingest.invoke(
        {
            "pdf_path": "/tmp/does_not_exist_resumini.pdf",
            "materia": "civil",
            "filename": "u1.md",
        }
    )
    assert "error" in result.lower()


def test_summarize_writes_output_and_returns_success(tmp_vault):
    vault = VaultManager(tmp_vault)
    vault.write_note(
        "civil", "u1_raw.md", "# U1\n\nLa prescripcion extintiva libera al deudor."
    )
    chroma = MagicMock()
    tools = make_task_tools(vault, chroma)
    summarize = _get_tool(tools, "summarize")
    with patch("resumini.agent.summarize_node.get_llm") as mock_llm:
        mock_llm.return_value.invoke.return_value = MagicMock(
            content="# Resumen\n\n- Prescripcion extintiva libera al deudor"
        )
        result = summarize.invoke(
            {
                "materia": "civil",
                "source_file": "u1_raw.md",
                "output_file": "u1_resumen.md",
            }
        )
    assert "u1_resumen.md" in result
    assert "Prescripcion" in vault.read_note("civil", "u1_resumen.md")


def test_summarize_default_output_filename(tmp_vault):
    vault = VaultManager(tmp_vault)
    vault.write_note("civil", "u1.md", "# U1\n\nContenido")
    chroma = MagicMock()
    tools = make_task_tools(vault, chroma)
    summarize = _get_tool(tools, "summarize")
    with patch("resumini.agent.summarize_node.get_llm") as mock_llm:
        mock_llm.return_value.invoke.return_value = MagicMock(
            content="# Resumen\n\nContenido"
        )
        summarize.invoke({"materia": "civil", "source_file": "u1.md"})
    assert "u1_resumen.md" in [
        f.name for f in (tmp_vault / "materias" / "civil").iterdir()
    ]


def test_summarize_with_instructions_includes_them_in_prompt(tmp_vault):
    vault = VaultManager(tmp_vault)
    vault.write_note("civil", "u1.md", "# U1\n\nContenido")
    chroma = MagicMock()
    tools = make_task_tools(vault, chroma)
    summarize = _get_tool(tools, "summarize")
    captured = {}
    with patch("resumini.agent.summarize_node.get_llm") as mock_llm:

        def capture(messages):
            captured["system"] = messages[0].content
            return MagicMock(content="# Resumen\n\nContenido")

        mock_llm.return_value.invoke.side_effect = capture
        summarize.invoke(
            {
                "materia": "civil",
                "source_file": "u1.md",
                "instructions": "foco en ejemplos practicos, sin tablas",
            }
        )
    assert "ejemplos practicos" in captured["system"]


def test_summarize_missing_source_returns_error(tmp_vault):
    vault = VaultManager(tmp_vault)
    chroma = MagicMock()
    tools = make_task_tools(vault, chroma)
    summarize = _get_tool(tools, "summarize")
    result = summarize.invoke({"materia": "civil", "source_file": "no_existe.md"})
    assert "error" in result.lower()


def test_edit_note_applies_and_returns_success(tmp_vault):
    vault = VaultManager(tmp_vault)
    vault.write_note("civil", "r.md", "# R\n\nOriginal")
    chroma = MagicMock()
    tools = make_task_tools(vault, chroma)
    edit = _get_tool(tools, "edit_note")
    with patch("resumini.agent.edit_node.get_llm") as mock_llm:
        mock_llm.return_value.invoke.return_value = MagicMock(
            content="# R\n\nOriginal\n\n## Nuevo\n\nContenido"
        )
        result = edit.invoke(
            {
                "materia": "civil",
                "filename": "r.md",
                "instruction": "agrega seccion Nuevo",
            }
        )
    assert "r.md" in result and "civil" in result
    assert "Nuevo" in vault.read_note("civil", "r.md")


def test_edit_note_missing_file_returns_error(tmp_vault):
    vault = VaultManager(tmp_vault)
    chroma = MagicMock()
    tools = make_task_tools(vault, chroma)
    edit = _get_tool(tools, "edit_note")
    result = edit.invoke(
        {"materia": "civil", "filename": "no.md", "instruction": "x"}
    )
    assert "error" in result.lower()


def test_make_readonly_tools_returns_three_named_tools(tmp_vault):
    vault = VaultManager(tmp_vault)
    tools = make_readonly_tools(vault)
    names = {t.name for t in tools}
    assert names == {"list_materias", "list_notes", "read_note"}


def test_list_materias_lists_existing_directories(tmp_vault):
    vault = VaultManager(tmp_vault)
    vault.write_note("civil", "u1.md", "x")
    vault.write_note("penal", "u1.md", "x")
    tools = make_readonly_tools(vault)
    result = _get_tool(tools, "list_materias").invoke({})
    assert "civil" in result and "penal" in result


def test_list_materias_empty_vault_returns_friendly_message(tmp_vault):
    vault = VaultManager(tmp_vault)
    tools = make_readonly_tools(vault)
    result = _get_tool(tools, "list_materias").invoke({})
    assert "no hay" in result.lower()


def test_list_notes_returns_files_in_materia(tmp_vault):
    vault = VaultManager(tmp_vault)
    vault.write_note("civil", "u1.md", "x")
    vault.write_note("civil", "u2.md", "x")
    tools = make_readonly_tools(vault)
    result = _get_tool(tools, "list_notes").invoke({"materia": "civil"})
    assert "u1.md" in result and "u2.md" in result


def test_list_notes_unknown_materia_returns_friendly_message(tmp_vault):
    vault = VaultManager(tmp_vault)
    tools = make_readonly_tools(vault)
    result = _get_tool(tools, "list_notes").invoke({"materia": "fantasma"})
    assert "fantasma" in result


def test_read_note_returns_full_content(tmp_vault):
    vault = VaultManager(tmp_vault)
    vault.write_note(
        "civil",
        "u1.md",
        "---\nmateria: civil\n---\n\n# U1\n\nContenido completo",
    )
    tools = make_readonly_tools(vault)
    result = _get_tool(tools, "read_note").invoke(
        {"materia": "civil", "filename": "u1.md"}
    )
    assert "materia: civil" in result
    assert "Contenido completo" in result


def test_read_note_missing_returns_error(tmp_vault):
    vault = VaultManager(tmp_vault)
    tools = make_readonly_tools(vault)
    result = _get_tool(tools, "read_note").invoke(
        {"materia": "civil", "filename": "no.md"}
    )
    assert "error" in result.lower()


def test_make_tools_combines_task_and_readonly(tmp_vault):
    vault = VaultManager(tmp_vault)
    chroma = MagicMock()
    tools = make_tools(vault, chroma)
    names = {t.name for t in tools}
    assert names == {
        "ingest_pdf",
        "summarize",
        "search_vault",
        "edit_note",
        "list_materias",
        "list_notes",
        "read_note",
    }

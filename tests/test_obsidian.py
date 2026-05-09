import pytest

from resumini.vault.obsidian import (
    ObsidianNote,
    extract_frontmatter,
    render_frontmatter,
    render_note,
    parse_note,
    resolve_wikilink,
    resolve_embed,
)


def test_extract_frontmatter_with_yaml():
    raw = "---\nmateria: civil\ntype: summary\n---\n\n# Content here"
    fm, body = extract_frontmatter(raw)
    assert fm == {"materia": "civil", "type": "summary"}
    assert body.strip() == "# Content here"


def test_extract_frontmatter_no_yaml():
    raw = "# Just a heading\n\nSome content"
    fm, body = extract_frontmatter(raw)
    assert fm == {}
    assert body == raw


def test_extract_frontmatter_empty_yaml():
    raw = "---\n---\n\n# Content"
    fm, body = extract_frontmatter(raw)
    assert fm == {}
    assert body.strip() == "# Content"


def test_render_frontmatter_simple():
    result = render_frontmatter({"materia": "civil", "type": "summary"})
    assert result.startswith("---\n")
    assert result.strip().endswith("---")
    assert "materia: civil" in result
    assert "type: summary" in result


def test_render_frontmatter_with_list():
    fm = {"tags": ["civil", "resumen"]}
    result = render_frontmatter(fm)
    assert "- civil" in result
    assert "- resumen" in result


def test_render_frontmatter_empty():
    result = render_frontmatter({})
    assert result == ""


def test_render_note_with_frontmatter():
    note = ObsidianNote(
        frontmatter={"materia": "civil", "type": "summary"},
        content="# My Note\n\nSome text",
        wikilinks=[],
        embeds=[],
        block_ids=[],
        tags=[],
        callouts=[],
    )
    rendered = render_note(note)
    assert rendered.startswith("---\n")
    assert "materia: civil" in rendered
    assert "# My Note" in rendered


def test_render_note_without_frontmatter():
    note = ObsidianNote(
        frontmatter={},
        content="# My Note\n\nSome text",
        wikilinks=[],
        embeds=[],
        block_ids=[],
        tags=[],
        callouts=[],
    )
    rendered = render_note(note)
    assert not rendered.startswith("---")
    assert "# My Note" in rendered


def test_parse_note_extracts_wikilinks():
    raw = "# Note\n\nSee [[prescripcion]] and [[derecho_civil/unidad_1]]"
    note = parse_note(raw)
    assert note.wikilinks == ["prescripcion", "derecho_civil/unidad_1"]


def test_parse_note_extracts_embeds():
    raw = "# Note\n\n![[def_prescripcion]] and ![[shared/concepto]]"
    note = parse_note(raw)
    assert note.embeds == ["def_prescripcion", "shared/concepto"]


def test_parse_note_extracts_block_ids():
    raw = "# Note\n\nKey term ^prescripcion-extintiva\nOther line\nAnother ^block-2"
    note = parse_note(raw)
    assert "prescripcion-extintiva" in note.block_ids
    assert "block-2" in note.block_ids


def test_parse_note_extracts_tags():
    raw = "# Note\n\nContent with #derecho and #civil/prescripcion tags"
    note = parse_note(raw)
    assert "derecho" in note.tags
    assert "civil/prescripcion" in note.tags


def test_parse_note_does_not_match_heading_as_tag():
    raw = "# Heading 1\n\n## Heading 2\n\nSome #real-tag here"
    note = parse_note(raw)
    assert "Heading" not in note.tags
    assert "Heading-1" not in note.tags
    assert "real-tag" in note.tags


def test_parse_note_extracts_callouts():
    raw = "# Note\n\n> [!def] Prescripcion extintiva\n> Instituto que libera al deudor\n> tras el plazo\n\nOther text"
    note = parse_note(raw)
    assert len(note.callouts) == 1
    assert note.callouts[0].type == "def"
    assert note.callouts[0].title == "Prescripcion extintiva"
    assert "libera al deudor" in note.callouts[0].content


def test_parse_note_multiple_callouts():
    raw = "# Note\n\n> [!def] Term A\n> Definition A\n\n> [!warning] Common mistake\n> Don't confuse X with Y"
    note = parse_note(raw)
    assert len(note.callouts) == 2
    assert note.callouts[0].type == "def"
    assert note.callouts[1].type == "warning"


def test_parse_note_callout_no_title():
    raw = "# Note\n\n> [!tip]\n> Study tip here"
    note = parse_note(raw)
    assert len(note.callouts) == 1
    assert note.callouts[0].type == "tip"
    assert note.callouts[0].title == ""


def test_parse_note_with_frontmatter_and_patterns():
    raw = (
        "---\nmateria: civil\ntype: summary\ntags:\n - civil\n---\n\n"
        "# Resumen\n\n> [!def] Prescripcion\n> Definicion aqui ^prescripcion\n\n"
        "See [[penal/unidad_1]] for comparison."
    )
    note = parse_note(raw)
    assert note.frontmatter == {"materia": "civil", "type": "summary", "tags": ["civil"]}
    assert "prescripcion" in note.block_ids
    assert "penal/unidad_1" in note.wikilinks
    assert len(note.callouts) == 1
    assert note.callouts[0].type == "def"


def test_parse_note_empty_content():
    note = parse_note("")
    assert note.frontmatter == {}
    assert note.content == ""
    assert note.wikilinks == []
    assert note.embeds == []
    assert note.block_ids == []
    assert note.tags == []
    assert note.callouts == []


@pytest.fixture
def vault_with_notes(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    materias = vault / "materias"
    materias.mkdir()
    civil = materias / "civil"
    civil.mkdir()
    penal = materias / "penal"
    penal.mkdir()
    (civil / "prescripcion.md").write_text("# Prescripcion civil\n\nContent")
    (penal / "prescripcion.md").write_text("# Prescripcion penal\n\nContent")
    (civil / "unidad_1.md").write_text("# Unidad 1\n\nContent")
    (vault / "profile.md").write_text("# Profile")
    return vault


def test_resolve_wikilink_same_materia(vault_with_notes):
    result = resolve_wikilink("prescripcion", vault_with_notes, current_materia="civil")
    assert result is not None
    assert result.name == "prescripcion.md"
    assert result.parent.name == "civil"


def test_resolve_wikilink_different_materia(vault_with_notes):
    result = resolve_wikilink("prescripcion", vault_with_notes, current_materia="penal")
    assert result is not None
    assert result.name == "prescripcion.md"
    assert result.parent.name == "penal"


def test_resolve_wikilink_no_current_materia(vault_with_notes):
    result = resolve_wikilink("prescripcion", vault_with_notes, current_materia=None)
    assert result is not None
    assert result.name == "prescripcion.md"


def test_resolve_wikilink_root_file(vault_with_notes):
    result = resolve_wikilink("profile", vault_with_notes, current_materia="civil")
    assert result is not None
    assert result.name == "profile.md"


def test_resolve_wikilink_not_found(vault_with_notes):
    result = resolve_wikilink("nonexistent_note", vault_with_notes)
    assert result is None


def test_resolve_embed_returns_content(vault_with_notes):
    content = resolve_embed("prescripcion", vault_with_notes, current_materia="civil")
    assert content is not None
    assert "Prescripcion civil" in content


def test_resolve_embed_not_found(vault_with_notes):
    content = resolve_embed("nonexistent_note", vault_with_notes)
    assert content is None


def test_resolve_wikilink_with_md_extension(vault_with_notes):
    result = resolve_wikilink("prescripcion.md", vault_with_notes, current_materia="civil")
    assert result is not None
    assert result.name == "prescripcion.md"

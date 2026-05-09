import pytest

from resumini.vault.manager import VaultManager
from resumini.vault.obsidian import ObsidianNote


@pytest.fixture
def vm(tmp_vault):
    return VaultManager(tmp_vault)


def test_read_profile(vm):
    content = vm.read_profile()
    assert "Estilo" in content


def test_write_materia_note(vm):
    vm.write_note("derecho_civil", "unidad_1.md", "# Unidad 1\n\nContenido")
    path = vm.vault_dir / "materias" / "derecho_civil" / "unidad_1.md"
    assert path.exists()
    assert "Unidad 1" in path.read_text()


def test_write_note_creates_materia_dir(vm):
    vm.write_note("penal", "unidad_1.md", "# Penal U1")
    assert (vm.vault_dir / "materias" / "penal").is_dir()


def test_read_note(vm):
    vm.write_note("civil", "test.md", "# Test content")
    content = vm.read_note("civil", "test.md")
    assert "Test content" in content


def test_read_note_not_found(vm):
    with pytest.raises(FileNotFoundError):
        vm.read_note("nonexistent", "missing.md")


def test_list_materias(vm):
    vm.write_note("civil", "u1.md", "c1")
    vm.write_note("penal", "u1.md", "p1")
    materias = vm.list_materias()
    assert "civil" in materias
    assert "penal" in materias


def test_list_notes_in_materia(vm):
    vm.write_note("civil", "u1.md", "c1")
    vm.write_note("civil", "u2.md", "c2")
    notes = vm.list_notes("civil")
    assert "u1.md" in notes
    assert "u2.md" in notes


def test_update_profile(vm):
    vm.update_profile("# Perfil\n\nEstilo: prosa")
    assert "prosa" in vm.read_profile()


def test_get_sessions_empty(vm):
    sessions = vm.get_sessions()
    assert sessions == []


def test_add_session(vm):
    vm.add_session({"id": "s1", "query": "resumir pdf", "timestamp": "2026-05-08"})
    sessions = vm.get_sessions()
    assert len(sessions) == 1
    assert sessions[0]["id"] == "s1"


def test_write_materia_index(vm):
    vm.write_materia_index("civil", {"profesor": "Garcia", "cuatrimestre": "1C2026"})
    index = vm.read_note("civil", "_index.md")
    assert "Garcia" in index


def test_read_obsidian_note(vm):
    vm.write_note(
        "civil",
        "test_obs.md",
        "---\nmateria: civil\ntype: summary\n---\n\n# Test\n\nSee [[penal/u1]]",
    )
    note = vm.read_obsidian_note("civil", "test_obs.md")
    assert note.frontmatter["materia"] == "civil"
    assert "penal/u1" in note.wikilinks


def test_write_obsidian_note(vm):
    note = ObsidianNote(
        frontmatter={"materia": "civil", "type": "summary"},
        content="# My Summary\n\nContent here",
        wikilinks=[],
        embeds=[],
        block_ids=[],
        tags=[],
        callouts=[],
    )
    vm.write_obsidian_note("civil", "summary.md", note)
    raw = vm.read_note("civil", "summary.md")
    assert "materia: civil" in raw
    assert "# My Summary" in raw


def test_find_note_by_title(vm):
    vm.write_note("civil", "prescripcion.md", "# Prescripcion civil")
    result = vm.find_note_by_title("prescripcion")
    assert result is not None
    assert "prescripcion.md" in str(result)


def test_find_note_by_title_not_found(vm):
    result = vm.find_note_by_title("nonexistent")
    assert result is None


def test_get_embed_content(vm):
    vm.write_note("civil", "shared_def.md", "# Shared Definition\n\nA legal concept")
    content = vm.get_embed_content("shared_def", current_materia="civil")
    assert content is not None
    assert "Shared Definition" in content


def test_get_embed_content_not_found(vm):
    content = vm.get_embed_content("nonexistent", current_materia="civil")
    assert content is None


def test_list_all_tags(vm):
    vm.write_note("civil", "u1.md", "---\ntags:\n - civil\n - prescripcion\n---\n\n# U1")
    vm.write_note("penal", "u1.md", "---\ntags:\n - penal\n - prescripcion\n---\n\n# U1")
    tag_map = vm.list_all_tags()
    assert "civil" in tag_map
    assert "penal" in tag_map
    assert "prescripcion" in tag_map


def test_backlinks(vm):
    vm.write_note("civil", "u1.md", "# U1\n\nSee [[penal/u1]] for comparison")
    vm.write_note("penal", "u1.md", "# Penal U1\n\nContent")
    links = vm.backlinks("penal/u1")
    assert len(links) >= 1

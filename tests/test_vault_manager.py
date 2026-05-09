import pytest

from resumini.vault.manager import VaultManager


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

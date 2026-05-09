from resumini.vault.migrate import migrate_vault
from resumini.vault.obsidian import parse_note


def test_migrate_adds_frontmatter_to_bare_file(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    materias = vault / "materias"
    civil = materias / "civil"
    civil.mkdir(parents=True)
    (civil / "unidad_1.md").write_text("# Unidad 1\n\nContenido de la unidad")
    (civil / "resumen_parcial.md").write_text("# Resumen parcial\n\nResumen aqui")
    (civil / "_index.md").write_text("# Civil\n\n- Profesor: Garcia")

    migrate_vault(vault)

    u1 = parse_note((civil / "unidad_1.md").read_text())
    assert u1.frontmatter.get("materia") == "civil"
    assert u1.frontmatter.get("type") == "note"

    resumen = parse_note((civil / "resumen_parcial.md").read_text())
    assert resumen.frontmatter.get("type") == "summary"

    index = parse_note((civil / "_index.md").read_text())
    assert index.frontmatter.get("type") == "index"


def test_migrate_preserves_existing_frontmatter(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    materias = vault / "materias"
    civil = materias / "civil"
    civil.mkdir(parents=True)
    (civil / "u1.md").write_text("---\nmateria: civil\ntype: raw_ingest\n---\n\n# U1\n\nContent")

    migrate_vault(vault)

    note = parse_note((civil / "u1.md").read_text())
    assert note.frontmatter.get("materia") == "civil"
    assert note.frontmatter.get("type") == "raw_ingest"
    assert "# U1" in note.content


def test_migrate_skips_non_md_files(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    materias = vault / "materias"
    civil = materias / "civil"
    civil.mkdir(parents=True)
    (civil / "image.png").write_bytes(b"fake image")

    migrate_vault(vault)

    assert (civil / "image.png").exists()


def test_migrate_handles_profile(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "profile.md").write_text("# Perfil\n\nEstilo: bullets")

    migrate_vault(vault)

    note = parse_note((vault / "profile.md").read_text())
    assert note.frontmatter.get("type") == "profile"
    assert "Perfil" in note.content


def test_migrate_handles_empty_vault(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "materias").mkdir()

    migrate_vault(vault)

from datetime import datetime
from pathlib import Path

from resumini.vault.obsidian import extract_frontmatter, render_frontmatter


def _infer_type(filename: str) -> str:
    if filename == "_index.md":
        return "index"
    if "resumen" in filename.lower():
        return "summary"
    if filename.startswith("clase_") or "apunte" in filename.lower():
        return "apunte"
    return "note"


def migrate_vault(vault_dir: Path):
    vault_dir = Path(vault_dir)
    for md_file in vault_dir.rglob("*.md"):
        try:
            raw = md_file.read_text()
        except OSError:
            continue

        fm, body = extract_frontmatter(raw)
        if fm:
            continue

        relative = md_file.relative_to(vault_dir)
        parts = relative.parts

        new_fm: dict = {}

        if len(parts) >= 3 and parts[0] == "materias":
            new_fm["materia"] = parts[1]

        if md_file.name == "profile.md" and len(parts) == 1:
            new_fm["type"] = "profile"
        else:
            new_fm["type"] = _infer_type(md_file.name)

        try:
            mtime = md_file.stat().st_mtime
            new_fm["date"] = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")
        except OSError:
            new_fm["date"] = datetime.now().strftime("%Y-%m-%d")

        rendered_fm = render_frontmatter(new_fm)
        new_content = rendered_fm + body if rendered_fm else body
        md_file.write_text(new_content)

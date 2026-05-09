import json
from pathlib import Path

from resumini.vault.obsidian import (
    ObsidianNote,
    parse_note,
    render_note,
    resolve_embed,
    resolve_wikilink,
)
from resumini.vault.templates import MATERIA_INDEX, PROFILE


class VaultManager:
    def __init__(self, vault_dir: Path):
        self.vault_dir = Path(vault_dir)
        self._ensure_structure()

    def _ensure_structure(self):
        for d in ["materias", "templates", ".meta"]:
            (self.vault_dir / d).mkdir(parents=True, exist_ok=True)

        profile_path = self.vault_dir / "profile.md"
        if not profile_path.exists():
            profile_path.write_text(
                PROFILE.format(
                    formato="bullet points",
                    detalle="moderado",
                    materias="- (ninguna)",
                    notas="- (ninguna)",
                )
            )

        sessions_path = self.vault_dir / ".meta" / "sessions.json"
        if not sessions_path.exists():
            sessions_path.write_text("[]")

    def read_profile(self) -> str:
        return (self.vault_dir / "profile.md").read_text()

    def update_profile(self, content: str):
        (self.vault_dir / "profile.md").write_text(content)

    def write_note(self, materia: str, filename: str, content: str):
        materia_dir = self.vault_dir / "materias" / materia
        materia_dir.mkdir(parents=True, exist_ok=True)
        (materia_dir / filename).write_text(content)

    def read_note(self, materia: str, filename: str) -> str:
        path = self.vault_dir / "materias" / materia / filename
        if not path.exists():
            raise FileNotFoundError(f"Note not found: {materia}/{filename}")
        return path.read_text()

    def list_materias(self) -> list[str]:
        materias_dir = self.vault_dir / "materias"
        if not materias_dir.exists():
            return []
        return [d.name for d in materias_dir.iterdir() if d.is_dir()]

    def list_notes(self, materia: str) -> list[str]:
        materia_dir = self.vault_dir / "materias" / materia
        if not materia_dir.exists():
            return []
        return [f.name for f in materia_dir.iterdir() if f.is_file() and f.name != "_index.md"]

    def write_materia_index(self, materia: str, metadata: dict):
        content = MATERIA_INDEX.format(
            nombre=materia,
            profesor=metadata.get("profesor", "N/A"),
            cuatrimestre=metadata.get("cuatrimestre", "N/A"),
            fuentes=metadata.get("fuentes", "N/A"),
        )
        self.write_note(materia, "_index.md", content)

    def get_sessions(self) -> list[dict]:
        path = self.vault_dir / ".meta" / "sessions.json"
        return json.loads(path.read_text())

    def add_session(self, session: dict):
        sessions = self.get_sessions()
        sessions.append(session)
        (self.vault_dir / ".meta" / "sessions.json").write_text(
            json.dumps(sessions, indent=2, ensure_ascii=False)
        )

    def read_obsidian_note(self, materia: str, filename: str) -> ObsidianNote:
        raw = self.read_note(materia, filename)
        return parse_note(raw)

    def write_obsidian_note(self, materia: str, filename: str, note: ObsidianNote):
        rendered = render_note(note)
        self.write_note(materia, filename, rendered)

    def find_note_by_title(self, title: str) -> Path | None:
        return resolve_wikilink(title, self.vault_dir)

    def get_embed_content(self, target: str, current_materia: str | None = None) -> str | None:
        return resolve_embed(target, self.vault_dir, current_materia=current_materia)

    def list_all_tags(self) -> dict[str, list[str]]:
        tag_map: dict[str, list[str]] = {}
        materias_dir = self.vault_dir / "materias"
        if not materias_dir.exists():
            return tag_map
        for materia_dir in materias_dir.iterdir():
            if not materia_dir.is_dir():
                continue
            for md_file in materia_dir.iterdir():
                if not md_file.is_file() or not md_file.suffix == ".md":
                    continue
                try:
                    raw = md_file.read_text()
                    note = parse_note(raw)
                    rel = str(md_file.relative_to(self.vault_dir))
                    all_tags = list(note.tags)
                    fm_tags = note.frontmatter.get("tags", [])
                    if isinstance(fm_tags, list):
                        all_tags.extend(str(t).lstrip("#") for t in fm_tags)
                    elif isinstance(fm_tags, str):
                        all_tags.append(fm_tags.lstrip("#"))
                    for tag in all_tags:
                        tag_map.setdefault(tag, []).append(rel)
                except OSError:
                    continue
        return tag_map

    def backlinks(self, note_title: str) -> list[Path]:
        links: list[Path] = []
        materias_dir = self.vault_dir / "materias"
        if not materias_dir.exists():
            return links
        for materia_dir in materias_dir.iterdir():
            if not materia_dir.is_dir():
                continue
            for md_file in materia_dir.iterdir():
                if not md_file.is_file() or not md_file.suffix == ".md":
                    continue
                try:
                    raw = md_file.read_text()
                    note = parse_note(raw)
                    if note_title in note.wikilinks:
                        links.append(md_file)
                except OSError:
                    continue
        return links

import json
from pathlib import Path

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

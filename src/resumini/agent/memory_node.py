import uuid
from datetime import datetime

from resumini.db.chroma import ChromaClient
from resumini.db.sqlite import SQLiteStore
from resumini.vault.manager import VaultManager


def update_memory(
    materia: str,
    filename: str,
    content: str,
    vault: VaultManager,
    chroma: ChromaClient,
    sqlite: SQLiteStore,
    session_id: str | None = None,
):
    vault.write_note(materia, filename, content)
    chroma.index_document(
        doc_id=f"{materia}_{filename.replace('.md', '')}",
        content=content,
        metadata={"materia": materia, "file": filename, "type": "auto"},
    )
    sid = session_id or str(uuid.uuid4())[:8]
    sqlite.add_session(sid, f"update {materia}/{filename}", datetime.now().isoformat())
    vault.add_session(
        {
            "id": sid,
            "action": "memory_update",
            "materia": materia,
            "file": filename,
            "timestamp": datetime.now().isoformat(),
        }
    )

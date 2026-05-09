from pathlib import Path

from resumini.db.chroma import ChromaClient
from resumini.ingest.pdf import ingest_pdf
from resumini.vault.manager import VaultManager


def run_ingest(
    pdf_path: Path, materia: str, filename: str, vault: VaultManager, chroma: ChromaClient
) -> str:
    raw_content = ingest_pdf(pdf_path)
    vault.write_note(materia, filename, raw_content)
    chroma.index_document(
        doc_id=f"{materia}_{filename.replace('.md', '')}",
        content=raw_content,
        metadata={"materia": materia, "file": filename, "type": "raw_ingest"},
    )
    return raw_content

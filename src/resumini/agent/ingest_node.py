from datetime import date
from pathlib import Path

from resumini.db.chroma import ChromaClient
from resumini.ingest.pdf import ingest_pdf
from resumini.vault.manager import VaultManager
from resumini.vault.obsidian import render_frontmatter


def run_ingest(
    pdf_path: Path, materia: str, filename: str, vault: VaultManager, chroma: ChromaClient
) -> str:
    raw_content = ingest_pdf(pdf_path)
    frontmatter = render_frontmatter({
        "materia": materia,
        "type": "raw_ingest",
        "source": str(pdf_path.name),
        "date": date.today().isoformat(),
    })
    content_with_fm = f"{frontmatter}\n{raw_content}" if frontmatter else raw_content
    vault.write_note(materia, filename, content_with_fm)
    chroma.index_document(
        doc_id=f"{materia}_{filename.replace('.md', '')}",
        content=content_with_fm,
        metadata={"materia": materia, "file": filename, "type": "raw_ingest"},
    )
    return content_with_fm

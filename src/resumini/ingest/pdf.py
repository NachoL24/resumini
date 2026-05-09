from pathlib import Path

import fitz


def ingest_pdf(pdf_path: Path) -> str:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        raise ValueError(f"Invalid PDF file: {pdf_path}") from e
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages.append(f"## Pagina {i + 1}\n\n{text.strip()}")
    doc.close()
    return "\n\n".join(pages)

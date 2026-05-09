from pathlib import Path

import fitz
import pytest

from resumini.ingest.pdf import ingest_pdf


def _make_pdf(text: str, path: Path):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=12)
    doc.save(str(path))
    doc.close()


def test_ingest_pdf_extracts_text(tmp_path):
    pdf_path = tmp_path / "test.pdf"
    _make_pdf("Este es un texto de prueba para el PDF.", pdf_path)
    result = ingest_pdf(pdf_path)
    assert "texto de prueba" in result


def test_ingest_pdf_multiple_pages(tmp_path):
    pdf_path = tmp_path / "multi.pdf"
    doc = fitz.open()
    for i in range(3):
        page = doc.new_page()
        page.insert_text((72, 72), f"Pagina {i + 1} contenido", fontsize=12)
    doc.save(str(pdf_path))
    doc.close()
    result = ingest_pdf(pdf_path)
    assert "Pagina 1" in result
    assert "Pagina 3" in result


def test_ingest_pdf_returns_markdown_structure(tmp_path):
    pdf_path = tmp_path / "struct.pdf"
    _make_pdf("Titulo\nContenido bajo el titulo", pdf_path)
    result = ingest_pdf(pdf_path)
    assert isinstance(result, str)
    assert len(result) > 0


def test_ingest_pdf_file_not_found():
    with pytest.raises(FileNotFoundError):
        ingest_pdf(Path("/nonexistent/file.pdf"))


def test_ingest_pdf_invalid_file(tmp_path):
    bad_path = tmp_path / "bad.pdf"
    bad_path.write_text("not a pdf")
    with pytest.raises(ValueError):
        ingest_pdf(bad_path)

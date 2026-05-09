import pytest

from resumini.db.chroma import ChromaClient


@pytest.fixture
def chroma(tmp_chroma):
    client = ChromaClient(persist_dir=str(tmp_chroma))
    return client


def test_index_document(chroma):
    chroma.index_document(
        "civil_u1",
        "La prescripcion extintiva es un instituto del derecho civil",
        metadata={"materia": "civil", "file": "u1.md"},
    )
    results = chroma.search("prescripcion", n_results=1)
    assert len(results) > 0
    assert "prescripcion" in results[0]["content"].lower()


def test_search_returns_metadata(chroma):
    chroma.index_document(
        "civil_u1", "contenido civil", metadata={"materia": "civil", "file": "u1.md"}
    )
    results = chroma.search("civil", n_results=1)
    assert results[0]["metadata"]["materia"] == "civil"


def test_index_multiple_and_search_cross_materia(chroma):
    chroma.index_document(
        "civil_u1",
        "La prescripcion en derecho civil libera al deudor",
        metadata={"materia": "civil", "file": "u1.md"},
    )
    chroma.index_document(
        "penal_u1",
        "La prescripcion de la accion penal extingue la persecucion",
        metadata={"materia": "penal", "file": "u1.md"},
    )
    results = chroma.search("prescripcion accion", n_results=2)
    assert len(results) == 2


def test_delete_document(chroma):
    chroma.index_document(
        "to_delete", "contenido temporal", metadata={"materia": "test", "file": "t.md"}
    )
    chroma.delete_document("to_delete")
    results = chroma.search("contenido temporal", n_results=1)
    assert len(results) == 0


def test_reindex_document(chroma):
    chroma.index_document("re_idx", "contenido viejo", metadata={"materia": "test", "file": "t.md"})
    chroma.index_document(
        "re_idx", "contenido nuevo actualizado", metadata={"materia": "test", "file": "t.md"}
    )
    results = chroma.search("contenido nuevo actualizado", n_results=1)
    assert len(results) == 1
    assert "actualizado" in results[0]["content"]

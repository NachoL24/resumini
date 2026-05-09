# Resumini F1 — MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the MVP: a CLI-based agent that ingests PDFs, generates personalized summaries stored in a markdown vault, and supports semantic search via ChromaDB.

**Architecture:** FastAPI server exposes the agent via a simple CLI. LangGraph orchestrates the agent flow (router → ingest/summarize/query/edit → validate → memory update → response). The vault (markdown files) is the source of truth; ChromaDB is a searchable index; SQLite stores metadata.

**Tech Stack:** Python 3.12, LangGraph, FastAPI, ChromaDB (embedded), SQLite, PyMuPDF, OpenAI API (LLM + embeddings)

---

## File Structure

```
resumini/
├── pyproject.toml
├── .env.example
├── src/
│   └── resumini/
│       ├── __init__.py
│       ├── main.py                  ← FastAPI app + CLI entry
│       ├── config.py                ← Settings from env vars
│       ├── vault/
│       │   ├── __init__.py
│       │   ├── manager.py           ← Read/write/search .md files
│       │   └── templates.py         ← Default vault templates
│       ├── db/
│       │   ├── __init__.py
│       │   ├── chroma.py            ← ChromaDB client + indexing
│       │   └── sqlite.py            ← SQLite metadata store
│       ├── ingest/
│       │   ├── __init__.py
│       │   └── pdf.py               ← PDF parsing with PyMuPDF
│       ├── agent/
│       │   ├── __init__.py
│       │   ├── graph.py             ← LangGraph state + graph definition
│       │   ├── router.py            ← Intent classification node
│       │   ├── ingest_node.py       ← Ingest processing node
│       │   ├── summarize_node.py    ← Summary generation node
│       │   ├── query_node.py        ← RAG query node
│       │   ├── edit_node.py         ← Edit existing summary node
│       │   ├── validate_node.py     ← Output validation node
│       │   ├── memory_node.py       ← Vault + ChromaDB update node
│       │   └── response_node.py     ← Final response formatting node
│       └── llm/
│           ├── __init__.py
│           └── client.py            ← LLM + embeddings client wrapper
├── vault/                           ← Default vault root (gitignored content)
│   ├── profile.md
│   ├── materias/
│   ├── templates/
│   │   ├── materia.md
│   │   ├── resumen.md
│   │   └── apunte_clase.md
│   └── .meta/
│       └── sessions.json
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_vault_manager.py
    ├── test_chroma.py
    ├── test_sqlite.py
    ├── test_pdf_ingest.py
    ├── test_agent_graph.py
    ├── test_router_node.py
    ├── test_summarize_node.py
    ├── test_query_node.py
    └── test_edit_node.py
```

---

### Task 1: Project Scaffolding + Config

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `src/resumini/__init__.py`
- Create: `src/resumini/config.py`

- [ ] **Step 1: Write pyproject.toml**

```toml
[project]
name = "resumini"
version = "0.1.0"
description = "Agente de resumenes para la facultad"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn>=0.34.0",
    "langgraph>=0.4.0",
    "langchain-openai>=0.3.0",
    "langchain-core>=0.3.0",
    "chromadb>=1.0.0",
    "pymupdf>=1.25.0",
    "python-dotenv>=1.1.0",
    "pydantic>=2.10.0",
    "pydantic-settings>=2.7.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3.0",
    "pytest-asyncio>=0.25.0",
    "pytest-cov>=6.0.0",
    "ruff>=0.8.0",
]

[project.scripts]
resumini = "resumini.main:cli"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/resumini"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"

[tool.ruff]
target-version = "py312"
line-length = 100
```

- [ ] **Step 2: Write .env.example**

```
OPENAI_API_KEY=sk-...
VAULT_DIR=./vault
CHROMA_DIR=./chroma_data
DB_PATH=./resumini.db
LLM_MODEL=gpt-4o
EMBEDDING_MODEL=text-embedding-3-small
```

- [ ] **Step 3: Write src/resumini/__init__.py**

```python
__version__ = "0.1.0"
```

- [ ] **Step 4: Write src/resumini/config.py**

```python
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str = ""
    vault_dir: Path = Path("./vault")
    chroma_dir: Path = Path("./chroma_data")
    db_path: Path = Path("./resumini.db")
    llm_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"
    llm_temperature: float = 0.3

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 5: Write failing test for config**

Create `tests/conftest.py`:

```python
import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_vault(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "materias").mkdir()
    (vault / "templates").mkdir()
    (vault / ".meta").mkdir()
    (vault / ".meta" / "sessions.json").write_text("[]")
    (vault / "profile.md").write_text("# Perfil\n\nEstilo: bullet points")
    return vault


@pytest.fixture
def tmp_db(tmp_path):
    return tmp_path / "test.db"


@pytest.fixture
def tmp_chroma(tmp_path):
    return tmp_path / "chroma"
```

Create `tests/test_config.py`:

```python
from resumini.config import Settings


def test_settings_defaults():
    s = Settings(openai_api_key="test-key")
    assert s.llm_model == "gpt-4o"
    assert s.embedding_model == "text-embedding-3-small"
    assert s.vault_dir == Path("./vault")


def test_settings_custom():
    s = Settings(openai_api_key="test-key", llm_model="gpt-4o-mini")
    assert s.llm_model == "gpt-4o-mini"
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd /home/nacho/Documents/resumini && python -m pytest tests/test_config.py -v`

Expected: PASS (pydantic-settings handles defaults)

- [ ] **Step 7: Install project in dev mode and run tests**

Run: `cd /home/nacho/Documents/resumini && pip install -e ".[dev]" && python -m pytest tests/ -v`

Expected: all PASS

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml .env.example src/ tests/conftest.py tests/test_config.py
git commit -m "feat: project scaffolding with config"
```

---

### Task 2: Vault Manager

**Files:**
- Create: `src/resumini/vault/__init__.py`
- Create: `src/resumini/vault/manager.py`
- Create: `src/resumini/vault/templates.py`
- Test: `tests/test_vault_manager.py`

- [ ] **Step 1: Write failing tests for VaultManager**

```python
import pytest
from pathlib import Path

from resumini.vault.manager import VaultManager


@pytest.fixture
def vm(tmp_vault):
    return VaultManager(tmp_vault)


def test_read_profile(vm):
    content = vm.read_profile()
    assert "Estilo" in content


def test_write_materia_note(vm):
    vm.write_note("derecho_civil", "unidad_1.md", "# Unidad 1\n\nContenido")
    path = vm.vault_dir / "materias" / "derecho_civil" / "unidad_1.md"
    assert path.exists()
    assert "Unidad 1" in path.read_text()


def test_write_note_creates_materia_dir(vm):
    vm.write_note("penal", "unidad_1.md", "# Penal U1")
    assert (vm.vault_dir / "materias" / "penal").is_dir()


def test_read_note(vm):
    vm.write_note("civil", "test.md", "# Test content")
    content = vm.read_note("civil", "test.md")
    assert "Test content" in content


def test_read_note_not_found(vm):
    with pytest.raises(FileNotFoundError):
        vm.read_note("nonexistent", "missing.md")


def test_list_materias(vm):
    vm.write_note("civil", "u1.md", "c1")
    vm.write_note("penal", "u1.md", "p1")
    materias = vm.list_materias()
    assert "civil" in materias
    assert "penal" in materias


def test_list_notes_in_materia(vm):
    vm.write_note("civil", "u1.md", "c1")
    vm.write_note("civil", "u2.md", "c2")
    notes = vm.list_notes("civil")
    assert "u1.md" in notes
    assert "u2.md" in notes


def test_update_profile(vm):
    vm.update_profile("# Perfil\n\nEstilo: prosa")
    assert "prosa" in vm.read_profile()


def test_get_sessions_empty(vm):
    sessions = vm.get_sessions()
    assert sessions == []


def test_add_session(vm):
    vm.add_session({"id": "s1", "query": "resumir pdf", "timestamp": "2026-05-08"})
    sessions = vm.get_sessions()
    assert len(sessions) == 1
    assert sessions[0]["id"] == "s1"


def test_write_materia_index(vm):
    vm.write_materia_index("civil", {"profesor": "Garcia", "cuatrimestre": "1C2026"})
    index = vm.read_note("civil", "_index.md")
    assert "Garcia" in index
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/nacho/Documents/resumini && python -m pytest tests/test_vault_manager.py -v`

Expected: FAIL — `ModuleNotFoundError: No module named 'resumini.vault.manager'`

- [ ] **Step 3: Write vault/__init__.py**

```python
```

- [ ] **Step 4: Write vault/templates.py**

```python
MATERIA_INDEX = """# {nombre}

- **Profesor**: {profesor}
- **Cuatrimestre**: {cuatrimestre}
- **Fuentes**: {fuentes}
"""

RESUMEN = """# {titulo}

## Ideas principales

{ideas}

## Conceptos clave

{conceptos}

## Notas

{notas}
"""

APUNTE_CLASE = """# Apunte de clase — {fecha}

## Tema

{tema}

## Apuntes

{apuntes}

## Dudas

{dudas}
"""

PROFILE = """# Perfil de estudio

## Preferencias de estilo

- **Formato preferido**: {formato}
- **Nivel de detalle**: {detalle}

## Materias activas

{materias}

## Notas sobre mi estilo

{notas}
"""
```

- [ ] **Step 5: Write vault/manager.py**

```python
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
                PROFILE.format(formato="bullet points", detalle="moderado", materias="- (ninguna)", notas="- (ninguna)")
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
        (self.vault_dir / ".meta" / "sessions.json").write_text(json.dumps(sessions, indent=2, ensure_ascii=False))
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd /home/nacho/Documents/resumini && python -m pytest tests/test_vault_manager.py -v`

Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add src/resumini/vault/ tests/test_vault_manager.py
git commit -m "feat: vault manager with read/write/list operations"
```

---

### Task 3: ChromaDB Client

**Files:**
- Create: `src/resumini/db/__init__.py`
- Create: `src/resumini/db/chroma.py`
- Test: `tests/test_chroma.py`

- [ ] **Step 1: Write failing tests for ChromaClient**

```python
import pytest

from resumini.db.chroma import ChromaClient


@pytest.fixture
def chroma(tmp_chroma):
    client = ChromaClient(persist_dir=str(tmp_chroma), embedding_model_name="text-embedding-3-small")
    return client


def test_index_document(chroma):
    chroma.index_document("civil_u1", "La prescripcion extintiva es un instituto del derecho civil", metadata={"materia": "civil", "file": "u1.md"})
    results = chroma.search("prescripcion", n_results=1)
    assert len(results) > 0
    assert "prescripcion" in results[0]["content"].lower()


def test_search_returns_metadata(chroma):
    chroma.index_document("civil_u1", "contenido civil", metadata={"materia": "civil", "file": "u1.md"})
    results = chroma.search("civil", n_results=1)
    assert results[0]["metadata"]["materia"] == "civil"


def test_index_multiple_and_search_cross_materia(chroma):
    chroma.index_document("civil_u1", "La prescripcion en derecho civil libera al deudor", metadata={"materia": "civil", "file": "u1.md"})
    chroma.index_document("penal_u1", "La prescripcion de la accion penal extingue la persecucion", metadata={"materia": "penal", "file": "u1.md"})
    results = chroma.search("prescripcion accion", n_results=2)
    assert len(results) == 2


def test_delete_document(chroma):
    chroma.index_document("to_delete", "contenido temporal", metadata={"materia": "test", "file": "t.md"})
    chroma.delete_document("to_delete")
    results = chroma.search("contenido temporal", n_results=1)
    assert len(results) == 0


def test_reindex_document(chroma):
    chroma.index_document("re_idx", "contenido viejo", metadata={"materia": "test", "file": "t.md"})
    chroma.index_document("re_idx", "contenido nuevo actualizado", metadata={"materia": "test", "file": "t.md"})
    results = chroma.search("contenido nuevo actualizado", n_results=1)
    assert len(results) == 1
    assert "actualizado" in results[0]["content"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/nacho/Documents/resumini && python -m pytest tests/test_chroma.py -v`

Expected: FAIL — `ModuleNotFoundError: No module named 'resumini.db.chroma'`

- [ ] **Step 3: Write db/__init__.py**

```python
```

- [ ] **Step 4: Write db/chroma.py**

```python
import chromadb
from chromadb.utils import embedding_functions


class ChromaClient:
    def __init__(self, persist_dir: str, embedding_model_name: str = "text-embedding-3-small", openai_api_key: str | None = None):
        self._client = chromadb.PersistentClient(path=persist_dir)
        ef_kwargs = {"model_name": embedding_model_name}
        if openai_api_key:
            ef_kwargs["api_key"] = openai_api_key
        self._ef = embedding_functions.OpenAIEmbeddingFunction(**ef_kwargs)
        self._collection = self._client.get_or_create_collection(
            name="resumini",
            embedding_function=self._ef,
        )

    def index_document(self, doc_id: str, content: str, metadata: dict):
        existing = self._collection.get(ids=[doc_id])
        if existing["ids"]:
            self._collection.update(ids=[doc_id], documents=[content], metadatas=[metadata])
        else:
            self._collection.add(ids=[doc_id], documents=[content], metadatas=[metadata])

    def search(self, query: str, n_results: int = 5, metadata_filter: dict | None = None) -> list[dict]:
        kwargs = {"query_texts": [query], "n_results": n_results}
        if metadata_filter:
            kwargs["where"] = metadata_filter
        results = self._collection.query(**kwargs)
        docs = []
        for i, doc_id in enumerate(results["ids"][0]):
            docs.append({
                "id": doc_id,
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i] if results.get("distances") else None,
            })
        return docs

    def delete_document(self, doc_id: str):
        self._collection.delete(ids=[doc_id])
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /home/nacho/Documents/resumini && python -m pytest tests/test_chroma.py -v`

Note: ChromaDB tests require OPENAI_API_KEY set in env. If running without key, mock the embedding function.

Expected: all PASS (with valid API key or mocked embeddings)

- [ ] **Step 6: Commit**

```bash
git add src/resumini/db/ tests/test_chroma.py
git commit -m "feat: ChromaDB client with index/search/delete"
```

---

### Task 4: SQLite Metadata Store

**Files:**
- Create: `src/resumini/db/sqlite.py`
- Test: `tests/test_sqlite.py`

- [ ] **Step 1: Write failing tests for SQLiteStore**

```python
import pytest

from resumini.db.sqlite import SQLiteStore


@pytest.fixture
def store(tmp_db):
    s = SQLiteStore(str(tmp_db))
    s.initialize()
    return s


def test_add_and_get_materia(store):
    store.add_materia("civil", {"profesor": "Garcia", "cuatrimestre": "1C2026"})
    m = store.get_materia("civil")
    assert m["nombre"] == "civil"
    assert m["profesor"] == "Garcia"


def test_list_materias(store):
    store.add_materia("civil", {"profesor": "G"})
    store.add_materia("penal", {"profesor": "L"})
    materias = store.list_materias()
    assert len(materias) == 2


def test_update_preference(store):
    store.set_preference("formato", "bullet points")
    assert store.get_preference("formato") == "bullet points"


def test_get_preference_missing(store):
    assert store.get_preference("missing") is None


def test_add_session(store):
    store.add_session("s1", "resumir pdf civil", "2026-05-08T10:00:00")
    sessions = store.get_recent_sessions(limit=10)
    assert len(sessions) == 1
    assert sessions[0]["session_id"] == "s1"


def test_get_recent_sessions_limit(store):
    for i in range(5):
        store.add_session(f"s{i}", f"query {i}", "2026-05-08T10:00:00")
    sessions = store.get_recent_sessions(limit=3)
    assert len(sessions) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/nacho/Documents/resumini && python -m pytest tests/test_sqlite.py -v`

Expected: FAIL — `ModuleNotFoundError: No module named 'resumini.db.sqlite'`

- [ ] **Step 3: Write db/sqlite.py**

```python
import json
import sqlite3
from pathlib import Path


class SQLiteStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row

    def initialize(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS materias (
                nombre TEXT PRIMARY KEY,
                metadata_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS preferences (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                query TEXT,
                timestamp TEXT,
                metadata_json TEXT
            );
        """)
        self._conn.commit()

    def add_materia(self, nombre: str, metadata: dict):
        self._conn.execute(
            "INSERT OR REPLACE INTO materias (nombre, metadata_json) VALUES (?, ?)",
            (nombre, json.dumps(metadata, ensure_ascii=False)),
        )
        self._conn.commit()

    def get_materia(self, nombre: str) -> dict | None:
        row = self._conn.execute("SELECT * FROM materias WHERE nombre = ?", (nombre,)).fetchone()
        if not row:
            return None
        meta = json.loads(row["metadata_json"])
        meta["nombre"] = row["nombre"]
        return meta

    def list_materias(self) -> list[dict]:
        rows = self._conn.execute("SELECT * FROM materias").fetchall()
        result = []
        for row in rows:
            meta = json.loads(row["metadata_json"])
            meta["nombre"] = row["nombre"]
            result.append(meta)
        return result

    def set_preference(self, key: str, value: str):
        self._conn.execute(
            "INSERT OR REPLACE INTO preferences (key, value) VALUES (?, ?)",
            (key, value),
        )
        self._conn.commit()

    def get_preference(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM preferences WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def add_session(self, session_id: str, query: str, timestamp: str, metadata: dict | None = None):
        meta_json = json.dumps(metadata or {}, ensure_ascii=False)
        self._conn.execute(
            "INSERT OR REPLACE INTO sessions (session_id, query, timestamp, metadata_json) VALUES (?, ?, ?, ?)",
            (session_id, query, timestamp, meta_json),
        )
        self._conn.commit()

    def get_recent_sessions(self, limit: int = 10) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM sessions ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(row) for row in rows]

    def close(self):
        self._conn.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/nacho/Documents/resumini && python -m pytest tests/test_sqlite.py -v`

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/resumini/db/sqlite.py tests/test_sqlite.py
git commit -m "feat: SQLite metadata store for materias, preferences, sessions"
```

---

### Task 5: PDF Ingest

**Files:**
- Create: `src/resumini/ingest/__init__.py`
- Create: `src/resumini/ingest/pdf.py`
- Test: `tests/test_pdf_ingest.py`

- [ ] **Step 1: Write failing tests for PDF ingester**

```python
import tempfile
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/nacho/Documents/resumini && python -m pytest tests/test_pdf_ingest.py -v`

Expected: FAIL — `ModuleNotFoundError: No module named 'resumini.ingest.pdf'`

- [ ] **Step 3: Write ingest/__init__.py**

```python
```

- [ ] **Step 4: Write ingest/pdf.py**

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /home/nacho/Documents/resumini && python -m pytest tests/test_pdf_ingest.py -v`

Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add src/resumini/ingest/ tests/test_pdf_ingest.py
git commit -m "feat: PDF ingestion with PyMuPDF"
```

---

### Task 6: LLM Client Wrapper

**Files:**
- Create: `src/resumini/llm/__init__.py`
- Create: `src/resumini/llm/client.py`

- [ ] **Step 1: Write llm/__init__.py**

```python
```

- [ ] **Step 2: Write llm/client.py**

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from resumini.config import Settings


def get_llm(settings: Settings | None = None) -> ChatOpenAI:
    if settings is None:
        from resumini.config import get_settings
        settings = get_settings()
    return ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        api_key=settings.openai_api_key,
    )


def get_embeddings(settings: Settings | None = None) -> OpenAIEmbeddings:
    if settings is None:
        from resumini.config import get_settings
        settings = get_settings()
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )
```

- [ ] **Step 3: Commit**

```bash
git add src/resumini/llm/
git commit -m "feat: LLM and embeddings client wrappers"
```

---

### Task 7: Agent Graph — Router Node

**Files:**
- Create: `src/resumini/agent/__init__.py`
- Create: `src/resumini/agent/router.py`
- Test: `tests/test_router_node.py`

- [ ] **Step 1: Write failing tests for router**

```python
import pytest
from unittest.mock import MagicMock

from resumini.agent.router import classify_intent, Intent


def test_classify_ingest():
    result = classify_intent("procesa este pdf de derecho civil")
    assert result == Intent.INGEST


def test_classify_summarize():
    result = classify_intent("haceme un resumen de la unidad 3")
    assert result == Intent.SUMMARIZE


def test_classify_query():
    result = classify_intent("que dice el libro sobre la prescripcion?")
    assert result == Intent.QUERY


def test_classify_edit():
    result = classify_intent("cambia el resumen de penal, agrega esto")
    assert result == Intent.EDIT


def test_classify_defaults_to_query():
    result = classify_intent("hola como andas")
    assert result == Intent.QUERY
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/nacho/Documents/resumini && python -m pytest tests/test_router_node.py -v`

Expected: FAIL — `ModuleNotFoundError: No module named 'resumini.agent.router'`

- [ ] **Step 3: Write agent/__init__.py**

```python
```

- [ ] **Step 4: Write agent/router.py**

```python
from enum import Enum

from langchain_core.messages import HumanMessage, SystemMessage

from resumini.llm.client import get_llm


class Intent(str, Enum):
    INGEST = "ingest"
    SUMMARIZE = "summarize"
    QUERY = "query"
    EDIT = "edit"


ROUTER_SYSTEM_PROMPT = """Eres un clasificador de intenciones para un agente de resumenes academicos.
Clasifica el mensaje del usuario en una de estas categorias:

- ingest: el usuario quiere procesar/ingestar material nuevo (PDF, video, apunte)
- summarize: el usuario quiere generar un resumen de material ya procesado
- query: el usuario pregunta sobre contenido estudiado
- edit: el usuario quiere modificar un resumen existente

Responde SOLO con la categoria, sin explicacion."""


def classify_intent(message: str) -> Intent:
    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=message),
    ])
    intent_str = response.content.strip().lower()
    intent_map = {
        "ingest": Intent.INGEST,
        "summarize": Intent.SUMMARIZE,
        "query": Intent.QUERY,
        "edit": Intent.EDIT,
    }
    return intent_map.get(intent_str, Intent.QUERY)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /home/nacho/Documents/resumini && python -m pytest tests/test_router_node.py -v`

Expected: all PASS (requires OPENAI_API_KEY)

- [ ] **Step 6: Commit**

```bash
git add src/resumini/agent/ tests/test_router_node.py
git commit -m "feat: intent classification router node"
```

---

### Task 8: Agent Graph — Ingest + Summarize + Query + Edit Nodes

**Files:**
- Create: `src/resumini/agent/ingest_node.py`
- Create: `src/resumini/agent/summarize_node.py`
- Create: `src/resumini/agent/query_node.py`
- Create: `src/resumini/agent/edit_node.py`
- Test: `tests/test_summarize_node.py`
- Test: `tests/test_query_node.py`
- Test: `tests/test_edit_node.py`

- [ ] **Step 1: Write ingest_node.py**

```python
from pathlib import Path

from resumini.ingest.pdf import ingest_pdf
from resumini.vault.manager import VaultManager
from resumini.db.chroma import ChromaClient


def run_ingest(pdf_path: Path, materia: str, filename: str, vault: VaultManager, chroma: ChromaClient) -> str:
    raw_content = ingest_pdf(pdf_path)
    vault.write_note(materia, filename, raw_content)
    chroma.index_document(
        doc_id=f"{materia}_{filename.replace('.md', '')}",
        content=raw_content,
        metadata={"materia": materia, "file": filename, "type": "raw_ingest"},
    )
    return raw_content
```

- [ ] **Step 2: Write failing tests for summarize_node**

```python
import pytest
from unittest.mock import MagicMock, patch

from resumini.agent.summarize_node import run_summarize


@pytest.fixture
def mocks(tmp_vault):
    from resumini.vault.manager import VaultManager
    vault = VaultManager(tmp_vault)
    vault.write_note("civil", "u1_raw.md", "# Unidad 1\n\nLa prescripcion extintiva es un instituto que libera al deudor. Requiere plazo y reclamacion.")
    chroma = MagicMock()
    chroma.search.return_value = [{"id": "civil_u1_raw", "content": "prescripcion extintiva", "metadata": {"materia": "civil", "file": "u1_raw.md"}}]
    return vault, chroma


def test_summarize_produces_output(mocks):
    vault, chroma = mocks
    with patch("resumini.agent.summarize_node.get_llm") as mock_llm:
        mock_llm.return_value.invoke.return_value = MagicMock(content="# Resumen U1\n\n- Prescripcion extintiva libera al deudor\n- Requiere plazo y reclamacion")
        result = run_summarize("civil", "u1_raw.md", "u1_resumen.md", vault, chroma)
    assert "Prescripcion" in result
    note = vault.read_note("civil", "u1_resumen.md")
    assert "Prescripcion" in note
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd /home/nacho/Documents/resumini && python -m pytest tests/test_summarize_node.py -v`

Expected: FAIL — `ModuleNotFoundError: No module named 'resumini.agent.summarize_node'`

- [ ] **Step 4: Write summarize_node.py**

```python
from langchain_core.messages import HumanMessage, SystemMessage

from resumini.llm.client import get_llm
from resumini.vault.manager import VaultManager
from resumini.db.chroma import ChromaClient

SUMMARIZE_PROMPT = """Eres un asistente academico que genera resumenes de material de estudio.
Usa el perfil del estudiante para adaptar el formato y nivel de detalle.
Genera un resumen claro, estructurado, en markdown.

Perfil del estudiante:
{profile}

Contenido fuente:
{content}

Genera el resumen:"""


def run_summarize(materia: str, source_file: str, output_file: str, vault: VaultManager, chroma: ChromaClient) -> str:
    source_content = vault.read_note(materia, source_file)
    profile = vault.read_profile()
    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content=SUMMARIZE_PROMPT.format(profile=profile, content=source_content)),
        HumanMessage(content=f"Genera un resumen de {source_file} para la materia {materia}"),
    ])
    summary = response.content
    vault.write_note(materia, output_file, summary)
    chroma.index_document(
        doc_id=f"{materia}_{output_file.replace('.md', '')}",
        content=summary,
        metadata={"materia": materia, "file": output_file, "type": "summary"},
    )
    return summary
```

- [ ] **Step 5: Write failing tests for query_node**

```python
import pytest
from unittest.mock import MagicMock, patch

from resumini.agent.query_node import run_query


def test_query_returns_answer():
    vault = MagicMock()
    chroma = MagicMock()
    chroma.search.return_value = [
        {"id": "civil_u1", "content": "La prescripcion extintiva libera al deudor tras el plazo", "metadata": {"materia": "civil", "file": "u1.md"}},
        {"id": "penal_u1", "content": "La prescripcion de la accion penal extingue la persecucion", "metadata": {"materia": "penal", "file": "u1.md"}},
    ]
    with patch("resumini.agent.query_node.get_llm") as mock_llm:
        mock_llm.return_value.invoke.return_value = MagicMock(content="La prescripcion funciona distinto en civil y penal...")
        result = run_query("que es la prescripcion?", vault, chroma)
    assert "prescripcion" in result.lower()
```

- [ ] **Step 6: Write query_node.py**

```python
from langchain_core.messages import HumanMessage, SystemMessage

from resumini.llm.client import get_llm
from resumini.vault.manager import VaultManager
from resumini.db.chroma import ChromaClient

QUERY_PROMPT = """Eres un asistente academico que responde preguntas basandote en resumenes y apuntes.
Usa el contexto proporcionado para dar una respuesta precisa.
Si la informacion no esta en el contexto, dilo claramente.

Perfil del estudiante:
{profile}

Contexto relevante:
{context}

Pregunta: {question}"""


def run_query(question: str, vault: VaultManager, chroma: ChromaDB) -> str:
    search_results = chroma.search(question, n_results=5)
    context_parts = []
    for r in search_results:
        context_parts.append(f"[{r['metadata']['materia']}/{r['metadata']['file']}] {r['content']}")
    context = "\n\n".join(context_parts)
    profile = vault.read_profile()
    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content=QUERY_PROMPT.format(profile=profile, context=context, question=question)),
        HumanMessage(content=question),
    ])
    return response.content
```

- [ ] **Step 7: Write failing tests for edit_node**

```python
import pytest
from unittest.mock import MagicMock, patch

from resumini.agent.edit_node import run_edit


def test_edit_updates_note(tmp_vault):
    from resumini.vault.manager import VaultManager
    vault = VaultManager(tmp_vault)
    vault.write_note("civil", "resumen.md", "# Resumen\n\nContenido original")
    chroma = MagicMock()
    with patch("resumini.agent.edit_node.get_llm") as mock_llm:
        mock_llm.return_value.invoke.return_value = MagicMock(content="# Resumen\n\nContenido original\n\n## Agregado\n\nNuevo contenido")
        result = run_edit("civil", "resumen.md", "agrega una seccion sobre prescripcion", vault, chroma)
    assert "Agregado" in result
    assert "Agregado" in vault.read_note("civil", "resumen.md")
```

- [ ] **Step 8: Write edit_node.py**

```python
from langchain_core.messages import HumanMessage, SystemMessage

from resumini.llm.client import get_llm
from resumini.vault.manager import VaultManager
from resumini.db.chroma import ChromaClient

EDIT_PROMPT = """Eres un asistente academico que edita resumenes existentes.
Aplica la edicion solicitada al contenido actual, manteniendo el formato y estilo.

Contenido actual:
{current_content}

Edicion solicitada:
{edit_instruction}

Devuelve el contenido completo editado en markdown:"""


def run_edit(materia: str, filename: str, instruction: str, vault: VaultManager, chroma: ChromaClient) -> str:
    current = vault.read_note(materia, filename)
    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content=EDIT_PROMPT.format(current_content=current, edit_instruction=instruction)),
        HumanMessage(content=instruction),
    ])
    edited = response.content
    vault.write_note(materia, filename, edited)
    chroma.index_document(
        doc_id=f"{materia}_{filename.replace('.md', '')}",
        content=edited,
        metadata={"materia": materia, "file": filename, "type": "summary"},
    )
    return edited
```

- [ ] **Step 9: Run all new tests**

Run: `cd /home/nacho/Documents/resumini && python -m pytest tests/test_summarize_node.py tests/test_query_node.py tests/test_edit_node.py -v`

Expected: all PASS (with mocked LLM)

- [ ] **Step 10: Commit**

```bash
git add src/resumini/agent/ingest_node.py src/resumini/agent/summarize_node.py src/resumini/agent/query_node.py src/resumini/agent/edit_node.py tests/test_summarize_node.py tests/test_query_node.py tests/test_edit_node.py
git commit -m "feat: ingest, summarize, query, and edit agent nodes"
```

---

### Task 9: Agent Graph — Validate + Memory Update + Response Nodes

**Files:**
- Create: `src/resumini/agent/validate_node.py`
- Create: `src/resumini/agent/memory_node.py`
- Create: `src/resumini/agent/response_node.py`

- [ ] **Step 1: Write validate_node.py**

```python
from resumini.vault.manager import VaultManager
from resumini.db.chroma import ChromaClient


def validate_output(content: str) -> dict:
    errors = []
    if not content or not content.strip():
        errors.append("empty_output")
    if len(content.strip()) < 20:
        errors.append("too_short")
    return {"valid": len(errors) == 0, "errors": errors}
```

- [ ] **Step 2: Write memory_node.py**

```python
import uuid
from datetime import datetime

from resumini.vault.manager import VaultManager
from resumini.db.chroma import ChromaClient
from resumini.db.sqlite import SQLiteStore


def update_memory(materia: str, filename: str, content: str, vault: VaultManager, chroma: ChromaClient, sqlite: SQLiteStore, session_id: str | None = None):
    vault.write_note(materia, filename, content)
    chroma.index_document(
        doc_id=f"{materia}_{filename.replace('.md', '')}",
        content=content,
        metadata={"materia": materia, "file": filename, "type": "auto"},
    )
    sid = session_id or str(uuid.uuid4())[:8]
    sqlite.add_session(sid, f"update {materia}/{filename}", datetime.now().isoformat())
    vault.add_session({"id": sid, "action": "memory_update", "materia": materia, "file": filename, "timestamp": datetime.now().isoformat()})
```

- [ ] **Step 3: Write response_node.py**

```python
def format_response(content: str, metadata: dict | None = None) -> str:
    parts = [content]
    if metadata:
        if metadata.get("materia"):
            parts.append(f"\n---\n*Materia: {metadata['materia']}*")
        if metadata.get("file"):
            parts.append(f" *Archivo: {metadata['file']}*")
    return "\n".join(parts)
```

- [ ] **Step 4: Commit**

```bash
git add src/resumini/agent/validate_node.py src/resumini/agent/memory_node.py src/resumini/agent/response_node.py
git commit -m "feat: validate, memory update, and response nodes"
```

---

### Task 10: Agent Graph — LangGraph Wiring

**Files:**
- Create: `src/resumini/agent/graph.py`
- Test: `tests/test_agent_graph.py`

- [ ] **Step 1: Write failing tests for the full graph**

```python
import pytest
from unittest.mock import MagicMock, patch

from resumini.agent.graph import AgentState, build_graph


def test_agent_state_has_required_fields():
    state = AgentState(message="haceme un resumen", materia="civil", intent=None, output=None)
    assert state.message == "haceme un resumen"


def test_build_graph_returns_compiled():
    with patch("resumini.agent.graph.get_llm") as mock_llm:
        mock_llm.return_value.invoke.return_value = MagicMock(content="summarize")
        graph = build_graph(MagicMock(), MagicMock(), MagicMock())
    assert graph is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/nacho/Documents/resumini && python -m pytest tests/test_agent_graph.py -v`

Expected: FAIL — `ModuleNotFoundError: No module named 'resumini.agent.graph'`

- [ ] **Step 3: Write agent/graph.py**

```python
from typing import TypedDict

from langgraph.graph import END, StateGraph

from resumini.agent.router import Intent, classify_intent
from resumini.agent.ingest_node import run_ingest
from resumini.agent.summarize_node import run_summarize
from resumini.agent.query_node import run_query
from resumini.agent.edit_node import run_edit
from resumini.agent.validate_node import validate_output
from resumini.agent.memory_node import update_memory
from resumini.agent.response_node import format_response
from resumini.vault.manager import VaultManager
from resumini.db.chroma import ChromaClient
from resumini.db.sqlite import SQLiteStore


class AgentState(TypedDict):
    message: str
    intent: Intent | None
    materia: str
    source_file: str | None
    output_file: str | None
    pdf_path: str | None
    edit_instruction: str | None
    output: str | None
    valid: bool
    errors: list[str]
    metadata: dict


def _route_intent(state: AgentState) -> str:
    if state["intent"] == Intent.INGEST:
        return "ingest"
    elif state["intent"] == Intent.SUMMARIZE:
        return "summarize"
    elif state["intent"] == Intent.EDIT:
        return "edit"
    return "query"


def build_graph(vault: VaultManager, chroma: ChromaClient, sqlite: SQLiteStore):
    graph = StateGraph(AgentState)

    def router_node(state: AgentState) -> dict:
        intent = classify_intent(state["message"])
        return {"intent": intent}

    def ingest_node(state: AgentState) -> dict:
        from pathlib import Path
        content = run_ingest(Path(state["pdf_path"]), state["materia"], state["source_file"], vault, chroma)
        return {"output": content, "metadata": {"materia": state["materia"], "file": state["source_file"]}}

    def summarize_node(state: AgentState) -> dict:
        source = state.get("source_file", "")
        output = state.get("output_file", source.replace(".md", "_resumen.md"))
        content = run_summarize(state["materia"], source, output, vault, chroma)
        return {"output": content, "metadata": {"materia": state["materia"], "file": output}}

    def query_node_fn(state: AgentState) -> dict:
        content = run_query(state["message"], vault, chroma)
        return {"output": content}

    def edit_node_fn(state: AgentState) -> dict:
        content = run_edit(state["materia"], state["source_file"], state["edit_instruction"], vault, chroma)
        return {"output": content, "metadata": {"materia": state["materia"], "file": state["source_file"]}}

    def validate_node(state: AgentState) -> dict:
        result = validate_output(state["output"] or "")
        return {"valid": result["valid"], "errors": result["errors"]}

    def memory_node(state: AgentState) -> dict:
        if state.get("materia") and state.get("source_file") and state.get("output"):
            update_memory(state["materia"], state["source_file"], state["output"], vault, chroma, sqlite)
        return {}

    def response_node(state: AgentState) -> dict:
        return {"output": format_response(state["output"] or "", state.get("metadata"))}

    graph.add_node("router", router_node)
    graph.add_node("ingest", ingest_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("query", query_node_fn)
    graph.add_node("edit", edit_node_fn)
    graph.add_node("validate", validate_node)
    graph.add_node("memory", memory_node)
    graph.add_node("response", response_node)

    graph.set_entry_point("router")
    graph.add_conditional_edges("router", _route_intent, {"ingest": "ingest", "summarize": "summarize", "query": "query", "edit": "edit"})
    graph.add_edge("ingest", "validate")
    graph.add_edge("summarize", "validate")
    graph.add_edge("query", "response")
    graph.add_edge("edit", "validate")
    graph.add_conditional_edges("validate", lambda s: "memory" if s["valid"] else "response", {"memory": "memory", "response": "response"})
    graph.add_edge("memory", "response")
    graph.add_edge("response", END)

    return graph.compile()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/nacho/Documents/resumini && python -m pytest tests/test_agent_graph.py -v`

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/resumini/agent/graph.py tests/test_agent_graph.py
git commit -m "feat: LangGraph agent graph with full pipeline"
```

---

### Task 11: FastAPI Server + CLI Entry Point

**Files:**
- Create: `src/resumini/main.py`

- [ ] **Step 1: Write main.py**

```python
import argparse
import sys
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from resumini.config import get_settings
from resumini.vault.manager import VaultManager
from resumini.db.chroma import ChromaClient
from resumini.db.sqlite import SQLiteStore
from resumini.agent.graph import AgentState, build_graph


def create_app() -> FastAPI:
    settings = get_settings()
    vault = VaultManager(settings.vault_dir)
    chroma = ChromaClient(persist_dir=str(settings.chroma_dir), embedding_model_name=settings.embedding_model, openai_api_key=settings.openai_api_key)
    sqlite = SQLiteStore(str(settings.db_path))
    sqlite.initialize()
    graph = build_graph(vault, chroma, sqlite)

    app = FastAPI(title="Resumini", version="0.1.0")
    app.state.vault = vault
    app.state.chroma = chroma
    app.state.sqlite = sqlite
    app.state.graph = graph

    class MessageRequest(BaseModel):
        message: str
        materia: str = ""
        source_file: str | None = None
        output_file: str | None = None
        pdf_path: str | None = None
        edit_instruction: str | None = None

    class MessageResponse(BaseModel):
        output: str
        intent: str | None
        valid: bool
        errors: list[str]

    @app.post("/chat", response_model=MessageResponse)
    async def chat(req: MessageRequest):
        state = AgentState(
            message=req.message,
            intent=None,
            materia=req.materia,
            source_file=req.source_file,
            output_file=req.output_file,
            pdf_path=req.pdf_path,
            edit_instruction=req.edit_instruction,
            output=None,
            valid=False,
            errors=[],
            metadata={},
        )
        result = graph.invoke(state)
        return MessageResponse(
            output=result.get("output", ""),
            intent=result.get("intent"),
            valid=result.get("valid", False),
            errors=result.get("errors", []),
        )

    return app


def cli():
    parser = argparse.ArgumentParser(description="Resumini - Agente de resumenes")
    parser.add_argument("--serve", action="store_true", help="Start the API server")
    parser.add_argument("--chat", type=str, help="Send a message to the agent")
    parser.add_argument("--materia", type=str, default="", help="Materia context")
    parser.add_argument("--ingest-pdf", type=str, help="Path to PDF to ingest")
    parser.add_argument("--source", type=str, help="Source file name")
    parser.add_argument("--output", type=str, help="Output file name")
    args = parser.parse_args()

    if args.serve:
        import uvicorn
        uvicorn.run(create_app(), host="0.0.0.0", port=8000)
    elif args.chat:
        settings = get_settings()
        vault = VaultManager(settings.vault_dir)
        chroma = ChromaClient(persist_dir=str(settings.chroma_dir), embedding_model_name=settings.embedding_model, openai_api_key=settings.openai_api_key)
        sqlite = SQLiteStore(str(settings.db_path))
        sqlite.initialize()
        graph = build_graph(vault, chroma, sqlite)
        state = AgentState(
            message=args.chat,
            intent=None,
            materia=args.materia,
            source_file=args.source,
            output_file=args.output,
            pdf_path=args.ingest_pdf,
            edit_instruction=None,
            output=None,
            valid=False,
            errors=[],
            metadata={},
        )
        result = graph.invoke(state)
        print(result.get("output", "Sin respuesta"))
    else:
        parser.print_help()


if __name__ == "__main__":
    cli()
```

- [ ] **Step 2: Test CLI help**

Run: `cd /home/nacho/Documents/resumini && python -m resumini.main --help`

Expected: shows help text with --serve, --chat, --materia, --ingest-pdf options

- [ ] **Step 3: Commit**

```bash
git add src/resumini/main.py
git commit -m "feat: FastAPI server and CLI entry point"
```

---

### Task 12: Default Vault Templates + Integration Test

**Files:**
- Create: `vault/profile.md`
- Create: `vault/templates/materia.md`
- Create: `vault/templates/resumen.md`
- Create: `vault/templates/apunte_clase.md`
- Create: `vault/.meta/sessions.json`
- Create: `tests/test_integration.py`
- Create: `.gitignore`

- [ ] **Step 1: Write .gitignore**

```
.env
__pycache__/
*.pyc
chroma_data/
resumini.db
*.egg-info/
dist/
build/
.pytest_cache/
vault/materias/
vault/.meta/sessions.json
.ruff_cache/
```

- [ ] **Step 2: Write vault/profile.md**

```markdown
# Perfil de estudio

## Preferencias de estilo

- **Formato preferido**: bullet points con highlights en negrita
- **Nivel de detalle**: moderado — ideas clave con contexto breve
- **Idioma**: espanol

## Materias activas

- (ninguna todavia)

## Notas sobre mi estilo

- Prefiero resumenes que conecten conceptos entre unidades
- Uso negritas para terminos clave
- Me gustan las tablas comparativas cuando hay opciones/teorias contrapuestas
```

- [ ] **Step 3: Write vault/templates/materia.md**

```markdown
# {nombre}

- **Profesor**: {profesor}
- **Cuatrimestre**: {cuatrimestre}
- **Fuentes**: {fuentes}
- **Estado**: en curso
```

- [ ] **Step 4: Write vault/templates/resumen.md**

```markdown
# {titulo}

## Ideas principales

{ideas}

## Conceptos clave

{conceptos}

## Conexiones con otras unidades

{conexiones}

## Notas

{notas}
```

- [ ] **Step 5: Write vault/templates/apunte_clase.md**

```markdown
# Apunte de clase — {fecha}

## Tema

{tema}

## Apuntes

{apuntes}

## Dudas

{dudas}
```

- [ ] **Step 6: Write vault/.meta/sessions.json**

```json
[]
```

- [ ] **Step 7: Write integration test**

```python
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from resumini.config import Settings
from resumini.vault.manager import VaultManager
from resumini.db.chroma import ChromaClient
from resumini.db.sqlite import SQLiteStore
from resumini.agent.graph import AgentState, build_graph


def test_full_pipeline_ingest_and_summarize(tmp_vault, tmp_db, tmp_chroma):
    settings = Settings(openai_api_key="test-key", vault_dir=tmp_vault, db_path=tmp_db, chroma_dir=tmp_chroma)
    vault = VaultManager(tmp_vault)
    sqlite = SQLiteStore(str(tmp_db))
    sqlite.initialize()

    with patch("resumini.agent.router.get_llm") as mock_router_llm, \
         patch("resumini.agent.summarize_node.get_llm") as mock_summ_llm:
        mock_router_llm.return_value.invoke.return_value = MagicMock(content="summarize")
        mock_summ_llm.return_value.invoke.return_value = MagicMock(content="# Resumen U1\n\n- **Prescripcion extintiva**: libera al deudor\n- Requiere plazo")

        chroma = MagicMock()
        graph = build_graph(vault, chroma, sqlite)

        state = AgentState(
            message="haceme un resumen",
            intent=None,
            materia="civil",
            source_file="u1_raw.md",
            output_file="u1_resumen.md",
            pdf_path=None,
            edit_instruction=None,
            output=None,
            valid=False,
            errors=[],
            metadata={},
        )

    vault.write_note("civil", "u1_raw.md", "# Unidad 1\n\nLa prescripcion extintiva es un instituto que libera al deudor.")

    with patch("resumini.agent.router.get_llm") as mock_router_llm, \
         patch("resumini.agent.summarize_node.get_llm") as mock_summ_llm:
        mock_router_llm.return_value.invoke.return_value = MagicMock(content="summarize")
        mock_summ_llm.return_value.invoke.return_value = MagicMock(content="# Resumen U1\n\n- **Prescripcion extintiva**: libera al deudor\n- Requiere plazo")

        chroma = MagicMock()
        graph = build_graph(vault, chroma, sqlite)
        result = graph.invoke(state)

    assert result["valid"] is True
    assert result["output"] is not None
```

- [ ] **Step 8: Run integration test**

Run: `cd /home/nacho/Documents/resumini && python -m pytest tests/test_integration.py -v`

Expected: PASS

- [ ] **Step 9: Run all tests**

Run: `cd /home/nacho/Documents/resumini && python -m pytest tests/ -v`

Expected: all PASS

- [ ] **Step 10: Commit**

```bash
git add .gitignore vault/ tests/test_integration.py
git commit -m "feat: default vault templates and integration test"
```

---

### Task 13: Lint + Final Verification

- [ ] **Step 1: Run ruff on all source files**

Run: `cd /home/nacho/Documents/resumini && ruff check src/ tests/ --fix`

Expected: no errors (fix any that appear)

- [ ] **Step 2: Run full test suite**

Run: `cd /home/nacho/Documents/resumini && python -m pytest tests/ -v --tb=short`

Expected: all PASS

- [ ] **Step 3: Final commit if lint fixes applied**

```bash
git add -A
git commit -m "chore: lint fixes and final verification"
```

---

## Spec Coverage Check

| Spec Requirement | Covered by Task |
|---|---|
| Vault (.md) as source of truth | Task 2 (VaultManager) |
| ChromaDB as vector index | Task 3 (ChromaClient) |
| SQLite for metadata | Task 4 (SQLiteStore) |
| PDF ingestion | Task 5 (PDF ingest) |
| LLM client | Task 6 (LLM wrapper) |
| Intent router | Task 7 (Router node) |
| Summarize node | Task 8 (Summarize node) |
| Query/RAG node | Task 8 (Query node) |
| Edit node | Task 8 (Edit node) |
| Validate node | Task 9 (Validate) |
| Memory update node | Task 9 (Memory) |
| Response formatting | Task 9 (Response) |
| LangGraph wiring | Task 10 (Graph) |
| FastAPI + CLI | Task 11 (Main) |
| Vault templates | Task 12 (Templates) |
| Integration test | Task 12 (Integration) |

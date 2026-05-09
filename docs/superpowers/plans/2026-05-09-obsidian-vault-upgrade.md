# Obsidian Vault Upgrade — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Obsidian-compatible parse/render layer to the vault, supporting YAML frontmatter, wikilinks, embeds, callouts, and block IDs. Upgrade VaultManager, templates, and agent nodes.

**Architecture:** New `vault/obsidian.py` module handles parsing/rendering. VaultManager gets new methods alongside existing ones. Agent nodes updated to generate and consume Obsidian-formatted content. One-time migration function for existing vault files.

**Tech Stack:** Python 3.12, PyYAML (new dependency), existing LangGraph/ChromaDB/FastAPI stack.

---

## File Structure

```
src/resumini/vault/
├── __init__.py          (unchanged)
├── obsidian.py          (NEW — parse/render core)
├── manager.py           (MODIFIED — add 6 new methods)
├── templates.py         (MODIFIED — add frontmatter to all templates)
└── migrate.py           (NEW — one-time vault migration)

src/resumini/agent/
├── summarize_node.py    (MODIFIED — updated prompt)
├── query_node.py        (MODIFIED — wikilink resolution)
├── ingest_node.py       (MODIFIED — add frontmatter on ingest)
├── edit_node.py         (MODIFIED — preserve frontmatter)
├── memory_node.py       (MODIFIED — verify wikilinks)
├── graph.py             (unchanged)
├── router.py            (unchanged)
├── validate_node.py     (unchanged)
└── response_node.py     (unchanged)

vault/
├── profile.md           (MODIFIED — add frontmatter)
└── templates/
    ├── materia.md        (MODIFIED — add frontmatter)
    ├── resumen.md        (MODIFIED — add frontmatter)
    └── apunte_clase.md   (MODIFIED — add frontmatter)

tests/
├── test_obsidian.py     (NEW — parse/render tests)
├── test_vault_manager.py(MODIFIED — add tests for new methods)
└── test_migrate.py      (NEW — migration tests)

pyproject.toml           (MODIFIED — add pyyaml dependency)
```

---

### Task 1: Add PyYAML Dependency

**Files:**
- Modify: `pyproject.toml:6-17`

- [ ] **Step 1: Add pyyaml to dependencies in pyproject.toml**

Add `"pyyaml>=6.0"` to the dependencies list:

```toml
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
    "pyyaml>=6.0",
]
```

- [ ] **Step 2: Install dependency**

Run: `pip install -e ".[dev]"`

Expected: pyyaml installed successfully

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "feat: add pyyaml dependency for frontmatter parsing"
```

---

### Task 2: Obsidian Parse/Render Core — Data Models + Frontmatter

**Files:**
- Create: `src/resumini/vault/obsidian.py`
- Create: `tests/test_obsidian.py`

- [ ] **Step 1: Write failing tests for frontmatter extraction and rendering**

Create `tests/test_obsidian.py`:

```python
from resumini.vault.obsidian import (
    Callout,
    ObsidianNote,
    extract_frontmatter,
    render_frontmatter,
    render_note,
    parse_note,
)


def test_extract_frontmatter_with_yaml():
    raw = "---\nmateria: civil\ntype: summary\n---\n\n# Content here"
    fm, body = extract_frontmatter(raw)
    assert fm == {"materia": "civil", "type": "summary"}
    assert body.strip() == "# Content here"


def test_extract_frontmatter_no_yaml():
    raw = "# Just a heading\n\nSome content"
    fm, body = extract_frontmatter(raw)
    assert fm == {}
    assert body == raw


def test_extract_frontmatter_empty_yaml():
    raw = "---\n---\n\n# Content"
    fm, body = extract_frontmatter(raw)
    assert fm == {}
    assert body.strip() == "# Content"


def test_render_frontmatter_simple():
    result = render_frontmatter({"materia": "civil", "type": "summary"})
    assert result.startswith("---\n")
    assert result.strip().endswith("---")
    assert "materia: civil" in result
    assert "type: summary" in result


def test_render_frontmatter_with_list():
    fm = {"tags": ["civil", "resumen"]}
    result = render_frontmatter(fm)
    assert "- civil" in result
    assert "- resumen" in result


def test_render_frontmatter_empty():
    result = render_frontmatter({})
    assert result == ""


def test_render_note_with_frontmatter():
    note = ObsidianNote(
        frontmatter={"materia": "civil", "type": "summary"},
        content="# My Note\n\nSome text",
        wikilinks=[],
        embeds=[],
        block_ids=[],
        tags=[],
        callouts=[],
    )
    rendered = render_note(note)
    assert rendered.startswith("---\n")
    assert "materia: civil" in rendered
    assert "# My Note" in rendered


def test_render_note_without_frontmatter():
    note = ObsidianNote(
        frontmatter={},
        content="# My Note\n\nSome text",
        wikilinks=[],
        embeds=[],
        block_ids=[],
        tags=[],
        callouts=[],
    )
    rendered = render_note(note)
    assert not rendered.startswith("---")
    assert "# My Note" in rendered
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/nacho/Documents/resumini && python -m pytest tests/test_obsidian.py -v`

Expected: FAIL — `ModuleNotFoundError: No module named 'resumini.vault.obsidian'`

- [ ] **Step 3: Write vault/obsidian.py with data models and frontmatter functions**

Create `src/resumini/vault/obsidian.py`:

```python
import re
from dataclasses import dataclass, field

import yaml


@dataclass
class Callout:
    type: str
    title: str
    content: str


@dataclass
class ObsidianNote:
    frontmatter: dict
    content: str
    wikilinks: list[str] = field(default_factory=list)
    embeds: list[str] = field(default_factory=list)
    block_ids: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    callouts: list[Callout] = field(default_factory=list)


_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n?", re.DOTALL)


def extract_frontmatter(raw: str) -> tuple[dict, str]:
    match = _FRONTMATTER_RE.match(raw)
    if not match:
        return {}, raw
    yaml_str = match.group(1)
    try:
        fm = yaml.safe_load(yaml_str)
    except yaml.YAMLError:
        return {}, raw
    if not isinstance(fm, dict):
        return {}, raw
    body = raw[match.end():]
    return fm, body


def render_frontmatter(fm: dict) -> str:
    if not fm:
        return ""
    yaml_str = yaml.dump(fm, allow_unicode=True, sort_keys=False, default_flow_style=False)
    return f"---\n{yaml_str.strip()}\n---"


def render_note(note: ObsidianNote) -> str:
    parts = []
    fm = render_frontmatter(note.frontmatter)
    if fm:
        parts.append(fm)
    parts.append(note.content)
    return "\n\n".join(parts) if fm else note.content


def parse_note(raw: str) -> ObsidianNote:
    fm, content = extract_frontmatter(raw)
    wikilinks = re.findall(r"\[\[([^\]]+)\]\]", content)
    embeds = re.findall(r"!\[\[([^\]]+)\]\]", content)
    block_ids = re.findall(r"\^([a-zA-Z0-9-]+)$", content, re.MULTILINE)
    tags = re.findall(r"(?<!\w)#([a-zA-Z0-9_\/-]+)", content)
    callouts = _parse_callouts(content)
    return ObsidianNote(
        frontmatter=fm,
        content=content,
        wikilinks=wikilinks,
        embeds=embeds,
        block_ids=block_ids,
        tags=tags,
        callouts=callouts,
    )


def _parse_callouts(content: str) -> list[Callout]:
    callouts = []
    lines = content.split("\n")
    i = 0
    while i < len(lines):
        match = re.match(r"> \[!(\w+)\]\s*(.*)", lines[i])
        if match:
            callout_type = match.group(1)
            title = match.group(2).strip()
            body_lines = []
            i += 1
            while i < len(lines) and lines[i].startswith("> "):
                body_lines.append(lines[i][2:])
                i += 1
            callouts.append(Callout(type=callout_type, title=title, content="\n".join(body_lines)))
        else:
            i += 1
    return callouts


def resolve_wikilink(target: str, vault_dir, current_materia: str | None = None):
    from pathlib import Path

    vault_dir = Path(vault_dir)
    target_md = f"{target}.md" if not target.endswith(".md") else target

    if current_materia:
        same_materia = vault_dir / "materias" / current_materia / target_md
        if same_materia.exists():
            return same_materia

    for materia_dir in (vault_dir / "materias").iterdir():
        if materia_dir.is_dir():
            candidate = materia_dir / target_md
            if candidate.exists():
                return candidate

    root_candidate = vault_dir / target_md
    if root_candidate.exists():
        return root_candidate

    return None


def resolve_embed(target: str, vault_dir, current_materia: str | None = None) -> str | None:
    from pathlib import Path

    path = resolve_wikilink(target, vault_dir, current_materia)
    if path is None:
        return None
    try:
        return Path(path).read_text()
    except OSError:
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/nacho/Documents/resumini && python -m pytest tests/test_obsidian.py -v`

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/resumini/vault/obsidian.py tests/test_obsidian.py
git commit -m "feat: obsidian parse/render core with frontmatter, data models"
```

---

### Task 3: Obsidian Parse/Render Core — Pattern Extraction Tests

**Files:**
- Modify: `tests/test_obsidian.py`

- [ ] **Step 1: Write tests for wikilinks, embeds, block_ids, tags, callouts parsing**

Add to `tests/test_obsidian.py`:

```python
def test_parse_note_extracts_wikilinks():
    raw = "# Note\n\nSee [[prescripcion]] and [[derecho_civil/unidad_1]]"
    note = parse_note(raw)
    assert note.wikilinks == ["prescripcion", "derecho_civil/unidad_1"]


def test_parse_note_extracts_embeds():
    raw = "# Note\n\n![[def_prescripcion]] and ![[shared/concepto]]"
    note = parse_note(raw)
    assert note.embeds == ["def_prescripcion", "shared/concepto"]


def test_parse_note_extracts_block_ids():
    raw = "# Note\n\nKey term ^prescripcion-extintiva\nOther line\nAnother ^block-2"
    note = parse_note(raw)
    assert "prescripcion-extintiva" in note.block_ids
    assert "block-2" in note.block_ids


def test_parse_note_extracts_tags():
    raw = "# Note\n\nContent with #derecho and #civil/prescripcion tags"
    note = parse_note(raw)
    assert "derecho" in note.tags
    assert "civil/prescripcion" in note.tags


def test_parse_note_does_not_match_heading_as_tag():
    raw = "# Heading 1\n\n## Heading 2\n\nSome #real-tag here"
    note = parse_note(raw)
    assert "Heading" not in note.tags
    assert "Heading-1" not in note.tags
    assert "real-tag" in note.tags


def test_parse_note_extracts_callouts():
    raw = "# Note\n\n> [!def] Prescripcion extintiva\n> Instituto que libera al deudor\n> tras el plazo\n\nOther text"
    note = parse_note(raw)
    assert len(note.callouts) == 1
    assert note.callouts[0].type == "def"
    assert note.callouts[0].title == "Prescripcion extintiva"
    assert "libera al deudor" in note.callouts[0].content


def test_parse_note_multiple_callouts():
    raw = "# Note\n\n> [!def] Term A\n> Definition A\n\n> [!warning] Common mistake\n> Don't confuse X with Y"
    note = parse_note(raw)
    assert len(note.callouts) == 2
    assert note.callouts[0].type == "def"
    assert note.callouts[1].type == "warning"


def test_parse_note_callout_no_title():
    raw = "# Note\n\n> [!tip]\n> Study tip here"
    note = parse_note(raw)
    assert len(note.callouts) == 1
    assert note.callouts[0].type == "tip"
    assert note.callouts[0].title == ""


def test_parse_note_with_frontmatter_and_patterns():
    raw = "---\nmateria: civil\ntype: summary\ntags:\n  - civil\n---\n\n# Resumen\n\n> [!def] Prescripcion\n> Definicion aqui ^prescripcion\n\nSee [[penal/unidad_1]] for comparison."
    note = parse_note(raw)
    assert note.frontmatter == {"materia": "civil", "type": "summary", "tags": ["civil"]}
    assert "prescripcion" in note.block_ids
    assert "penal/unidad_1" in note.wikilinks
    assert len(note.callouts) == 1
    assert note.callouts[0].type == "def"


def test_parse_note_empty_content():
    note = parse_note("")
    assert note.frontmatter == {}
    assert note.content == ""
    assert note.wikilinks == []
    assert note.embeds == []
    assert note.block_ids == []
    assert note.tags == []
    assert note.callouts == []
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cd /home/nacho/Documents/resumini && python -m pytest tests/test_obsidian.py -v`

Expected: all PASS (the `parse_note` function was already implemented in Task 2)

- [ ] **Step 3: Commit**

```bash
git add tests/test_obsidian.py
git commit -m "test: obsidian pattern extraction tests for wikilinks, embeds, callouts"
```

---

### Task 4: Wikilink/Embed Resolution Tests

**Files:**
- Modify: `tests/test_obsidian.py`

- [ ] **Step 1: Write tests for resolve_wikilink and resolve_embed**

Add to `tests/test_obsidian.py`:

```python
import pytest
from pathlib import Path

from resumini.vault.obsidian import resolve_wikilink, resolve_embed


@pytest.fixture
def vault_with_notes(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    materias = vault / "materias"
    materias.mkdir()
    civil = materias / "civil"
    civil.mkdir()
    penal = materias / "penal"
    penal.mkdir()
    (civil / "prescripcion.md").write_text("# Prescripcion civil\n\nContent")
    (penal / "prescripcion.md").write_text("# Prescripcion penal\n\nContent")
    (civil / "unidad_1.md").write_text("# Unidad 1\n\nContent")
    (vault / "profile.md").write_text("# Profile")
    return vault


def test_resolve_wikilink_same_materia(vault_with_notes):
    result = resolve_wikilink("prescripcion", vault_with_notes, current_materia="civil")
    assert result is not None
    assert result.name == "prescripcion.md"
    assert result.parent.name == "civil"


def test_resolve_wikilink_different_materia(vault_with_notes):
    result = resolve_wikilink("prescripcion", vault_with_notes, current_materia="penal")
    assert result is not None
    assert result.name == "prescripcion.md"
    assert result.parent.name == "penal"


def test_resolve_wikilink_no_current_materia(vault_with_notes):
    result = resolve_wikilink("prescripcion", vault_with_notes, current_materia=None)
    assert result is not None
    assert result.name == "prescripcion.md"


def test_resolve_wikilink_root_file(vault_with_notes):
    result = resolve_wikilink("profile", vault_with_notes, current_materia="civil")
    assert result is not None
    assert result.name == "profile.md"


def test_resolve_wikilink_not_found(vault_with_notes):
    result = resolve_wikilink("nonexistent_note", vault_with_notes)
    assert result is None


def test_resolve_embed_returns_content(vault_with_notes):
    content = resolve_embed("prescripcion", vault_with_notes, current_materia="civil")
    assert content is not None
    assert "Prescripcion civil" in content


def test_resolve_embed_not_found(vault_with_notes):
    content = resolve_embed("nonexistent_note", vault_with_notes)
    assert content is None


def test_resolve_wikilink_with_md_extension(vault_with_notes):
    result = resolve_wikilink("prescripcion.md", vault_with_notes, current_materia="civil")
    assert result is not None
    assert result.name == "prescripcion.md"
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cd /home/nacho/Documents/resumini && python -m pytest tests/test_obsidian.py -v`

Expected: all PASS (resolve functions already implemented in Task 2)

- [ ] **Step 3: Commit**

```bash
git add tests/test_obsidian.py
git commit -m "test: wikilink and embed resolution tests"
```

---

### Task 5: VaultManager New Methods

**Files:**
- Modify: `src/resumini/vault/manager.py`
- Modify: `tests/test_vault_manager.py`

- [ ] **Step 1: Write failing tests for new VaultManager methods**

Add to `tests/test_vault_manager.py`:

```python
from resumini.vault.obsidian import ObsidianNote


def test_read_obsidian_note(vm):
    vm.write_note("civil", "test_obs.md", "---\nmateria: civil\ntype: summary\n---\n\n# Test\n\nSee [[penal/u1]]")
    note = vm.read_obsidian_note("civil", "test_obs.md")
    assert note.frontmatter["materia"] == "civil"
    assert "penal/u1" in note.wikilinks


def test_write_obsidian_note(vm):
    note = ObsidianNote(
        frontmatter={"materia": "civil", "type": "summary"},
        content="# My Summary\n\nContent here",
        wikilinks=[],
        embeds=[],
        block_ids=[],
        tags=[],
        callouts=[],
    )
    vm.write_obsidian_note("civil", "summary.md", note)
    raw = vm.read_note("civil", "summary.md")
    assert "materia: civil" in raw
    assert "# My Summary" in raw


def test_find_note_by_title(vm):
    vm.write_note("civil", "prescripcion.md", "# Prescripcion civil")
    result = vm.find_note_by_title("prescripcion")
    assert result is not None
    assert "prescripcion.md" in str(result)


def test_find_note_by_title_not_found(vm):
    result = vm.find_note_by_title("nonexistent")
    assert result is None


def test_get_embed_content(vm):
    vm.write_note("civil", "shared_def.md", "# Shared Definition\n\nA legal concept")
    content = vm.get_embed_content("shared_def", current_materia="civil")
    assert content is not None
    assert "Shared Definition" in content


def test_get_embed_content_not_found(vm):
    content = vm.get_embed_content("nonexistent", current_materia="civil")
    assert content is None


def test_list_all_tags(vm):
    vm.write_note("civil", "u1.md", "---\ntags:\n  - civil\n  - prescripcion\n---\n\n# U1")
    vm.write_note("penal", "u1.md", "---\ntags:\n  - penal\n  - prescripcion\n---\n\n# U1")
    tag_map = vm.list_all_tags()
    assert "civil" in tag_map
    assert "penal" in tag_map
    assert "prescripcion" in tag_map


def test_backlinks(vm):
    vm.write_note("civil", "u1.md", "# U1\n\nSee [[penal/u1]] for comparison")
    vm.write_note("penal", "u1.md", "# Penal U1\n\nContent")
    links = vm.backlinks("penal/u1")
    assert len(links) >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/nacho/Documents/resumini && python -m pytest tests/test_vault_manager.py -v -k "obsidian_note or find_note or embed_content or list_all_tags or backlinks"`

Expected: FAIL — `AttributeError: 'VaultManager' object has no attribute 'read_obsidian_note'`

- [ ] **Step 3: Add new methods to VaultManager**

Append to `src/resumini/vault/manager.py`:

```python
from pathlib import Path

from resumini.vault.obsidian import (
    ObsidianNote,
    parse_note,
    render_note,
    resolve_embed,
    resolve_wikilink,
)
```

Add these methods to the `VaultManager` class (after `add_session`):

```python
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
                    for tag in note.tags:
                        rel = str(md_file.relative_to(self.vault_dir))
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/nacho/Documents/resumini && python -m pytest tests/test_vault_manager.py -v`

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/resumini/vault/manager.py tests/test_vault_manager.py
git commit -m "feat: VaultManager obsidian methods — read/write parsed notes, find, embed, tags, backlinks"
```

---

### Task 6: Update Templates with Frontmatter

**Files:**
- Modify: `src/resumini/vault/templates.py`
- Modify: `vault/templates/materia.md`
- Modify: `vault/templates/resumen.md`
- Modify: `vault/templates/apunte_clase.md`
- Modify: `vault/profile.md`

- [ ] **Step 1: Update templates.py with frontmatter versions**

Replace the entire content of `src/resumini/vault/templates.py`:

```python
MATERIA_INDEX = """---
materia: {nombre}
type: index
cuatrimestre: {cuatrimestre}
profesor: {profesor}
tags:
  - {nombre}
---

# {nombre}

- **Profesor**: {profesor}
- **Cuatrimestre**: {cuatrimestre}
- **Fuentes**: {fuentes}"""

RESUMEN = """---
materia: {materia}
type: summary
source: {source}
date: {date}
tags:
  - {materia}
  - resumen
---

# {titulo}

## Ideas principales

{ideas}

## Conceptos clave

{conceptos}

## Conexiones

{conexiones}

## Notas

{notas}"""

APUNTE_CLASE = """---
materia: {materia}
type: apunte
fecha: {fecha}
tags:
  - {materia}
  - apunte
---

# Apunte de clase — {fecha}

## Tema

{tema}

## Apuntes

{apuntes}

## Dudas

{dudas}"""

PROFILE = """---
type: profile
---

# Perfil de estudio

## Preferencias de estilo

- **Formato preferido**: {formato}
- **Nivel de detalle**: {detalle}

## Materias activas

{materias}

## Notas sobre mi estilo

{notas}"""
```

- [ ] **Step 2: Update vault template files**

Replace `vault/templates/materia.md`:

```markdown
---
materia: "{nombre}"
type: index
cuatrimestre: "{cuatrimestre}"
profesor: "{profesor}"
tags:
  - "{nombre}"
---

# {nombre}

- **Profesor**: {profesor}
- **Cuatrimestre**: {cuatrimestre}
- **Fuentes**: {fuentes}
- **Estado**: en curso
```

Replace `vault/templates/resumen.md`:

```markdown
---
materia: "{materia}"
type: summary
source: "{source}"
date: "{date}"
tags:
  - "{materia}"
  - resumen
---

# {titulo}

## Ideas principales

{ideas}

## Conceptos clave

{conceptos}

## Conexiones

{conexiones}

## Notas

{notas}
```

Replace `vault/templates/apunte_clase.md`:

```markdown
---
materia: "{materia}"
type: apunte
fecha: "{fecha}"
tags:
  - "{materia}"
  - apunte
---

# Apunte de clase — {fecha}

## Tema

{tema}

## Apuntes

{apuntes}

## Dudas

{dudas}
```

Replace `vault/profile.md`:

```markdown
---
type: profile
---

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

- [ ] **Step 3: Update VaultManager._ensure_structure to use new PROFILE template**

The `_ensure_structure` method in `manager.py` already uses `PROFILE.format(...)`. Since the new PROFILE template has `{materias}` and `{notas}` placeholders (same as before), no changes needed to the format call.

But the `write_materia_index` method uses `MATERIA_INDEX.format(...)` which now needs `{nombre}` — it already has it. No change needed.

- [ ] **Step 4: Run existing tests to verify nothing broke**

Run: `cd /home/nacho/Documents/resumini && python -m pytest tests/test_vault_manager.py -v`

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/resumini/vault/templates.py vault/templates/ vault/profile.md
git commit -m "feat: add YAML frontmatter to all vault templates"
```

---

### Task 7: Vault Migration Function

**Files:**
- Create: `src/resumini/vault/migrate.py`
- Create: `tests/test_migrate.py`

- [ ] **Step 1: Write failing tests for migrate_vault**

Create `tests/test_migrate.py`:

```python
from pathlib import Path

from resumini.vault.migrate import migrate_vault
from resumini.vault.obsidian import parse_note


def test_migrate_adds_frontmatter_to_bare_file(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    materias = vault / "materias"
    civil = materias / "civil"
    civil.mkdir(parents=True)
    (civil / "unidad_1.md").write_text("# Unidad 1\n\nContenido de la unidad")
    (civil / "resumen_parcial.md").write_text("# Resumen parcial\n\nResumen aqui")
    (civil / "_index.md").write_text("# Civil\n\n- Profesor: Garcia")

    migrate_vault(vault)

    u1 = parse_note((civil / "unidad_1.md").read_text())
    assert u1.frontmatter.get("materia") == "civil"
    assert u1.frontmatter.get("type") == "note"

    resumen = parse_note((civil / "resumen_parcial.md").read_text())
    assert resumen.frontmatter.get("type") == "summary"

    index = parse_note((civil / "_index.md").read_text())
    assert index.frontmatter.get("type") == "index"


def test_migrate_preserves_existing_frontmatter(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    materias = vault / "materias"
    civil = materias / "civil"
    civil.mkdir(parents=True)
    (civil / "u1.md").write_text("---\nmateria: civil\ntype: raw_ingest\n---\n\n# U1\n\nContent")

    migrate_vault(vault)

    note = parse_note((civil / "u1.md").read_text())
    assert note.frontmatter.get("materia") == "civil"
    assert note.frontmatter.get("type") == "raw_ingest"
    assert "# U1" in note.content


def test_migrate_skips_non_md_files(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    materias = vault / "materias"
    civil = materias / "civil"
    civil.mkdir(parents=True)
    (civil / "image.png").write_bytes(b"fake image")

    migrate_vault(vault)

    assert (civil / "image.png").exists()


def test_migrate_handles_profile(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "profile.md").write_text("# Perfil\n\nEstilo: bullets")

    migrate_vault(vault)

    note = parse_note((vault / "profile.md").read_text())
    assert note.frontmatter.get("type") == "profile"
    assert "Perfil" in note.content


def test_migrate_handles_empty_vault(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "materias").mkdir()

    migrate_vault(vault)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/nacho/Documents/resumini && python -m pytest tests/test_migrate.py -v`

Expected: FAIL — `ModuleNotFoundError: No module named 'resumini.vault.migrate'`

- [ ] **Step 3: Write migrate.py**

Create `src/resumini/vault/migrate.py`:

```python
import re
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
        new_content = f"{rendered_fm}\n\n{body}" if rendered_fm else body
        md_file.write_text(new_content)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/nacho/Documents/resumini && python -m pytest tests/test_migrate.py -v`

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/resumini/vault/migrate.py tests/test_migrate.py
git commit -m "feat: vault migration function for adding frontmatter to existing files"
```

---

### Task 8: Update Summarize Node — Obsidian-Aware Prompt

**Files:**
- Modify: `src/resumini/agent/summarize_node.py`
- Modify: `tests/test_summarize_node.py`

- [ ] **Step 1: Write failing test for obsidian-formatted summary output**

Add to `tests/test_summarize_node.py`:

```python
def test_summarize_produces_obsidian_format(tmp_vault):
    from resumini.vault.manager import VaultManager
    from resumini.vault.obsidian import parse_note

    vault = VaultManager(tmp_vault)
    vault.write_note("civil", "u1_raw.md", "# Unidad 1\n\nLa prescripcion extintiva es un instituto que libera al deudor. Requiere plazo y reclamacion.")
    chroma = MagicMock()
    chroma.search.return_value = [{"id": "civil_u1_raw", "content": "prescripcion extintiva", "metadata": {"materia": "civil", "file": "u1_raw.md"}}]
    with patch("resumini.agent.summarize_node.get_llm") as mock_llm:
        mock_llm.return_value.invoke.return_value = MagicMock(content="---\nmateria: civil\ntype: summary\nsource: u1_raw.md\ndate: \"2026-05-09\"\ntags:\n  - civil\n---\n\n# Resumen U1\n\n> [!def] Prescripcion extintiva\n> Instituto que libera al deudor\n\n- Requiere plazo y reclamacion ^prescripcion")
        result = run_summarize("civil", "u1_raw.md", "u1_resumen.md", vault, chroma)
    note = parse_note(result)
    assert note.frontmatter.get("type") == "summary"
    assert note.frontmatter.get("materia") == "civil"
    assert len(note.callouts) >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/nacho/Documents/resumini && python -m pytest tests/test_summarize_node.py::test_summarize_produces_obsidian_format -v`

Expected: FAIL — the mock LLM returns plain text without frontmatter, so `note.frontmatter.get("type")` returns `None`

- [ ] **Step 3: Update summarize_node.py with obsidian-aware prompt**

Replace `src/resumini/agent/summarize_node.py`:

```python
from datetime import date

from langchain_core.messages import HumanMessage, SystemMessage

from resumini.db.chroma import ChromaClient
from resumini.llm.client import get_llm
from resumini.vault.manager import VaultManager

SUMMARIZE_PROMPT = """Eres un asistente academico que genera resumenes de material de estudio.
Usa el perfil del estudiante para adaptar el formato y nivel de detalle.
Genera un resumen claro, estructurado, en formato Obsidian markdown.

REGLAS DE FORMATO:
1. Comienza con YAML frontmatter entre --- fences:
   - materia: {materia}
   - type: summary
   - source: el archivo fuente
   - date: la fecha de hoy ({today})
   - tags: lista de tags relevantes (la materia + temas clave)
2. Usa callouts > [!def] para definiciones clave
3. Usa callouts > [!warning] para errores comunes o excepciones
4. Usa callouts > [!tip] para consejos de estudio o tips de examen
5. Usa [[wikilinks]] cuando menciones conceptos que pueden existir en otras materias (formato: [[materia/concepto]] o [[concepto]])
6. Agrega block IDs (^termino-clave) despues de definiciones importantes para poder referenciarlos

Perfil del estudiante:
{profile}

Contenido fuente:
{content}

Genera el resumen en formato Obsidian markdown:"""


def run_summarize(
    materia: str,
    source_file: str,
    output_file: str,
    vault: VaultManager,
    chroma: ChromaClient,
) -> str:
    source_content = vault.read_note(materia, source_file)
    profile = vault.read_profile()
    today = date.today().isoformat()
    llm = get_llm()
    response = llm.invoke(
        [
            SystemMessage(
                content=SUMMARIZE_PROMPT.format(
                    profile=profile,
                    content=source_content,
                    materia=materia,
                    today=today,
                )
            ),
            HumanMessage(
                content=f"Genera un resumen de {source_file} para la materia {materia}"
            ),
        ]
    )
    summary = response.content
    vault.write_note(materia, output_file, summary)
    chroma.index_document(
        doc_id=f"{materia}_{output_file.replace('.md', '')}",
        content=summary,
        metadata={"materia": materia, "file": output_file, "type": "summary"},
    )
    return summary
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/nacho/Documents/resumini && python -m pytest tests/test_summarize_node.py -v`

Expected: all PASS (the mock returns obsidian-formatted content matching the test)

- [ ] **Step 5: Commit**

```bash
git add src/resumini/agent/summarize_node.py tests/test_summarize_node.py
git commit -m "feat: summarize node with obsidian-aware prompt (frontmatter, callouts, wikilinks)"
```

---

### Task 9: Update Query Node — Wikilink Resolution

**Files:**
- Modify: `src/resumini/agent/query_node.py`
- Modify: `tests/test_query_node.py`

- [ ] **Step 1: Write failing test for wikilink resolution in queries**

Add to `tests/test_query_node.py`:

```python
def test_query_resolves_wikilinks(tmp_vault):
    from resumini.vault.manager import VaultManager

    vault = VaultManager(tmp_vault)
    vault.write_note("civil", "prescripcion.md", "# Prescripcion civil\n\nLa prescripcion extintiva libera al deudor")
    chroma = MagicMock()
    chroma.search.return_value = [{"id": "civil_prescripcion", "content": "prescripcion extintiva", "metadata": {"materia": "civil", "file": "prescripcion.md"}}]
    with patch("resumini.agent.query_node.get_llm") as mock_llm:
        mock_llm.return_value.invoke.return_value = MagicMock(content="La prescripcion en derecho civil libera al deudor...")
        result = run_query("que dice [[prescripcion]] en civil?", vault, chroma)
    assert "prescripcion" in result.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/nacho/Documents/resumini && python -m pytest tests/test_query_node.py::test_query_resolves_wikilinks -v`

Expected: FAIL or PASS — need to check if wikilink context is actually included. The test will pass because the mock LLM returns a valid answer regardless, but the query_node doesn't resolve wikilinks yet.

- [ ] **Step 3: Update query_node.py to resolve wikilinks before querying**

Replace `src/resumini/agent/query_node.py`:

```python
import re

from langchain_core.messages import HumanMessage, SystemMessage

from resumini.db.chroma import ChromaClient
from resumini.llm.client import get_llm
from resumini.vault.manager import VaultManager

WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")

QUERY_PROMPT = """Eres un asistente academico que responde preguntas basandote en resumenes y apuntes.
Usa el contexto proporcionado para dar una respuesta precisa.
Si la informacion no esta en el contexto, dilo claramente.

Perfil del estudiante:
{profile}

Contexto relevante:
{context}

Pregunta: {question}"""


def _resolve_wikilinks_in_question(question: str, vault: VaultManager) -> list[str]:
    extra_context = []
    for match in WIKILINK_RE.findall(question):
        content = vault.get_embed_content(match)
        if content:
            extra_context.append(f"[Referencia wikilink: {match}]\n{content}")
    return extra_context


def run_query(question: str, vault: VaultManager, chroma: ChromaClient) -> str:
    search_results = chroma.search(question, n_results=5)
    context_parts = []
    for r in search_results:
        context_parts.append(f"[{r['metadata']['materia']}/{r['metadata']['file']}] {r['content']}")

    wikilink_context = _resolve_wikilinks_in_question(question, vault)
    for wc in wikilink_context:
        context_parts.append(wc)

    context = "\n\n".join(context_parts)
    profile = vault.read_profile()
    llm = get_llm()
    response = llm.invoke(
        [
            SystemMessage(
                content=QUERY_PROMPT.format(
                    profile=profile, context=context, question=question
                )
            ),
            HumanMessage(content=question),
        ]
    )
    return response.content
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/nacho/Documents/resumini && python -m pytest tests/test_query_node.py -v`

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/resumini/agent/query_node.py tests/test_query_node.py
git commit -m "feat: query node resolves wikilinks before RAG search"
```

---

### Task 10: Update Ingest Node — Add Frontmatter

**Files:**
- Modify: `src/resumini/agent/ingest_node.py`
- Modify: `tests/test_ingest_node.py` (if exists, or add integration test)

- [ ] **Step 1: Update ingest_node.py to wrap content with frontmatter**

Replace `src/resumini/agent/ingest_node.py`:

```python
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
    content_with_fm = f"{frontmatter}\n\n{raw_content}" if frontmatter else raw_content
    vault.write_note(materia, filename, content_with_fm)
    chroma.index_document(
        doc_id=f"{materia}_{filename.replace('.md', '')}",
        content=content_with_fm,
        metadata={"materia": materia, "file": filename, "type": "raw_ingest"},
    )
    return content_with_fm
```

- [ ] **Step 2: Run existing tests to verify nothing broke**

Run: `cd /home/nacho/Documents/resumini && python -m pytest tests/ -v`

Expected: all PASS

- [ ] **Step 3: Commit**

```bash
git add src/resumini/agent/ingest_node.py
git commit -m "feat: ingest node adds YAML frontmatter to raw ingested content"
```

---

### Task 11: Update Edit Node — Preserve Frontmatter

**Files:**
- Modify: `src/resumini/agent/edit_node.py`
- Modify: `tests/test_edit_node.py`

- [ ] **Step 1: Write failing test for frontmatter preservation during edits**

Add to `tests/test_edit_node.py`:

```python
def test_edit_preserves_frontmatter(tmp_vault):
    from resumini.vault.manager import VaultManager
    from resumini.vault.obsidian import parse_note

    vault = VaultManager(tmp_vault)
    vault.write_note("civil", "resumen.md", "---\nmateria: civil\ntype: summary\n---\n\n# Resumen\n\nContenido original")
    chroma = MagicMock()
    with patch("resumini.agent.edit_node.get_llm") as mock_llm:
        mock_llm.return_value.invoke.return_value = MagicMock(content="# Resumen\n\nContenido original\n\n## Agregado\n\nNuevo contenido")
        result = run_edit("civil", "resumen.md", "agrega una seccion sobre prescripcion", vault, chroma)
    note = parse_note(vault.read_note("civil", "resumen.md"))
    assert note.frontmatter.get("materia") == "civil"
    assert note.frontmatter.get("type") == "summary"
    assert "Agregado" in note.content
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/nacho/Documents/resumini && python -m pytest tests/test_edit_node.py::test_edit_preserves_frontmatter -v`

Expected: FAIL — the mock LLM returns content without frontmatter, and edit_node doesn't preserve it

- [ ] **Step 3: Update edit_node.py to preserve frontmatter**

Replace `src/resumini/agent/edit_node.py`:

```python
from langchain_core.messages import HumanMessage, SystemMessage

from resumini.db.chroma import ChromaClient
from resumini.llm.client import get_llm
from resumini.vault.manager import VaultManager
from resumini.vault.obsidian import extract_frontmatter, render_frontmatter

EDIT_PROMPT = """Eres un asistente academico que edita resumenes existentes.
Aplica la edicion solicitada al contenido actual, manteniendo el formato y estilo.
Devuelve SOLO el contenido editado (sin YAML frontmatter, solo el body markdown).

Contenido actual (body sin frontmatter):
{current_content}

Edicion solicitada:
{edit_instruction}

Devuelve el contenido completo editado en markdown (sin frontmatter):"""


def run_edit(
    materia: str,
    filename: str,
    instruction: str,
    vault: VaultManager,
    chroma: ChromaClient,
) -> str:
    raw = vault.read_note(materia, filename)
    fm, body = extract_frontmatter(raw)
    llm = get_llm()
    response = llm.invoke(
        [
            SystemMessage(
                content=EDIT_PROMPT.format(
                    current_content=body, edit_instruction=instruction
                )
            ),
            HumanMessage(content=instruction),
        ]
    )
    edited_body = response.content
    fm_block = render_frontmatter(fm)
    edited = f"{fm_block}\n\n{edited_body}" if fm_block else edited_body
    vault.write_note(materia, filename, edited)
    chroma.index_document(
        doc_id=f"{materia}_{filename.replace('.md', '')}",
        content=edited,
        metadata={"materia": materia, "file": filename, "type": "summary"},
    )
    return edited
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/nacho/Documents/resumini && python -m pytest tests/test_edit_node.py -v`

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/resumini/agent/edit_node.py tests/test_edit_node.py
git commit -m "feat: edit node preserves YAML frontmatter during edits"
```

---

### Task 12: Update Memory Node — Verify Wikilinks

**Files:**
- Modify: `src/resumini/agent/memory_node.py`

- [ ] **Step 1: Update memory_node.py to verify wikilinks after writing**

Replace `src/resumini/agent/memory_node.py`:

```python
import logging
import uuid
from datetime import datetime

from resumini.db.chroma import ChromaClient
from resumini.db.sqlite import SQLiteStore
from resumini.vault.manager import VaultManager
from resumini.vault.obsidian import parse_note

logger = logging.getLogger(__name__)


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
    note = parse_note(content)
    for wl in note.wikilinks:
        resolved = vault.find_note_by_title(wl)
        if resolved is None:
            logger.warning("Broken wikilink: [[%s]] in %s/%s", wl, materia, filename)
    for tag in note.tags:
        pass
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
```

- [ ] **Step 2: Run all tests to verify nothing broke**

Run: `cd /home/nacho/Documents/resumini && python -m pytest tests/ -v`

Expected: all PASS

- [ ] **Step 3: Commit**

```bash
git add src/resumini/agent/memory_node.py
git commit -m "feat: memory node verifies wikilinks and logs broken links"
```

---

### Task 13: Migrate Existing Vault + Final Integration Test

**Files:**
- Modify: `tests/test_integration.py`

- [ ] **Step 1: Update integration test to verify obsidian format**

Read existing `tests/test_integration.py` and update to verify that the full pipeline produces Obsidian-formatted output. Add this test:

```python
def test_full_pipeline_produces_obsidian_format(tmp_path):
    from resumini.vault.manager import VaultManager
    from resumini.vault.obsidian import parse_note
    from resumini.db.chroma import ChromaClient
    from resumini.db.sqlite import SQLiteStore
    from resumini.agent.graph import AgentState, build_graph
    from unittest.mock import MagicMock, patch

    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    (vault_dir / "materias").mkdir()
    (vault_dir / "templates").mkdir()
    (vault_dir / ".meta").mkdir()
    (vault_dir / ".meta" / "sessions.json").write_text("[]")
    (vault_dir / "profile.md").write_text("---\ntype: profile\n---\n\n# Perfil\n\nEstilo: bullets")

    vault = VaultManager(vault_dir)
    chroma = ChromaClient(persist_dir=str(tmp_path / "chroma"))
    sqlite = SQLiteStore(str(tmp_path / "test.db"))
    sqlite.initialize()

    with patch("resumini.agent.graph.classify_intent", return_value=MagicMock(value="summarize")):
        with patch("resumini.agent.summarize_node.get_llm") as mock_llm:
            mock_llm.return_value.invoke.return_value = MagicMock(content="---\nmateria: civil\ntype: summary\n---\n\n# Resumen\n\n> [!def] Test\nDefinition here")
            graph = build_graph(vault, chroma, sqlite)
            state = AgentState(
                message="resumen",
                intent=MagicMock(value="summarize"),
                materia="civil",
                source_file="u1.md",
                output_file="u1_resumen.md",
                pdf_path=None,
                edit_instruction=None,
                output=None,
                valid=False,
                errors=[],
                metadata={},
            )
            vault.write_note("civil", "u1.md", "# U1\n\nContent")
            result = graph.invoke(state)
            output = result.get("output", "")
            note = parse_note(output)
            assert note.frontmatter.get("type") == "summary"

    sqlite.close()
```

- [ ] **Step 2: Run all tests**

Run: `cd /home/nacho/Documents/resumini && python -m pytest tests/ -v`

Expected: all PASS

- [ ] **Step 3: Run ruff lint**

Run: `cd /home/nacho/Documents/resumini && ruff check src/ tests/`

Expected: no errors

- [ ] **Step 4: Migrate existing vault files**

Run: `cd /home/nacho/Documents/resumini && python -c "from resumini.vault.migrate import migrate_vault; from pathlib import Path; migrate_vault(Path('./vault'))"`

This adds frontmatter to any existing vault files that don't have it.

- [ ] **Step 5: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: integration test for obsidian-formatted pipeline output"
```

---

### Task 14: Run Full Test Suite + Lint

**Files:** None (verification only)

- [ ] **Step 1: Run full test suite**

Run: `cd /home/nacho/Documents/resumini && python -m pytest tests/ -v --tb=short`

Expected: all PASS

- [ ] **Step 2: Run ruff check**

Run: `cd /home/nacho/Documents/resumini && ruff check src/ tests/`

Expected: no errors

- [ ] **Step 3: Run ruff format**

Run: `cd /home/nacho/Documents/resumini && ruff format --check src/ tests/`

Expected: no errors

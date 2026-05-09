# Resumini — Obsidian Vault Upgrade Design

## Goal

Upgrade the vault layer to support full Obsidian-compatible markdown: YAML frontmatter, wikilinks, embeds, callouts, and block IDs. This makes the vault a real Obsidian vault, enables cross-materia linking, and gives the agent structured access to note metadata and relationships.

## Approach

**Parse/Render Layer** — add a `vault/obsidian.py` module that parses Obsidian markdown into structured data and renders it back. VaultManager gets new methods alongside existing ones for backward compatibility. Agent nodes updated to generate and consume Obsidian-formatted content.

## New Module: `vault/obsidian.py`

### Data Models

```python
@dataclass
class Callout:
    type: str       # "def", "warning", "tip", "info", "note", "example", "quote"
    title: str      # Optional title after > [!type] Title
    content: str    # Body of the callout (without > prefix)

@dataclass
class ObsidianNote:
    frontmatter: dict          # YAML properties parsed from --- fences
    content: str               # Body markdown (without frontmatter block)
    wikilinks: list[str]       # Extracted [[target]] references
    embeds: list[str]          # Extracted ![[target]] references
    block_ids: list[str]       # Extracted ^block-id markers
    tags: list[str]            # Extracted #tag values
    callouts: list[Callout]    # Parsed > [!type] blocks
```

### Core Functions

- `parse_note(raw: str) -> ObsidianNote` — splits frontmatter from body, extracts all patterns
- `render_note(note: ObsidianNote) -> str` — converts back to valid Obsidian markdown (YAML fences + body)
- `resolve_wikilink(target: str, vault_dir: Path) -> Path | None` — finds the actual file in vault
- `resolve_embed(target: str, vault_dir: Path) -> str | None` — resolves and returns embedded content
- `extract_frontmatter(raw: str) -> tuple[dict, str]` — split YAML frontmatter from body
- `render_frontmatter(fm: dict) -> str` — serialize dict to YAML fences

### Pattern Regexes

| Pattern | Regex | Notes |
|---------|-------|-------|
| Frontmatter | `^---\n(.*?)\n---` | YAML between fences at start of file |
| Wikilinks | `\[\[([^\]]+)\]\]` | Captures link target |
| Embeds | `!\[\[([^\]]+)\]\]` | Captures embed target |
| Block IDs | `\^([a-zA-Z0-9-]+)$` | End of line only |
| Tags | `(?<!\w)#([a-zA-Z0-9_\/-]+)` | Not inside a word |
| Callouts | `> \[!(\w+)\]\s*(.*)` + continuation `> (.*)` | Type + optional title + body |

### Wikilink Resolution Strategy

Wikilinks use note titles (without `.md` extension). Resolution order:
1. Exact match in same materia directory: `materias/{current_materia}/{target}.md`
2. Search all materia directories: `materias/*/{target}.md`
3. Search vault root: `{target}.md`
4. Return `None` if not found (broken link)

Embeds follow the same resolution, then read and return the file content.

## VaultManager Integration

### New Methods (alongside existing ones for backward compat)

| Method | Signature | Purpose |
|--------|-----------|---------|
| `read_obsidian_note` | `(materia, filename) -> ObsidianNote` | Parsed version of read_note |
| `write_obsidian_note` | `(materia, filename, note: ObsidianNote)` | Renders and writes |
| `find_note_by_title` | `(title: str) -> Path \| None` | Resolve wikilinks across materias |
| `get_embed_content` | `(target: str) -> str \| None` | Resolve and return embedded content |
| `list_all_tags` | `() -> dict[str, list[str]]` | Tag → list of files mapping |
| `backlinks` | `(note_title: str) -> list[Path]` | Which notes link TO this one |

### Existing Methods — Unchanged

- `read_note()`, `write_note()`, `list_materias()`, `list_notes()`, etc. — all stay the same
- New methods are additions, not replacements

## Template Updates

All templates get YAML frontmatter. Content body adds callouts and wikilinks.

### Materia Index (`_index.md`)

```markdown
---
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
- **Fuentes**: {fuentes}
```

### Resumen

```markdown
---
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

{notas}
```

### Apunte de Clase

```markdown
---
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

{dudas}
```

### Profile

```markdown
---
type: profile
---

# Perfil de estudio

## Preferencias de estilo

- **Formato preferido**: {formato}
- **Nivel de detalle**: {detalle}

## Materias activas

{materias}

## Notas sobre mi estilo

{notas}
```

## Agent Node Updates

### Summarize Node

Prompt updated to instruct the LLM:
- Add YAML frontmatter with `materia`, `type: summary`, `source`, `date`, `tags`
- Use `> [!def] Term` callouts for key definitions
- Use `> [!warning]` for common mistakes or exceptions
- Use `> [!tip]` for study tips or exam hints
- Add `[[wikilinks]]` when referencing concepts that exist in other materias
- Add `^term-name` block IDs after key definitions for cross-referencing

### Query Node

Before querying ChromaDB:
- Resolve any `[[wikilinks]]` in the user's question
- Include linked note content as additional context in the RAG prompt
- Example: "que dice [[prescripcion]] en penal?" → include civil's definition too

### Ingest Node

When writing raw content to vault:
- Add basic frontmatter: `type: raw_ingest`, `materia`, `source`, `date`
- Content body unchanged (PDF extraction is plain markdown)

### Edit Node

- Parse the note with `parse_note()` before editing
- Apply edits to content body, preserve frontmatter
- If the edit instruction mentions metadata (e.g. "add tag #importante"), update frontmatter accordingly

### Memory Node

After writing a note:
- Scan for wikilinks and verify they resolve
- Log broken links as warnings (not errors — the target may be created later)
- Update `sessions.json` with the action

## Vault Migration

A one-time `migrate_vault(vault_dir: Path)` function:

1. Walk all `.md` files in vault
2. If file has no frontmatter fence (`---`), infer properties:
   - `materia`: from parent directory name (if inside `materias/`)
   - `type`: from filename pattern (`_index.md` → `index`, `*resumen*` → `summary`, `clase_*` → `apunte`, else `note`)
   - `date`: from file modification time
3. Prepend frontmatter block
4. Leave content body unchanged
5. Write back to same file

No data loss — only adds structured metadata on top of existing content.

## Files Changed

| File | Action |
|------|--------|
| `src/resumini/vault/obsidian.py` | **New** — parse/render module |
| `src/resumini/vault/manager.py` | **Modified** — add 5 new methods |
| `src/resumini/vault/templates.py` | **Modified** — add frontmatter to all templates |
| `src/resumini/vault/migrate.py` | **New** — one-time vault migration |
| `src/resumini/agent/summarize_node.py` | **Modified** — updated prompt |
| `src/resumini/agent/query_node.py` | **Modified** — wikilink resolution before RAG |
| `src/resumini/agent/ingest_node.py` | **Modified** — add frontmatter on ingest |
| `src/resumini/agent/edit_node.py` | **Modified** — preserve frontmatter |
| `src/resumini/agent/memory_node.py` | **Modified** — verify wikilinks |
| `src/resumini/agent/graph.py` | **Unchanged** |
| `tests/test_obsidian.py` | **New** — tests for parse/render |
| `tests/test_vault_manager.py` | **Modified** — add tests for new methods |
| `tests/test_migrate.py` | **New** — tests for migration |
| `vault/templates/*.md` | **Modified** — add frontmatter |
| `vault/profile.md` | **Modified** — add frontmatter |

## Out of Scope

- `.base` files (Obsidian Bases) — query/view files, not needed for summaries
- `.canvas` files (JSON Canvas) — visual graph layout, not needed
- Obsidian plugin compatibility — we generate compatible markdown, not plugins
- Wikilink aliases `[[target|display]]` — simple links only for now
- Nested callouts — single-level callouts only

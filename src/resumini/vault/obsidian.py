import re
from dataclasses import dataclass
from pathlib import Path

import yaml


_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---", re.DOTALL)
_FRONTMATTER_EMPTY_RE = re.compile(r"^---\n---\n?")
_WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")
_EMBED_RE = re.compile(r"!\[\[([^\]]+)\]\]")
_BLOCK_ID_RE = re.compile(r"\^([a-zA-Z0-9-]+)$", re.MULTILINE)
_TAG_RE = re.compile(r"(?<!\w)#([a-zA-Z0-9_\/-]+)")
_CALLOUT_START_RE = re.compile(r"^> \[!(\w+)\]\s*(.*)", re.MULTILINE)


@dataclass
class Callout:
    type: str
    title: str
    content: str


@dataclass
class ObsidianNote:
    frontmatter: dict
    content: str
    wikilinks: list[str]
    embeds: list[str]
    block_ids: list[str]
    tags: list[str]
    callouts: list[Callout]


def extract_frontmatter(raw: str) -> tuple[dict, str]:
    empty_match = _FRONTMATTER_EMPTY_RE.match(raw)
    if empty_match:
        body = raw[empty_match.end() :]
        if body.startswith("\n"):
            body = body[1:]
        return {}, body
    match = _FRONTMATTER_RE.match(raw)
    if not match:
        return {}, raw
    yaml_str = match.group(1)
    fm = yaml.safe_load(yaml_str) or {}
    body = raw[match.end() :]
    if body.startswith("\n"):
        body = body[1:]
    return fm, body


def render_frontmatter(fm: dict) -> str:
    if not fm:
        return ""
    yaml_str = yaml.dump(fm, allow_unicode=True, sort_keys=False, default_flow_style=False)
    return f"---\n{yaml_str}---\n"


def render_note(note: ObsidianNote) -> str:
    if note.frontmatter:
        return render_frontmatter(note.frontmatter) + "\n" + note.content
    return note.content


def _extract_callouts(body: str) -> list[Callout]:
    callouts: list[Callout] = []
    lines = body.split("\n")
    i = 0
    while i < len(lines):
        m = _CALLOUT_START_RE.match(lines[i])
        if m:
            callout_type = m.group(1)
            title = m.group(2).strip()
            content_lines: list[str] = []
            i += 1
            while i < len(lines):
                cont = re.match(r"^> (.*)", lines[i])
                if cont:
                    content_lines.append(cont.group(1))
                    i += 1
                else:
                    break
            callouts.append(
                Callout(type=callout_type, title=title, content="\n".join(content_lines))
            )
        else:
            i += 1
    return callouts


def parse_note(raw: str) -> ObsidianNote:
    fm, body = extract_frontmatter(raw)
    embeds = _EMBED_RE.findall(body)
    wikilinks = [w for w in _WIKILINK_RE.findall(body) if w not in embeds]
    block_ids = _BLOCK_ID_RE.findall(body)
    tags = [t for t in _TAG_RE.findall(body) if not _is_heading_tag(body, t)]
    callouts = _extract_callouts(body)
    return ObsidianNote(
        frontmatter=fm,
        content=body,
        wikilinks=wikilinks,
        embeds=embeds,
        block_ids=block_ids,
        tags=tags,
        callouts=callouts,
    )


_HEADING_LINE_RE = re.compile(r"^#{1,6}\s+", re.MULTILINE)


def _is_heading_tag(body: str, tag: str) -> bool:
    pattern = r"^#{1,6}\s+.*(?:\s+|^)#" + re.escape(tag) + r"(?:\s|$)"
    if re.search(pattern, body, re.MULTILINE):
        return True
    for m in _HEADING_LINE_RE.finditer(body):
        start = m.start()
        end = body.find("\n", start)
        if end == -1:
            end = len(body)
        heading_text = body[start:end].strip()
        if heading_text == f"#{tag}" or heading_text.endswith(f" #{tag}"):
            return True
    return False


def resolve_wikilink(target: str, vault_dir, current_materia: str | None = None) -> Path | None:
    vault_dir = Path(vault_dir)
    stem = target.removesuffix(".md")
    filename = f"{stem}.md"
    if current_materia:
        same_materia = vault_dir / "materias" / current_materia / filename
        if same_materia.exists():
            return same_materia
    materias_dir = vault_dir / "materias"
    if materias_dir.exists():
        for d in sorted(materias_dir.iterdir()):
            if d.is_dir():
                candidate = d / filename
                if candidate.exists():
                    return candidate
    root_file = vault_dir / filename
    if root_file.exists():
        return root_file
    return None


def resolve_embed(target: str, vault_dir, current_materia: str | None = None) -> str | None:
    path = resolve_wikilink(target, vault_dir, current_materia)
    if path is None:
        return None
    return path.read_text()

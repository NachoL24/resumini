"""Post-write hook: verify wikilinks resolve and log warnings for broken ones."""

import logging

from resumini.vault.manager import VaultManager
from resumini.vault.obsidian import parse_note

logger = logging.getLogger(__name__)


def verify_wikilinks(
    materia: str, filename: str, content: str, vault: VaultManager
) -> list[str]:
    """Check wikilinks in `content`. Return the list of broken targets and log warnings."""
    note = parse_note(content)
    broken: list[str] = []
    for wl in note.wikilinks:
        if vault.find_note_by_title(wl) is None:
            broken.append(wl)
            logger.warning(
                "Broken wikilink: [[%s]] in %s/%s", wl, materia, filename
            )
    return broken

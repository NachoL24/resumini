"""Tests for the wikilink verification hook."""

import logging

from resumini.agent.memory_node import verify_wikilinks
from resumini.vault.manager import VaultManager


def test_verify_wikilinks_returns_broken_targets(tmp_vault):
    vault = VaultManager(tmp_vault)
    content = "Vinculo a [[no_existe]] aqui."
    broken = verify_wikilinks("civil", "f.md", content, vault)
    assert broken == ["no_existe"]


def test_verify_wikilinks_silent_when_link_resolves(tmp_vault, caplog):
    vault = VaultManager(tmp_vault)
    vault.write_note("civil", "target.md", "# Target")
    content = "Vinculo a [[target]] aqui."
    with caplog.at_level(logging.WARNING):
        broken = verify_wikilinks("civil", "f.md", content, vault)
    assert broken == []
    assert "broken wikilink" not in caplog.text.lower()


def test_verify_wikilinks_logs_warning_for_broken_link(tmp_vault, caplog):
    vault = VaultManager(tmp_vault)
    content = "Vinculo a [[missing]]"
    with caplog.at_level(logging.WARNING):
        verify_wikilinks("civil", "f.md", content, vault)
    assert "missing" in caplog.text

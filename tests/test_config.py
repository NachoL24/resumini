from pathlib import Path

from resumini.config import Settings


def test_settings_defaults():
    s = Settings(openai_api_key="test-key")
    assert s.llm_model == "gpt-4o"
    assert s.embedding_model == "text-embedding-3-small"
    assert s.vault_dir == Path("./vault")


def test_settings_custom():
    s = Settings(openai_api_key="test-key", llm_model="gpt-4o-mini")
    assert s.llm_model == "gpt-4o-mini"

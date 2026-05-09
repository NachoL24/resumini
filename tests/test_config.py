from pathlib import Path

from resumini.config import Settings


def test_settings_defaults():
    s = Settings(nvidia_api_key="test-key")
    assert s.llm_model == "z-ai/glm-5.1"
    assert s.embedding_model == "nvidia/nv-embedqa-e5-v5"
    assert s.vault_dir == Path("./vault")


def test_settings_custom():
    s = Settings(nvidia_api_key="test-key", llm_model="meta/llama-3.3-70b-instruct")
    assert s.llm_model == "meta/llama-3.3-70b-instruct"

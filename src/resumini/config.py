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

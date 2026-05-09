from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    nvidia_api_key: str = ""
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    vault_dir: Path = Path("./vault")
    chroma_dir: Path = Path("./chroma_data")
    db_path: Path = Path("./resumini.db")
    llm_model: str = "z-ai/glm-5.1"
    embedding_model: str = "nvidia/nv-embedqa-e5-v5"
    llm_temperature: float = 0.3

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


def get_settings() -> Settings:
    return Settings()

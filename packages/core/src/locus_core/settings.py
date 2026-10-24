from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Locus configuration, read from LOCUS_* environment variables."""

    model_config = SettingsConfigDict(env_prefix="LOCUS_")

    data_dir: Path = Field(default_factory=lambda: Path.home() / ".locus")
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_api_key: str | None = None

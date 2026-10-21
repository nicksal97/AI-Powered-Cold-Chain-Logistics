"""Typed, environment-driven configuration.

Everything the service needs is declared here once. Nothing else in the codebase
reads os.environ directly, which keeps config testable and makes missing values
fail at startup with a clear message instead of halfway through a request.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="COLDCHAIN_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    env: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"
    log_json: bool = True

    # --- storage -------------------------------------------------------------
    database_url: str = "postgresql+psycopg://coldchain:coldchain@localhost:5432/coldchain"
    db_pool_size: int = 5

    # --- LLM -----------------------------------------------------------------
    # openai | deepseek | ollama (ollama is reached through its OpenAI-compatible /v1 API)
    llm_provider: Literal["openai", "deepseek", "ollama"] = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_base_url: str | None = None
    llm_api_key: SecretStr | None = None
    llm_temperature: float = 0.0
    llm_timeout_s: float = 60.0
    agent_max_steps: int = 12

    # --- embeddings / retrieval ---------------------------------------------
    # "hash" is a deterministic offline embedder used for tests and air-gapped dev.
    embedding_provider: Literal["openai", "hash"] = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    retrieval_k: int = 4
    retrieval_candidates: int = 20

    # --- agent memory --------------------------------------------------------
    checkpointer: Literal["postgres", "memory"] = "postgres"

    # --- external services ---------------------------------------------------
    weather_base_url: str = "https://api.open-meteo.com/v1/forecast"
    weather_timeout_s: float = 6.0

    # --- API -----------------------------------------------------------------
    api_keys: list[str] = Field(default_factory=list)
    cors_origins: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_prod(self) -> Settings:
        if self.env == "prod" and not self.api_keys:
            raise ValueError("COLDCHAIN_API_KEYS must be set when COLDCHAIN_ENV=prod")
        return self

    @property
    def psycopg_dsn(self) -> str:
        """Plain libpq DSN for the LangGraph Postgres checkpointer."""
        return self.database_url.replace("postgresql+psycopg://", "postgresql://", 1)


@lru_cache
def get_settings() -> Settings:
    return Settings()

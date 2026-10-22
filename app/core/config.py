from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "fastapi-learning"
    environment: str = "local"
    debug: bool = False
    api_prefix: str = "/api"

    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-5"

    # Attaches TraceMiddleware to the deepagents engine — logs every model
    # call and tool call, and provides the breakpoint surface for stepping
    # through a turn (see app/deepagents_engine/tracing.py). Set
    # DEEPAGENTS_TRACE=false in .env to detach it.
    deepagents_trace: bool = True

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "postgres"
    postgres_user: str = "postgres"
    postgres_password: str | None = None

    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "learning_documents"

    docling_serve_url: str = "http://127.0.0.1:5001"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()

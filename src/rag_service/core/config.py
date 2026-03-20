from functools import lru_cache
from typing import Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "rag-service"
    app_env: str = "development"
    debug: bool = False
    log_level: str = "INFO"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/rag_service"
    database_pool_size: int = 5
    database_max_overflow: int = 10

    # Vector settings
    embedding_dimension: int = 1536

    # ==========================================================================
    # Embedding Providers
    # ==========================================================================

    # OpenAI
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"

    # Azure OpenAI
    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_api_version: str = "2024-02-01"

    # HuggingFace
    huggingface_api_key: str = ""

    # Ollama
    ollama_base_url: str = "http://localhost:11434"

    # Default provider settings
    default_embedding_provider: str = "openai"
    default_embedding_model: str = "text-embedding-3-small"

    # ==========================================================================
    # Chunking Settings
    # ==========================================================================

    chunk_size: int = 512
    chunk_overlap: int = 50
    default_chunking_strategy: str = "fixed"

    # ==========================================================================
    # RAG Settings
    # ==========================================================================

    top_k: int = 5
    min_similarity: float = 0.5

    # ==========================================================================
    # File Upload Settings
    # ==========================================================================

    max_file_size_mb: int = 50
    allowed_extensions: list[str] = [
        "pdf", "txt", "docx", "md", "csv", "xlsx", "xls", "html", "htm", "rtf"
    ]

    @field_validator("allowed_extensions", mode="before")
    @classmethod
    def parse_extensions(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            import json
            return json.loads(v)
        return v

    # ==========================================================================
    # Authentication
    # ==========================================================================

    secret_key: str = "change-me-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # ==========================================================================
    # RBAC Settings
    # ==========================================================================

    enable_public_documents: bool = False
    default_user_role: str = "viewer"

    # ==========================================================================
    # CORS
    # ==========================================================================

    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:8080"]

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            import json
            return json.loads(v)
        return v

    # ==========================================================================
    # Computed Properties
    # ==========================================================================

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024


# Embedding model configurations
EMBEDDING_MODELS = {
    "openai": {
        "text-embedding-3-small": {"dimension": 1536, "max_tokens": 8191},
        "text-embedding-3-large": {"dimension": 3072, "max_tokens": 8191},
        "text-embedding-ada-002": {"dimension": 1536, "max_tokens": 8191},
    },
    "azure": {
        "text-embedding-3-small": {"dimension": 1536, "max_tokens": 8191},
        "text-embedding-3-large": {"dimension": 3072, "max_tokens": 8191},
        "text-embedding-ada-002": {"dimension": 1536, "max_tokens": 8191},
    },
    "huggingface": {
        "all-MiniLM-L6-v2": {"dimension": 384, "max_tokens": 256},
        "all-MiniLM-L12-v2": {"dimension": 384, "max_tokens": 256},
        "all-mpnet-base-v2": {"dimension": 768, "max_tokens": 384},
        "e5-small-v2": {"dimension": 384, "max_tokens": 512},
        "e5-base-v2": {"dimension": 768, "max_tokens": 512},
        "e5-large-v2": {"dimension": 1024, "max_tokens": 512},
        "bge-small-en-v1.5": {"dimension": 384, "max_tokens": 512},
        "bge-base-en-v1.5": {"dimension": 768, "max_tokens": 512},
        "bge-large-en-v1.5": {"dimension": 1024, "max_tokens": 512},
    },
    "ollama": {
        "nomic-embed-text": {"dimension": 768, "max_tokens": 8192},
        "mxbai-embed-large": {"dimension": 1024, "max_tokens": 512},
        "all-minilm": {"dimension": 384, "max_tokens": 256},
    },
}


def get_embedding_dimension(provider: str, model: str) -> int:
    """Get embedding dimension for a provider/model combination."""
    provider_models = EMBEDDING_MODELS.get(provider, {})
    model_config = provider_models.get(model, {})
    return model_config.get("dimension", 1536)


@lru_cache
def get_settings() -> Settings:
    return Settings()

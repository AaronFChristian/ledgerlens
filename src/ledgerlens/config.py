"""Central application settings — loaded from .env via pydantic-settings."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Anthropic ─────────────────────────────────────────
    anthropic_api_key: str = Field(..., alias="ANTHROPIC_API_KEY")
    claude_model: str = Field("claude-sonnet-4-6", alias="CLAUDE_MODEL")

    # ── Extraction ────────────────────────────────────────
    confidence_threshold: float = Field(0.75, alias="CONFIDENCE_THRESHOLD")
    max_image_size_px: int = 1568  # Claude's optimal long-edge size
    extraction_max_tokens: int = 2048

    # ── Neo4j (Day 2) ────────────────────────────────────
    neo4j_uri: str = Field("bolt://localhost:7687", alias="NEO4J_URI")
    neo4j_username: str = Field("neo4j", alias="NEO4J_USERNAME")
    neo4j_password: str = Field("", alias="NEO4J_PASSWORD")

    # ── Langfuse (Day 3) ─────────────────────────────────
    langfuse_public_key: str = Field("", alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str = Field("", alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field("https://cloud.langfuse.com", alias="LANGFUSE_HOST")


# Single shared instance — import this everywhere
settings = Settings()

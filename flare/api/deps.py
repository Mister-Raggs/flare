"""Shared dependencies and configuration for the API layer."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache


@dataclass
class Settings:
    """Application settings loaded from environment variables."""

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-20250514"
    log_level: str = "INFO"
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = field(default_factory=lambda: ["*"])

    @classmethod
    def from_env(cls) -> Settings:
        """Load settings from environment variables."""
        origins_raw = os.environ.get("FLARE_CORS_ORIGINS", "*")
        origins = [o.strip() for o in origins_raw.split(",")]
        return cls(
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            anthropic_model=os.environ.get(
                "FLARE_MODEL", "claude-sonnet-4-20250514"
            ),
            log_level=os.environ.get("FLARE_LOG_LEVEL", "INFO"),
            host=os.environ.get("FLARE_HOST", "0.0.0.0"),
            port=int(os.environ.get("FLARE_PORT", "8000")),
            cors_origins=origins,
        )


@lru_cache
def get_settings() -> Settings:
    """Cached singleton for app settings."""
    return Settings.from_env()

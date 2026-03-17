"""Health check endpoint."""

from __future__ import annotations

import os

from fastapi import APIRouter

from flare import __version__
from flare.api.models import HealthResponse

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns API status, Anthropic API connectivity, model info, and version.",
)
async def health() -> HealthResponse:
    """Check API health and Anthropic connectivity."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    anthropic_status = "reachable" if api_key else "unreachable"

    # If key exists, try a lightweight validation
    if api_key:
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=api_key)
            client.models.list(limit=1)
        except Exception:
            anthropic_status = "unreachable"

    from flare.api.deps import get_settings

    settings = get_settings()

    return HealthResponse(
        status="ok",
        anthropic_api=anthropic_status,
        model=settings.anthropic_model,
        version=__version__,
    )

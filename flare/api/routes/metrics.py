"""Prometheus-compatible /metrics endpoint."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from flare.api.metrics import get_metrics

router = APIRouter()


@router.get(
    "/metrics",
    response_class=PlainTextResponse,
    summary="Prometheus metrics",
    description=(
        "Exposes request counts, latencies, detection stats, and LLM usage "
        "in Prometheus text exposition format. Scrapable by Prometheus/Grafana."
    ),
)
async def metrics() -> PlainTextResponse:
    """Return all collected metrics in Prometheus text format."""
    body = get_metrics().export()
    return PlainTextResponse(content=body, media_type="text/plain; version=0.0.4")

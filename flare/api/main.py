"""FastAPI application: lifespan, middleware, exception handling."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from flare import __version__
from flare.api.deps import get_settings
from flare.api.metrics import get_metrics
from flare.api.models import ErrorResponse

logger = logging.getLogger("flare.api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: startup and shutdown logic."""
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    logger.info("Flare API v%s starting up", __version__)
    logger.info(
        "Dashboard: http://localhost:%s  |  Docs: http://localhost:%s/docs",
        settings.port,
        settings.port,
    )
    yield
    logger.info("Flare API shutting down")


app = FastAPI(
    title="Flare — Log Anomaly Detection API",
    description=(
        "LLM-powered log anomaly detection and incident summarization. "
        "Upload raw logs, detect anomalies via Isolation Forest, "
        "and get plain-English incident summaries from Claude."
    ),
    version=__version__,
    lifespan=lifespan,
    responses={
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)

# ── CORS ────────────────────────────────────────────────────────
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request logging + metrics middleware ───────────────────────
@app.middleware("http")
async def request_logging(request: Request, call_next):  # type: ignore[no-untyped-def]
    """Log every request with a unique request_id and record metrics."""
    request_id = uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    path = request.url.path
    method = request.method
    logger.info("req=%s method=%s path=%s", request_id, method, path)

    m = get_metrics()
    start = time.monotonic()
    response = await call_next(request)
    duration = time.monotonic() - start

    status = str(response.status_code)
    m.inc("flare_http_requests_total", method=method, path=path, status=status)
    m.observe("flare_http_request_duration_seconds", duration, method=method, path=path)

    logger.info("req=%s status=%s duration=%.3fs", request_id, status, duration)
    response.headers["X-Request-ID"] = request_id
    return response


# ── Global exception handler ───────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch unhandled exceptions — never leak stack traces to clients."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.exception("req=%s Unhandled exception: %s", request_id, exc)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": ""},
    )


# ── Register routes ────────────────────────────────────────────
from flare.api.routes.demo import router as demo_router  # noqa: E402
from flare.api.routes.detect import router as detect_router  # noqa: E402
from flare.api.routes.health import router as health_router  # noqa: E402
from flare.api.routes.metrics import router as metrics_router  # noqa: E402
from flare.api.routes.summarize import router as summarize_router  # noqa: E402

app.include_router(health_router, tags=["Health"])
app.include_router(metrics_router, tags=["Observability"])
app.include_router(detect_router, tags=["Detection"])
app.include_router(summarize_router, tags=["Summarization"])
app.include_router(demo_router, tags=["Demo"])

# ── Root redirect to dashboard ────────────────────────────────
@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    """Redirect root to the dashboard."""
    return RedirectResponse(url="/dashboard")


# ── Serve dashboard as static files ────────────────────────────
_dashboard_dir = Path(__file__).resolve().parent.parent.parent / "dashboard"
if _dashboard_dir.is_dir():
    app.mount("/dashboard", StaticFiles(directory=str(_dashboard_dir), html=True))

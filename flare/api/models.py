"""Pydantic request/response models for the Flare API."""

from __future__ import annotations

from pydantic import BaseModel, Field

from flare.llm.schemas import LLMSummary, UsageStats  # noqa: F401 (used in type hints)

# ── Request models ──────────────────────────────────────────────


class DetectRequest(BaseModel):
    """Request body for POST /detect."""

    log_text: str = Field(
        description="Raw log text to analyze (newline-separated log lines).",
        min_length=1,
        json_schema_extra={
            "example": (
                "081109 203518 148 INFO dfs.DataNode$DataXceiver: "
                "Receiving block blk_-123 src: /10.0.0.1 dest: /10.0.0.2"
            )
        },
    )
    contamination: float = Field(
        default=0.03,
        ge=0.0,
        le=0.5,
        description="Expected anomaly ratio for Isolation Forest (0.0-0.5).",
    )


class SummarizeRequest(BaseModel):
    """Request body for POST /summarize."""

    incidents: list[IncidentPayload] = Field(
        description="Incident objects from the /detect response."
    )
    run_eval: bool = Field(
        default=False,
        description="If true, run LLM-as-judge quality scoring on each summary.",
    )


class AnalyzeRequest(BaseModel):
    """Request body for POST /analyze (end-to-end pipeline)."""

    log_text: str = Field(
        description="Raw log text to analyze.",
        min_length=1,
    )
    contamination: float = Field(default=0.03, ge=0.0, le=0.5)
    run_eval: bool = Field(
        default=False,
        description="Run quality evaluation on LLM summaries.",
    )


# ── Shared payload models ───────────────────────────────────────


class IncidentPayload(BaseModel):
    """Incident data as exchanged over the API."""

    incident_id: int
    block_ids: list[str]
    severity: float
    mean_anomaly_score: float = 0.0
    log_lines: list[str] = Field(default_factory=list)
    templates: list[str] = Field(default_factory=list)
    time_range: list[str] = Field(default_factory=lambda: ["", ""])

    model_config = {"json_schema_extra": {
        "example": {
            "incident_id": 0,
            "block_ids": ["blk_-3544583377289625738"],
            "severity": 0.231,
            "mean_anomaly_score": -0.231,
            "log_lines": ["Receiving block blk_-3544583377289625738 src: /10.0.0.1"],
            "templates": ["Receiving block <*> src: <*> dest: <*>"],
            "time_range": ["081109 203518", "081109 203520"],
        }
    }}


class QualityScorePayload(BaseModel):
    """Quality score from LLM-as-judge evaluation."""

    relevance: int = Field(ge=1, le=5)
    specificity: int = Field(ge=1, le=5)
    actionability: int = Field(ge=1, le=5)
    reasoning: str


class EvalReport(BaseModel):
    """Aggregated LLM evaluation report."""

    mean_relevance: float
    mean_specificity: float
    mean_actionability: float
    mean_quality: float
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float
    mean_latency_ms: float
    num_incidents_evaluated: int


# ── Response models ─────────────────────────────────────────────


class DetectResponse(BaseModel):
    """Response body for POST /detect."""

    incidents: list[IncidentPayload]
    anomaly_count: int
    total_blocks: int
    total_events: int
    templates_discovered: int
    processing_time_ms: int


class SummaryPayload(BaseModel):
    """A single summarized incident in the response."""

    incident_id: int
    block_ids: list[str]
    severity_score: float
    llm_summary: LLMSummary
    usage: UsageStats


class SummarizeResponse(BaseModel):
    """Response body for POST /summarize."""

    summaries: list[SummaryPayload]
    eval_results: EvalReport | None = None
    total_tokens: int
    estimated_cost_usd: float


class AnalyzeResponse(BaseModel):
    """Response body for POST /analyze (full pipeline)."""

    detection: DetectResponse
    summaries: list[SummaryPayload]
    eval_results: EvalReport | None = None
    total_tokens: int
    estimated_cost_usd: float
    total_processing_time_ms: int


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str = "ok"
    anthropic_api: str = Field(description="reachable or unreachable")
    model: str
    version: str


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: str = ""


# Fix forward reference — SummarizeRequest references IncidentPayload
SummarizeRequest.model_rebuild()

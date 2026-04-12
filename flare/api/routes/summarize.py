"""Summarization and full-pipeline endpoints."""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException

from flare.api.deps import get_settings
from flare.api.models import (
    AnalyzeRequest,
    AnalyzeResponse,
    EvalReport,
    IncidentPayload,
    SummarizeRequest,
    SummarizeResponse,
    SummaryPayload,
)
from flare.api.routes.detect import _run_detection

logger = logging.getLogger("flare.api.summarize")
router = APIRouter()


def _to_incident(payload: IncidentPayload):  # type: ignore[no-untyped-def]
    """Convert API payload to internal Incident dataclass."""
    from flare.clustering.clusterer import Incident

    return Incident(
        incident_id=payload.incident_id,
        block_ids=payload.block_ids,
        severity=payload.severity,
        anomaly_scores=[payload.mean_anomaly_score],
        log_lines=payload.log_lines,
        templates=payload.templates,
        time_range=(payload.time_range[0], payload.time_range[1])
        if len(payload.time_range) == 2
        else ("", ""),
    )


def _run_summarization(
    incident_payloads: list[IncidentPayload],
    run_eval: bool,
    mlflow_run_id: str | None = None,
) -> SummarizeResponse:
    """Run LLM summarization (and optional eval) on incidents."""
    settings = get_settings()

    if not settings.anthropic_api_key:
        raise HTTPException(
            status_code=503,
            detail="ANTHROPIC_API_KEY is not configured.",
        )

    from flare.llm.client import AnthropicClient
    from flare.llm.schemas import QualityScore, UsageStats
    from flare.llm.summarizer import IncidentSummarizer

    client = AnthropicClient(
        api_key=settings.anthropic_api_key,
        model=settings.anthropic_model,
    )
    summarizer = IncidentSummarizer(client=client)

    incidents = [_to_incident(p) for p in incident_payloads]
    summarized = summarizer.summarize_all(incidents)

    summaries = [
        SummaryPayload(
            incident_id=s.incident_id,
            block_ids=s.block_ids,
            severity_score=s.severity_score,
            llm_summary=s.llm_summary,
            usage=s.usage,
        )
        for s in summarized
    ]

    eval_results = None
    eval_usage_list: list[UsageStats] = []

    if run_eval:
        quality_scores: list[QualityScore] = []
        for s, inc in zip(summarized, incidents):
            score, usage = summarizer.evaluate_quality(inc, s.llm_summary)
            quality_scores.append(score)
            eval_usage_list.append(usage)

        from flare.eval.benchmark import Benchmark

        bench = Benchmark()
        llm_eval = bench.evaluate_llm(
            summarized, quality_scores, eval_usage_list, run_id=mlflow_run_id
        )
        eval_results = EvalReport(
            mean_relevance=round(llm_eval.mean_relevance, 2),
            mean_specificity=round(llm_eval.mean_specificity, 2),
            mean_actionability=round(llm_eval.mean_actionability, 2),
            mean_quality=round(llm_eval.mean_quality, 2),
            total_input_tokens=llm_eval.total_input_tokens,
            total_output_tokens=llm_eval.total_output_tokens,
            total_cost_usd=round(llm_eval.total_cost_usd, 6),
            mean_latency_ms=round(llm_eval.mean_latency_ms, 1),
            num_incidents_evaluated=len(quality_scores),
        )

    all_usage = [s.usage for s in summarized] + eval_usage_list
    total_tokens = sum(u.input_tokens + u.output_tokens for u in all_usage)
    total_cost = sum(u.estimated_cost_usd for u in all_usage)

    # Record LLM metrics
    from flare.api.metrics import get_metrics

    m = get_metrics()
    m.inc("flare_llm_requests_total", value=len(summarized))
    m.inc("flare_llm_tokens_total", value=total_tokens, type="all")
    for u in all_usage:
        m.inc("flare_llm_tokens_total", value=u.input_tokens, type="input")
        m.inc("flare_llm_tokens_total", value=u.output_tokens, type="output")
        m.observe("flare_llm_latency_seconds", u.latency_ms / 1000.0)
    m.inc("flare_llm_cost_usd_total", value=total_cost)

    return SummarizeResponse(
        summaries=summaries,
        eval_results=eval_results,
        total_tokens=total_tokens,
        estimated_cost_usd=round(total_cost, 6),
    )


@router.post(
    "/summarize",
    response_model=SummarizeResponse,
    summary="Summarize incidents with LLM",
    description=(
        "Send detected incidents to Claude for plain-English explanation, "
        "severity assessment, root cause analysis, and remediation steps."
    ),
)
async def summarize(request: SummarizeRequest) -> SummarizeResponse:
    """Generate LLM summaries for detected incidents."""
    logger.info(
        "Summarize request: %d incidents, eval=%s",
        len(request.incidents),
        request.run_eval,
    )
    return _run_summarization(request.incidents, request.run_eval)


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Full pipeline: detect + summarize",
    description=(
        "End-to-end pipeline: parse logs, detect anomalies, cluster incidents, "
        "and summarize each with Claude. Convenience endpoint combining /detect "
        "and /summarize."
    ),
)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """Run the full detection + summarization pipeline."""
    logger.info(
        "Analyze request: %d chars, contamination=%.3f, eval=%s",
        len(request.log_text),
        request.contamination,
        request.run_eval,
    )

    overall_start = time.monotonic()

    # Step 1: Detection
    detection = _run_detection(request.log_text, request.contamination, request.use_registry)

    if not detection.incidents:
        return AnalyzeResponse(
            detection=detection,
            summaries=[],
            eval_results=None,
            total_tokens=0,
            estimated_cost_usd=0.0,
            total_processing_time_ms=detection.processing_time_ms,
        )

    # Step 2: Summarization
    summarize_resp = _run_summarization(
        detection.incidents, request.run_eval, mlflow_run_id=detection.mlflow_run_id
    )

    total_ms = int((time.monotonic() - overall_start) * 1000)

    return AnalyzeResponse(
        detection=detection,
        summaries=summarize_resp.summaries,
        eval_results=summarize_resp.eval_results,
        total_tokens=summarize_resp.total_tokens,
        estimated_cost_usd=summarize_resp.estimated_cost_usd,
        total_processing_time_ms=total_ms,
    )

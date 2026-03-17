"""Tests for LLM evaluation metrics in the benchmark module."""

import pytest

from flare.eval.benchmark import Benchmark, LLMEvalResult
from flare.llm.schemas import (
    LLMSummary,
    QualityScore,
    SummarizedIncident,
    UsageStats,
)


def _make_summarized_incident(
    incident_id: int = 0,
    input_tokens: int = 800,
    output_tokens: int = 350,
    latency_ms: float = 1500.0,
    cost: float = 0.00765,
) -> SummarizedIncident:
    summary = LLMSummary(
        incident_id=incident_id,
        explanation="Test explanation",
        severity="high",
        severity_reasoning="Test reasoning",
        remediation=[],
        root_cause="Test root cause",
        confidence=0.85,
        confidence_reasoning="Test confidence",
    )
    return SummarizedIncident(
        incident_id=incident_id,
        block_ids=["blk_1"],
        severity_score=0.5,
        llm_summary=summary,
        usage=UsageStats(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            estimated_cost_usd=cost,
        ),
    )


class TestLLMEvalResult:
    def test_to_dict(self) -> None:
        result = LLMEvalResult(
            mean_relevance=4.5,
            mean_specificity=3.5,
            mean_actionability=4.0,
            mean_quality=4.0,
            total_input_tokens=5000,
            total_output_tokens=2000,
            total_cost_usd=0.045,
            mean_latency_ms=1200.0,
        )
        d = result.to_dict()
        assert d["mean_quality"] == 4.0
        assert d["total_input_tokens"] == 5000
        assert "num_incidents_evaluated" in d


class TestBenchmarkEvaluateLLM:
    def test_evaluate_single_incident(self) -> None:
        bench = Benchmark()
        summarized = [_make_summarized_incident()]
        quality_scores = [
            QualityScore(
                relevance=5,
                specificity=4,
                actionability=4,
                reasoning="Good analysis",
            )
        ]
        eval_usage = [
            UsageStats(
                input_tokens=600,
                output_tokens=200,
                latency_ms=1000.0,
                estimated_cost_usd=0.005,
            )
        ]

        result = bench.evaluate_llm(summarized, quality_scores, eval_usage)

        assert result.mean_relevance == 5.0
        assert result.mean_specificity == 4.0
        assert result.mean_actionability == 4.0
        assert result.mean_quality == pytest.approx(13 / 3)
        # Total tokens = summarization (800+350) + eval (600+200)
        assert result.total_input_tokens == 1400
        assert result.total_output_tokens == 550

    def test_evaluate_multiple_incidents(self) -> None:
        bench = Benchmark()
        summarized = [
            _make_summarized_incident(0, input_tokens=800, output_tokens=300, cost=0.007),
            _make_summarized_incident(1, input_tokens=900, output_tokens=400, cost=0.009),
        ]
        quality_scores = [
            QualityScore(relevance=5, specificity=4, actionability=3, reasoning="ok"),
            QualityScore(relevance=3, specificity=2, actionability=5, reasoning="ok"),
        ]
        eval_usage = [
            UsageStats(input_tokens=500, output_tokens=150, estimated_cost_usd=0.003),
            UsageStats(input_tokens=500, output_tokens=150, estimated_cost_usd=0.003),
        ]

        result = bench.evaluate_llm(summarized, quality_scores, eval_usage)

        assert result.mean_relevance == 4.0  # (5+3)/2
        assert result.mean_specificity == 3.0  # (4+2)/2
        assert result.mean_actionability == 4.0  # (3+5)/2

    def test_evaluate_no_quality_scores(self) -> None:
        bench = Benchmark()
        summarized = [_make_summarized_incident()]
        result = bench.evaluate_llm(summarized, [], [])

        assert result.mean_quality == 0.0
        # Still counts summarization usage
        assert result.total_input_tokens == 800

    def test_cost_aggregation(self) -> None:
        bench = Benchmark()
        summarized = [
            _make_summarized_incident(0, cost=0.01),
            _make_summarized_incident(1, cost=0.02),
        ]
        eval_usage = [
            UsageStats(estimated_cost_usd=0.005),
            UsageStats(estimated_cost_usd=0.005),
        ]

        result = bench.evaluate_llm(summarized, [], eval_usage)

        # Total = 0.01 + 0.02 + 0.005 + 0.005 = 0.04
        assert result.total_cost_usd == pytest.approx(0.04)

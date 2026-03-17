"""Tests for LLM schema validation."""

import pytest
from pydantic import ValidationError

from flare.llm.schemas import (
    LLMSummary,
    QualityScore,
    RemediationStep,
    SeverityLevel,
    SummarizedIncident,
    UsageStats,
)


class TestSeverityLevel:
    def test_values(self) -> None:
        assert SeverityLevel.LOW == "low"
        assert SeverityLevel.CRITICAL == "critical"

    def test_from_string(self) -> None:
        assert SeverityLevel("high") == SeverityLevel.HIGH


class TestRemediationStep:
    def test_valid(self) -> None:
        step = RemediationStep(action="Restart the DataNode", priority="immediate")
        assert step.action == "Restart the DataNode"
        assert step.priority == "immediate"


class TestLLMSummary:
    def _make_valid_summary(self, **overrides: object) -> dict:
        base = {
            "incident_id": 0,
            "explanation": "A connection was reset during block transfer.",
            "severity": "high",
            "severity_reasoning": "Data transfer failure with IOException.",
            "remediation": [
                {"action": "Check network connectivity", "priority": "immediate"},
                {"action": "Review DataNode logs", "priority": "short-term"},
            ],
            "root_cause": "Network instability between DataNodes.",
            "confidence": 0.85,
            "confidence_reasoning": "Clear IOException pattern in logs.",
        }
        base.update(overrides)
        return base

    def test_valid_summary(self) -> None:
        data = self._make_valid_summary()
        summary = LLMSummary(**data)
        assert summary.severity == SeverityLevel.HIGH
        assert summary.confidence == 0.85
        assert len(summary.remediation) == 2

    def test_invalid_severity(self) -> None:
        data = self._make_valid_summary(severity="catastrophic")
        with pytest.raises(ValidationError):
            LLMSummary(**data)

    def test_confidence_out_of_range(self) -> None:
        data = self._make_valid_summary(confidence=1.5)
        with pytest.raises(ValidationError):
            LLMSummary(**data)

    def test_confidence_negative(self) -> None:
        data = self._make_valid_summary(confidence=-0.1)
        with pytest.raises(ValidationError):
            LLMSummary(**data)

    def test_all_severity_levels(self) -> None:
        for level in ["low", "medium", "high", "critical"]:
            data = self._make_valid_summary(severity=level)
            summary = LLMSummary(**data)
            assert summary.severity.value == level


class TestQualityScore:
    def test_valid(self) -> None:
        score = QualityScore(
            relevance=4,
            specificity=3,
            actionability=5,
            reasoning="Good analysis overall.",
        )
        assert score.mean_score == pytest.approx(4.0)

    def test_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            QualityScore(
                relevance=6, specificity=3, actionability=5, reasoning="test"
            )

    def test_zero_not_allowed(self) -> None:
        with pytest.raises(ValidationError):
            QualityScore(
                relevance=0, specificity=3, actionability=5, reasoning="test"
            )

    def test_mean_score(self) -> None:
        score = QualityScore(
            relevance=3, specificity=3, actionability=3, reasoning="test"
        )
        assert score.mean_score == 3.0


class TestUsageStats:
    def test_defaults(self) -> None:
        usage = UsageStats()
        assert usage.input_tokens == 0
        assert usage.estimated_cost_usd == 0.0

    def test_with_values(self) -> None:
        usage = UsageStats(
            input_tokens=1000,
            output_tokens=500,
            latency_ms=1200.5,
            estimated_cost_usd=0.0105,
        )
        assert usage.input_tokens == 1000
        assert usage.latency_ms == 1200.5


class TestSummarizedIncident:
    def test_valid(self) -> None:
        summary = LLMSummary(
            incident_id=0,
            explanation="Test",
            severity="high",
            severity_reasoning="Test",
            remediation=[],
            root_cause="Test",
            confidence=0.5,
            confidence_reasoning="Test",
        )
        si = SummarizedIncident(
            incident_id=0,
            block_ids=["blk_1"],
            severity_score=0.5,
            llm_summary=summary,
        )
        assert si.usage.input_tokens == 0  # default
        assert si.block_ids == ["blk_1"]

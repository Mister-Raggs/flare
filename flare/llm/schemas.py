"""Pydantic models for LLM summarization outputs."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class SeverityLevel(StrEnum):
    """Incident severity classification."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RemediationStep(BaseModel):
    """A single suggested remediation action."""

    action: str = Field(description="What to do")
    priority: str = Field(description="Priority: immediate, short-term, or long-term")


class LLMSummary(BaseModel):
    """Structured output from LLM incident summarization.

    Attributes:
        incident_id: ID of the incident being summarized.
        explanation: Plain-English explanation of what likely happened.
        severity: Assessed severity level.
        severity_reasoning: Why this severity was assigned.
        remediation: Ordered list of suggested remediation steps.
        root_cause: Likely root cause of the incident.
        confidence: Model's confidence in the analysis (0.0-1.0).
        confidence_reasoning: Why the model assigned this confidence.
    """

    incident_id: int = Field(description="ID of the incident")
    explanation: str = Field(description="Plain-English explanation of what happened")
    severity: SeverityLevel = Field(description="Severity: low, medium, high, critical")
    severity_reasoning: str = Field(description="Why this severity was assigned")
    remediation: list[RemediationStep] = Field(
        description="Ordered remediation steps"
    )
    root_cause: str = Field(description="Likely root cause")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0-1")
    confidence_reasoning: str = Field(
        description="Why this confidence level was assigned"
    )


class QualityScore(BaseModel):
    """LLM-as-judge quality assessment of an explanation.

    Attributes:
        relevance: How relevant the explanation is to the log data (1-5).
        specificity: How specific and detailed the explanation is (1-5).
        actionability: How actionable the remediation steps are (1-5).
        reasoning: The judge's reasoning for the scores.
    """

    relevance: int = Field(ge=1, le=5, description="Relevance to log data (1-5)")
    specificity: int = Field(ge=1, le=5, description="Specificity and detail (1-5)")
    actionability: int = Field(ge=1, le=5, description="Actionability of remediation (1-5)")
    reasoning: str = Field(description="Judge reasoning for scores")

    @property
    def mean_score(self) -> float:
        """Average quality score across all dimensions."""
        return (self.relevance + self.specificity + self.actionability) / 3.0


class UsageStats(BaseModel):
    """Token usage and cost tracking for a single LLM call.

    Attributes:
        input_tokens: Number of tokens in the prompt.
        output_tokens: Number of tokens in the response.
        latency_ms: API call latency in milliseconds.
        estimated_cost_usd: Estimated cost in USD.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    estimated_cost_usd: float = 0.0


class SummarizedIncident(BaseModel):
    """An incident enriched with LLM summary and usage metadata.

    Attributes:
        incident_id: ID of the incident.
        block_ids: Block IDs in this incident.
        severity_score: Numerical severity from detection (0-1).
        llm_summary: The LLM-generated summary.
        usage: Token usage and cost stats.
    """

    incident_id: int
    block_ids: list[str]
    severity_score: float
    llm_summary: LLMSummary
    usage: UsageStats = Field(default_factory=UsageStats)

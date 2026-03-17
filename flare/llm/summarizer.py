"""Incident summarizer: generates LLM explanations for detected incidents."""

from __future__ import annotations

import logging

from flare.clustering.clusterer import Incident
from flare.llm.client import AnthropicClient, LLMResponse
from flare.llm.prompts import (
    QUALITY_EVAL_SYSTEM_PROMPT,
    QUALITY_EVAL_USER_PROMPT,
    SUMMARIZE_SYSTEM_PROMPT,
    SUMMARIZE_USER_PROMPT,
)
from flare.llm.schemas import (
    LLMSummary,
    QualityScore,
    SummarizedIncident,
    UsageStats,
)

logger = logging.getLogger(__name__)


class IncidentSummarizer:
    """Summarizes incidents using the Anthropic API.

    Takes Incident objects from the clustering stage and generates
    plain-English explanations with severity assessments and
    remediation suggestions.

    Example:
        >>> summarizer = IncidentSummarizer()
        >>> results = summarizer.summarize_all(incidents)
        >>> for r in results:
        ...     print(r.llm_summary.explanation)
    """

    def __init__(self, client: AnthropicClient | None = None) -> None:
        """Initialize the summarizer.

        Args:
            client: Anthropic API client. If None, creates one from env vars.
        """
        self._client = client or AnthropicClient()

    def summarize(self, incident: Incident) -> SummarizedIncident:
        """Generate an LLM summary for a single incident.

        Args:
            incident: The incident to summarize.

        Returns:
            SummarizedIncident with the LLM's analysis.
        """
        user_prompt = self._build_prompt(incident)

        response = self._client.complete(
            system=SUMMARIZE_SYSTEM_PROMPT,
            user=user_prompt,
        )

        summary = self._parse_summary(response, incident.incident_id)

        return SummarizedIncident(
            incident_id=incident.incident_id,
            block_ids=incident.block_ids,
            severity_score=incident.severity,
            llm_summary=summary,
            usage=response.usage,
        )

    def summarize_all(self, incidents: list[Incident]) -> list[SummarizedIncident]:
        """Generate LLM summaries for all incidents.

        Args:
            incidents: List of incidents to summarize.

        Returns:
            List of SummarizedIncident objects.
        """
        results: list[SummarizedIncident] = []
        for incident in incidents:
            logger.info("Summarizing incident %d...", incident.incident_id)
            result = self.summarize(incident)
            results.append(result)
        return results

    def evaluate_quality(
        self,
        incident: Incident,
        summary: LLMSummary,
    ) -> tuple[QualityScore, UsageStats]:
        """Use a second LLM call to evaluate the quality of a summary.

        Args:
            incident: The original incident.
            summary: The LLM summary to evaluate.

        Returns:
            Tuple of (QualityScore, UsageStats for the eval call).
        """
        remediation_text = "\n".join(
            f"- [{step.priority}] {step.action}"
            for step in summary.remediation
        )

        user_prompt = QUALITY_EVAL_USER_PROMPT.format(
            log_lines="\n".join(incident.log_lines[:30]),
            explanation=summary.explanation,
            severity=summary.severity.value,
            root_cause=summary.root_cause,
            remediation_steps=remediation_text,
        )

        response = self._client.complete(
            system=QUALITY_EVAL_SYSTEM_PROMPT,
            user=user_prompt,
        )

        score = QualityScore(**response.content)
        return score, response.usage

    def _build_prompt(self, incident: Incident) -> str:
        """Build the user prompt for an incident."""
        import numpy as np

        log_lines_text = "\n".join(
            f"  {line}" for line in incident.log_lines[:50]
        )
        templates_text = "\n".join(
            f"  - {tmpl}" for tmpl in incident.templates
        )

        return SUMMARIZE_USER_PROMPT.format(
            incident_id=incident.incident_id,
            time_range_start=incident.time_range[0],
            time_range_end=incident.time_range[1],
            block_ids=", ".join(incident.block_ids),
            mean_anomaly_score=float(np.mean(incident.anomaly_scores)),
            severity=incident.severity,
            log_line_count=len(incident.log_lines),
            log_lines=log_lines_text or "  (no log lines available)",
            templates=templates_text or "  (no templates available)",
        )

    def _parse_summary(
        self, response: LLMResponse, incident_id: int
    ) -> LLMSummary:
        """Parse the LLM response into a validated LLMSummary."""
        data = response.content
        # Ensure incident_id matches
        data["incident_id"] = incident_id
        return LLMSummary(**data)

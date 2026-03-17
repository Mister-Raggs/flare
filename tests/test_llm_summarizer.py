"""Tests for the LLM summarizer with mocked Anthropic API."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from flare.clustering.clusterer import Incident
from flare.llm.client import AnthropicClient, LLMResponse
from flare.llm.schemas import LLMSummary, QualityScore, UsageStats
from flare.llm.summarizer import IncidentSummarizer


def _make_incident(incident_id: int = 0) -> Incident:
    """Create a test incident with realistic data."""
    return Incident(
        incident_id=incident_id,
        block_ids=["blk_-3544583377289625738"],
        severity=0.231,
        anomaly_scores=[-0.231],
        log_lines=[
            "Receiving block blk_-3544583377289625738 "
            "src: /10.250.14.224:42458 dest: /10.250.14.224:50010",
            "10.250.14.224:50010:DataXceiver: "
            "java.io.IOException: Connection reset by peer",
            "writeBlock blk_-3544583377289625738 received exception "
            "java.io.IOException: Connection reset by peer",
        ],
        templates=[
            "Receiving block <*> src: <*> dest: <*>",
            "<*> java.io.IOException: Connection reset by peer",
            "writeBlock <*> received exception java.io.IOException: Connection reset by peer",
        ],
        time_range=("081109 204005", "081109 204006"),
    )


def _mock_summary_response() -> dict:
    """Return a realistic LLM summary response."""
    return {
        "incident_id": 0,
        "explanation": (
            "A block transfer operation failed due to a network connection "
            "being reset by the remote peer. The DataXceiver thread encountered "
            "a java.io.IOException while receiving block data."
        ),
        "severity": "high",
        "severity_reasoning": (
            "The IOException indicates an active failure in block replication. "
            "This could lead to under-replicated blocks and potential data loss "
            "if the block has insufficient replicas."
        ),
        "remediation": [
            {
                "action": "Check network connectivity between DataNodes 10.250.14.224 and peers",
                "priority": "immediate",
            },
            {
                "action": "Verify block replication status for blk_-3544583377289625738",
                "priority": "immediate",
            },
            {
                "action": "Review DataNode heap and thread dumps for resource exhaustion",
                "priority": "short-term",
            },
        ],
        "root_cause": (
            "Network instability or resource exhaustion on the remote DataNode "
            "caused the TCP connection to be reset during block transfer."
        ),
        "confidence": 0.85,
        "confidence_reasoning": (
            "The IOException: Connection reset by peer pattern is well-known "
            "and clearly visible in the logs. High confidence in the diagnosis."
        ),
    }


def _mock_quality_response() -> dict:
    """Return a realistic quality evaluation response."""
    return {
        "relevance": 5,
        "specificity": 4,
        "actionability": 4,
        "reasoning": (
            "The analysis correctly identifies the IOException "
            "pattern and provides specific remediation steps."
        ),
    }


class TestIncidentSummarizer:
    def _make_mock_client(
        self, response_data: dict, usage: UsageStats | None = None
    ) -> MagicMock:
        """Create a mock AnthropicClient that returns a preset response."""
        if usage is None:
            usage = UsageStats(
                input_tokens=800,
                output_tokens=350,
                latency_ms=1500.0,
                estimated_cost_usd=0.00765,
            )
        mock_response = LLMResponse(
            content=response_data,
            raw_text=json.dumps(response_data),
            usage=usage,
        )
        mock_client = MagicMock(spec=AnthropicClient)
        mock_client.complete.return_value = mock_response
        return mock_client

    def test_summarize_single_incident(self) -> None:
        mock_client = self._make_mock_client(_mock_summary_response())
        summarizer = IncidentSummarizer(client=mock_client)
        incident = _make_incident()

        result = summarizer.summarize(incident)

        assert result.incident_id == 0
        assert result.llm_summary.severity.value == "high"
        assert result.llm_summary.confidence == 0.85
        assert len(result.llm_summary.remediation) == 3
        assert result.usage.input_tokens == 800
        mock_client.complete.assert_called_once()

    def test_summarize_all(self) -> None:
        mock_client = self._make_mock_client(_mock_summary_response())
        summarizer = IncidentSummarizer(client=mock_client)
        incidents = [_make_incident(0), _make_incident(1)]

        results = summarizer.summarize_all(incidents)

        assert len(results) == 2
        assert mock_client.complete.call_count == 2
        # incident_id should be overridden to match
        assert results[0].incident_id == 0
        assert results[1].incident_id == 1

    def test_summarize_uses_correct_prompt(self) -> None:
        mock_client = self._make_mock_client(_mock_summary_response())
        summarizer = IncidentSummarizer(client=mock_client)
        incident = _make_incident()

        summarizer.summarize(incident)

        call_args = mock_client.complete.call_args
        user_prompt = call_args.kwargs.get("user") or call_args[1].get("user")
        # Verify the prompt contains incident data
        assert "blk_-3544583377289625738" in user_prompt
        assert "IOException" in user_prompt
        assert "Incident #0" in user_prompt

    def test_evaluate_quality(self) -> None:
        mock_client = self._make_mock_client(_mock_quality_response())
        summarizer = IncidentSummarizer(client=mock_client)
        incident = _make_incident()

        summary = LLMSummary(**_mock_summary_response())
        score, usage = summarizer.evaluate_quality(incident, summary)

        assert isinstance(score, QualityScore)
        assert score.relevance == 5
        assert score.specificity == 4
        assert score.actionability == 4
        assert score.mean_score == pytest.approx(13 / 3)

    def test_prompt_truncates_long_logs(self) -> None:
        """Verify that very long log lines lists are truncated in the prompt."""
        mock_client = self._make_mock_client(_mock_summary_response())
        summarizer = IncidentSummarizer(client=mock_client)

        incident = _make_incident()
        incident.log_lines = [f"Log line {i}" for i in range(100)]

        summarizer.summarize(incident)

        call_args = mock_client.complete.call_args
        user_prompt = call_args.kwargs.get("user") or call_args[1].get("user")
        # Should only include first 50 lines
        assert "Log line 49" in user_prompt
        assert "Log line 50" not in user_prompt


class TestAnthropicClient:
    @patch("flare.llm.client.anthropic.Anthropic")
    def test_parse_json_clean(self, mock_anthropic_cls: MagicMock) -> None:
        """Test JSON parsing from clean response."""
        client = AnthropicClient(api_key="test-key")
        result = client._parse_json('{"key": "value"}')
        assert result == {"key": "value"}

    @patch("flare.llm.client.anthropic.Anthropic")
    def test_parse_json_with_fences(self, mock_anthropic_cls: MagicMock) -> None:
        """Test JSON parsing with markdown code fences."""
        client = AnthropicClient(api_key="test-key")
        result = client._parse_json('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    @patch("flare.llm.client.anthropic.Anthropic")
    def test_cost_estimation(self, mock_anthropic_cls: MagicMock) -> None:
        """Test USD cost calculation."""
        client = AnthropicClient(api_key="test-key")
        cost = client._estimate_cost(input_tokens=1_000_000, output_tokens=100_000)
        # 1M input * $3/M + 100K output * $15/M = $3 + $1.5 = $4.5
        assert cost == pytest.approx(4.5, abs=0.01)

    def test_missing_api_key_raises(self) -> None:
        """Test that missing API key raises ValueError."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                AnthropicClient()

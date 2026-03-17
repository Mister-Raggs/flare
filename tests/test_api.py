"""Integration tests for the FastAPI REST API.

All LLM calls are mocked — no real Anthropic API calls are made.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from flare.api.main import app

client = TestClient(app)

# Minimal HDFS log text with both normal and anomalous patterns
SAMPLE_LOG = """\
081109 203518 148 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687919862906 src: /10.250.19.102:54106 dest: /10.250.19.102:50010
081109 203518 148 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687919862906 src: /10.250.8.203:47900 dest: /10.250.8.203:50010
081109 203519 148 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_-1608999687919862906 terminating
081109 203519 148 INFO dfs.DataNode$PacketResponder: PacketResponder 0 for block blk_-1608999687919862906 terminating
081109 203519 148 INFO dfs.DataNode$PacketResponder: Received block blk_-1608999687919862906 of size 67108864 from /10.250.19.102
081109 203520 35 INFO dfs.DataNode$DataXceiver: Receiving block blk_7503483334202473044 src: /10.250.14.224:42420 dest: /10.250.14.224:50010
081109 203520 35 INFO dfs.DataNode$DataXceiver: Receiving block blk_7503483334202473044 src: /10.250.8.203:47898 dest: /10.250.8.203:50010
081109 203521 35 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_7503483334202473044 terminating
081109 203521 35 INFO dfs.DataNode$PacketResponder: PacketResponder 0 for block blk_7503483334202473044 terminating
081109 203521 35 INFO dfs.DataNode$PacketResponder: Received block blk_7503483334202473044 of size 67108864 from /10.250.14.224
081109 204005 89 INFO dfs.DataNode$DataXceiver: Receiving block blk_3544583377289625738 src: /10.250.11.85:34� dest: /10.250.11.85:50010
081109 204006 89 WARN dfs.DataNode$DataXceiver: IOException in BlockReceiver for block blk_3544583377289625738
081109 204007 89 ERROR dfs.DataNode$DataXceiver: Exception in receiveBlock for block blk_3544583377289625738 java.io.IOException: Connection reset by peer
081109 204100 12 INFO dfs.DataNode$DataXceiver: Receiving block blk_1111111111111111111 src: /10.0.0.1:40000 dest: /10.0.0.1:50010
081109 204101 12 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_1111111111111111111 terminating
081109 204101 12 INFO dfs.DataNode$PacketResponder: PacketResponder 0 for block blk_1111111111111111111 terminating
081109 204101 12 INFO dfs.DataNode$PacketResponder: Received block blk_1111111111111111111 of size 67108864 from /10.0.0.1
081109 204200 44 INFO dfs.DataNode$DataXceiver: Receiving block blk_2222222222222222222 src: /10.0.0.2:40001 dest: /10.0.0.2:50010
081109 204201 44 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_2222222222222222222 terminating
081109 204201 44 INFO dfs.DataNode$PacketResponder: PacketResponder 0 for block blk_2222222222222222222 terminating
081109 204201 44 INFO dfs.DataNode$PacketResponder: Received block blk_2222222222222222222 of size 67108864 from /10.0.0.2
"""  # noqa: E501


# ── Health endpoint ──────────────────────────────────────────


class TestHealth:
    """Tests for GET /health."""

    def test_health_returns_ok(self) -> None:
        """Health endpoint returns status ok."""
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert data["anthropic_api"] in ("reachable", "unreachable")

    def test_health_includes_model(self) -> None:
        """Health endpoint includes model info."""
        resp = client.get("/health")
        data = resp.json()
        assert "model" in data
        assert len(data["model"]) > 0


# ── Detection endpoint ───────────────────────────────────────


class TestDetect:
    """Tests for POST /detect."""

    def test_detect_with_log_text(self) -> None:
        """Detection returns incidents from valid log text."""
        resp = client.post(
            "/detect",
            json={"log_text": SAMPLE_LOG, "contamination": 0.15},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "incidents" in data
        assert "anomaly_count" in data
        assert "total_blocks" in data
        assert "total_events" in data
        assert "processing_time_ms" in data
        assert data["total_events"] > 0
        assert data["total_blocks"] > 0

    def test_detect_returns_incident_structure(self) -> None:
        """Each incident has required fields."""
        resp = client.post(
            "/detect",
            json={"log_text": SAMPLE_LOG, "contamination": 0.15},
        )
        data = resp.json()
        if data["incidents"]:
            inc = data["incidents"][0]
            assert "incident_id" in inc
            assert "block_ids" in inc
            assert "severity" in inc
            assert "log_lines" in inc
            assert "templates" in inc
            assert "time_range" in inc

    def test_detect_empty_log(self) -> None:
        """Detection with no parseable content returns empty results."""
        resp = client.post(
            "/detect",
            json={"log_text": "this is not a valid log line\nanother invalid line"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["incidents"] == []
        assert data["anomaly_count"] == 0

    def test_detect_validation_error(self) -> None:
        """Detection rejects empty log_text."""
        resp = client.post("/detect", json={"log_text": ""})
        assert resp.status_code == 422

    def test_detect_upload(self) -> None:
        """Detection via file upload works."""
        resp = client.post(
            "/detect/upload",
            files={"file": ("test.log", SAMPLE_LOG.encode(), "text/plain")},
            data={"contamination": "0.15"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_events"] > 0


# ── Summarize endpoint ───────────────────────────────────────


MOCK_LLM_RESPONSE = {
    "incident_id": 0,
    "explanation": "Block transfer failed due to connection reset.",
    "severity": "high",
    "severity_reasoning": "IOException during block receive indicates data loss risk.",
    "remediation": [
        {"action": "Check network connectivity", "priority": "immediate"},
    ],
    "root_cause": "Network instability caused TCP connection reset.",
    "confidence": 0.85,
    "confidence_reasoning": "Clear error pattern in logs.",
}

MOCK_QUALITY_RESPONSE = {
    "relevance": 4,
    "specificity": 4,
    "actionability": 3,
    "reasoning": "Good explanation with specific references to log evidence.",
}


def _make_mock_client() -> MagicMock:
    """Create a mock AnthropicClient that returns expected responses."""
    from flare.llm.schemas import UsageStats

    mock_client = MagicMock()

    usage = UsageStats(
        input_tokens=500,
        output_tokens=200,
        latency_ms=150.0,
        estimated_cost_usd=0.0045,
    )

    # First call: summarization, second call: quality eval
    mock_client.complete.side_effect = [
        MagicMock(content=MOCK_LLM_RESPONSE, raw_text="{}", usage=usage),
        MagicMock(content=MOCK_QUALITY_RESPONSE, raw_text="{}", usage=usage),
    ]
    return mock_client


class TestSummarize:
    """Tests for POST /summarize."""

    def test_summarize_missing_api_key(self) -> None:
        """Summarize returns 503 when no API key is configured."""
        with patch("flare.api.routes.summarize.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                anthropic_api_key="",
                anthropic_model="claude-sonnet-4-20250514",
            )
            resp = client.post(
                "/summarize",
                json={
                    "incidents": [{
                        "incident_id": 0,
                        "block_ids": ["blk_123"],
                        "severity": 0.5,
                        "mean_anomaly_score": -0.5,
                        "log_lines": ["test log line"],
                        "templates": ["test <*>"],
                        "time_range": ["081109 203518", "081109 203520"],
                    }],
                    "run_eval": False,
                },
            )
            assert resp.status_code == 503

    @patch("flare.api.routes.summarize.get_settings")
    @patch("flare.llm.summarizer.IncidentSummarizer")
    @patch("flare.llm.client.AnthropicClient")
    def test_summarize_with_mock_llm(
        self,
        mock_client_cls: MagicMock,
        mock_summarizer_cls: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Summarize works with mocked LLM client."""
        from flare.llm.schemas import (
            LLMSummary,
            RemediationStep,
            SeverityLevel,
            SummarizedIncident,
            UsageStats,
        )

        mock_settings.return_value = MagicMock(
            anthropic_api_key="test-key",
            anthropic_model="claude-sonnet-4-20250514",
        )

        usage = UsageStats(
            input_tokens=500,
            output_tokens=200,
            latency_ms=150.0,
            estimated_cost_usd=0.0045,
        )

        mock_summary = SummarizedIncident(
            incident_id=0,
            block_ids=["blk_123"],
            severity_score=0.5,
            llm_summary=LLMSummary(
                incident_id=0,
                explanation="Block transfer failed.",
                severity=SeverityLevel.HIGH,
                severity_reasoning="IOException present.",
                remediation=[
                    RemediationStep(
                        action="Check network", priority="immediate"
                    ),
                ],
                root_cause="Network reset.",
                confidence=0.85,
                confidence_reasoning="Clear error.",
            ),
            usage=usage,
        )

        mock_summarizer = MagicMock()
        mock_summarizer.summarize_all.return_value = [mock_summary]
        mock_summarizer_cls.return_value = mock_summarizer

        resp = client.post(
            "/summarize",
            json={
                "incidents": [{
                    "incident_id": 0,
                    "block_ids": ["blk_123"],
                    "severity": 0.5,
                    "mean_anomaly_score": -0.5,
                    "log_lines": ["test log line"],
                    "templates": ["test <*>"],
                    "time_range": ["081109 203518", "081109 203520"],
                }],
                "run_eval": False,
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["summaries"]) == 1
        assert data["summaries"][0]["llm_summary"]["severity"] == "high"
        assert data["total_tokens"] > 0


# ── Analyze endpoint ─────────────────────────────────────────


class TestAnalyze:
    """Tests for POST /analyze (detection only, LLM mocked)."""

    @patch("flare.api.routes.summarize.get_settings")
    @patch("flare.llm.summarizer.IncidentSummarizer")
    @patch("flare.llm.client.AnthropicClient")
    def test_analyze_end_to_end(
        self,
        mock_client_cls: MagicMock,
        mock_summarizer_cls: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Full pipeline returns detection + summaries."""
        from flare.llm.schemas import (
            LLMSummary,
            RemediationStep,
            SeverityLevel,
            SummarizedIncident,
            UsageStats,
        )

        mock_settings.return_value = MagicMock(
            anthropic_api_key="test-key",
            anthropic_model="claude-sonnet-4-20250514",
        )

        usage = UsageStats(
            input_tokens=500,
            output_tokens=200,
            latency_ms=150.0,
            estimated_cost_usd=0.0045,
        )

        mock_summary = SummarizedIncident(
            incident_id=0,
            block_ids=["blk_123"],
            severity_score=0.5,
            llm_summary=LLMSummary(
                incident_id=0,
                explanation="Block transfer failed.",
                severity=SeverityLevel.HIGH,
                severity_reasoning="IOException present.",
                remediation=[
                    RemediationStep(
                        action="Check network", priority="immediate"
                    ),
                ],
                root_cause="Network reset.",
                confidence=0.85,
                confidence_reasoning="Clear error.",
            ),
            usage=usage,
        )

        mock_summarizer = MagicMock()
        # Return one summary per incident the pipeline finds
        mock_summarizer.summarize_all.side_effect = (
            lambda incidents: [mock_summary] * len(incidents)
        )
        mock_summarizer_cls.return_value = mock_summarizer

        resp = client.post(
            "/analyze",
            json={
                "log_text": SAMPLE_LOG,
                "contamination": 0.15,
                "run_eval": False,
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert "detection" in data
        assert "summaries" in data
        assert data["detection"]["total_events"] > 0
        assert data["total_processing_time_ms"] > 0

    def test_analyze_empty_logs(self) -> None:
        """Analyze with no parseable content returns empty results."""
        resp = client.post(
            "/analyze",
            json={"log_text": "invalid log line\nanother one"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["detection"]["incidents"] == []
        assert data["summaries"] == []


# ── OpenAPI docs ─────────────────────────────────────────────


class TestDocs:
    """Tests for auto-generated OpenAPI documentation."""

    def test_openapi_schema_available(self) -> None:
        """OpenAPI schema is accessible at /openapi.json."""
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert "paths" in schema
        assert "/detect" in schema["paths"]
        assert "/summarize" in schema["paths"]
        assert "/analyze" in schema["paths"]
        assert "/health" in schema["paths"]

    def test_docs_ui_available(self) -> None:
        """Swagger UI is accessible at /docs."""
        resp = client.get("/docs")
        assert resp.status_code == 200

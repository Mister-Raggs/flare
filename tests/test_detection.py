"""Tests for the classical anomaly detection module."""

from pathlib import Path

from flare.detection.detector import AnomalyDetector, AnomalyResult
from flare.ingestion.parser import LogParser

SAMPLE_LOG = Path(__file__).parent.parent / "logs" / "hdfs_sample.log"


class TestAnomalyResult:
    def test_to_dict(self) -> None:
        result = AnomalyResult(
            block_id="blk_123",
            is_anomaly=True,
            anomaly_score=-0.1234,
            event_count=5,
            template_ids=[0, 1, 2],
        )
        d = result.to_dict()
        assert d["block_id"] == "blk_123"
        assert d["is_anomaly"] is True
        assert d["anomaly_score"] == -0.1234
        assert d["event_count"] == 5


class TestAnomalyDetector:
    def _get_events(self) -> list:
        parser = LogParser()
        batch = parser.parse_file(SAMPLE_LOG)
        return batch.events

    def test_detect_returns_results(self) -> None:
        events = self._get_events()
        detector = AnomalyDetector(contamination=0.1)
        results = detector.detect(events)

        assert len(results) > 0
        assert all(isinstance(r, AnomalyResult) for r in results)

    def test_detect_finds_anomalies(self) -> None:
        events = self._get_events()
        detector = AnomalyDetector(contamination=0.1)
        results = detector.detect(events)

        anomalies = [r for r in results if r.is_anomaly]
        normals = [r for r in results if not r.is_anomaly]

        # With contamination=0.1 we expect some anomalies
        assert len(anomalies) > 0
        assert len(normals) > 0

    def test_detect_empty_events(self) -> None:
        detector = AnomalyDetector()
        results = detector.detect([])
        assert results == []

    def test_anomaly_scores_are_floats(self) -> None:
        events = self._get_events()
        detector = AnomalyDetector(contamination=0.1)
        results = detector.detect(events)

        for result in results:
            assert isinstance(result.anomaly_score, float)

    def test_all_blocks_have_ids(self) -> None:
        events = self._get_events()
        detector = AnomalyDetector(contamination=0.1)
        results = detector.detect(events)

        for result in results:
            assert result.block_id.startswith("blk_")
            assert result.event_count > 0

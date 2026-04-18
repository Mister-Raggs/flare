"""Tests for the classical anomaly detection module."""

from pathlib import Path

import numpy as np

from flare.detection.detector import AnomalyDetector, AnomalyResult
from flare.ingestion.models import LogEvent, LogLevel
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


def _make_event(
    line_id: int,
    template_id: int,
    block_id: str = "blk_1",
    level: LogLevel = LogLevel.INFO,
) -> LogEvent:
    return LogEvent(
        line_id=line_id,
        timestamp="081109 203615 581",
        level=level,
        component="dfs.DataNode",
        content="test log line",
        template="test <*>",
        template_id=template_id,
        block_id=block_id,
    )


class TestBuildFeatures:
    """Unit tests for the feature engineering layer."""

    def _build(self, events: list[LogEvent]) -> np.ndarray:
        d = AnomalyDetector()
        d._build_vocab(events)
        blocks = d._group_by_block(events)
        block_ids = sorted(blocks.keys())
        return d._build_features(blocks, block_ids)

    def _run(self, events: list[LogEvent]) -> tuple[AnomalyDetector, np.ndarray]:
        d = AnomalyDetector()
        d._build_vocab(events)
        blocks = d._group_by_block(events)
        mat = d._build_features(blocks, sorted(blocks.keys()))
        return d, mat

    def test_feature_matrix_width(self) -> None:
        events = [_make_event(i, i % 3) for i in range(6)]
        d, mat = self._run(events)
        assert mat.shape[1] == len(d._template_vocab) + AnomalyDetector._N_EXTRA

    def test_entropy_zero_for_single_template(self) -> None:
        events = [_make_event(i, 5) for i in range(5)]
        d, mat = self._run(events)
        assert mat[0, len(d._template_vocab) + 3] == 0.0

    def test_entropy_positive_for_mixed_templates(self) -> None:
        events = [_make_event(i, i % 3) for i in range(6)]
        d, mat = self._run(events)
        assert mat[0, len(d._template_vocab) + 3] > 0.0

    def test_repeat_ratio_all_same(self) -> None:
        # A→A→A→A: every pair is a repeat → ratio = 1.0
        events = [_make_event(i, 7) for i in range(4)]
        d, mat = self._run(events)
        assert mat[0, len(d._template_vocab) + 4] == 1.0

    def test_repeat_ratio_no_repeats(self) -> None:
        # A→B→A→B: no consecutive same-template pairs → ratio = 0.0
        events = [_make_event(i, i % 2) for i in range(4)]
        d, mat = self._run(events)
        assert mat[0, len(d._template_vocab) + 4] == 0.0

    def test_unique_bigrams(self) -> None:
        # A→B→C→A: bigrams are (A,B),(B,C),(C,A) → 3 unique
        events = [_make_event(i, [0, 1, 2, 0][i]) for i in range(4)]
        d, mat = self._run(events)
        assert mat[0, len(d._template_vocab) + 5] == 3.0

    def test_error_and_warn_counts(self) -> None:
        events = [
            _make_event(0, 1, level=LogLevel.INFO),
            _make_event(1, 2, level=LogLevel.ERROR),
            _make_event(2, 3, level=LogLevel.FATAL),
            _make_event(3, 4, level=LogLevel.WARN),
        ]
        d, mat = self._run(events)
        vocab_size = len(d._template_vocab)
        assert mat[0, vocab_size + 6] == 2.0  # ERROR + FATAL
        assert mat[0, vocab_size + 7] == 1.0  # WARN

    def test_event_span(self) -> None:
        events = [_make_event(10, 1), _make_event(20, 2), _make_event(50, 3)]
        d, mat = self._run(events)
        assert mat[0, len(d._template_vocab) + 2] == 40.0  # 50 - 10


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

    def test_last_feature_matrix_stored(self) -> None:
        events = self._get_events()
        detector = AnomalyDetector(contamination=0.1)
        detector.detect(events, track=False)
        assert detector._last_feature_matrix is not None
        vocab_size = len(detector._template_vocab)
        assert detector._last_feature_matrix.shape[1] == vocab_size + AnomalyDetector._N_EXTRA

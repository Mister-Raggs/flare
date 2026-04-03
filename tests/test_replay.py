"""Tests for the log replayer."""

from __future__ import annotations

from pathlib import Path

import pytest

from flare.replay import LogReplayer, WindowResult

SAMPLE_LOG = Path(__file__).parent.parent / "logs" / "hdfs_sample.log"


# ---------------------------------------------------------------------------
# WindowResult
# ---------------------------------------------------------------------------


class TestWindowResult:
    def test_anomaly_count(self) -> None:
        from flare.detection.detector import AnomalyResult

        r = WindowResult(
            window_index=0,
            lines_processed=10,
            anomalies=[
                AnomalyResult(
                    block_id="blk_1",
                    is_anomaly=True,
                    anomaly_score=0.6,
                    feature_vector=[],
                )
            ],
        )
        assert r.anomaly_count == 1

    def test_incident_count(self) -> None:
        r = WindowResult(window_index=0, lines_processed=10)
        assert r.incident_count == 0


# ---------------------------------------------------------------------------
# LogReplayer construction
# ---------------------------------------------------------------------------


class TestLogReplayerInit:
    def test_raises_for_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            LogReplayer("nonexistent.log")

    def test_default_params(self) -> None:
        r = LogReplayer(SAMPLE_LOG)
        assert r.rate == 10.0
        assert r.window == 50

    def test_max_speed_when_rate_zero(self) -> None:
        # rate=0 means None internally — no sleep
        r = LogReplayer(SAMPLE_LOG, rate=0)
        assert r.rate == 0


# ---------------------------------------------------------------------------
# Replay behaviour (max speed so tests don't wait)
# ---------------------------------------------------------------------------


class TestReplay:
    def test_yields_window_results(self) -> None:
        r = LogReplayer(SAMPLE_LOG, rate=0, window=20)
        results = list(r.replay())
        assert len(results) > 0
        assert all(isinstance(res, WindowResult) for res in results)

    def test_window_index_increments(self) -> None:
        r = LogReplayer(SAMPLE_LOG, rate=0, window=20)
        results = list(r.replay())
        for i, res in enumerate(results):
            assert res.window_index == i

    def test_lines_processed_matches_window_size(self) -> None:
        r = LogReplayer(SAMPLE_LOG, rate=0, window=20)
        results = list(r.replay())
        # All windows except possibly the last should have exactly window lines
        for res in results[:-1]:
            assert res.lines_processed == 20

    def test_all_lines_covered(self) -> None:
        total_lines = len(
            [line for line in SAMPLE_LOG.read_text().splitlines() if line.strip()]
        )
        r = LogReplayer(SAMPLE_LOG, rate=0, window=20)
        results = list(r.replay())
        assert sum(res.lines_processed for res in results) == total_lines

    def test_events_parsed(self) -> None:
        r = LogReplayer(SAMPLE_LOG, rate=0, window=50)
        results = list(r.replay())
        total_events = sum(len(res.events) for res in results)
        assert total_events > 0

    def test_elapsed_ms_positive(self) -> None:
        r = LogReplayer(SAMPLE_LOG, rate=0, window=50)
        results = list(r.replay())
        assert all(res.elapsed_ms >= 0 for res in results)

    def test_small_window_produces_more_results(self) -> None:
        r_small = LogReplayer(SAMPLE_LOG, rate=0, window=10)
        r_large = LogReplayer(SAMPLE_LOG, rate=0, window=50)
        assert len(list(r_small.replay())) > len(list(r_large.replay()))


# ---------------------------------------------------------------------------
# Persistence integration
# ---------------------------------------------------------------------------


class TestReplayPersistence:
    def test_persist_file_created(self, tmp_path) -> None:
        snap = str(tmp_path / "drain.bin")
        r = LogReplayer(SAMPLE_LOG, rate=0, window=50, persist_path=snap)
        list(r.replay())
        assert Path(snap).exists()

    def test_second_run_loads_existing_templates(self, tmp_path) -> None:
        snap = str(tmp_path / "drain.bin")

        r1 = LogReplayer(SAMPLE_LOG, rate=0, window=50, persist_path=snap)
        list(r1.replay())
        templates_after_first = len(r1._parser._miner.drain.clusters)

        r2 = LogReplayer(SAMPLE_LOG, rate=0, window=50, persist_path=snap)
        # Templates loaded before any lines are parsed
        assert len(r2._parser._miner.drain.clusters) == templates_after_first

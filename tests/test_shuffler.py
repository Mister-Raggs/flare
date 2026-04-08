"""Tests for the log block shuffler."""

from __future__ import annotations

from pathlib import Path

import pytest

from flare.replay.shuffler import shuffled_stream

SAMPLE_LOG = Path(__file__).parent.parent / "logs" / "hdfs_sample.log"
DEMO_LOG = Path(__file__).parent.parent / "logs" / "hdfs_demo.log"


class TestShuffledStream:
    def test_returns_list_of_strings(self) -> None:
        result = shuffled_stream(SAMPLE_LOG, n_lines=50)
        assert isinstance(result, list)
        assert all(isinstance(line, str) for line in result)

    def test_respects_n_lines_soft_limit(self) -> None:
        # Result may slightly exceed n_lines due to block boundary
        result = shuffled_stream(SAMPLE_LOG, n_lines=30)
        # Should be at least 1 line and not wildly over the limit
        assert len(result) >= 1
        assert len(result) <= 30 + 20  # max block size buffer

    def test_raises_for_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            shuffled_stream("nonexistent.log", n_lines=50)

    def test_seed_gives_reproducible_output(self) -> None:
        r1 = shuffled_stream(SAMPLE_LOG, n_lines=50, seed=42)
        r2 = shuffled_stream(SAMPLE_LOG, n_lines=50, seed=42)
        assert r1 == r2

    def test_different_seeds_give_different_output(self) -> None:
        r1 = shuffled_stream(SAMPLE_LOG, n_lines=86, seed=1)
        r2 = shuffled_stream(SAMPLE_LOG, n_lines=86, seed=2)
        # With 18 blocks there's a chance they match, but very unlikely
        assert r1 != r2

    def test_all_lines_are_non_empty(self) -> None:
        result = shuffled_stream(SAMPLE_LOG, n_lines=86)
        assert all(line.strip() for line in result)

    def test_demo_log_has_enough_blocks(self) -> None:
        if not DEMO_LOG.exists():
            pytest.skip("hdfs_demo.log not present")
        result = shuffled_stream(DEMO_LOG, n_lines=200)
        assert len(result) >= 100

    def test_n_lines_500_on_demo_log(self) -> None:
        if not DEMO_LOG.exists():
            pytest.skip("hdfs_demo.log not present")
        result = shuffled_stream(DEMO_LOG, n_lines=500)
        assert len(result) >= 100

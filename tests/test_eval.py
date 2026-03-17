"""Tests for the evaluation benchmark module."""

from pathlib import Path

import pytest

from flare.detection.detector import AnomalyResult
from flare.eval.benchmark import Benchmark, BenchmarkResult

SAMPLE_LABELS = Path(__file__).parent.parent / "logs" / "sample_labels.csv"


class TestBenchmarkResult:
    def test_to_dict(self) -> None:
        result = BenchmarkResult(
            precision=0.8,
            recall=0.9,
            f1=0.8471,
            true_positives=9,
            false_positives=2,
            false_negatives=1,
            true_negatives=10,
            total_blocks=22,
        )
        d = result.to_dict()
        assert d["precision"] == 0.8
        assert d["f1"] == 0.8471


class TestBenchmark:
    def test_load_labels(self) -> None:
        bench = Benchmark()
        labels = bench.load_labels(SAMPLE_LABELS)

        assert len(labels) > 0
        # Check known labels from sample
        assert labels["blk_-3544583377289625738"] is True  # Anomaly
        assert labels["blk_-1608999687919862906"] is False  # Normal

    def test_load_labels_not_found(self) -> None:
        bench = Benchmark()
        with pytest.raises(FileNotFoundError):
            bench.load_labels("nonexistent.csv")

    def test_evaluate_perfect(self) -> None:
        bench = Benchmark()
        labels = {"blk_1": True, "blk_2": False, "blk_3": True}
        results = [
            AnomalyResult("blk_1", is_anomaly=True, anomaly_score=-0.3),
            AnomalyResult("blk_2", is_anomaly=False, anomaly_score=0.5),
            AnomalyResult("blk_3", is_anomaly=True, anomaly_score=-0.2),
        ]
        bm = bench.evaluate(results, labels)
        assert bm.precision == 1.0
        assert bm.recall == 1.0
        assert bm.f1 == 1.0
        assert bm.true_positives == 2
        assert bm.true_negatives == 1

    def test_evaluate_all_wrong(self) -> None:
        bench = Benchmark()
        labels = {"blk_1": True, "blk_2": False}
        results = [
            AnomalyResult("blk_1", is_anomaly=False, anomaly_score=0.5),
            AnomalyResult("blk_2", is_anomaly=True, anomaly_score=-0.3),
        ]
        bm = bench.evaluate(results, labels)
        assert bm.precision == 0.0
        assert bm.recall == 0.0
        assert bm.f1 == 0.0

    def test_evaluate_skips_unknown_blocks(self) -> None:
        bench = Benchmark()
        labels = {"blk_1": True}
        results = [
            AnomalyResult("blk_1", is_anomaly=True, anomaly_score=-0.3),
            AnomalyResult("blk_unknown", is_anomaly=True, anomaly_score=-0.5),
        ]
        bm = bench.evaluate(results, labels)
        assert bm.total_blocks == 1  # only blk_1 counted

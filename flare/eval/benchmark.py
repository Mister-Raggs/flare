"""Evaluation framework: precision, recall, F1 against HDFS ground truth labels.

Also includes LLM-specific evaluation: quality rubric scoring, cost tracking,
and severity accuracy assessment.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from flare.detection.detector import AnomalyResult
from flare.llm.schemas import QualityScore, SummarizedIncident, UsageStats


@dataclass
class BenchmarkResult:
    """Benchmark metrics for an anomaly detection run.

    Attributes:
        precision: Fraction of detected anomalies that are true anomalies.
        recall: Fraction of true anomalies that were detected.
        f1: Harmonic mean of precision and recall.
        true_positives: Count of correctly detected anomalies.
        false_positives: Count of normal blocks incorrectly flagged.
        false_negatives: Count of anomalies that were missed.
        true_negatives: Count of correctly identified normal blocks.
        total_blocks: Total number of blocks evaluated.
    """

    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    total_blocks: int

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "true_negatives": self.true_negatives,
            "total_blocks": self.total_blocks,
        }


@dataclass
class LLMEvalResult:
    """Aggregated LLM evaluation metrics across all incidents.

    Attributes:
        quality_scores: Per-incident quality scores from LLM-as-judge.
        usage_stats: Per-incident token usage and cost.
        mean_relevance: Average relevance score across incidents.
        mean_specificity: Average specificity score across incidents.
        mean_actionability: Average actionability score across incidents.
        mean_quality: Overall mean quality across all dimensions.
        total_input_tokens: Total input tokens across all LLM calls.
        total_output_tokens: Total output tokens across all LLM calls.
        total_cost_usd: Total estimated cost in USD.
        mean_latency_ms: Average API call latency.
    """

    quality_scores: list[QualityScore] = field(default_factory=list)
    usage_stats: list[UsageStats] = field(default_factory=list)
    mean_relevance: float = 0.0
    mean_specificity: float = 0.0
    mean_actionability: float = 0.0
    mean_quality: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    mean_latency_ms: float = 0.0

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "mean_relevance": round(self.mean_relevance, 2),
            "mean_specificity": round(self.mean_specificity, 2),
            "mean_actionability": round(self.mean_actionability, 2),
            "mean_quality": round(self.mean_quality, 2),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "mean_latency_ms": round(self.mean_latency_ms, 1),
            "num_incidents_evaluated": len(self.quality_scores),
        }


class Benchmark:
    """Evaluates anomaly detection results against ground truth labels.

    Supports the HDFS dataset label format where each line contains
    a block ID and its label (Normal/Anomaly).

    Also provides LLM-specific evaluation: quality rubric scoring
    using LLM-as-judge and cost/latency tracking.

    Example:
        >>> bench = Benchmark()
        >>> labels = bench.load_labels("logs/anomaly_label.csv")
        >>> result = bench.evaluate(detection_results, labels)
        >>> print(f"F1: {result.f1:.3f}")
    """

    def load_labels(self, filepath: str | Path) -> dict[str, bool]:
        """Load ground truth labels from a CSV file.

        Expects a CSV with columns: BlockId, Label
        where Label is 'Anomaly' or 'Normal'.

        Args:
            filepath: Path to the labels CSV file.

        Returns:
            Dictionary mapping block_id to is_anomaly (True/False).

        Raises:
            FileNotFoundError: If the labels file does not exist.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Labels file not found: {filepath}")

        labels: dict[str, bool] = {}

        with open(filepath, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                block_id = row.get("BlockId", "").strip()
                label = row.get("Label", "").strip()
                if block_id:
                    labels[block_id] = label == "Anomaly"

        return labels

    def evaluate(
        self,
        results: list[AnomalyResult],
        labels: dict[str, bool],
        run_id: str | None = None,
    ) -> BenchmarkResult:
        """Compute precision, recall, F1 against ground truth labels.

        Only evaluates blocks that appear in both the results and labels.

        Args:
            results: Anomaly detection results.
            labels: Ground truth labels (block_id -> is_anomaly).

        Returns:
            BenchmarkResult with computed metrics.
        """
        tp = fp = fn = tn = 0

        for result in results:
            if result.block_id not in labels:
                continue

            predicted = result.is_anomaly
            actual = labels[result.block_id]

            if predicted and actual:
                tp += 1
            elif predicted and not actual:
                fp += 1
            elif not predicted and actual:
                fn += 1
            else:
                tn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        if run_id:
            import mlflow
            with mlflow.start_run(run_id=run_id):
                mlflow.log_metric("precision", round(precision, 4))
                mlflow.log_metric("recall", round(recall, 4))
                mlflow.log_metric("f1", round(f1, 4))
                mlflow.log_metric("true_positives", tp)
                mlflow.log_metric("false_positives", fp)
                mlflow.log_metric("false_negatives", fn)

        return BenchmarkResult(
            precision=precision,
            recall=recall,
            f1=f1,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            true_negatives=tn,
            total_blocks=tp + fp + fn + tn,
        )

    def evaluate_llm(
        self,
        summarized: list[SummarizedIncident],
        quality_scores: list[QualityScore],
        eval_usage: list[UsageStats],
        run_id: str | None = None,
    ) -> LLMEvalResult:
        """Compute aggregated LLM evaluation metrics.

        Args:
            summarized: List of LLM-summarized incidents.
            quality_scores: Per-incident quality scores from LLM-as-judge.
            eval_usage: Token usage for the eval calls themselves.

        Returns:
            LLMEvalResult with aggregated metrics.
        """
        # Aggregate quality scores
        if quality_scores:
            mean_rel = sum(q.relevance for q in quality_scores) / len(quality_scores)
            mean_spec = sum(q.specificity for q in quality_scores) / len(quality_scores)
            mean_act = sum(q.actionability for q in quality_scores) / len(quality_scores)
            mean_q = (mean_rel + mean_spec + mean_act) / 3.0
        else:
            mean_rel = mean_spec = mean_act = mean_q = 0.0

        # Aggregate usage from summarization calls
        all_usage = [s.usage for s in summarized] + eval_usage
        total_in = sum(u.input_tokens for u in all_usage)
        total_out = sum(u.output_tokens for u in all_usage)
        total_cost = sum(u.estimated_cost_usd for u in all_usage)
        latencies = [u.latency_ms for u in all_usage if u.latency_ms > 0]
        mean_lat = sum(latencies) / len(latencies) if latencies else 0.0

        if run_id:
            import mlflow
            with mlflow.start_run(run_id=run_id):
                mlflow.log_metric("mean_relevance", round(mean_rel, 4))
                mlflow.log_metric("mean_specificity", round(mean_spec, 4))
                mlflow.log_metric("mean_actionability", round(mean_act, 4))
                mlflow.log_metric("mean_quality", round(mean_q, 4))
                mlflow.log_metric("total_cost_usd", round(total_cost, 6))
                mlflow.log_metric("mean_latency_ms", round(mean_lat, 2))

        return LLMEvalResult(
            quality_scores=quality_scores,
            usage_stats=all_usage,
            mean_relevance=mean_rel,
            mean_specificity=mean_spec,
            mean_actionability=mean_act,
            mean_quality=mean_q,
            total_input_tokens=total_in,
            total_output_tokens=total_out,
            total_cost_usd=total_cost,
            mean_latency_ms=mean_lat,
        )

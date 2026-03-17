"""Classical anomaly detection using Isolation Forest on log features."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import IsolationForest

from flare.ingestion.models import LogEvent


@dataclass
class AnomalyResult:
    """Result of anomaly detection for a single block/session.

    Attributes:
        block_id: The log block/session identifier.
        is_anomaly: Whether this block is classified as anomalous.
        anomaly_score: Raw anomaly score from the model (lower = more anomalous).
        feature_vector: The feature vector used for detection.
        event_count: Number of log events in this block.
        template_ids: Unique template IDs observed in this block.
    """

    block_id: str
    is_anomaly: bool
    anomaly_score: float
    feature_vector: list[float] = field(default_factory=list)
    event_count: int = 0
    template_ids: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "block_id": self.block_id,
            "is_anomaly": bool(self.is_anomaly),
            "anomaly_score": round(self.anomaly_score, 4),
            "event_count": self.event_count,
            "template_ids": self.template_ids,
        }


class AnomalyDetector:
    """Detects anomalous log blocks using Isolation Forest.

    Builds feature vectors from log template frequency distributions
    per block, then applies Isolation Forest to identify outliers.

    The feature space represents each block as a vector of template
    occurrence counts, capturing the "shape" of log activity.

    Example:
        >>> detector = AnomalyDetector(contamination=0.05)
        >>> results = detector.detect(log_events)
        >>> anomalies = [r for r in results if r.is_anomaly]
    """

    def __init__(
        self,
        contamination: float = 0.03,
        n_estimators: int = 200,
        random_state: int = 42,
    ) -> None:
        """Initialize the detector.

        Args:
            contamination: Expected proportion of anomalies in the data.
            n_estimators: Number of trees in the Isolation Forest.
            random_state: Random seed for reproducibility.
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self._model: IsolationForest | None = None
        self._template_vocab: list[int] = []

    def detect(self, events: list[LogEvent]) -> list[AnomalyResult]:
        """Run anomaly detection on a list of log events.

        Groups events by block_id, builds feature vectors from template
        frequency, and applies Isolation Forest.

        Args:
            events: List of parsed log events.

        Returns:
            List of AnomalyResult, one per block.
        """
        # Group events by block_id
        blocks = self._group_by_block(events)
        if not blocks:
            return []

        # Build feature matrix
        block_ids = sorted(blocks.keys())
        self._build_vocab(events)
        feature_matrix = self._build_features(blocks, block_ids)

        # Fit and predict with Isolation Forest
        self._model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
        )
        predictions = self._model.fit_predict(feature_matrix)
        scores = self._model.decision_function(feature_matrix)

        # Build results
        results: list[AnomalyResult] = []
        for i, bid in enumerate(block_ids):
            block_events = blocks[bid]
            template_ids = sorted({e.template_id for e in block_events})
            results.append(
                AnomalyResult(
                    block_id=bid,
                    is_anomaly=predictions[i] == -1,
                    anomaly_score=float(scores[i]),
                    feature_vector=feature_matrix[i].tolist(),
                    event_count=len(block_events),
                    template_ids=template_ids,
                )
            )

        return results

    def _group_by_block(self, events: list[LogEvent]) -> dict[str, list[LogEvent]]:
        """Group log events by their block_id."""
        blocks: dict[str, list[LogEvent]] = {}
        for event in events:
            if event.block_id:
                blocks.setdefault(event.block_id, []).append(event)
        return blocks

    def _build_vocab(self, events: list[LogEvent]) -> None:
        """Build the template vocabulary from all events."""
        template_ids = {e.template_id for e in events if e.template_id >= 0}
        self._template_vocab = sorted(template_ids)

    def _build_features(
        self, blocks: dict[str, list[LogEvent]], block_ids: list[str]
    ) -> np.ndarray:
        """Build feature matrix: rows=blocks, cols=template frequencies.

        Each block is represented by a vector where each dimension
        is the count of a particular log template occurring in that block.
        Additional features include total event count and unique template count.
        """
        vocab_index = {tid: idx for idx, tid in enumerate(self._template_vocab)}
        n_blocks = len(block_ids)
        n_features = len(self._template_vocab) + 2  # +2 for count features

        matrix = np.zeros((n_blocks, n_features), dtype=np.float64)

        for i, bid in enumerate(block_ids):
            events = blocks[bid]
            counts = Counter(e.template_id for e in events if e.template_id >= 0)

            # Template frequency features
            for tid, count in counts.items():
                if tid in vocab_index:
                    matrix[i, vocab_index[tid]] = count

            # Aggregate features
            matrix[i, -2] = len(events)  # total event count
            matrix[i, -1] = len(counts)  # unique template count

        return matrix

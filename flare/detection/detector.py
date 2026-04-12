"""Classical anomaly detection using Isolation Forest on log features."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import mlflow
import mlflow.sklearn
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
        use_registry: bool = False,
    ) -> None:
        """Initialize the detector.

        Args:
            contamination: Expected proportion of anomalies in the data.
            n_estimators: Number of trees in the Isolation Forest.
            random_state: Random seed for reproducibility.
            use_registry: If True, attempt to load the Production model from
                the MLflow Model Registry instead of training a new one.
                Falls back to training if no Production model exists.
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.use_registry = use_registry
        self._model: IsolationForest | None = None
        self._template_vocab: list[int] = []
        self.mlflow_run_id: str | None = None

    def detect(self, events: list[LogEvent]) -> list[AnomalyResult]:
        """Run anomaly detection on a list of log events.

        If ``use_registry=True``, loads the Production model from the MLflow
        Model Registry and runs inference only (no retraining). Falls back to
        training if no Production model is available.

        Args:
            events: List of parsed log events.

        Returns:
            List of AnomalyResult, one per block.
        """
        blocks = self._group_by_block(events)
        if not blocks:
            return []

        block_ids = sorted(blocks.keys())
        self._build_vocab(events)
        feature_matrix = self._build_features(blocks, block_ids)

        loaded_from_registry = False
        if self.use_registry:
            loaded_from_registry = self._try_load_production(feature_matrix)

        if not loaded_from_registry:
            self._model = IsolationForest(
                contamination=self.contamination,
                n_estimators=self.n_estimators,
                random_state=self.random_state,
            )
            self._model.fit(feature_matrix)

        assert self._model is not None
        predictions = self._model.predict(feature_matrix)
        scores = self._model.decision_function(feature_matrix)

        self._log_to_mlflow(
            block_ids=block_ids,
            predictions=predictions,
            scores=scores,
            feature_matrix=feature_matrix,
            register=not loaded_from_registry,
        )

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

    def _try_load_production(self, feature_matrix: np.ndarray) -> bool:
        """Attempt to load the Production model from the MLflow registry.

        Also loads the saved vocab so the feature space matches.

        Returns:
            True if a Production model was loaded successfully, False otherwise.
        """
        import json
        import tempfile

        try:
            model = mlflow.sklearn.load_model("models:/flare-isolation-forest/Production")
        except Exception:
            return False

        # Load the vocab saved with this model version
        try:
            client = mlflow.tracking.MlflowClient()
            versions = client.search_model_versions(
                "name='flare-isolation-forest' and current_stage='Production'"
            )
            if not versions:
                return False
            run_id = versions[0].run_id or ""
            with tempfile.TemporaryDirectory() as tmp:
                local = client.download_artifacts(run_id, "vocab.json", tmp)
                with open(local) as f:
                    self._template_vocab = json.load(f)
        except Exception:
            # No vocab artifact — model was registered before vocab saving was added
            pass

        self._model = model
        return True

    def _log_to_mlflow(
        self,
        block_ids: list[str],
        predictions: np.ndarray,
        scores: np.ndarray,
        feature_matrix: np.ndarray,
        register: bool,
    ) -> None:
        """Log params, metrics, and optionally register the model."""
        import json
        import tempfile

        mlflow.set_experiment("flare-detection")
        with mlflow.start_run() as run:
            mlflow.log_param("contamination", self.contamination)
            mlflow.log_param("n_estimators", self.n_estimators)
            mlflow.log_param("random_state", self.random_state)
            mlflow.log_param("total_blocks", len(block_ids))
            mlflow.log_param("vocab_size", len(self._template_vocab))
            mlflow.log_param("from_registry", not register)

            anomaly_count = int(sum(predictions == -1))
            mlflow.log_metric("anomaly_count", anomaly_count)
            mlflow.log_metric("anomaly_rate", round(anomaly_count / len(block_ids), 4))
            mlflow.log_metric("mean_anomaly_score", round(float(scores.mean()), 4))
            mlflow.log_metric("min_anomaly_score", round(float(scores.min()), 4))

            if register:
                mlflow.sklearn.log_model(self._model, "isolation-forest")

                # Save vocab alongside model so it can be reloaded later
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False
                ) as f:
                    json.dump(self._template_vocab, f)
                    vocab_path = f.name
                mlflow.log_artifact(vocab_path, artifact_path="")
                import os
                os.unlink(vocab_path)

                mlflow.register_model(
                    f"runs:/{run.info.run_id}/isolation-forest",
                    "flare-isolation-forest",
                )

            self.mlflow_run_id = run.info.run_id

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

"""Classical anomaly detection using Isolation Forest on log features."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import IsolationForest

from flare.ingestion.models import LogEvent, LogLevel


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
        feature_set: str = "full",
    ) -> None:
        """Initialize the detector.

        Args:
            contamination: Expected proportion of anomalies in the data.
            n_estimators: Number of trees in the Isolation Forest.
            random_state: Random seed for reproducibility.
            use_registry: If True, attempt to load the Production model from
                the MLflow Model Registry instead of training a new one.
                Falls back to training if no Production model exists.
            feature_set: ``"full"`` includes all 8 sequence/level columns on
                top of template frequencies.  ``"freq_only"`` uses template
                frequency counts only — the pre-Phase-4 baseline.
        """
        if feature_set not in ("full", "freq_only"):
            raise ValueError(
                f"feature_set must be 'full' or 'freq_only', got {feature_set!r}"
            )
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.use_registry = use_registry
        self.feature_set = feature_set
        self._model: IsolationForest | None = None
        self._template_vocab: list[int] = []
        self._last_feature_matrix: np.ndarray | None = None
        self.mlflow_run_id: str | None = None

    # Number of non-vocab feature columns appended after template frequencies.
    # Layout: [template freqs × V] [event_count] [unique_templates] [span]
    #         [entropy] [repeat_ratio] [unique_bigrams] [error_count] [warn_count]
    _N_EXTRA: int = 8
    _N_EXTRA_FREQ_ONLY: int = 0

    def detect(
        self,
        events: list[LogEvent],
        track: bool = True,
        source_path: str | None = None,
    ) -> list[AnomalyResult]:
        """Run anomaly detection on a list of log events.

        If ``use_registry=True``, loads the Production model from the MLflow
        Model Registry and runs inference only (no retraining). Falls back to
        training if no Production model is available.

        Args:
            events: List of parsed log events.
            track: If False, skip MLflow logging entirely (useful for sweeps).
            source_path: Optional path to the source log file, used for
                dataset lineage tracking in MLflow.

        Returns:
            List of AnomalyResult, one per block.
        """
        blocks = self._group_by_block(events)
        if not blocks:
            return []

        block_ids = sorted(blocks.keys())
        self._build_vocab(events)
        feature_matrix = self._build_features(blocks, block_ids)
        self._last_feature_matrix = feature_matrix

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

        if track:
            self._log_to_mlflow(
                block_ids=block_ids,
                predictions=predictions,
                scores=scores,
                feature_matrix=feature_matrix,
                register=not loaded_from_registry,
                source_path=source_path,
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
            import mlflow
            import mlflow.sklearn
            import mlflow.tracking
        except ImportError:
            return False

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
                    data = json.load(f)
                # Support both old format (bare list) and new format (dict)
                if isinstance(data, list):
                    self._template_vocab = data
                else:
                    self._template_vocab = data.get("template_vocab", [])
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
        source_path: str | None = None,
    ) -> None:
        """Log params, metrics, artifacts, and optionally register the model."""
        import json
        import os
        import subprocess
        import sys
        import tempfile
        from pathlib import Path

        try:
            import mlflow
            import mlflow.sklearn
            from mlflow.models import infer_signature
        except ImportError:
            return

        mlflow.set_experiment("flare-detection")
        with mlflow.start_run() as run:
            # ── params ───────────────────────────────────────────────────────
            mlflow.log_param("contamination", self.contamination)
            mlflow.log_param("n_estimators", self.n_estimators)
            mlflow.log_param("random_state", self.random_state)
            mlflow.log_param("total_blocks", len(block_ids))
            mlflow.log_param("vocab_size", len(self._template_vocab))
            mlflow.log_param("from_registry", not register)

            # ── metrics ──────────────────────────────────────────────────────
            anomaly_count = int(sum(predictions == -1))
            mlflow.log_metric("anomaly_count", anomaly_count)
            mlflow.log_metric("anomaly_rate", round(anomaly_count / len(block_ids), 4))
            mlflow.log_metric("mean_anomaly_score", round(float(scores.mean()), 4))
            mlflow.log_metric("min_anomaly_score", round(float(scores.min()), 4))

            # ── tags: reproducibility metadata ───────────────────────────────
            try:
                git_sha = subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"],
                    stderr=subprocess.DEVNULL,
                ).decode().strip()
            except Exception:
                git_sha = "unknown"

            mlflow.set_tags({
                "git_sha": git_sha,
                "python_version": sys.version.split()[0],
                "dataset": Path(source_path).stem if source_path else "unknown",
                "experimenter": "flare-detector",
            })

            # ── dataset lineage ──────────────────────────────────────────────
            try:
                import mlflow.data
                dataset = mlflow.data.from_numpy(
                    feature_matrix,
                    source=source_path or "in-memory",
                    name=f"{Path(source_path).stem if source_path else 'log'}-features",
                )
                mlflow.log_input(dataset, context="training")
            except Exception:
                pass  # dataset logging is optional (requires mlflow >= 2.4)

            if register:
                # ── model signature ──────────────────────────────────────────
                signature = infer_signature(feature_matrix, predictions)
                mlflow.sklearn.log_model(
                    self._model,
                    "isolation-forest",
                    signature=signature,
                )

                # Save vocab + feature schema alongside model
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False
                ) as f:
                    json.dump(
                        {"template_vocab": self._template_vocab, "n_extra": self._N_EXTRA},
                        f,
                    )
                    vocab_path = f.name
                mlflow.log_artifact(vocab_path, artifact_path="")
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

    def _n_extra(self) -> int:
        """Extra columns appended after template frequencies for this feature_set."""
        return self._N_EXTRA_FREQ_ONLY if self.feature_set == "freq_only" else self._N_EXTRA

    def _build_features(
        self, blocks: dict[str, list[LogEvent]], block_ids: list[str]
    ) -> np.ndarray:
        """Build feature matrix.

        ``freq_only`` layout (V = vocab size):
          [0 … V-1]  template frequency counts only

        ``full`` layout adds 8 columns:
          [V+0]      total event count
          [V+1]      unique template count
          [V+2]      event span  (max line_id - min line_id)
          [V+3]      sequence entropy  (Shannon entropy of template ID sequence)
          [V+4]      repeat ratio  (fraction of consecutive same-template pairs)
          [V+5]      unique bigram count  (distinct A→B template transitions)
          [V+6]      error / fatal event count
          [V+7]      warn event count
        """
        vocab_index = {tid: idx for idx, tid in enumerate(self._template_vocab)}
        vocab_size = len(self._template_vocab)
        n_blocks = len(block_ids)
        n_extra = self._n_extra()
        matrix = np.zeros((n_blocks, vocab_size + n_extra), dtype=np.float64)

        for i, bid in enumerate(block_ids):
            events = blocks[bid]
            template_seq = [e.template_id for e in events if e.template_id >= 0]
            counts = Counter(template_seq)

            # ── template frequency features (always) ──────────────────────────
            for tid, count in counts.items():
                if tid in vocab_index:
                    matrix[i, vocab_index[tid]] = count

            if self.feature_set == "freq_only":
                continue

            # ── aggregate counts ──────────────────────────────────────────────
            matrix[i, vocab_size] = len(events)
            matrix[i, vocab_size + 1] = len(counts)

            # ── event span (line_id range as proxy for block duration) ────────
            line_ids = [e.line_id for e in events]
            matrix[i, vocab_size + 2] = (max(line_ids) - min(line_ids)) if len(line_ids) > 1 else 0

            if template_seq:
                total = len(template_seq)

                # ── sequence entropy ──────────────────────────────────────────
                matrix[i, vocab_size + 3] = -sum(
                    (c / total) * math.log2(c / total) for c in counts.values()
                )

                if total > 1:
                    pairs = list(zip(template_seq, template_seq[1:]))

                    # ── repeat ratio (retry-storm detector) ───────────────────
                    matrix[i, vocab_size + 4] = sum(a == b for a, b in pairs) / len(pairs)

                    # ── unique bigram transitions ─────────────────────────────
                    matrix[i, vocab_size + 5] = len(set(pairs))

            # ── log level counts ──────────────────────────────────────────────
            matrix[i, vocab_size + 6] = sum(
                1 for e in events if e.level in (LogLevel.ERROR, LogLevel.FATAL)
            )
            matrix[i, vocab_size + 7] = sum(
                1 for e in events if e.level == LogLevel.WARN
            )

        return matrix

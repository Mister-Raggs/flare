"""Inference-only model server for production serving.

Separates the training path (offline sweeps) from the serving path (API).
A ModelServer loads a pre-trained model + vocab from the MLflow registry
once at startup and serves all subsequent requests via feature building +
predict only — no fit() call on the request path.

Typical use:
    server = ModelServer.from_registry()   # once at startup
    results = server.infer(events)         # on every request

Why this matters:
    Training IF on 36k blocks takes ~10-30s per request.
    Inference on the same data takes <1s.
    Pre-loading the model at startup eliminates this entirely.
"""

from __future__ import annotations

import json
import logging
import tempfile
from typing import Any

from flare.detection.detector import AnomalyDetector, AnomalyResult
from flare.ingestion.models import LogEvent

logger = logging.getLogger("flare.detection.server")

DEFAULT_MODEL_NAME = "flare-isolation-forest"
DEFAULT_STAGE = "Production"


class ModelNotFoundError(Exception):
    """Raised when no suitable model exists in the registry."""


class ModelServer:
    """Inference-only wrapper around a pre-trained model and its vocab.

    Keeps the model in memory and builds features from the stored vocab
    so each infer() call is pure feature-building + predict — no fit().

    Attributes:
        model_name: MLflow registered model name.
        stage: Registry stage this model was loaded from.
        run_id: MLflow run ID the model artifact came from.
        feature_set: Feature engineering mode ('full' or 'freq_only').
        vocab_size: Number of template IDs in the training vocabulary.
    """

    def __init__(
        self,
        model: Any,
        vocab: list[int],
        feature_set: str = "full",
        model_name: str = DEFAULT_MODEL_NAME,
        stage: str = DEFAULT_STAGE,
        run_id: str = "",
    ) -> None:
        self._model = model
        self._vocab = vocab
        self.feature_set = feature_set
        self.model_name = model_name
        self.stage = stage
        self.run_id = run_id

        # Reuse AnomalyDetector's feature building with the stored vocab
        self._detector = AnomalyDetector(feature_set=feature_set)
        self._detector._template_vocab = vocab

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    @classmethod
    def from_registry(
        cls,
        stage: str = DEFAULT_STAGE,
        model_name: str = DEFAULT_MODEL_NAME,
    ) -> ModelServer:
        """Load the latest model at ``stage`` from the MLflow registry.

        Args:
            stage: Registry stage to load from (e.g. 'Production', 'Staging').
            model_name: Registered model name in MLflow.

        Returns:
            A ready-to-serve ModelServer.

        Raises:
            ModelNotFoundError: If no model exists at the given stage.
            ImportError: If mlflow is not installed.
        """
        try:
            import mlflow
            import mlflow.sklearn
        except ImportError as e:
            raise ImportError("mlflow required for registry serving") from e

        uri = f"models:/{model_name}/{stage}"
        try:
            model = mlflow.sklearn.load_model(uri)
        except Exception as exc:
            raise ModelNotFoundError(
                f"No model at {uri}. Run `flare model sweep --promote` first."
            ) from exc

        vocab, feature_set, run_id = cls._load_vocab(model_name, stage)
        logger.info(
            "Loaded %s/%s  run=%s  vocab=%d  feature_set=%s",
            model_name, stage, run_id[:8], len(vocab), feature_set,
        )
        return cls(
            model=model,
            vocab=vocab,
            feature_set=feature_set,
            model_name=model_name,
            stage=stage,
            run_id=run_id,
        )

    @classmethod
    def from_run(
        cls,
        run_id: str,
        model_artifact: str = "model",
        model_name: str = DEFAULT_MODEL_NAME,
    ) -> ModelServer:
        """Load a model directly from an MLflow run ID.

        Useful for serving a specific child run before it has been promoted.

        Args:
            run_id: MLflow run ID to load from.
            model_artifact: Name of the model artifact within the run.
            model_name: Used only for logging / metadata.
        """
        try:
            import mlflow
            import mlflow.sklearn
        except ImportError as e:
            raise ImportError("mlflow required for run-based serving") from e

        uri = f"runs:/{run_id}/{model_artifact}"
        try:
            model = mlflow.sklearn.load_model(uri)
        except Exception as exc:
            raise ModelNotFoundError(f"No model artifact at {uri}") from exc

        vocab, feature_set = cls._load_vocab_from_run(run_id)
        logger.info(
            "Loaded run %s  vocab=%d  feature_set=%s", run_id[:8], len(vocab), feature_set
        )
        return cls(
            model=model,
            vocab=vocab,
            feature_set=feature_set,
            model_name=model_name,
            stage="run",
            run_id=run_id,
        )

    def infer(self, events: list[LogEvent]) -> list[AnomalyResult]:
        """Run inference on a list of log events.

        Uses the stored vocab to build features — new template IDs not seen
        during training contribute zero to those columns, which is the correct
        behaviour (the model has no signal for unseen templates).

        Args:
            events: Parsed log events.

        Returns:
            List of AnomalyResult, one per block.
        """
        blocks = self._detector._group_by_block(events)
        if not blocks:
            return []

        block_ids = sorted(blocks.keys())
        feature_matrix = self._detector._build_features(blocks, block_ids)

        predictions = self._model.predict(feature_matrix)

        # Pipelines expose score_samples; bare sklearn models expose both
        if hasattr(self._model, "decision_function"):
            scores = self._model.decision_function(feature_matrix)
        else:
            scores = self._model.score_samples(feature_matrix)

        results: list[AnomalyResult] = []
        for i, bid in enumerate(block_ids):
            block_events = blocks[bid]
            template_ids = sorted({e.template_id for e in block_events})
            results.append(
                AnomalyResult(
                    block_id=bid,
                    is_anomaly=bool(predictions[i] == -1),
                    anomaly_score=float(scores[i]),
                    feature_vector=feature_matrix[i].tolist(),
                    event_count=len(block_events),
                    template_ids=template_ids,
                )
            )
        return results

    def summary(self) -> dict[str, object]:
        """Return metadata about the loaded model."""
        return {
            "model_name": self.model_name,
            "stage": self.stage,
            "run_id": self.run_id,
            "feature_set": self.feature_set,
            "vocab_size": self.vocab_size,
            "model_type": type(self._model).__name__,
        }

    # ── private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _load_vocab(
        model_name: str, stage: str
    ) -> tuple[list[int], str, str]:
        """Download vocab.json from the run associated with the staged model."""
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
        staged = [v for v in versions if v.current_stage == stage]
        if not staged:
            return [], "full", ""

        run_id = staged[0].run_id or ""
        vocab, feature_set = ModelServer._load_vocab_from_run(run_id)
        return vocab, feature_set, run_id

    @staticmethod
    def _load_vocab_from_run(run_id: str) -> tuple[list[int], str]:
        """Download and parse vocab.json for a given run ID."""
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                local = client.download_artifacts(run_id, "vocab.json", tmp)
                with open(local) as f:
                    data = json.load(f)
            if isinstance(data, list):
                return data, "full"
            vocab = data.get("template_vocab", [])
            feature_set = data.get("feature_set", "full")
            return vocab, feature_set
        except Exception:
            logger.warning("run %s has no vocab.json — vocab will be empty", run_id[:8])
            return [], "full"

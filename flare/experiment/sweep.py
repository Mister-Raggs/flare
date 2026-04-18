"""Hyperparameter sweep with MLflow nested runs.

Supports grid search across multiple anomaly detection model families
(Isolation Forest, Local Outlier Factor, One-Class SVM, Elliptic Envelope).
Each (model, params) combination is logged as a *child run* nested under a
single *parent run*, so the MLflow UI shows the full cross-model comparison
in one place and lets you sort by F1.

After all children finish the parent is annotated with the best F1, best
model class, and the child run ID that produced it.  Optionally, the best
model is registered in the MLflow Model Registry and promoted to Staging.

Example — compare all four models with their default grids:
    >>> from flare.experiment.sweep import HyperparamSweep, MODELS
    >>> sweep = HyperparamSweep(model_names=["isolation_forest", "lof", "ocsvm"])
    >>> result = sweep.run("logs/hdfs_sample.log", "logs/sample_labels.csv")
    >>> print(f"Best: {result.best_model}  F1={result.best_f1:.4f}")

Example — IF-only hyperparameter sweep (original behaviour):
    >>> sweep = HyperparamSweep(
    ...     contamination_values=[0.01, 0.03, 0.05],
    ...     n_estimators_values=[100, 200, 300],
    ... )
    >>> result = sweep.run("logs/hdfs_sample.log", "logs/sample_labels.csv")
"""

from __future__ import annotations

import itertools
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

# ── Model spec registry ────────────────────────────────────────────────────────

@dataclass
class ModelSpec:
    """One anomaly detection model family with its default sweep param grid.

    Attributes:
        name: Short key used on the CLI (e.g. ``"lof"``).
        display_name: Human-readable label for tables and MLflow tags.
        estimator_class: Sklearn-compatible class (must support fit + predict).
        default_params: Constructor kwargs always passed to the estimator
            (e.g. ``{"novelty": True}`` for LOF).
        param_grid: Mapping of param name → list of values to sweep over.
    """

    name: str
    display_name: str
    estimator_class: type
    default_params: dict[str, Any]
    param_grid: dict[str, list[Any]]


#: Built-in model registry — extend by adding entries here.
MODELS: dict[str, ModelSpec] = {
    "isolation_forest": ModelSpec(
        name="isolation_forest",
        display_name="Isolation Forest",
        estimator_class=IsolationForest,
        default_params={"random_state": 42},
        param_grid={
            "contamination": [0.01, 0.03, 0.05],
            "n_estimators": [100, 200],
        },
    ),
    "lof": ModelSpec(
        name="lof",
        display_name="Local Outlier Factor",
        estimator_class=LocalOutlierFactor,
        # novelty=True is required to call predict() on new data after fit()
        default_params={"novelty": True},
        param_grid={
            "contamination": [0.01, 0.03, 0.05],
            "n_neighbors": [10, 20],
        },
    ),
    "ocsvm": ModelSpec(
        name="ocsvm",
        display_name="One-Class SVM",
        estimator_class=OneClassSVM,
        default_params={"kernel": "rbf"},
        # nu ≈ upper bound on the fraction of outliers (analogous to contamination)
        param_grid={
            "nu": [0.01, 0.05, 0.10],
        },
    ),
    "elliptic": ModelSpec(
        name="elliptic",
        display_name="Elliptic Envelope",
        estimator_class=EllipticEnvelope,
        default_params={},
        param_grid={
            "contamination": [0.01, 0.03, 0.05],
        },
    ),
}


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class SweepResult:
    """Summary of a completed hyperparameter sweep.

    Attributes:
        best_f1: Highest F1 score achieved across all (model, params) combos.
        best_run_id: MLflow run ID of the child run with the best F1.
        best_model: Display name of the model class that produced the best F1.
        best_params: Hyperparameters that produced the best F1.
        parent_run_id: MLflow run ID of the parent sweep run.
        all_results: Per-combination metrics, sorted by F1 descending.
            Each entry has keys: run_id, model, model_key, params, f1,
            precision, recall, skipped (bool).
    """

    best_f1: float
    best_run_id: str
    best_model: str
    best_params: dict[str, Any]
    parent_run_id: str
    all_results: list[dict[str, Any]] = field(default_factory=list)


# ── Sweep class ────────────────────────────────────────────────────────────────

class HyperparamSweep:
    """Grid search over anomaly detection models and their hyperparameters.

    Logs all combinations as nested MLflow runs under a single parent.  The
    parent is annotated with ``best_f1``, ``best_model``, and ``best_run_id``
    so the outcome is visible without drilling into children.

    When ``model_names`` contains only ``"isolation_forest"`` (the default),
    the ``contamination_values`` and ``n_estimators_values`` arguments let you
    customise the IF grid — matching the original single-model sweep interface.
    For all other models their default :data:`MODELS` param grids are used.

    Args:
        model_names: One or more keys from :data:`MODELS` to compare.
            Defaults to ``["isolation_forest"]``.
        contamination_values: Override contamination grid for Isolation Forest.
        n_estimators_values: Override n_estimators grid for Isolation Forest.
        experiment_name: MLflow experiment to log into.
        model_registry_name: Registry name used when ``promote_best=True``.
    """

    EXPERIMENT_NAME = "flare-sweep"
    MODEL_REGISTRY_NAME = "flare-isolation-forest"

    def __init__(
        self,
        model_names: list[str] | None = None,
        contamination_values: list[float] | None = None,
        n_estimators_values: list[int] | None = None,
        experiment_name: str = EXPERIMENT_NAME,
        model_registry_name: str = MODEL_REGISTRY_NAME,
    ) -> None:
        self.model_names = model_names or ["isolation_forest"]
        for name in self.model_names:
            if name not in MODELS:
                valid = list(MODELS.keys())
                raise ValueError(f"Unknown model '{name}'. Valid options: {valid}")
        self.contamination_values = contamination_values
        self.n_estimators_values = n_estimators_values
        self.experiment_name = experiment_name
        self.model_registry_name = model_registry_name

    def run(
        self,
        log_path: str | Path,
        labels_path: str | Path,
        promote_best: bool = False,
    ) -> SweepResult:
        """Execute the grid search and log results as nested MLflow runs.

        Logs are parsed once and reused across all child runs.  Each child
        trains a fresh estimator, evaluates against ground truth labels, and
        logs params + F1/precision/recall + a confusion matrix PNG.

        Args:
            log_path: Path to the raw log file to train on.
            labels_path: Path to the ground truth labels CSV.
            promote_best: If True, register the best child's model and promote
                it to Staging in the MLflow Model Registry.

        Returns:
            :class:`SweepResult` with best outcome and per-combination detail.

        Raises:
            RuntimeError: If mlflow is not installed.
        """
        try:
            import mlflow
            import mlflow.sklearn
            from mlflow.models import infer_signature
        except ImportError:
            raise RuntimeError(
                "mlflow is required for sweep. "
                "Install with: pip install 'flare-log-analyzer[tracking]'"
            )

        from flare.detection.detector import AnomalyDetector
        from flare.eval.benchmark import Benchmark
        from flare.ingestion.parser import LogParser

        log_path = Path(log_path)
        parser = LogParser()
        batch = parser.parse_file(str(log_path))

        bench = Benchmark()
        labels = bench.load_labels(labels_path)

        git_sha = _get_git_sha()
        n_combos = sum(
            _combo_count(self._get_param_grid(MODELS[m])) for m in self.model_names
        )

        mlflow.set_experiment(self.experiment_name)

        best_f1 = -1.0
        best_run_id = ""
        best_model_display = ""
        best_params: dict[str, Any] = {}
        all_results: list[dict[str, Any]] = []

        with mlflow.start_run(run_name=f"sweep-{log_path.stem}") as parent_run:
            mlflow.set_tags({
                "git_sha": git_sha,
                "python_version": sys.version.split()[0],
                "log_file": log_path.name,
                "sweep.type": "grid",
                "models_compared": ",".join(self.model_names),
            })
            mlflow.log_params({
                "log_file": str(log_path),
                "n_combinations": n_combos,
                "models": str(self.model_names),
            })

            for model_key in self.model_names:
                spec = MODELS[model_key]
                param_grid = self._get_param_grid(spec)
                param_names = sorted(param_grid.keys())

                for combo in itertools.product(*[param_grid[k] for k in param_names]):
                    combo_params = dict(zip(param_names, combo))
                    run_name = f"{model_key}_{_params_to_str(combo_params)}"

                    with mlflow.start_run(run_name=run_name, nested=True) as child_run:
                        mlflow.log_param("model_class", spec.display_name)
                        mlflow.log_param("model_key", model_key)
                        mlflow.log_params(combo_params)
                        mlflow.set_tags({
                            "git_sha": git_sha,
                            "python_version": sys.version.split()[0],
                            "dataset": log_path.stem,
                            "experimenter": "flare-sweep",
                        })

                        # ── train ─────────────────────────────────────────
                        # Reuse IF vocab/features from AnomalyDetector;
                        # for other models build features independently.
                        detector = AnomalyDetector()
                        detection_results = detector.detect(batch.events, track=False)
                        assert detector._last_feature_matrix is not None
                        feature_matrix = detector._last_feature_matrix

                        raw_estimator = spec.estimator_class(
                            **spec.default_params, **combo_params
                        )
                        estimator = Pipeline([
                            ("scaler", StandardScaler()),
                            ("model", raw_estimator),
                        ])
                        mlflow.log_param("preprocessing", "StandardScaler")
                        try:
                            estimator.fit(feature_matrix)
                            predictions = estimator.predict(feature_matrix)
                        except Exception as exc:
                            # EllipticEnvelope can fail on rank-deficient matrices
                            mlflow.log_param("skipped_reason", str(exc)[:200])
                            mlflow.log_metric("f1", -1.0)
                            all_results.append({
                                "run_id": child_run.info.run_id,
                                "model": spec.display_name,
                                "model_key": model_key,
                                "params": combo_params,
                                "f1": -1.0,
                                "precision": 0.0,
                                "recall": 0.0,
                                "skipped": True,
                            })
                            continue

                        # ── evaluate ──────────────────────────────────────
                        # Map sklearn predictions {-1, 1} → AnomalyResult
                        from flare.detection.detector import AnomalyResult
                        eval_results = [
                            AnomalyResult(
                                block_id=r.block_id,
                                is_anomaly=bool(predictions[i] == -1),
                                anomaly_score=float(
                                    estimator.score_samples(
                                        feature_matrix[i : i + 1]
                                    )[0]
                                    if hasattr(estimator, "score_samples")
                                    else float(predictions[i])
                                ),
                                event_count=r.event_count,
                                template_ids=r.template_ids,
                            )
                            for i, r in enumerate(detection_results)
                        ]
                        bench_result = bench.evaluate(eval_results, labels)

                        # ── log metrics ───────────────────────────────────
                        n_anomalies = sum(1 for r in eval_results if r.is_anomaly)
                        mlflow.log_params({
                            "vocab_size": len(detector._template_vocab),
                            "total_blocks": len(eval_results),
                        })
                        mlflow.log_metrics({
                            "precision": round(bench_result.precision, 4),
                            "recall": round(bench_result.recall, 4),
                            "f1": round(bench_result.f1, 4),
                            "true_positives": bench_result.true_positives,
                            "false_positives": bench_result.false_positives,
                            "false_negatives": bench_result.false_negatives,
                            "anomaly_rate": round(
                                n_anomalies / max(len(eval_results), 1), 4
                            ),
                        })

                        # ── dataset lineage ───────────────────────────────
                        try:
                            import mlflow.data
                            ds = mlflow.data.from_numpy(
                                feature_matrix,
                                source=str(log_path),
                                name=f"{log_path.stem}-features",
                            )
                            mlflow.log_input(ds, context="training")
                        except Exception:
                            pass

                        # ── model + signature ─────────────────────────────
                        signature = infer_signature(feature_matrix, predictions)
                        mlflow.sklearn.log_model(
                            estimator, "model", signature=signature
                        )

                        # ── confusion matrix ──────────────────────────────
                        self._log_confusion_matrix(
                            eval_results, labels, spec.display_name, combo_params
                        )

                        entry: dict[str, Any] = {
                            "run_id": child_run.info.run_id,
                            "model": spec.display_name,
                            "model_key": model_key,
                            "params": combo_params,
                            "f1": bench_result.f1,
                            "precision": bench_result.precision,
                            "recall": bench_result.recall,
                            "skipped": False,
                        }
                        all_results.append(entry)

                        if bench_result.f1 > best_f1:
                            best_f1 = bench_result.f1
                            best_run_id = child_run.info.run_id
                            best_model_display = spec.display_name
                            best_params = {"model": model_key, **combo_params}

            # ── annotate parent ───────────────────────────────────────────
            mlflow.log_metric("best_f1", round(best_f1, 4))
            mlflow.log_params({
                "best_model": best_model_display,
                "best_child_run_id": best_run_id,
                **{f"best_{k}": v for k, v in best_params.items()},
            })

            parent_run_id = parent_run.info.run_id

        if promote_best and best_run_id:
            self._promote_best(best_run_id)

        all_results.sort(key=lambda r: r["f1"], reverse=True)
        return SweepResult(
            best_f1=best_f1,
            best_run_id=best_run_id,
            best_model=best_model_display,
            best_params=best_params,
            parent_run_id=parent_run_id,
            all_results=all_results,
        )

    # ── private helpers ────────────────────────────────────────────────────────

    def _get_param_grid(self, spec: ModelSpec) -> dict[str, list[Any]]:
        """Return the param grid for a spec, applying IF-specific overrides."""
        grid = dict(spec.param_grid)
        if spec.name == "isolation_forest":
            if self.contamination_values is not None:
                grid["contamination"] = self.contamination_values
            if self.n_estimators_values is not None:
                grid["n_estimators"] = self.n_estimators_values
        return grid

    def _log_confusion_matrix(
        self,
        detection_results: list,
        labels: dict[str, bool],
        model_label: str,
        params: dict[str, Any],
    ) -> None:
        """Log a confusion matrix PNG as an MLflow artifact."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import mlflow
            from sklearn.metrics import ConfusionMatrixDisplay
        except ImportError:
            return

        actuals, predicteds = [], []
        for r in detection_results:
            if r.block_id in labels:
                actuals.append(int(labels[r.block_id]))
                predicteds.append(int(r.is_anomaly))

        if not actuals:
            return

        params_str = _params_to_str(params)
        fig, ax = plt.subplots(figsize=(4, 4))
        ConfusionMatrixDisplay.from_predictions(
            actuals,
            predicteds,
            display_labels=["Normal", "Anomaly"],
            ax=ax,
            colorbar=False,
        )
        ax.set_title(f"{model_label}\n{params_str}", fontsize=8)
        fig.tight_layout()
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

    def _promote_best(self, best_run_id: str) -> None:
        """Register best child model and promote to Staging."""
        try:
            import mlflow
            from mlflow.tracking import MlflowClient
        except ImportError:
            return

        mv = mlflow.register_model(
            f"runs:/{best_run_id}/model",
            self.model_registry_name,
        )
        client = MlflowClient()
        for v in client.search_model_versions(f"name='{self.model_registry_name}'"):
            if v.current_stage == "Staging" and v.version != mv.version:
                client.transition_model_version_stage(
                    name=self.model_registry_name,
                    version=v.version,
                    stage="Archived",
                )
        client.transition_model_version_stage(
            name=self.model_registry_name,
            version=mv.version,
            stage="Staging",
        )


# ── module-level helpers ───────────────────────────────────────────────────────

def _params_to_str(params: dict[str, Any]) -> str:
    return " ".join(f"{k}={v}" for k, v in sorted(params.items()))


def _combo_count(grid: dict[str, list[Any]]) -> int:
    count = 1
    for v in grid.values():
        count *= len(v)
    return count


def _get_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"

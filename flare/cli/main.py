"""Flare CLI — log anomaly detection from the command line."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from flare import __version__

console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="flare")
def cli() -> None:
    """Flare — LLM-powered log anomaly detection and incident summarization."""


@cli.command()
@click.option(
    "--input", "-i",
    "input_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to raw log file.",
)
@click.option(
    "--output", "-o",
    "output_path",
    default=None,
    type=click.Path(),
    help="Path to write JSON results (optional).",
)
@click.option(
    "--labels",
    default=None,
    type=click.Path(exists=True),
    help="Path to ground truth labels CSV for evaluation.",
)
@click.option(
    "--contamination",
    default=0.03,
    type=float,
    help="Expected anomaly ratio (0.0-0.5).",
)
@click.option(
    "--use-registry",
    "use_registry",
    is_flag=True,
    default=False,
    help="Load Production model from MLflow registry instead of retraining.",
)
def detect(
    input_path: str,
    output_path: str | None,
    labels: str | None,
    contamination: float,
    use_registry: bool,
) -> None:
    """Parse logs, detect anomalies, and cluster into incidents."""
    from flare.clustering import IncidentClusterer
    from flare.detection import AnomalyDetector
    from flare.eval import Benchmark
    from flare.ingestion import LogParser

    # Step 1: Parse logs
    console.print(Panel("[bold cyan]Step 1/3[/] Parsing logs...", expand=False))
    parser = LogParser()
    batch = parser.parse_file(input_path)

    console.print(
        f"  Parsed [green]{batch.total_lines}[/] lines → "
        f"[green]{len(batch.events)}[/] events, "
        f"[green]{batch.template_count}[/] templates, "
        f"[yellow]{batch.parse_errors}[/] errors"
    )

    if not batch.events:
        console.print("[red]No events parsed. Check the log format.[/]")
        sys.exit(1)

    # Step 2: Detect anomalies
    console.print(Panel("[bold cyan]Step 2/3[/] Detecting anomalies...", expand=False))
    detector = AnomalyDetector(contamination=contamination, use_registry=use_registry)
    results = detector.detect(batch.events, source_path=input_path)

    anomalies = [r for r in results if r.is_anomaly]
    console.print(
        f"  Analyzed [green]{len(results)}[/] blocks → "
        f"[red]{len(anomalies)}[/] anomalous "
        f"({len(anomalies)/len(results)*100:.1f}%)"
    )

    # Step 3: Cluster into incidents
    console.print(Panel("[bold cyan]Step 3/3[/] Clustering incidents...", expand=False))
    clusterer = IncidentClusterer()
    incidents = clusterer.cluster(results, events=batch.events)
    console.print(f"  Grouped into [magenta]{len(incidents)}[/] incidents")

    # Display results table
    _display_incidents(incidents)

    # Evaluate against ground truth if labels provided
    benchmark_result = None
    if labels:
        console.print()
        console.print(Panel("[bold yellow]Evaluation[/]", expand=False))
        bench = Benchmark()
        ground_truth = bench.load_labels(labels)
        benchmark_result = bench.evaluate(results, ground_truth)
        _display_benchmark(benchmark_result)

    # Write output
    if output_path:
        output = {
            "summary": {
                "total_lines": batch.total_lines,
                "total_events": len(batch.events),
                "total_blocks": len(results),
                "total_anomalies": len(anomalies),
                "total_incidents": len(incidents),
                "templates_discovered": batch.template_count,
            },
            "incidents": [inc.to_dict() for inc in incidents],
            "anomalies": [r.to_dict() for r in results if r.is_anomaly],
        }
        if benchmark_result:
            output["benchmark"] = benchmark_result.to_dict()

        Path(output_path).write_text(json.dumps(output, indent=2, default=str))
        console.print(f"\nResults written to [blue]{output_path}[/]")


@cli.command()
@click.option(
    "--input", "-i",
    "input_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to detection results JSON (output of 'flare detect').",
)
@click.option(
    "--output", "-o",
    "output_path",
    default=None,
    type=click.Path(),
    help="Path to write summarized incidents JSON.",
)
@click.option(
    "--eval",
    "run_eval",
    is_flag=True,
    default=False,
    help="Run quality rubric scoring on each summary.",
)
def summarize(
    input_path: str,
    output_path: str | None,
    run_eval: bool,
) -> None:
    """Summarize detected incidents using the Anthropic LLM."""
    from flare.eval.benchmark import Benchmark
    from flare.llm.client import AnthropicClient
    from flare.llm.schemas import QualityScore, UsageStats
    from flare.llm.summarizer import IncidentSummarizer

    # Load detection results
    console.print(Panel("[bold cyan]Step 1/3[/] Loading detection results...", expand=False))
    raw = json.loads(Path(input_path).read_text())

    incidents = _rebuild_incidents(raw.get("incidents", []))
    if not incidents:
        console.print("[red]No incidents found in input file.[/]")
        sys.exit(1)

    console.print(f"  Loaded [magenta]{len(incidents)}[/] incidents")

    # Summarize with LLM
    console.print(Panel("[bold cyan]Step 2/3[/] Summarizing with LLM...", expand=False))
    try:
        client = AnthropicClient()
    except ValueError as e:
        console.print(f"[red]{e}[/]")
        sys.exit(1)

    summarizer = IncidentSummarizer(client=client)
    summarized = summarizer.summarize_all(incidents)

    for s in summarized:
        console.print(
            f"  [green]Incident {s.incident_id}[/]: "
            f"severity={s.llm_summary.severity.value}, "
            f"confidence={s.llm_summary.confidence:.2f}, "
            f"tokens={s.usage.input_tokens}+{s.usage.output_tokens}"
        )

    # Display summaries
    _display_summaries(summarized)

    # Run quality evaluation if requested
    eval_result = None
    if run_eval:
        console.print(Panel("[bold cyan]Step 3/3[/] Evaluating quality...", expand=False))
        quality_scores: list[QualityScore] = []
        eval_usage: list[UsageStats] = []

        for s, inc in zip(summarized, incidents):
            score, usage = summarizer.evaluate_quality(inc, s.llm_summary)
            quality_scores.append(score)
            eval_usage.append(usage)
            console.print(
                f"  [green]Incident {s.incident_id}[/]: "
                f"relevance={score.relevance}, "
                f"specificity={score.specificity}, "
                f"actionability={score.actionability}"
            )

        bench = Benchmark()

        # Create an MLflow run so LLM eval metrics land in the tracking server
        mlflow_run_id: str | None = None
        try:
            import mlflow
            mlflow.set_experiment("flare-llm-eval")
            with mlflow.start_run(run_name="summarize") as _run:
                mlflow.log_param("num_incidents", len(incidents))
                mlflow.log_param("model", client.model)
                mlflow_run_id = _run.info.run_id
        except ImportError:
            pass

        eval_result = bench.evaluate_llm(
            summarized, quality_scores, eval_usage, run_id=mlflow_run_id
        )
        if mlflow_run_id:
            console.print(f"  [dim]MLflow run: {mlflow_run_id[:8]}[/]")
        _display_llm_eval(eval_result)
    else:
        console.print(Panel("[dim]Step 3/3[/] Skipping eval (use --eval to enable)", expand=False))

    # Write output
    if output_path:
        output: dict[str, Any] = {
            "summarized_incidents": [s.model_dump() for s in summarized],
        }
        if eval_result:
            output["llm_eval"] = eval_result.to_dict()

        Path(output_path).write_text(json.dumps(output, indent=2, default=str))
        console.print(f"\nResults written to [blue]{output_path}[/]")


def _rebuild_incidents(raw_incidents: list[dict]) -> list:
    """Rebuild Incident objects from serialized detection results."""
    from flare.clustering.clusterer import Incident

    incidents = []
    for raw in raw_incidents:
        incidents.append(
            Incident(
                incident_id=raw["incident_id"],
                block_ids=raw["block_ids"],
                severity=raw["severity"],
                anomaly_scores=[raw.get("mean_anomaly_score", 0.0)],
                log_lines=raw.get("log_lines", []),
                templates=raw.get("templates", []),
                time_range=tuple(raw.get("time_range", ["", ""])),
            )
        )
    return incidents


def _display_incidents(incidents: list) -> None:
    """Display incidents in a Rich table."""
    if not incidents:
        console.print("[dim]No incidents detected.[/]")
        return

    table = Table(title="Detected Incidents", show_lines=True)
    table.add_column("ID", style="bold", width=4)
    table.add_column("Blocks", style="cyan", width=8)
    table.add_column("Severity", width=10)
    table.add_column("Sample Block IDs", style="dim", max_width=60)

    for inc in incidents:
        severity = inc.severity
        if severity > 0.5:
            sev_style = "[red]"
        elif severity > 0.2:
            sev_style = "[yellow]"
        else:
            sev_style = "[green]"

        sample_ids = ", ".join(inc.block_ids[:3])
        if len(inc.block_ids) > 3:
            sample_ids += f" (+{len(inc.block_ids) - 3} more)"

        table.add_row(
            str(inc.incident_id),
            str(inc.size),
            f"{sev_style}{severity:.3f}[/]",
            sample_ids,
        )

    console.print()
    console.print(table)


def _display_summaries(summarized: list) -> None:
    """Display LLM summaries in Rich panels."""
    console.print()
    for s in summarized:
        summary = s.llm_summary
        sev = summary.severity.value
        sev_colors = {"low": "green", "medium": "yellow", "high": "red", "critical": "bold red"}
        color = sev_colors.get(sev, "white")

        panel_content = (
            f"[bold]Severity:[/] [{color}]{sev.upper()}[/{color}] "
            f"(confidence: {summary.confidence:.0%})\n\n"
            f"[bold]Explanation:[/]\n{summary.explanation}\n\n"
            f"[bold]Root Cause:[/]\n{summary.root_cause}\n\n"
            f"[bold]Remediation:[/]"
        )
        for step in summary.remediation:
            panel_content += f"\n  [{step.priority}] {step.action}"

        console.print(
            Panel(
                panel_content,
                title=f"Incident {s.incident_id} — {len(s.block_ids)} block(s)",
                border_style=color,
                expand=False,
            )
        )


def _display_benchmark(result: object) -> None:
    """Display benchmark metrics in a Rich table."""
    table = Table(title="Benchmark Results", show_lines=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Precision", f"{result.precision:.4f}")  # type: ignore[attr-defined]
    table.add_row("Recall", f"{result.recall:.4f}")  # type: ignore[attr-defined]
    table.add_row("F1 Score", f"[bold]{result.f1:.4f}[/]")  # type: ignore[attr-defined]
    table.add_row("True Positives", str(result.true_positives))  # type: ignore[attr-defined]
    table.add_row("False Positives", str(result.false_positives))  # type: ignore[attr-defined]
    table.add_row("False Negatives", str(result.false_negatives))  # type: ignore[attr-defined]
    table.add_row("True Negatives", str(result.true_negatives))  # type: ignore[attr-defined]

    console.print(table)


def _display_llm_eval(eval_result: object) -> None:
    """Display LLM evaluation metrics."""
    table = Table(title="LLM Quality Evaluation", show_lines=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Mean Relevance", f"{eval_result.mean_relevance:.2f} / 5")  # type: ignore[attr-defined]
    table.add_row("Mean Specificity", f"{eval_result.mean_specificity:.2f} / 5")  # type: ignore[attr-defined]
    table.add_row("Mean Actionability", f"{eval_result.mean_actionability:.2f} / 5")  # type: ignore[attr-defined]
    table.add_row("Mean Quality", f"[bold]{eval_result.mean_quality:.2f} / 5[/]")  # type: ignore[attr-defined]
    table.add_row("", "")
    table.add_row("Total Input Tokens", f"{eval_result.total_input_tokens:,}")  # type: ignore[attr-defined]
    table.add_row("Total Output Tokens", f"{eval_result.total_output_tokens:,}")  # type: ignore[attr-defined]
    table.add_row("Total Cost (USD)", f"${eval_result.total_cost_usd:.4f}")  # type: ignore[attr-defined]
    table.add_row("Mean Latency", f"{eval_result.mean_latency_ms:.0f}ms")  # type: ignore[attr-defined]

    console.print()
    console.print(table)


@cli.command()
@click.option(
    "--replay", "-r",
    "replay_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to log file to replay.",
)
@click.option(
    "--rate",
    default=10.0,
    type=float,
    help="Lines to emit per second (0 = as fast as possible).",
    show_default=True,
)
@click.option(
    "--window",
    default=50,
    type=int,
    help="Number of lines per detection window.",
    show_default=True,
)
@click.option(
    "--format", "log_format",
    default="auto",
    type=str,
    help="Log format: auto, hdfs, openssh, syslog, generic.",
    show_default=True,
)
@click.option(
    "--contamination",
    default=0.03,
    type=float,
    help="Expected anomaly fraction (0.0–0.5).",
    show_default=True,
)
@click.option(
    "--persist",
    "persist_path",
    default=None,
    type=click.Path(),
    help="File path for Drain3 template persistence across restarts.",
)
def collect(
    replay_path: str,
    rate: float,
    window: int,
    log_format: str,
    contamination: float,
    persist_path: str | None,
) -> None:
    """Replay a log file through the detection pipeline in real time."""
    from flare.replay import LogReplayer

    effective_rate: float | None = rate if rate > 0 else None

    console.print(
        Panel(
            f"[bold cyan]Replaying[/] [blue]{replay_path}[/]\n"
            f"rate=[green]{rate or 'max'}[/] lines/s  "
            f"window=[green]{window}[/] lines  "
            f"format=[green]{log_format}[/]",
            title="flare collect",
            expand=False,
        )
    )

    try:
        replayer = LogReplayer(
            filepath=replay_path,
            rate=effective_rate,
            window=window,
            log_format=log_format,
            contamination=contamination,
            persist_path=persist_path,
        )
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/]")
        sys.exit(1)

    total_anomalies = 0
    total_incidents = 0

    for result in replayer.replay():
        if result.incidents:
            total_anomalies += result.anomaly_count
            total_incidents += result.incident_count
            for inc in result.incidents:
                sev = inc.severity
                color = "red" if sev > 0.5 else "yellow" if sev > 0.2 else "green"
                console.print(
                    f"  [dim]w{result.window_index:03d}[/] "
                    f"[{color}]INCIDENT {inc.incident_id}[/{color}]  "
                    f"blocks={len(inc.block_ids)}  "
                    f"severity={sev:.3f}  "
                    f"[dim]{result.elapsed_ms:.0f}ms[/]"
                )
        else:
            console.print(
                f"  [dim]w{result.window_index:03d}  "
                f"{result.lines_processed} lines  "
                f"{len(result.events)} events  "
                f"no anomalies  "
                f"{result.elapsed_ms:.0f}ms[/]"
            )

    console.print()
    console.print(
        Panel(
            f"Total anomalous blocks: [red]{total_anomalies}[/]\n"
            f"Total incidents: [magenta]{total_incidents}[/]",
            title="Replay complete",
            expand=False,
        )
    )


@cli.group()
def model() -> None:
    """Manage registered MLflow models."""


@model.command("list")
@click.option("--name", default="flare-isolation-forest", show_default=True)
def model_list(name: str) -> None:
    """List all versions of a registered model."""
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    try:
        versions = client.search_model_versions(f"name='{name}'")
    except Exception as e:
        console.print(f"[red]{e}[/]")
        return

    if not versions:
        console.print(f"[yellow]No versions found for '{name}'.[/]")
        return

    table = Table(title=f"Model: {name}", show_lines=True)
    table.add_column("Version", style="bold", width=8)
    table.add_column("Stage", width=12)
    table.add_column("Run ID", style="dim", width=36)
    table.add_column("Created", width=20)

    stage_colors = {"Production": "green", "Staging": "yellow", "Archived": "dim", "None": "white"}
    for v in sorted(versions, key=lambda x: int(x.version)):
        stage = v.current_stage
        color = stage_colors.get(stage or "None", "white")
        import datetime
        created = datetime.datetime.fromtimestamp(
            v.creation_timestamp / 1000
        ).strftime("%Y-%m-%d %H:%M")
        table.add_row(str(v.version), f"[{color}]{stage}[/{color}]", str(v.run_id or ""), created)

    console.print(table)


@model.command("promote")
@click.argument("version", type=int)
@click.argument("stage", type=click.Choice(
    ["Staging", "Production", "Archived"], case_sensitive=False
))
@click.option("--name", default="flare-isolation-forest", show_default=True)
@click.option("--archive-existing", is_flag=True, default=True,
              help="Archive any existing model in the target stage.")
def model_promote(version: int, stage: str, name: str, archive_existing: bool) -> None:
    """Promote a model version to Staging, Production, or Archived.

    \b
    Examples:
      flare model promote 3 Production
      flare model promote 2 Staging --no-archive-existing
    """
    from mlflow.tracking import MlflowClient

    client = MlflowClient()

    if archive_existing:
        try:
            existing = client.search_model_versions(f"name='{name}'")
            for v in existing:
                if v.current_stage == stage and int(v.version) != version:
                    client.transition_model_version_stage(
                        name=name, version=v.version, stage="Archived"
                    )
                    console.print(f"  Archived v{v.version} (was in {stage})")
        except Exception:
            pass

    try:
        client.transition_model_version_stage(
            name=name,
            version=str(version),
            stage=stage,
        )
        color = {"Production": "green", "Staging": "yellow", "Archived": "dim"}.get(stage, "white")
        console.print(
            Panel(
                f"[{color}]v{version} → {stage}[/{color}]\n"
                f"[dim]Model: {name}[/]",
                title="Promoted",
                expand=False,
            )
        )
    except Exception as e:
        console.print(f"[red]{e}[/]")


@model.command("compare")
@click.option("--name", default="flare-detection", show_default=True,
              help="MLflow experiment name to compare runs from.")
@click.option("--n", default=5, show_default=True, help="Number of recent runs to show.")
def model_compare(name: str, n: int) -> None:
    """Compare recent detection runs by key metrics."""
    import mlflow

    runs = mlflow.search_runs(  # type: ignore[assignment]
        experiment_names=[name],
        order_by=["start_time DESC"],
        max_results=n,
    )

    if runs.empty:  # type: ignore[union-attr]
        console.print(f"[yellow]No runs found in experiment '{name}'.[/]")
        return

    table = Table(title=f"Recent runs — {name}", show_lines=True)
    table.add_column("Run", style="dim", width=8)
    table.add_column("contamination", width=14)
    table.add_column("vocab_size", width=10)
    table.add_column("anomaly_rate", width=12)
    table.add_column("mean_score", width=12)
    table.add_column("anomaly_count", width=14)

    for _, row in runs.iterrows():  # type: ignore[union-attr]
        p = row.get
        table.add_row(
            str(row.get("run_id", ""))[:8],
            str(p("params.contamination", "-")),
            str(p("params.vocab_size", "-")),
            str(p("metrics.anomaly_rate", "-")),
            str(p("metrics.mean_anomaly_score", "-")),
            str(p("metrics.anomaly_count", "-")),
        )

    console.print(table)


@model.command("sweep")
@click.option(
    "--input", "-i",
    "input_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to raw log file to train on.",
)
@click.option(
    "--labels",
    required=True,
    type=click.Path(exists=True),
    help="Path to ground truth labels CSV for F1 evaluation.",
)
@click.option(
    "--models",
    "models_str",
    default="isolation_forest",
    show_default=True,
    help=(
        "Comma-separated model keys to compare. "
        "Available: isolation_forest, lof, ocsvm, elliptic."
    ),
)
@click.option(
    "--contamination-values",
    "contamination_str",
    default=None,
    help=(
        "Override contamination grid for isolation_forest only "
        "(e.g. '0.01,0.03,0.05'). Uses model default when omitted."
    ),
)
@click.option(
    "--n-estimators-values",
    "n_estimators_str",
    default=None,
    help=(
        "Override n_estimators grid for isolation_forest only "
        "(e.g. '100,200,300'). Uses model default when omitted."
    ),
)
@click.option(
    "--promote",
    is_flag=True,
    default=False,
    help="Register best model and promote to Staging after sweep.",
)
def model_sweep(
    input_path: str,
    labels: str,
    models_str: str,
    contamination_str: str | None,
    n_estimators_str: str | None,
    promote: bool,
) -> None:
    """Compare anomaly detection models via a grid search, logged as nested MLflow runs.

    \b
    Each (model, params) combination trains a fresh estimator, evaluates it
    against ground truth labels, and logs params + F1/precision/recall + a
    confusion matrix PNG as a child run nested under a parent sweep run.

    \b
    Available models:
      isolation_forest   Isolation Forest (sklearn)
      lof                Local Outlier Factor (sklearn, novelty=True)
      ocsvm              One-Class SVM (sklearn, rbf kernel)
      elliptic           Elliptic Envelope (sklearn, assumes Gaussian)

    \b
    Examples:
      # Compare all four models with their default param grids
      flare model sweep -i logs/hdfs_sample.log --labels logs/sample_labels.csv \\
          --models isolation_forest,lof,ocsvm,elliptic

      # IF-only hyperparameter sweep with custom grid
      flare model sweep -i logs/hdfs_sample.log --labels logs/sample_labels.csv \\
          --contamination-values 0.01,0.03,0.05 --n-estimators-values 100,200 --promote
    """
    from flare.experiment.sweep import MODELS, HyperparamSweep

    model_names = [m.strip() for m in models_str.split(",")]
    for name in model_names:
        if name not in MODELS:
            console.print(
                f"[red]Unknown model '{name}'. "
                f"Valid: {', '.join(MODELS.keys())}[/]"
            )
            return

    contamination_values = (
        [float(x.strip()) for x in contamination_str.split(",")]
        if contamination_str else None
    )
    n_estimators_values = (
        [int(x.strip()) for x in n_estimators_str.split(",")]
        if n_estimators_str else None
    )

    from flare.experiment.sweep import _combo_count
    n_combos = sum(
        _combo_count(
            HyperparamSweep(
                model_names=[m],
                contamination_values=contamination_values,
                n_estimators_values=n_estimators_values,
            )._get_param_grid(MODELS[m])
        )
        for m in model_names
    )

    console.print(
        Panel(
            f"[bold]models:[/]       {model_names}\n"
            f"[bold]combinations:[/] {n_combos}\n"
            f"[bold]labels:[/]       {labels}\n"
            + ("[green]auto-promote best → Staging[/]"
               if promote else "[dim]--promote to auto-stage best[/]"),
            title="flare model sweep",
            expand=False,
        )
    )

    sweep = HyperparamSweep(
        model_names=model_names,
        contamination_values=contamination_values,
        n_estimators_values=n_estimators_values,
    )

    try:
        result = sweep.run(input_path, labels, promote_best=promote)
    except (RuntimeError, ValueError) as e:
        console.print(f"[red]{e}[/]")
        return

    # Results table — sorted best-first by sweep.run
    table = Table(title="Sweep Results (ranked by F1)", show_lines=True)
    table.add_column("Run ID", style="dim", width=10)
    table.add_column("Model", width=22)
    table.add_column("Params", width=28)
    table.add_column("F1", width=10)
    table.add_column("Precision", width=10)
    table.add_column("Recall", width=8)

    for r in result.all_results:
        is_best = r["run_id"] == result.best_run_id
        if r.get("skipped"):
            f1_str = "[dim]skipped[/]"
            prec_str = rec_str = "[dim]-[/]"
        else:
            f1_str = (
                f"[bold green]{r['f1']:.4f} ★[/]"
                if is_best else f"{r['f1']:.4f}"
            )
            prec_str = f"{r['precision']:.4f}"
            rec_str = f"{r['recall']:.4f}"

        from flare.experiment.sweep import _params_to_str
        table.add_row(
            r["run_id"][:8],
            r["model"],
            _params_to_str(r["params"]),
            f1_str,
            prec_str,
            rec_str,
        )

    console.print()
    console.print(table)

    params_str = "  ".join(f"{k}={v}" for k, v in result.best_params.items())
    promoted_line = (
        "[green]Best model registered and promoted → Staging[/]"
        if promote
        else "[dim]Run with --promote to register and stage the best model[/]"
    )
    console.print(
        Panel(
            f"Best F1:     [bold green]{result.best_f1:.4f}[/]\n"
            f"Best model:  [cyan]{result.best_model}[/]\n"
            f"Best params: [dim]{params_str}[/]\n"
            f"Best run:    [dim]{result.best_run_id[:8]}[/]\n"
            f"Parent run:  [dim]{result.parent_run_id[:8]}[/]\n"
            f"{promoted_line}",
            title="Sweep complete",
            expand=False,
        )
    )


if __name__ == "__main__":
    cli()

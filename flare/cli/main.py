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
def detect(
    input_path: str,
    output_path: str | None,
    labels: str | None,
    contamination: float,
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
    detector = AnomalyDetector(contamination=contamination)
    results = detector.detect(batch.events)

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
        eval_result = bench.evaluate_llm(summarized, quality_scores, eval_usage)
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


if __name__ == "__main__":
    cli()

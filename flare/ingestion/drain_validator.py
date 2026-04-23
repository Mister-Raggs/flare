"""Drain3 parameter validation — supervised and unsupervised metrics.

Sweeps (sim_th, depth) combinations and computes a set of intrinsic metrics
that work without ground truth labels, plus ARI when a reference template
file is provided.

Unsupervised metrics (always available):
  - n_templates     : number of distinct templates discovered
  - coverage_rate   : fraction of lines matched to an existing template
  - wildcard_density: avg wildcards-per-token across all templates
  - template_entropy: Shannon entropy of the template frequency distribution
  - convergence     : 1 - (templates at 50% / templates at 100%)
                      High = settled early → stable / generalising

Supervised metric (requires reference templates CSV):
  - ari             : Adjusted Rand Index vs ground-truth template assignments

The sweep runs on the first ``sample_lines`` lines of the log so each combo
takes seconds even on 1.5 GB files.  Drain3 quality is stable across samples
— the template set converges well before 50 k lines for typical HDFS logs.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

# ── Result dataclasses ─────────────────────────────────────────────────────────

@dataclass
class DrainMetrics:
    """Per-(sim_th, depth) metrics from a Drain3 validation run.

    Attributes:
        sim_th: Drain3 similarity threshold used.
        depth: Drain3 parse-tree depth used.
        n_templates: Number of distinct templates discovered.
        coverage_rate: Fraction of lines matched to a template (0-1).
        wildcard_density: Mean wildcard tokens / total tokens per template.
        template_entropy: Shannon entropy of the template frequency distribution.
        convergence: 1 - (templates at 50 % / templates at 100 %).
        ari: Adjusted Rand Index vs ground truth. None when unsupervised.
        composite: Weighted unsupervised score (higher = better).
    """

    sim_th: float
    depth: int
    n_templates: int
    coverage_rate: float
    wildcard_density: float
    template_entropy: float
    convergence: float
    ari: float | None = None
    composite: float = 0.0


@dataclass
class DrainValidationResult:
    """Output of :meth:`DrainValidator.validate`.

    Attributes:
        results: All (sim_th, depth) combos, sorted by composite score desc.
        best: Combo with the highest composite score.
        best_ari: Combo with the highest ARI (None if unsupervised).
        sample_lines: Number of log lines used for the sweep.
        log_path: Source log file path.
    """

    results: list[DrainMetrics]
    best: DrainMetrics
    best_ari: DrainMetrics | None
    sample_lines: int
    log_path: Path
    gt_path: Path | None = None


# ── Validator ──────────────────────────────────────────────────────────────────

class DrainValidator:
    """Sweep Drain3 (sim_th, depth) settings and score template quality.

    Example — unsupervised sweep:
        >>> v = DrainValidator()
        >>> result = v.validate("logs/hdfs_20k.log")
        >>> print(f"Best sim_th={result.best.sim_th}  depth={result.best.depth}")

    Example — supervised sweep with ground truth:
        >>> result = v.validate(
        ...     "HDFS_v1/HDFS.log",
        ...     gt_templates_path="HDFS_v1/preprocessed/HDFS.log_templates.csv",
        ... )
        >>> print(f"Best ARI={result.best_ari.ari:.4f}")
    """

    DEFAULT_SIM_TH = [0.3, 0.4, 0.5, 0.6]
    DEFAULT_DEPTH = [3, 4, 5]

    def __init__(
        self,
        sim_th_values: list[float] | None = None,
        depth_values: list[int] | None = None,
        sample_lines: int = 50_000,
    ) -> None:
        self.sim_th_values = sim_th_values or self.DEFAULT_SIM_TH
        self.depth_values = depth_values or self.DEFAULT_DEPTH
        self.sample_lines = sample_lines

    def validate(
        self,
        log_path: str | Path,
        gt_templates_path: str | Path | None = None,
    ) -> DrainValidationResult:
        """Run the sweep and return ranked metrics.

        Args:
            log_path: Path to the raw log file.
            gt_templates_path: Optional path to LogHub-format templates CSV.
                Columns: EventId, EventTemplate.  Wildcards written as ``[*]``.

        Returns:
            :class:`DrainValidationResult` with all combos ranked.
        """
        log_path = Path(log_path)
        gt_path = Path(gt_templates_path) if gt_templates_path else None

        lines = _read_sample(log_path, self.sample_lines)
        gt_templates = _load_gt_templates(gt_path) if gt_path else []

        results: list[DrainMetrics] = []

        for sim_th in self.sim_th_values:
            for depth in self.depth_values:
                metrics = self._run_combo(lines, sim_th, depth, gt_templates)
                results.append(metrics)

        # Score and rank
        for m in results:
            m.composite = _composite_score(m)
        results.sort(key=lambda m: m.composite, reverse=True)

        best_ari: DrainMetrics | None = None
        if any(m.ari is not None for m in results):
            best_ari = max(
                (m for m in results if m.ari is not None),
                key=lambda m: m.ari or -1,
            )

        return DrainValidationResult(
            results=results,
            best=results[0],
            best_ari=best_ari,
            sample_lines=len(lines),
            log_path=log_path,
            gt_path=gt_path,
        )

    # ── private ────────────────────────────────────────────────────────────────

    def _run_combo(
        self,
        lines: list[str],
        sim_th: float,
        depth: int,
        gt_templates: list[re.Pattern[str]],
    ) -> DrainMetrics:
        config = TemplateMinerConfig()
        config.drain_sim_th = sim_th
        config.drain_depth = depth
        config.profiling_enabled = False
        miner = TemplateMiner(persistence_handler=None, config=config)

        template_counts: dict[int, int] = {}
        checkpoint = len(lines) // 2  # record template count at 50%
        templates_at_50pct = 0

        drain_assignments: list[int] = []

        for idx, line in enumerate(lines):
            result = miner.add_log_message(line)
            cid = result.get("cluster_id", -1) if isinstance(result, dict) else -1
            drain_assignments.append(cid)
            if cid >= 0:
                template_counts[cid] = template_counts.get(cid, 0) + 1
            if idx == checkpoint - 1:
                templates_at_50pct = len(miner.drain.clusters)

        n_templates = len(miner.drain.clusters)
        total = len(lines)

        # Coverage: lines whose cid is in our template set
        covered = sum(1 for cid in drain_assignments if cid >= 0)
        coverage_rate = covered / total if total else 0.0

        # Wildcard density across all templates
        wildcard_density = _wildcard_density(miner)

        # Shannon entropy of template frequency distribution
        template_entropy = _entropy(list(template_counts.values()))

        # Convergence: how much did template count grow after the 50% mark
        convergence = (
            1.0 - (templates_at_50pct / n_templates)
            if n_templates > 0
            else 0.0
        )

        # Supervised ARI
        ari: float | None = None
        if gt_templates:
            gt_assignments = _assign_gt(lines, gt_templates)
            ari = _adjusted_rand_index(drain_assignments, gt_assignments)

        return DrainMetrics(
            sim_th=sim_th,
            depth=depth,
            n_templates=n_templates,
            coverage_rate=coverage_rate,
            wildcard_density=wildcard_density,
            template_entropy=template_entropy,
            convergence=convergence,
            ari=ari,
        )


# ── helpers ────────────────────────────────────────────────────────────────────

def _read_sample(path: Path, n: int) -> list[str]:
    """Read up to n non-empty lines, stripping the structured prefix."""
    lines: list[str] = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            # Strip common HDFS prefix (date time level component) to get content
            parts = raw.split(None, 4)
            content = parts[4] if len(parts) >= 5 else raw
            lines.append(content)
            if len(lines) >= n:
                break
    return lines


def _load_gt_templates(path: Path) -> list[re.Pattern[str]]:
    """Load LogHub template CSV and compile each template to a regex."""
    import csv
    patterns: list[re.Pattern[str]] = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            tmpl = row.get("EventTemplate", "")
            # Escape everything except [*] wildcards, then replace [*] → .*
            escaped = re.escape(tmpl).replace(r"\[\*\]", ".*")
            try:
                patterns.append(re.compile(escaped, re.IGNORECASE))
            except re.error:
                pass
    return patterns


def _assign_gt(lines: list[str], patterns: list[re.Pattern[str]]) -> list[int]:
    """Assign each line a ground-truth template index (-1 = unmatched)."""
    assignments: list[int] = []
    for line in lines:
        matched = -1
        for i, pat in enumerate(patterns):
            if pat.search(line):
                matched = i
                break
        assignments.append(matched)
    return assignments


def _wildcard_density(miner: TemplateMiner) -> float:
    """Mean fraction of wildcard tokens across all templates."""
    densities: list[float] = []
    for cluster in miner.drain.clusters:
        tokens = cluster.log_template_tokens
        if not tokens:
            continue
        wc = sum(1 for t in tokens if t == "<*>")
        densities.append(wc / len(tokens))
    return sum(densities) / len(densities) if densities else 0.0


def _entropy(counts: list[int]) -> float:
    """Shannon entropy of a frequency distribution."""
    total = sum(counts)
    if total == 0:
        return 0.0
    return -sum(
        (c / total) * math.log2(c / total) for c in counts if c > 0
    )


def _adjusted_rand_index(a: list[int], b: list[int]) -> float:
    """Compute ARI between two cluster assignment lists."""
    try:
        from sklearn.metrics import adjusted_rand_score
        return float(adjusted_rand_score(a, b))
    except ImportError:
        return float("nan")


def _composite_score(m: DrainMetrics) -> float:
    """Weighted unsupervised quality score (higher = better).

    Weights:
      coverage_rate   0.35 — lines we can actually represent
      convergence     0.30 — settled early = generalises well
      template_entropy 0.20 — captures diversity of log patterns
      wildcard_density -0.15 — penalise over-merging (too many wildcards)
    """
    return (
        0.35 * m.coverage_rate
        + 0.30 * m.convergence
        + 0.20 * min(m.template_entropy / 6.0, 1.0)  # normalise to ~[0,1]
        - 0.15 * m.wildcard_density
    )

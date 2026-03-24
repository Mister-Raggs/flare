"""Lightweight in-memory metrics collector — no external dependencies.

Exposes counters, gauges, and histograms in Prometheus text exposition format.
Designed for observability demos; in production you'd use prometheus_client.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class _Histogram:
    """Simple histogram with fixed buckets."""

    buckets: tuple[float, ...] = (
        0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
    )
    counts: dict[float, int] = field(default_factory=dict)
    total: float = 0.0
    count: int = 0

    def __post_init__(self) -> None:
        for b in self.buckets:
            self.counts[b] = 0
        self.counts[float("inf")] = 0

    def observe(self, value: float) -> None:
        self.total += value
        self.count += 1
        for b in self.buckets:
            if value <= b:
                self.counts[b] += 1
        self.counts[float("inf")] += 1


class MetricsCollector:
    """Thread-safe metrics collector with Prometheus text format export."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, float] = defaultdict(float)
        self._gauges: dict[str, float] = defaultdict(float)
        self._histograms: dict[str, _Histogram] = {}
        self._start_time = time.monotonic()

    # ── Counter ───────────────────────────────────────────────
    def inc(self, name: str, value: float = 1.0, **labels: str) -> None:
        """Increment a counter."""
        key = self._label_key(name, labels)
        with self._lock:
            self._counters[key] += value

    # ── Gauge ─────────────────────────────────────────────────
    def set_gauge(self, name: str, value: float, **labels: str) -> None:
        """Set a gauge value."""
        key = self._label_key(name, labels)
        with self._lock:
            self._gauges[key] = value

    # ── Histogram ─────────────────────────────────────────────
    def observe(self, name: str, value: float, **labels: str) -> None:
        """Record a histogram observation."""
        key = self._label_key(name, labels)
        with self._lock:
            if key not in self._histograms:
                self._histograms[key] = _Histogram()
            self._histograms[key].observe(value)

    # ── Export ────────────────────────────────────────────────
    def export(self) -> str:
        """Export all metrics in Prometheus text exposition format."""
        lines: list[str] = []

        with self._lock:
            # Uptime gauge
            uptime = time.monotonic() - self._start_time
            lines.append("# HELP flare_uptime_seconds Time since API startup")
            lines.append("# TYPE flare_uptime_seconds gauge")
            lines.append(f"flare_uptime_seconds {uptime:.3f}")
            lines.append("")

            # Counters
            grouped_counters = self._group_by_name(self._counters)
            for name, entries in sorted(grouped_counters.items()):
                lines.append(f"# HELP {name} Counter metric")
                lines.append(f"# TYPE {name} counter")
                for label_str, value in sorted(entries):
                    if label_str:
                        lines.append(f"{name}{{{label_str}}} {value}")
                    else:
                        lines.append(f"{name} {value}")
                lines.append("")

            # Gauges
            grouped_gauges = self._group_by_name(self._gauges)
            for name, entries in sorted(grouped_gauges.items()):
                lines.append(f"# HELP {name} Gauge metric")
                lines.append(f"# TYPE {name} gauge")
                for label_str, value in sorted(entries):
                    if label_str:
                        lines.append(f"{name}{{{label_str}}} {value}")
                    else:
                        lines.append(f"{name} {value}")
                lines.append("")

            # Histograms
            for key, hist in sorted(self._histograms.items()):
                name, label_str = self._parse_key(key)
                lines.append(f"# HELP {name} Histogram metric")
                lines.append(f"# TYPE {name} histogram")
                prefix = f"{{{label_str}," if label_str else "{"
                for bucket, count in sorted(hist.counts.items()):
                    le = "+Inf" if bucket == float("inf") else str(bucket)
                    lines.append(f'{name}_bucket{prefix}le="{le}"}} {count}')
                lines.append(f"{name}_sum{{{label_str}}}" if label_str
                             else f"{name}_sum" + f" {hist.total:.6f}")
                lines.append(f"{name}_count{{{label_str}}}" if label_str
                             else f"{name}_count" + f" {hist.count}")
                lines.append("")

        return "\n".join(lines) + "\n"

    # ── Internals ─────────────────────────────────────────────
    @staticmethod
    def _label_key(name: str, labels: dict[str, str]) -> str:
        if not labels:
            return name
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}|{label_str}"

    @staticmethod
    def _parse_key(key: str) -> tuple[str, str]:
        if "|" in key:
            name, label_str = key.split("|", 1)
            return name, label_str
        return key, ""

    @staticmethod
    def _group_by_name(
        store: dict[str, float],
    ) -> dict[str, list[tuple[str, float]]]:
        groups: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for key, value in store.items():
            name, label_str = MetricsCollector._parse_key(key)
            groups[name].append((label_str, value))
        return groups


# ── Singleton ─────────────────────────────────────────────────
_metrics: MetricsCollector | None = None


def get_metrics() -> MetricsCollector:
    """Return the global metrics collector singleton."""
    global _metrics  # noqa: PLW0603
    if _metrics is None:
        _metrics = MetricsCollector()
    return _metrics

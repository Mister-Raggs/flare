"""Log replayer — streams a log file through the detection pipeline at controlled speed."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from flare.clustering.clusterer import Incident, IncidentClusterer
from flare.detection.detector import AnomalyDetector, AnomalyResult
from flare.ingestion.models import LogEvent
from flare.ingestion.parser import LogParser


@dataclass
class WindowResult:
    """Detection output for a single tumbling window of log lines.

    Attributes:
        window_index: Zero-based window number in the replay sequence.
        lines_processed: Number of raw lines consumed in this window.
        events: Parsed log events from this window.
        anomalies: Blocks flagged as anomalous by the detector.
        incidents: Clustered incidents derived from the anomalies.
        elapsed_ms: Wall-clock time to process this window in milliseconds.
    """

    window_index: int
    lines_processed: int
    events: list[LogEvent] = field(default_factory=list)
    anomalies: list[AnomalyResult] = field(default_factory=list)
    incidents: list[Incident] = field(default_factory=list)
    elapsed_ms: float = 0.0

    @property
    def anomaly_count(self) -> int:
        return len(self.anomalies)

    @property
    def incident_count(self) -> int:
        return len(self.incidents)


class LogReplayer:
    """Replays a log file through the detection pipeline at a controlled rate.

    Lines are emitted at ``rate`` lines-per-second. Every ``window`` lines,
    the accumulated events are run through anomaly detection and incident
    clustering — producing a :class:`WindowResult` for each window.

    Template state is preserved across windows via a shared ``LogParser``
    instance. If ``persist_path`` is supplied, Drain3's cluster state
    survives process restarts too.

    Example::

        replayer = LogReplayer("logs/hdfs_sample.log", rate=20, window=30)
        for result in replayer.replay():
            if result.incidents:
                print(f"Window {result.window_index}: {result.incident_count} incident(s)")

    Args:
        filepath: Path to the log file to replay.
        rate: Lines to emit per second. Use ``0`` or ``None`` to replay as
            fast as possible (useful for testing).
        window: Number of lines per tumbling detection window.
        log_format: Passed directly to :class:`LogParser` — ``"auto"``,
            ``"hdfs"``, ``"generic"``, etc.
        contamination: Expected anomaly fraction passed to
            :class:`AnomalyDetector`.
        persist_path: Optional file path for Drain3 template persistence.
    """

    def __init__(
        self,
        filepath: str | Path,
        rate: float | None = 10.0,
        window: int = 50,
        log_format: str = "auto",
        contamination: float = 0.03,
        persist_path: str | None = None,
    ) -> None:
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"Log file not found: {self.filepath}")

        self.rate = rate
        self.window = window
        self.contamination = contamination

        self._parser = LogParser(log_format, persist_path=persist_path)
        self._clusterer = IncidentClusterer()

    def replay(self) -> Iterator[WindowResult]:
        """Stream the log file through the pipeline, yielding one WindowResult per window.

        Sleeps between lines to honour the configured ``rate``. Detection runs
        at the end of each complete window; a partial trailing window is also
        processed if it contains at least two distinct blocks (minimum needed
        for Isolation Forest).

        Yields:
            WindowResult for each window of ``self.window`` lines.
        """
        lines = self.filepath.read_text(encoding="utf-8", errors="replace").splitlines()
        lines = [line for line in lines if line.strip()]

        sleep_per_line = (1.0 / self.rate) if self.rate else 0.0

        buffer: list[str] = []
        window_index = 0

        for raw_line in lines:
            buffer.append(raw_line)

            if sleep_per_line:
                time.sleep(sleep_per_line)

            if len(buffer) >= self.window:
                result = self._process_window(buffer, window_index)
                yield result
                window_index += 1
                buffer = []

        # Trailing partial window
        if buffer:
            result = self._process_window(buffer, window_index)
            yield result

    def _process_window(self, lines: list[str], window_index: int) -> WindowResult:
        """Parse, detect, and cluster a single window of raw log lines."""
        t0 = time.monotonic()

        events: list[LogEvent] = []
        for line_id, raw_line in enumerate(lines, start=1):
            event = self._parser.parse_line(raw_line, line_id)
            if event is not None:
                events.append(event)

        anomalies: list[AnomalyResult] = []
        incidents: list[Incident] = []

        # Need at least 2 blocks for Isolation Forest to be meaningful
        unique_blocks = {e.block_id for e in events if e.block_id}
        if len(unique_blocks) >= 2:
            detector = AnomalyDetector(contamination=self.contamination)
            all_results = detector.detect(events)
            anomalies = [r for r in all_results if r.is_anomaly]
            if anomalies:
                incidents = self._clusterer.cluster(all_results, events=events)

        elapsed_ms = (time.monotonic() - t0) * 1000

        return WindowResult(
            window_index=window_index,
            lines_processed=len(lines),
            events=events,
            anomalies=anomalies,
            incidents=incidents,
            elapsed_ms=elapsed_ms,
        )

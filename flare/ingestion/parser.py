"""Log parser with Drain3 templating — supports registered and unknown formats."""

from __future__ import annotations

import re
from pathlib import Path

from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence
from drain3.template_miner_config import TemplateMinerConfig

from flare.ingestion.formats import (
    FORMATS,
    LogFormat,
    detect_entity_field,
    detect_format,
    parse_line_generic,
)
from flare.ingestion.models import LogEvent, LogLevel, ParsedLogBatch


def _parse_level(raw: str) -> LogLevel:
    """Convert a raw level string to LogLevel enum."""
    try:
        return LogLevel(raw.upper())
    except ValueError:
        return LogLevel.UNKNOWN


def _create_template_miner(
    sim_th: float = 0.4,
    depth: int = 4,
    persist_path: str | None = None,
) -> TemplateMiner:
    """Create a configured Drain3 TemplateMiner instance.

    If persist_path is given, templates are loaded from and saved to that
    file automatically by Drain3's FilePersistence handler.
    """
    config = TemplateMinerConfig()
    config.drain_sim_th = sim_th
    config.drain_depth = depth
    config.profiling_enabled = False

    persistence = FilePersistence(persist_path) if persist_path else None
    return TemplateMiner(persistence_handler=persistence, config=config)


class LogParser:
    """Parses raw log files and applies Drain3 template mining.

    Supports three modes:
      - Named format: ``LogParser("hdfs")`` — uses the registered format.
      - Auto-detect:  ``LogParser("auto")`` — sniffs the first lines of the
        file and picks the best registered format, falling back to generic.
      - Generic:      ``LogParser("generic")`` — heuristic parsing for any
        log format without a registered definition.

    Example:
        >>> parser = LogParser("auto")
        >>> batch = parser.parse_file("logs/hdfs_sample.log")
        >>> print(f"Parsed {batch.total_lines} lines, {batch.template_count} templates")
    """

    def __init__(
        self,
        log_format: str = "auto",
        persist_path: str | None = None,
    ) -> None:
        """Initialize the parser.

        Args:
            log_format: Format name from the registry, "auto" for
                auto-detection, or "generic" for heuristic parsing.
            persist_path: Optional file path for Drain3 template persistence.
                When set, templates are loaded on startup and saved after each
                call to ``add_log_message``, so state survives restarts.

        Raises:
            ValueError: If a named format is not found in the registry.
        """
        self.log_format = log_format
        self._persist_path = persist_path
        self._format: LogFormat | None = None
        self._entity_pattern: re.Pattern[str] | None = None

        if log_format not in ("auto", "generic"):
            if log_format not in FORMATS:
                raise ValueError(
                    f"Unknown log format '{log_format}'. "
                    f"Available: {', '.join(FORMATS)}, auto, generic"
                )
            self._format = FORMATS[log_format]
            self._entity_pattern = self._format.entity_pattern

        sim_th = self._format.drain_sim_th if self._format else 0.4
        depth = self._format.drain_depth if self._format else 4
        self._miner = _create_template_miner(sim_th, depth, persist_path)

    def parse_file(self, filepath: str | Path) -> ParsedLogBatch:
        """Parse an entire log file into structured events.

        For auto mode, reads the first lines to detect the format before
        parsing the full file.

        Args:
            filepath: Path to the raw log file.

        Returns:
            ParsedLogBatch containing all parsed events and metadata.

        Raises:
            FileNotFoundError: If the log file does not exist.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Log file not found: {filepath}")

        lines = self._read_lines(filepath)
        if not lines:
            return ParsedLogBatch(events=[], total_lines=0)

        # Auto-detect format if needed
        if self.log_format == "auto" and self._format is None:
            self._resolve_format(lines)

        events: list[LogEvent] = []
        parse_errors = 0

        for line_id, raw_line in enumerate(lines, start=1):
            event = self.parse_line(raw_line, line_id)
            if event is None:
                parse_errors += 1
            else:
                events.append(event)

        return ParsedLogBatch(
            events=events,
            total_lines=len(lines),
            parse_errors=parse_errors,
            template_count=len(self._miner.drain.clusters),
        )

    def parse_line(self, raw_line: str, line_id: int = 0) -> LogEvent | None:
        """Parse a single log line into a structured LogEvent.

        Uses the registered format's regex if available, otherwise
        falls back to heuristic generic parsing.

        Args:
            raw_line: Raw log line string.
            line_id: Line number in the source file.

        Returns:
            LogEvent if parsing succeeds, None otherwise.
        """
        if self._format is not None:
            return self._parse_with_format(raw_line, line_id)

        # Auto mode without a resolved format: try each registered format
        if self.log_format == "auto":
            for fmt in FORMATS.values():
                if fmt.line_pattern.match(raw_line):
                    self._format = fmt
                    self._entity_pattern = fmt.entity_pattern
                    self._miner = _create_template_miner(
                        fmt.drain_sim_th, fmt.drain_depth
                    )
                    return self._parse_with_format(raw_line, line_id)

        return self._parse_generic(raw_line, line_id)

    def _parse_with_format(
        self, raw_line: str, line_id: int
    ) -> LogEvent | None:
        """Parse a line using a registered LogFormat."""
        fmt = self._format
        assert fmt is not None

        match = fmt.line_pattern.match(raw_line)
        if not match:
            return None

        groups = match.groupdict()
        content = groups.get("content", raw_line)

        # Build timestamp from configured keys
        ts_parts = [groups[k] for k in fmt.timestamp_keys if k in groups]
        timestamp = " ".join(ts_parts)

        # Drain3 template mining
        result = self._miner.add_log_message(content)
        cluster_id = (
            result.get("cluster_id", -1)
            if isinstance(result, dict)
            else -1
        )
        template = self._get_template(content)

        # Entity extraction
        entity_id = ""
        if self._entity_pattern:
            entity_match = self._entity_pattern.search(content)
            if not entity_match:
                # Also try the full line for patterns like sshd[pid]
                entity_match = self._entity_pattern.search(raw_line)
            if entity_match:
                entity_id = entity_match.group(1)

        params = self._extract_params(content, template)

        return LogEvent(
            line_id=line_id,
            timestamp=timestamp,
            level=_parse_level(groups.get("level", "UNKNOWN")),
            component=groups.get("component", ""),
            content=content,
            template=template,
            template_id=cluster_id,
            block_id=entity_id,
            params=params,
        )

    def _parse_generic(
        self, raw_line: str, line_id: int
    ) -> LogEvent | None:
        """Parse a line using heuristic generic parsing."""
        if not raw_line.strip():
            return None

        parsed = parse_line_generic(raw_line)
        content = parsed.content

        # Drain3 template mining
        result = self._miner.add_log_message(content)
        cluster_id = (
            result.get("cluster_id", -1)
            if isinstance(result, dict)
            else -1
        )
        template = self._get_template(content)

        # Entity extraction from auto-detected pattern
        entity_id = ""
        if self._entity_pattern:
            entity_match = self._entity_pattern.search(raw_line)
            if entity_match:
                entity_id = (
                    entity_match.group(1)
                    if entity_match.lastindex
                    else entity_match.group(0)
                )

        params = self._extract_params(content, template)

        return LogEvent(
            line_id=line_id,
            timestamp=parsed.timestamp,
            level=_parse_level(parsed.level),
            component=parsed.component,
            content=content,
            template=template,
            template_id=cluster_id,
            block_id=entity_id,
            params=params,
        )

    def _resolve_format(self, lines: list[str]) -> None:
        """Auto-detect format and configure the parser accordingly."""
        detected = detect_format(lines)
        if detected:
            self._format = detected
            self._entity_pattern = detected.entity_pattern
            # Re-init miner with format-specific Drain3 config + persistence
            self._miner = _create_template_miner(
                detected.drain_sim_th, detected.drain_depth, self._persist_path
            )
        else:
            # Generic fallback — try to find an entity ID pattern
            self._entity_pattern = detect_entity_field(lines)

    def _get_template(self, content: str) -> str:
        """Get the Drain3 template for a log message."""
        matched_cluster = self._miner.match(content)
        if matched_cluster:
            return str(matched_cluster.get_template())
        return content

    def _extract_params(self, content: str, template: str) -> list[str]:
        """Extract parameter values by comparing content against its template."""
        params: list[str] = []
        template_tokens = template.split()
        content_tokens = content.split()

        for t_tok, c_tok in zip(template_tokens, content_tokens):
            if t_tok == "<*>":
                params.append(c_tok)

        return params

    def reset(self, clear_persistence: bool = False) -> None:
        """Reset the template miner state and format detection.

        Args:
            clear_persistence: If True and a persist_path was configured,
                delete the snapshot file so the next run starts from scratch.
        """
        if clear_persistence and self._persist_path:
            snap = Path(self._persist_path)
            if snap.exists():
                snap.unlink()

        sim_th = self._format.drain_sim_th if self._format else 0.4
        depth = self._format.drain_depth if self._format else 4
        self._miner = _create_template_miner(sim_th, depth, self._persist_path)
        if self.log_format == "auto":
            self._format = None
            self._entity_pattern = None

    @staticmethod
    def _read_lines(filepath: Path) -> list[str]:
        """Read and strip non-empty lines from a file."""
        with open(filepath, encoding="utf-8", errors="replace") as f:
            return [line.strip() for line in f if line.strip()]

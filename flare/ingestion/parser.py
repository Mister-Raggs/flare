"""Log parser with Drain3 templating for HDFS and generic log formats."""

from __future__ import annotations

import re
from pathlib import Path

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

from flare.ingestion.models import LogEvent, LogLevel, ParsedLogBatch

# HDFS log line pattern:
# 081109 203518 148 INFO dfs.DataNode$DataXceiver: Receiving block blk_-123 src: /x dest: /y
_HDFS_PATTERN = re.compile(
    r"^(?P<date>\d{6})\s+"
    r"(?P<time>\d{6})\s+"
    r"(?P<pid>\d+)\s+"
    r"(?P<level>\w+)\s+"
    r"(?P<component>\S+):\s+"
    r"(?P<content>.+)$"
)

# Extract HDFS block IDs like blk_-1608999687919862906 or blk_7503483334202473044
_BLOCK_ID_PATTERN = re.compile(r"(blk_-?\d+)")


def _parse_level(raw: str) -> LogLevel:
    """Convert a raw level string to LogLevel enum."""
    try:
        return LogLevel(raw.upper())
    except ValueError:
        return LogLevel.UNKNOWN


def _create_template_miner() -> TemplateMiner:
    """Create a configured Drain3 TemplateMiner instance."""
    config = TemplateMinerConfig()
    config.drain_sim_th = 0.4
    config.drain_depth = 4
    config.profiling_enabled = False
    return TemplateMiner(config=config)


class LogParser:
    """Parses raw log files and applies Drain3 template mining.

    Supports HDFS log format out of the box. Can be extended to handle
    other formats by providing custom regex patterns.

    Example:
        >>> parser = LogParser()
        >>> batch = parser.parse_file("logs/hdfs_sample.log")
        >>> print(f"Parsed {batch.total_lines} lines, {batch.template_count} templates")
    """

    def __init__(self, log_format: str = "hdfs") -> None:
        """Initialize the parser.

        Args:
            log_format: Log format to use. Currently supports 'hdfs'.
        """
        self.log_format = log_format
        self._miner = _create_template_miner()
        self._pattern = _HDFS_PATTERN

    def parse_file(self, filepath: str | Path) -> ParsedLogBatch:
        """Parse an entire log file into structured events.

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

        events: list[LogEvent] = []
        parse_errors = 0

        with open(filepath, encoding="utf-8", errors="replace") as f:
            for line_id, raw_line in enumerate(f, start=1):
                raw_line = raw_line.strip()
                if not raw_line:
                    continue

                event = self.parse_line(raw_line, line_id)
                if event is None:
                    parse_errors += 1
                else:
                    events.append(event)

        return ParsedLogBatch(
            events=events,
            total_lines=line_id if events or parse_errors else 0,
            parse_errors=parse_errors,
            template_count=len(self._miner.drain.clusters),
        )

    def parse_line(self, raw_line: str, line_id: int = 0) -> LogEvent | None:
        """Parse a single log line into a structured LogEvent.

        Args:
            raw_line: Raw log line string.
            line_id: Line number in the source file.

        Returns:
            LogEvent if parsing succeeds, None otherwise.
        """
        match = self._pattern.match(raw_line)
        if not match:
            return None

        groups = match.groupdict()
        content = groups["content"]

        # Apply Drain3 template mining
        result = self._miner.add_log_message(content)
        cluster = result.get("cluster_id", -1) if isinstance(result, dict) else -1

        # Get the template for this cluster
        template = self._get_template(content)

        # Extract block IDs from content
        block_ids = _BLOCK_ID_PATTERN.findall(content)
        block_id = block_ids[0] if block_ids else ""

        # Extract parameters (values that differ from template)
        params = self._extract_params(content, template)

        return LogEvent(
            line_id=line_id,
            timestamp=f"{groups['date']} {groups['time']}",
            level=_parse_level(groups["level"]),
            component=groups["component"],
            content=content,
            template=template,
            template_id=cluster,
            block_id=block_id,
            params=params,
        )

    def _get_template(self, content: str) -> str:
        """Get the Drain3 template for a log message."""
        matched_cluster = self._miner.match(content)
        if matched_cluster:
            return matched_cluster.get_template()
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

    def reset(self) -> None:
        """Reset the template miner state."""
        self._miner = _create_template_miner()

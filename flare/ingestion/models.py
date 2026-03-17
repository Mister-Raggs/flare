"""Data models for structured log events."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class LogLevel(StrEnum):
    """Standard log severity levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"
    FATAL = "FATAL"
    UNKNOWN = "UNKNOWN"


@dataclass
class LogEvent:
    """A single structured log event after parsing.

    Attributes:
        line_id: Original line number in the raw log file.
        timestamp: Raw timestamp string from the log line.
        level: Parsed log severity level.
        component: Source component or process that emitted the log.
        content: Original raw log message content.
        template: Drain3-extracted log template (parameterized form).
        template_id: Unique identifier for the log template cluster.
        block_id: Session/block identifier (e.g., HDFS block ID).
        params: Extracted parameter values from the template.
    """

    line_id: int
    timestamp: str
    level: LogLevel
    component: str
    content: str
    template: str = ""
    template_id: int = -1
    block_id: str = ""
    params: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "line_id": self.line_id,
            "timestamp": self.timestamp,
            "level": self.level.value,
            "component": self.component,
            "content": self.content,
            "template": self.template,
            "template_id": self.template_id,
            "block_id": self.block_id,
            "params": self.params,
        }


@dataclass
class ParsedLogBatch:
    """A batch of parsed log events with metadata.

    Attributes:
        events: List of parsed log events.
        total_lines: Total number of lines processed.
        parse_errors: Number of lines that failed to parse.
        template_count: Number of unique templates discovered.
    """

    events: list[LogEvent]
    total_lines: int = 0
    parse_errors: int = 0
    template_count: int = 0

    @property
    def block_ids(self) -> set[str]:
        """Return unique block IDs found in this batch."""
        return {e.block_id for e in self.events if e.block_id}

    def events_for_block(self, block_id: str) -> list[LogEvent]:
        """Return all events belonging to a specific block."""
        return [e for e in self.events if e.block_id == block_id]

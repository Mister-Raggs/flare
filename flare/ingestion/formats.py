"""Log format definitions and auto-detection for multi-format parsing."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class LogFormat:
    """Defines how to parse a specific log format.

    Attributes:
        name: Short identifier (e.g., "hdfs", "openssh", "syslog").
        line_pattern: Regex with named groups. Must capture at least 'content'.
            Optional groups: 'timestamp', 'level', 'component'.
        entity_pattern: Regex to extract a grouping key (block ID, session ID,
            request ID) from the content field. None means no entity grouping.
        timestamp_keys: Named groups in line_pattern that form the timestamp.
            Joined with a space if multiple (e.g., ["date", "time"] -> "081109 203518").
        drain_sim_th: Drain3 similarity threshold tuned for this format.
        drain_depth: Drain3 parse tree depth tuned for this format.
        sample_lines: A few example lines used for auto-detection scoring.
    """

    name: str
    line_pattern: re.Pattern[str]
    entity_pattern: re.Pattern[str] | None = None
    timestamp_keys: list[str] = field(default_factory=lambda: ["timestamp"])
    drain_sim_th: float = 0.4
    drain_depth: int = 4
    sample_lines: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Built-in format registry
# ---------------------------------------------------------------------------

HDFS = LogFormat(
    name="hdfs",
    line_pattern=re.compile(
        r"^(?P<date>\d{6})\s+"
        r"(?P<time>\d{6})\s+"
        r"(?P<pid>\d+)\s+"
        r"(?P<level>\w+)\s+"
        r"(?P<component>\S+):\s+"
        r"(?P<content>.+)$"
    ),
    entity_pattern=re.compile(r"(blk_-?\d+)"),
    timestamp_keys=["date", "time"],
    drain_sim_th=0.4,
    drain_depth=4,
)

OPENSSH = LogFormat(
    name="openssh",
    line_pattern=re.compile(
        r"^(?P<timestamp>\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+"
        r"(?P<component>\S+)\s+"
        r"sshd\[(?P<pid>\d+)\]:\s+"
        r"(?P<content>.+)$"
    ),
    entity_pattern=re.compile(r"sshd\[(\d+)\]"),
    timestamp_keys=["timestamp"],
    drain_sim_th=0.5,
    drain_depth=4,
)

SYSLOG = LogFormat(
    name="syslog",
    line_pattern=re.compile(
        r"^(?P<timestamp>\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+"
        r"(?P<host>\S+)\s+"
        r"(?P<component>[^\[:]+)(?:\[(?P<pid>\d+)\])?:\s+"
        r"(?P<content>.+)$"
    ),
    entity_pattern=None,
    timestamp_keys=["timestamp"],
    drain_sim_th=0.5,
    drain_depth=3,
)

# Registry: name -> LogFormat
FORMATS: dict[str, LogFormat] = {
    fmt.name: fmt for fmt in [HDFS, OPENSSH, SYSLOG]
}


# ---------------------------------------------------------------------------
# Generic (fallback) parsing helpers
# ---------------------------------------------------------------------------

# Ordered by specificity — first match wins
_TIMESTAMP_PATTERNS: list[re.Pattern[str]] = [
    # ISO 8601: 2024-03-17T10:23:45.123Z or 2024-03-17 10:23:45,123
    re.compile(
        r"^(?P<timestamp>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}[.,]?\d*Z?)\s+"
    ),
    # Syslog: Mar 17 10:23:45
    re.compile(
        r"^(?P<timestamp>\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+"
    ),
    # HDFS-style: 081109 203518
    re.compile(r"^(?P<timestamp>\d{6}\s+\d{6})\s+"),
    # Epoch seconds/millis: 1710672225 or 1710672225123
    re.compile(r"^(?P<timestamp>\d{10,13})\s+"),
    # Common date: 2024/03/17 10:23:45 or 17/Mar/2024:10:23:45
    re.compile(
        r"^(?P<timestamp>\d{2,4}[/\-]\w{2,3}[/\-]\d{2,4}[: ]\d{2}:\d{2}:\d{2})\s+"
    ),
]

_LEVEL_PATTERN = re.compile(
    r"\b(?P<level>DEBUG|INFO|WARN(?:ING)?|ERROR|FATAL|CRITICAL|TRACE|NOTICE|ALERT|EMERG)\b",
    re.IGNORECASE,
)

# Component patterns: [component], component:, process[pid]:
_COMPONENT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\[(?P<component>[a-zA-Z][\w./-]{1,60})\]"),
    re.compile(r"(?P<component>[a-zA-Z][\w./-]{1,60})\[\d+\]:"),
    re.compile(r"(?P<component>[a-zA-Z][\w./-]{1,60}):\s"),
]

# Entity ID candidates: UUIDs, hex IDs, common key=value ID patterns
_ENTITY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I),
    re.compile(
        r"(?:req|request|trace|span|session|tx|correlation)"
        r"[_\-]?[iI][dD][=: ]+([a-zA-Z0-9\-_]{8,})"
    ),
    re.compile(r"\b[0-9a-f]{12,32}\b"),
]


@dataclass
class GenericParseResult:
    """Result of heuristic parsing on a single line."""

    timestamp: str
    level: str
    component: str
    content: str


def parse_line_generic(raw_line: str) -> GenericParseResult:
    """Best-effort parse of an arbitrary log line.

    Extracts timestamp, level, and component via heuristics,
    leaving the remainder as content for Drain3.
    """
    remaining = raw_line
    timestamp = ""
    level = ""
    component = ""

    # 1. Extract timestamp
    for pat in _TIMESTAMP_PATTERNS:
        m = pat.match(remaining)
        if m:
            timestamp = m.group("timestamp")
            remaining = remaining[m.end():]
            break

    # 2. Extract level
    m = _LEVEL_PATTERN.search(remaining)
    if m:
        level = m.group("level").upper()
        # Remove level from remaining, keeping surrounding text as content
        start, end = m.span()
        remaining = (remaining[:start] + remaining[end:]).strip()

    # 3. Extract component (first match only)
    for pat in _COMPONENT_PATTERNS:
        m = pat.search(remaining)
        if m:
            component = m.group("component")
            break

    content = remaining.strip()
    if not content:
        content = raw_line

    return GenericParseResult(
        timestamp=timestamp,
        level=level if level else "UNKNOWN",
        component=component,
        content=content,
    )


def detect_entity_field(lines: list[str], threshold: float = 0.3) -> re.Pattern[str] | None:
    """Scan lines for a recurring ID-like token to use as grouping key.

    Returns the pattern that matches in >= threshold fraction of lines,
    or None if no good candidate is found.
    """
    if not lines:
        return None

    sample = lines[:200]
    best_pattern: re.Pattern[str] | None = None
    best_rate = 0.0

    for pat in _ENTITY_PATTERNS:
        hits = sum(1 for line in sample if pat.search(line))
        rate = hits / len(sample)
        if rate >= threshold and rate > best_rate:
            best_rate = rate
            best_pattern = pat

    return best_pattern


def detect_format(lines: list[str]) -> LogFormat | None:
    """Auto-detect log format by trying registered formats against sample lines.

    Returns the best matching LogFormat, or None if no format matches
    well enough (triggers generic fallback).
    """
    if not lines:
        return None

    sample = lines[:20]
    best_format: LogFormat | None = None
    best_score = 0.0

    for fmt in FORMATS.values():
        hits = sum(1 for line in sample if fmt.line_pattern.match(line))
        score = hits / len(sample)
        if score > best_score:
            best_score = score
            best_format = fmt

    # Require >80% match to confidently pick a format
    if best_score >= 0.8 and best_format is not None:
        return best_format

    return None

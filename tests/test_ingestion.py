"""Tests for the log ingestion module."""

from pathlib import Path

import pytest

from flare.ingestion.formats import (
    FORMATS,
    detect_entity_field,
    detect_format,
    parse_line_generic,
)
from flare.ingestion.models import LogEvent, LogLevel, ParsedLogBatch
from flare.ingestion.parser import LogParser

SAMPLE_LOG = Path(__file__).parent.parent / "logs" / "hdfs_sample.log"


class TestLogEvent:
    def test_to_dict(self) -> None:
        event = LogEvent(
            line_id=1,
            timestamp="081109 203518",
            level=LogLevel.INFO,
            component="dfs.DataNode",
            content="Receiving block blk_123",
            template="Receiving block <*>",
            template_id=0,
            block_id="blk_123",
            params=["blk_123"],
        )
        d = event.to_dict()
        assert d["line_id"] == 1
        assert d["level"] == "INFO"
        assert d["block_id"] == "blk_123"

    def test_log_level_parsing(self) -> None:
        assert LogLevel("INFO") == LogLevel.INFO
        assert LogLevel("ERROR") == LogLevel.ERROR
        assert LogLevel("WARN") == LogLevel.WARN


class TestParsedLogBatch:
    def test_block_ids(self) -> None:
        events = [
            LogEvent(1, "t", LogLevel.INFO, "c", "msg", block_id="blk_1"),
            LogEvent(2, "t", LogLevel.INFO, "c", "msg", block_id="blk_2"),
            LogEvent(3, "t", LogLevel.INFO, "c", "msg", block_id="blk_1"),
        ]
        batch = ParsedLogBatch(events=events)
        assert batch.block_ids == {"blk_1", "blk_2"}

    def test_events_for_block(self) -> None:
        events = [
            LogEvent(1, "t", LogLevel.INFO, "c", "msg", block_id="blk_1"),
            LogEvent(2, "t", LogLevel.INFO, "c", "msg", block_id="blk_2"),
            LogEvent(3, "t", LogLevel.INFO, "c", "msg", block_id="blk_1"),
        ]
        batch = ParsedLogBatch(events=events)
        assert len(batch.events_for_block("blk_1")) == 2


class TestLogParser:
    def test_parse_single_line(self) -> None:
        parser = LogParser()
        line = (
            "081109 203518 148 INFO dfs.DataNode$DataXceiver: "
            "Receiving block blk_-1608999687919862906 "
            "src: /10.250.19.102:54106 dest: /10.250.19.102:50010"
        )
        event = parser.parse_line(line, line_id=1)

        assert event is not None
        assert event.line_id == 1
        assert event.timestamp == "081109 203518"
        assert event.level == LogLevel.INFO
        assert event.component == "dfs.DataNode$DataXceiver"
        assert "Receiving block" in event.content
        assert event.block_id == "blk_-1608999687919862906"

    def test_parse_error_line(self) -> None:
        parser = LogParser()
        line = (
            "081109 204005 550 ERROR dfs.DataNode$DataXceiver: "
            "10.250.14.224:50010:DataXceiver: java.io.IOException: "
            "Connection reset by peer"
        )
        event = parser.parse_line(line, line_id=1)

        assert event is not None
        assert event.level == LogLevel.ERROR

    def test_parse_invalid_line_with_known_format(self) -> None:
        parser = LogParser("hdfs")
        event = parser.parse_line("this is not a valid log line", line_id=1)
        assert event is None

    def test_parse_invalid_line_generic_fallback(self) -> None:
        parser = LogParser()
        event = parser.parse_line("this is not a valid log line", line_id=1)
        # Generic mode parses anything — no format match, best-effort result
        assert event is not None
        assert event.level == LogLevel.UNKNOWN
        assert event.block_id == ""

    def test_parse_file(self) -> None:
        parser = LogParser()
        batch = parser.parse_file(SAMPLE_LOG)

        assert batch.total_lines > 0
        assert len(batch.events) > 0
        assert batch.template_count > 0
        assert batch.parse_errors == 0

    def test_parse_file_not_found(self) -> None:
        parser = LogParser()
        with pytest.raises(FileNotFoundError):
            parser.parse_file("nonexistent.log")

    def test_block_ids_extracted(self) -> None:
        parser = LogParser()
        batch = parser.parse_file(SAMPLE_LOG)
        assert len(batch.block_ids) > 0
        assert all(bid.startswith("blk_") for bid in batch.block_ids)

    def test_reset(self) -> None:
        parser = LogParser()
        line = (
            "081109 203518 148 INFO dfs.DataNode$DataXceiver: "
            "Receiving block blk_123 src: /a dest: /b"
        )
        parser.parse_line(line, line_id=1)
        parser.reset()
        # After reset, template miner should be fresh
        assert len(parser._miner.drain.clusters) == 0


# ---------------------------------------------------------------------------
# Format registry tests
# ---------------------------------------------------------------------------


class TestFormatRegistry:
    def test_hdfs_in_registry(self) -> None:
        assert "hdfs" in FORMATS

    def test_openssh_in_registry(self) -> None:
        assert "openssh" in FORMATS

    def test_syslog_in_registry(self) -> None:
        assert "syslog" in FORMATS

    def test_unknown_format_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown log format"):
            LogParser("nonexistent_format")

    def test_hdfs_pattern_matches(self) -> None:
        line = (
            "081109 203518 148 INFO dfs.DataNode$DataXceiver: "
            "Receiving block blk_123 src: /a dest: /b"
        )
        assert FORMATS["hdfs"].line_pattern.match(line)

    def test_openssh_pattern_matches(self) -> None:
        line = (
            "Dec 10 06:55:48 LabSZ sshd[24200]: "
            "Invalid user webmaster from 173.234.31.186"
        )
        assert FORMATS["openssh"].line_pattern.match(line)

    def test_syslog_pattern_matches(self) -> None:
        line = (
            "Jan  5 14:32:01 myhost kernel[0]: "
            "en0: received unsolicited IPv6 router advertisement"
        )
        assert FORMATS["syslog"].line_pattern.match(line)


# ---------------------------------------------------------------------------
# Auto-detection tests
# ---------------------------------------------------------------------------


class TestDetectFormat:
    def test_detect_hdfs(self) -> None:
        lines = [
            "081109 203518 148 INFO dfs.DataNode$DataXceiver: Receiving block blk_1",
            "081109 203518 148 INFO dfs.DataNode$DataXceiver: Receiving block blk_2",
            "081109 203519 148 INFO dfs.DataNode$PacketResponder: "
            "PacketResponder 1 for block blk_1 terminating",
        ]
        fmt = detect_format(lines)
        assert fmt is not None
        assert fmt.name == "hdfs"

    def test_detect_openssh(self) -> None:
        lines = [
            "Dec 10 06:55:48 LabSZ sshd[24200]: Invalid user webmaster from 173.234.31.186",
            "Dec 10 06:55:48 LabSZ sshd[24200]: pam_unix(sshd:auth): check pass; user unknown",
            "Dec 10 06:55:48 LabSZ sshd[24200]: Failed password for invalid user webmaster",
        ]
        fmt = detect_format(lines)
        assert fmt is not None
        assert fmt.name == "openssh"

    def test_detect_returns_none_for_unknown(self) -> None:
        lines = [
            "just some random text",
            "another random line here",
            "nothing structured about this",
        ]
        fmt = detect_format(lines)
        assert fmt is None

    def test_detect_empty_lines(self) -> None:
        assert detect_format([]) is None


# ---------------------------------------------------------------------------
# Generic parsing tests
# ---------------------------------------------------------------------------


class TestGenericParser:
    def test_iso8601_timestamp(self) -> None:
        result = parse_line_generic(
            "2024-03-17T10:23:45.123Z ERROR [app.server] Connection refused"
        )
        assert result.timestamp == "2024-03-17T10:23:45.123Z"
        assert result.level == "ERROR"

    def test_syslog_timestamp(self) -> None:
        result = parse_line_generic(
            "Mar 17 10:23:45 myhost kernel: segfault at 0000000"
        )
        assert result.timestamp == "Mar 17 10:23:45"

    def test_level_extraction(self) -> None:
        result = parse_line_generic("some prefix FATAL something crashed")
        assert result.level == "FATAL"

    def test_warning_variant(self) -> None:
        result = parse_line_generic(
            "2024-01-01T00:00:00Z WARNING disk usage high"
        )
        assert result.level == "WARNING"

    def test_no_level_defaults_unknown(self) -> None:
        result = parse_line_generic("just a plain message with no level")
        assert result.level == "UNKNOWN"

    def test_component_bracket_style(self) -> None:
        result = parse_line_generic(
            "2024-03-17T10:23:45Z INFO [my.component] starting up"
        )
        assert result.component == "my.component"

    def test_component_colon_style(self) -> None:
        result = parse_line_generic(
            "Mar 17 10:23:45 myhost nginx: request completed"
        )
        assert result.component == "nginx"

    def test_content_never_empty(self) -> None:
        result = parse_line_generic("INFO")
        assert result.content != ""

    def test_full_generic_pipeline(self) -> None:
        """Generic parser feeds into LogParser end-to-end."""
        parser = LogParser("generic")
        event = parser.parse_line(
            "2024-03-17T10:23:45Z ERROR [api.handler] "
            "request_id=abc123def456 timeout after 30s",
            line_id=1,
        )
        assert event is not None
        assert event.level == LogLevel.ERROR
        assert event.template_id >= 0


# ---------------------------------------------------------------------------
# Entity detection tests
# ---------------------------------------------------------------------------


class TestEntityDetection:
    def test_detect_uuid_entity(self) -> None:
        lines = [
            "req started trace_id=550e8400-e29b-41d4-a716-446655440000",
            "processing trace_id=550e8400-e29b-41d4-a716-446655440000",
            "req started trace_id=661f9511-f3ac-52e5-b827-557766551111",
            "done trace_id=661f9511-f3ac-52e5-b827-557766551111",
        ]
        pat = detect_entity_field(lines)
        assert pat is not None

    def test_detect_request_id_entity(self) -> None:
        lines = [
            "handling request_id=abcdef1234567890 GET /api",
            "query request_id=abcdef1234567890 SELECT *",
            "response request_id=abcdef1234567890 200 OK",
            "handling request_id=1234567890abcdef GET /health",
        ]
        pat = detect_entity_field(lines)
        assert pat is not None

    def test_no_entity_in_plain_text(self) -> None:
        lines = [
            "the quick brown fox",
            "jumped over the lazy dog",
            "no ids here at all",
        ]
        pat = detect_entity_field(lines)
        assert pat is None

    def test_empty_lines(self) -> None:
        assert detect_entity_field([]) is None


# ---------------------------------------------------------------------------
# Auto-mode integration tests
# ---------------------------------------------------------------------------


class TestAutoMode:
    def test_auto_detects_hdfs_from_file(self) -> None:
        parser = LogParser("auto")
        batch = parser.parse_file(SAMPLE_LOG)
        assert parser._format is not None
        assert parser._format.name == "hdfs"
        assert batch.parse_errors == 0
        assert len(batch.block_ids) > 0

    def test_auto_detects_hdfs_from_single_line(self) -> None:
        parser = LogParser("auto")
        line = (
            "081109 203518 148 INFO dfs.DataNode$DataXceiver: "
            "Receiving block blk_123 src: /a dest: /b"
        )
        event = parser.parse_line(line, line_id=1)
        assert event is not None
        assert event.block_id == "blk_123"
        assert parser._format is not None
        assert parser._format.name == "hdfs"

    def test_auto_falls_back_to_generic(self) -> None:
        parser = LogParser("auto")
        event = parser.parse_line(
            "2024-03-17T10:23:45Z ERROR something broke", line_id=1
        )
        assert event is not None
        assert event.level == LogLevel.ERROR
        # No registered format matched, so _format stays None
        assert parser._format is None

    def test_explicit_hdfs_same_as_auto(self) -> None:
        auto = LogParser("auto")
        explicit = LogParser("hdfs")
        line = (
            "081109 203518 148 INFO dfs.DataNode$DataXceiver: "
            "Receiving block blk_999 src: /a dest: /b"
        )
        e_auto = auto.parse_line(line, line_id=1)
        e_explicit = explicit.parse_line(line, line_id=1)
        assert e_auto is not None
        assert e_explicit is not None
        assert e_auto.block_id == e_explicit.block_id
        assert e_auto.component == e_explicit.component
        assert e_auto.timestamp == e_explicit.timestamp

    def test_reset_clears_auto_detection(self) -> None:
        parser = LogParser("auto")
        line = (
            "081109 203518 148 INFO dfs.DataNode$DataXceiver: "
            "Receiving block blk_123 src: /a dest: /b"
        )
        parser.parse_line(line, line_id=1)
        assert parser._format is not None
        parser.reset()
        assert parser._format is None


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------


class TestPersistence:
    LINE = (
        "081109 203518 148 INFO dfs.DataNode$DataXceiver: "
        "Receiving block blk_123 src: /a dest: /b"
    )

    def test_snapshot_file_created(self, tmp_path) -> None:
        snap = str(tmp_path / "drain.bin")
        parser = LogParser("hdfs", persist_path=snap)
        parser.parse_line(self.LINE, line_id=1)
        assert Path(snap).exists()

    def test_templates_survive_restart(self, tmp_path) -> None:
        snap = str(tmp_path / "drain.bin")

        # First parser: learns the template
        p1 = LogParser("hdfs", persist_path=snap)
        p1.parse_line(self.LINE, line_id=1)
        templates_after_first = len(p1._miner.drain.clusters)
        assert templates_after_first > 0

        # Second parser: loads snapshot, should already know the template
        p2 = LogParser("hdfs", persist_path=snap)
        assert len(p2._miner.drain.clusters) == templates_after_first

    def test_reset_without_clear_keeps_snapshot(self, tmp_path) -> None:
        snap = str(tmp_path / "drain.bin")
        parser = LogParser("hdfs", persist_path=snap)
        parser.parse_line(self.LINE, line_id=1)
        assert Path(snap).exists()

        parser.reset(clear_persistence=False)
        assert Path(snap).exists()

    def test_reset_with_clear_deletes_snapshot(self, tmp_path) -> None:
        snap = str(tmp_path / "drain.bin")
        parser = LogParser("hdfs", persist_path=snap)
        parser.parse_line(self.LINE, line_id=1)
        assert Path(snap).exists()

        parser.reset(clear_persistence=True)
        assert not Path(snap).exists()

    def test_no_persist_path_leaves_no_file(self, tmp_path) -> None:
        parser = LogParser("hdfs")
        parser.parse_line(self.LINE, line_id=1)
        # No snapshot anywhere in tmp_path
        assert list(tmp_path.iterdir()) == []

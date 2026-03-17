"""Tests for the log ingestion module."""

from pathlib import Path

import pytest

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

    def test_parse_invalid_line(self) -> None:
        parser = LogParser()
        event = parser.parse_line("this is not a valid log line", line_id=1)
        assert event is None

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

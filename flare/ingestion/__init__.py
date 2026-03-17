"""Log ingestion: parsing, templating, and structuring raw log data."""

from flare.ingestion.models import LogEvent, ParsedLogBatch
from flare.ingestion.parser import LogParser

__all__ = ["LogParser", "LogEvent", "ParsedLogBatch"]

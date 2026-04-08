"""Demo endpoint — streams a shuffled log sample through the detection pipeline via SSE."""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/demo", tags=["demo"])

_DEMO_LOG = Path(__file__).parent.parent.parent.parent / "logs" / "hdfs_demo.log"
_FALLBACK_LOG = Path(__file__).parent.parent.parent.parent / "logs" / "hdfs_sample.log"

_DEMO_RATE = 5.0
_DEMO_WINDOW = 50
_MIN_LINES = 50
_MAX_LINES = 500


def _sse(event: str, data: dict) -> str:
    """Format a single SSE message."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _stream_demo(n_lines: int):
    """Generator that replays a shuffled log stream and yields SSE events."""
    from flare.replay import LogReplayer, shuffled_stream

    source = _DEMO_LOG if _DEMO_LOG.exists() else _FALLBACK_LOG
    if not source.exists():
        yield _sse("error", {"message": "Demo log not found on server."})
        return

    lines = shuffled_stream(source, n_lines=n_lines)

    yield _sse("start", {
        "message": f"Starting demo — {len(lines)} lines shuffled from {source.name}",
        "lines": len(lines),
        "rate": _DEMO_RATE,
        "window": _DEMO_WINDOW,
    })

    # Write shuffled stream to a temp file for the replayer
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".log", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write("\n".join(lines) + "\n")
        tmp_path = tmp.name

    try:
        replayer = LogReplayer(
            filepath=tmp_path,
            rate=_DEMO_RATE,
            window=_DEMO_WINDOW,
            log_format="hdfs",
            contamination=0.03,
        )

        total_anomalies = 0
        total_events = 0

        for result in replayer.replay():
            total_events += len(result.events)
            total_anomalies += result.anomaly_count

            payload = {
                "window_index": result.window_index,
                "lines_processed": result.lines_processed,
                "events_parsed": len(result.events),
                "templates_seen": len(replayer._parser._miner.drain.clusters),
                "anomaly_count": result.anomaly_count,
                "incident_count": result.incident_count,
                "elapsed_ms": round(result.elapsed_ms, 1),
                "incidents": [
                    {
                        "incident_id": inc.incident_id,
                        "block_ids": inc.block_ids,
                        "severity": round(inc.severity, 4),
                        "templates": inc.templates,
                        "log_lines": inc.log_lines[:5],
                    }
                    for inc in result.incidents
                ],
            }
            yield _sse("window", payload)

        yield _sse("done", {
            "total_events": total_events,
            "total_anomalies": total_anomalies,
            "message": "Replay complete",
        })
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@router.get("/stream")
async def demo_stream(
    lines: int = Query(
        default=100,
        ge=_MIN_LINES,
        le=_MAX_LINES,
        description="Number of log lines to sample (50–500).",
    ),
):
    """Stream a shuffled sample of HDFS logs through the detection pipeline.

    Each call shuffles blocks randomly so results vary between runs.

    Query params:
    - ``lines``: how many lines to include (default 100, range 50–500)

    Returns a Server-Sent Events stream with ``start``, ``window``, ``done``,
    and ``error`` event types.
    """
    return StreamingResponse(
        _stream_demo(lines),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

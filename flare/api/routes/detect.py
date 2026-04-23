"""Detection endpoint: parse logs, detect anomalies, cluster incidents."""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, Request, UploadFile

from flare.api.models import DetectRequest, DetectResponse, IncidentPayload

logger = logging.getLogger("flare.api.detect")
router = APIRouter()


def _run_detection(
    log_text: str,
    contamination: float,
    use_registry: bool = False,
    model_server: Any = None,
) -> DetectResponse:
    """Run the full ingestion → detection → clustering pipeline.

    When ``model_server`` is provided (pre-loaded at API startup), detection
    is pure inference — no fit() on the request path.  Falls back to training
    a fresh model when no server is available.

    This is synchronous because Drain3 and scikit-learn are CPU-bound.
    FastAPI runs it in a threadpool automatically for async endpoints.
    """
    from flare.clustering import IncidentClusterer
    from flare.detection import AnomalyDetector
    from flare.ingestion import LogParser

    start = time.monotonic()

    # Write log text to a temp file for the parser (no cache for temp files)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".log", delete=False, encoding="utf-8"
    ) as f:
        f.write(log_text)
        tmp_path = f.name

    mlflow_run_id = None
    try:
        parser = LogParser()
        batch = parser.parse_file(tmp_path, use_cache=False)

        if not batch.events:
            return DetectResponse(
                incidents=[],
                anomaly_count=0,
                total_blocks=0,
                total_events=0,
                templates_discovered=0,
                processing_time_ms=0,
            )

        if model_server is not None:
            # ── inference-only path (zero training) ───────────────────────
            results = model_server.infer(batch.events)
        else:
            # ── fallback: train a fresh model on the incoming data ─────────
            detector = AnomalyDetector(
                contamination=contamination, use_registry=use_registry
            )
            results = detector.detect(batch.events)
            mlflow_run_id = getattr(detector, "mlflow_run_id", None)

        anomalies = [r for r in results if r.is_anomaly]
        clusterer = IncidentClusterer()
        incidents = clusterer.cluster(results, events=batch.events)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    elapsed_ms = int((time.monotonic() - start) * 1000)

    # Record detection metrics
    from flare.api.metrics import get_metrics

    m = get_metrics()
    m.inc("flare_detection_runs_total")
    m.observe("flare_detection_duration_seconds", (time.monotonic() - start))
    m.inc("flare_anomalies_detected_total", value=len(anomalies))
    m.inc("flare_incidents_clustered_total", value=len(incidents))
    m.set_gauge("flare_last_detection_blocks", float(len(results)))
    m.set_gauge("flare_last_detection_anomalies", float(len(anomalies)))

    incident_payloads = []
    for inc in incidents:
        incident_payloads.append(
            IncidentPayload(
                incident_id=inc.incident_id,
                block_ids=inc.block_ids,
                severity=round(inc.severity, 4),
                mean_anomaly_score=round(float(np.mean(inc.anomaly_scores)), 4),
                log_lines=inc.log_lines,
                templates=inc.templates,
                time_range=list(inc.time_range),
            )
        )

    return DetectResponse(
        incidents=incident_payloads,
        anomaly_count=len(anomalies),
        total_blocks=len(results),
        total_events=len(batch.events),
        templates_discovered=batch.template_count,
        processing_time_ms=elapsed_ms,
        mlflow_run_id=mlflow_run_id,
    )


@router.post(
    "/detect",
    response_model=DetectResponse,
    summary="Detect anomalies in log data",
    description=(
        "Parse raw log text, detect anomalies via Isolation Forest, "
        "and cluster anomalous blocks into incidents."
    ),
)
async def detect(request: DetectRequest, req: Request) -> DetectResponse:
    """Run anomaly detection on raw log text."""
    logger.info(
        "Detection request: %d chars, contamination=%.3f",
        len(request.log_text),
        request.contamination,
    )
    server = getattr(req.app.state, "model_server", None)
    return _run_detection(
        request.log_text, request.contamination, request.use_registry, server
    )


@router.post(
    "/detect/upload",
    response_model=DetectResponse,
    summary="Detect anomalies from uploaded log file",
    description="Upload a .log file for anomaly detection.",
)
async def detect_upload(
    file: UploadFile,
    req: Request,
    contamination: float = 0.03,
    use_registry: bool = False,
) -> DetectResponse:
    """Run anomaly detection on an uploaded log file."""
    content = await file.read()
    log_text = content.decode("utf-8", errors="replace")
    logger.info(
        "Detection upload: file=%s, %d bytes, contamination=%.3f",
        file.filename,
        len(content),
        contamination,
    )
    server = getattr(req.app.state, "model_server", None)
    return _run_detection(log_text, contamination, use_registry, server)

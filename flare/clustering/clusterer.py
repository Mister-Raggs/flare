"""Incident clustering: groups related anomalous blocks using DBSCAN."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from flare.detection.detector import AnomalyResult
from flare.ingestion.models import LogEvent


@dataclass
class Incident:
    """A group of related anomalous blocks forming a single incident.

    Attributes:
        incident_id: Unique identifier for this incident.
        block_ids: List of block IDs belonging to this incident.
        severity: Computed severity score (0-1 scale).
        anomaly_scores: List of anomaly scores for constituent blocks.
        log_lines: Raw log line contents for LLM context.
        templates: Unique log templates observed across this incident.
        time_range: Tuple of (earliest_timestamp, latest_timestamp).
        summary: Human-readable summary (populated by LLM).
    """

    incident_id: int
    block_ids: list[str]
    severity: float
    anomaly_scores: list[float] = field(default_factory=list)
    log_lines: list[str] = field(default_factory=list)
    templates: list[str] = field(default_factory=list)
    time_range: tuple[str, str] = ("", "")
    summary: str = ""

    @property
    def size(self) -> int:
        """Number of blocks in this incident."""
        return len(self.block_ids)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "incident_id": self.incident_id,
            "block_ids": self.block_ids,
            "severity": round(self.severity, 4),
            "size": self.size,
            "mean_anomaly_score": round(float(np.mean(self.anomaly_scores)), 4),
            "log_lines": self.log_lines,
            "templates": self.templates,
            "time_range": list(self.time_range),
            "summary": self.summary,
        }


class IncidentClusterer:
    """Clusters anomalous blocks into incidents using DBSCAN.

    Takes the feature vectors of anomalous blocks and groups them
    by similarity, treating each group as a distinct incident.
    Enriches each incident with the original log lines, templates,
    and time range for downstream LLM summarization.

    Example:
        >>> clusterer = IncidentClusterer(eps=0.5, min_samples=2)
        >>> incidents = clusterer.cluster(anomaly_results, events=log_events)
    """

    def __init__(self, eps: float = 0.5, min_samples: int = 2) -> None:
        """Initialize the clusterer.

        Args:
            eps: DBSCAN neighborhood radius.
            min_samples: Minimum samples to form a cluster.
        """
        self.eps = eps
        self.min_samples = min_samples

    def cluster(
        self,
        anomaly_results: list[AnomalyResult],
        events: list[LogEvent] | None = None,
    ) -> list[Incident]:
        """Cluster anomalous blocks into incidents.

        Args:
            anomaly_results: List of AnomalyResult objects (only anomalies
                will be clustered).
            events: Optional list of original log events. When provided,
                incidents are enriched with log lines, templates, and time ranges.

        Returns:
            List of Incident objects, one per cluster.
        """
        # Filter to anomalies only
        anomalies = [r for r in anomaly_results if r.is_anomaly]
        if not anomalies:
            return []

        # Build block-to-events index for enrichment
        block_events: dict[str, list[LogEvent]] = {}
        if events:
            for e in events:
                if e.block_id:
                    block_events.setdefault(e.block_id, []).append(e)

        # Handle single anomaly case
        if len(anomalies) == 1:
            a = anomalies[0]
            incident = Incident(
                incident_id=0,
                block_ids=[a.block_id],
                severity=self._compute_severity([a.anomaly_score]),
                anomaly_scores=[a.anomaly_score],
            )
            self._enrich_incident(incident, block_events)
            return [incident]

        # Build feature matrix from anomaly feature vectors
        features = np.array([r.feature_vector for r in anomalies])

        # Normalize features for distance computation
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Run DBSCAN
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = dbscan.fit_predict(features_scaled)

        # Group results by cluster label
        incidents: list[Incident] = []
        unique_labels = set(labels)

        for label in sorted(unique_labels):
            mask = labels == label
            cluster_results = [a for a, m in zip(anomalies, mask) if m]

            # DBSCAN label -1 = noise; treat each as its own incident
            if label == -1:
                for r in cluster_results:
                    incident = Incident(
                        incident_id=len(incidents),
                        block_ids=[r.block_id],
                        severity=self._compute_severity([r.anomaly_score]),
                        anomaly_scores=[r.anomaly_score],
                    )
                    self._enrich_incident(incident, block_events)
                    incidents.append(incident)
            else:
                scores = [r.anomaly_score for r in cluster_results]
                incident = Incident(
                    incident_id=len(incidents),
                    block_ids=[r.block_id for r in cluster_results],
                    severity=self._compute_severity(scores),
                    anomaly_scores=scores,
                )
                self._enrich_incident(incident, block_events)
                incidents.append(incident)

        return incidents

    def _enrich_incident(
        self,
        incident: Incident,
        block_events: dict[str, list[LogEvent]],
    ) -> None:
        """Populate log_lines, templates, and time_range from source events."""
        if not block_events:
            return

        all_events: list[LogEvent] = []
        for bid in incident.block_ids:
            all_events.extend(block_events.get(bid, []))

        if not all_events:
            return

        incident.log_lines = [e.content for e in all_events]
        incident.templates = sorted({e.template for e in all_events if e.template})

        timestamps = [e.timestamp for e in all_events if e.timestamp]
        if timestamps:
            incident.time_range = (min(timestamps), max(timestamps))

    def _compute_severity(self, anomaly_scores: list[float]) -> float:
        """Compute severity from anomaly scores.

        Converts Isolation Forest decision function scores (where more negative
        = more anomalous) to a 0-1 severity scale.
        """
        mean_score = float(np.mean(anomaly_scores))
        # Decision function: negative = anomalous, positive = normal
        # Map to 0-1 where 1 = most severe
        severity = max(0.0, min(1.0, -mean_score))
        return severity

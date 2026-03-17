"""Tests for the incident clustering module."""

from flare.clustering.clusterer import Incident, IncidentClusterer
from flare.detection.detector import AnomalyResult


class TestIncident:
    def test_size(self) -> None:
        inc = Incident(
            incident_id=0,
            block_ids=["blk_1", "blk_2", "blk_3"],
            severity=0.5,
        )
        assert inc.size == 3

    def test_to_dict(self) -> None:
        inc = Incident(
            incident_id=0,
            block_ids=["blk_1"],
            severity=0.75,
            anomaly_scores=[-0.3, -0.5],
        )
        d = inc.to_dict()
        assert d["incident_id"] == 0
        assert d["size"] == 1
        assert "mean_anomaly_score" in d


class TestIncidentClusterer:
    def _make_anomaly(
        self, block_id: str, score: float, features: list[float]
    ) -> AnomalyResult:
        return AnomalyResult(
            block_id=block_id,
            is_anomaly=True,
            anomaly_score=score,
            feature_vector=features,
            event_count=5,
        )

    def _make_normal(self, block_id: str) -> AnomalyResult:
        return AnomalyResult(
            block_id=block_id,
            is_anomaly=False,
            anomaly_score=0.5,
            feature_vector=[1.0, 1.0, 1.0],
            event_count=5,
        )

    def test_cluster_empty(self) -> None:
        clusterer = IncidentClusterer()
        incidents = clusterer.cluster([])
        assert incidents == []

    def test_cluster_no_anomalies(self) -> None:
        clusterer = IncidentClusterer()
        results = [self._make_normal("blk_1"), self._make_normal("blk_2")]
        incidents = clusterer.cluster(results)
        assert incidents == []

    def test_cluster_single_anomaly(self) -> None:
        clusterer = IncidentClusterer()
        results = [self._make_anomaly("blk_1", -0.3, [1.0, 0.0, 0.0])]
        incidents = clusterer.cluster(results)
        assert len(incidents) == 1
        assert incidents[0].block_ids == ["blk_1"]

    def test_cluster_similar_anomalies(self) -> None:
        clusterer = IncidentClusterer(eps=1.0, min_samples=2)
        results = [
            self._make_anomaly("blk_1", -0.3, [1.0, 1.0, 0.0]),
            self._make_anomaly("blk_2", -0.4, [1.1, 1.0, 0.0]),
            self._make_anomaly("blk_3", -0.2, [1.0, 0.9, 0.0]),
            self._make_normal("blk_4"),
        ]
        incidents = clusterer.cluster(results)
        # Similar anomalies should be grouped together
        assert len(incidents) >= 1
        total_blocks = sum(inc.size for inc in incidents)
        assert total_blocks == 3  # only anomalies

    def test_severity_computed(self) -> None:
        clusterer = IncidentClusterer()
        results = [self._make_anomaly("blk_1", -0.3, [1.0, 0.0])]
        incidents = clusterer.cluster(results)
        assert incidents[0].severity > 0

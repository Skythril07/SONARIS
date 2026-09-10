"""World-space aggregation (plan gate P2.1).

Pure geodesic clustering + persistence counting, exercised on hand-placed points a known number of
metres apart -- the same coordinate-truth discipline as the geometry gate, one level up.
"""
from __future__ import annotations

import pytest

from sonaris.aggregate.cluster import (
    aggregate_detections,
    cluster_indices,
    geodesic_distance_m,
)
from sonaris.report.schema import Detection

BASE_LAT, BASE_LON = 8.7, 78.1
M_PER_DEG_LAT = 111_320.0


def _lat_north(m: float) -> float:
    """A latitude ``m`` metres north of BASE_LAT."""
    return BASE_LAT + m / M_PER_DEG_LAT


def _det(did: str, lat: float, lon: float, conf: float) -> Detection:
    return Detection(
        detection_id=did, line_id="L1", cls="fishing_gear",
        confidence=conf, confidence_raw=conf, lat=lat, lon=lon,
        position_uncertainty_m=3.0, length_m=2.0, width_m=1.0,
        ground_range_m=25.0, channel="port", ping_index=0,
    )


def test_geodesic_distance_matches_metre_offset():
    d = geodesic_distance_m(BASE_LAT, BASE_LON, _lat_north(50), BASE_LON)
    assert d == pytest.approx(50.0, abs=0.5)


def test_cluster_indices_merges_near_splits_far():
    pts = [(BASE_LAT, BASE_LON), (_lat_north(2), BASE_LON), (_lat_north(50), BASE_LON)]
    groups = sorted(cluster_indices(pts, radius_m=3.0), key=lambda g: g[0])
    assert groups == [[0, 1], [2]]


def test_aggregate_collapses_duplicates_and_counts_persistence():
    dets = [
        _det("a", BASE_LAT, BASE_LON, 0.6),
        _det("b", _lat_north(2), BASE_LON, 0.9),    # within 3 m of a -> same object
        _det("c", _lat_north(50), BASE_LON, 0.5),   # separate object
    ]
    out = aggregate_detections(dets, radius_m=3.0, alpha=0.9)

    assert len(out) == 2
    merged = max(out, key=lambda d: d.persistence_count)
    assert merged.persistence_count == 2
    assert merged.detection_id == "b"                        # highest-confidence survivor
    assert merged.lat == pytest.approx(_lat_north(2))        # keeps its own coordinate
    assert "a" in merged.evidence["merged_ids"]
    assert merged.evidence["cluster_spread_m"] == pytest.approx(2.0, abs=0.5)
    assert out[0].evidence["salience"] >= out[1].evidence["salience"]  # strongest first


def test_aggregate_empty_is_empty():
    assert aggregate_detections([], 3.0) == []

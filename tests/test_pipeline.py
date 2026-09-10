"""End-to-end vertical-slice test (plan gate P1, section 4.2).

Run the whole pipeline on a synthetic file and assert it recovers the planted object's lat/lon to
within a metre, on the correct side, and writes every deliverable. This is the Phase 1 definition
of done exercised in milliseconds -- with the classical blob detector standing in for YOLO.
"""
from __future__ import annotations

from pyproj import Geod

from sonaris.pipeline import run_pipeline
from tests.synthetic import write_synthetic_xtf

_GEOD = Geod(ellps="WGS84")


def test_run_pipeline_recovers_object_and_writes_reports(tmp_path):
    truth = write_synthetic_xtf(tmp_path / "synthetic.xtf")
    out = tmp_path / "run"

    dets = run_pipeline(truth.path, out)
    assert dets, "pipeline produced no detections"

    nearest = min(dets, key=lambda d: _GEOD.inv(d.lon, d.lat, truth.object_lon, truth.object_lat)[2])
    dist = _GEOD.inv(nearest.lon, nearest.lat, truth.object_lon, truth.object_lat)[2]
    assert dist < 1.0, f"nearest detection off by {dist:.3f} m"
    assert nearest.channel == truth.object_side
    assert nearest.position_uncertainty_m > 0

    for name in (
        "detections.geojson", "detections.csv", "detections.gpx", "map.html",
        f"{truth.object_side}_annotated.png",
    ):
        assert (out / name).exists(), f"missing deliverable: {name}"


def test_pipeline_no_object_side_stays_quiet(tmp_path):
    """The channel without the planted object should not invent detections from flat seabed."""
    truth = write_synthetic_xtf(tmp_path / "synthetic.xtf")
    dets = run_pipeline(truth.path, tmp_path / "run")
    assert all(d.channel == truth.object_side for d in dets)

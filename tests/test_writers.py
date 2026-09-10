"""Report writer and report-CLI coverage for plan gate P1.8."""
from __future__ import annotations

import csv
import json
from xml.etree import ElementTree as ET

from sonaris.cli import main
from sonaris.report.schema import Detection
from sonaris.report.writers import write_reports
from tests.synthetic import write_synthetic_xtf


def _detections() -> list[Detection]:
    return [
        Detection("line-0", "line", "fishing_gear", 0.91, 0.93, 8.76, 78.10, 3.2,
                  2.5, 1.0, 15.0, "stbd", 8, evidence={"source": "test"}),
        Detection("line-1", "line", "pipe_cable", 0.72, 0.74, 8.761, 78.101, 4.1,
                  8.0, 0.3, 20.0, "port", 9),
    ]


def test_writers_emit_geojson_csv_and_gpx(tmp_path):
    reports = write_reports(_detections(), tmp_path)
    with open(reports["geojson"], encoding="utf-8") as f:
        geojson = json.load(f)
    assert geojson["features"][0]["geometry"]["coordinates"] == [78.10, 8.76]

    with open(reports["csv"], encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert (rows[1]["lat"], rows[1]["lon"]) == ("8.761", "78.101")
    assert json.loads(rows[0]["evidence"]) == {"source": "test"}

    namespace = {"gpx": "http://www.topografix.com/GPX/1/1"}
    waypoints = ET.parse(reports["gpx"]).getroot().findall("gpx:wpt", namespace)
    assert [(w.attrib["lat"], w.attrib["lon"]) for w in waypoints] == [
        ("8.76", "78.1"), ("8.761", "78.101")
    ]


def test_report_cli_converts_stub_geojson(tmp_path):
    truth = write_synthetic_xtf(tmp_path / "survey.xtf")
    detections, reports = tmp_path / "detections.geojson", tmp_path / "reports"
    assert main(["infer", truth.path, "--stub", "--out", str(detections)]) == 0
    assert main(["report", str(detections), "--out", str(reports)]) == 0
    for suffix in ("geojson", "csv", "gpx"):
        assert (reports / f"detections.{suffix}").is_file()

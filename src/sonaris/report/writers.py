"""Report writers: list[Detection] -> GeoJSON (plan gate P1.3 minimal; CSV/GPX at P1.8).

Writers are dumb and total (layout section 5): they serialise whatever fields a Detection carries
and never reach back to raw files or recompute geometry. GeoJSON comes first because it is the
one artefact you can eyeball on a map to confirm the whole coordinate chain landed a detection on
the right spot and side -- the visual half of the star gate P1.3.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from xml.etree import ElementTree as ET

from ..config import load_config
from .schema import Detection

# GeoJSON coordinate order is [longitude, latitude] (RFC 7946), the opposite of how we say it.


def detections_to_geojson(detections: list[Detection]) -> dict:
    """Build a GeoJSON FeatureCollection; each detection is a Point with its full record."""
    features = []
    for det in detections:
        props = asdict(det)
        props.pop("lat", None)
        props.pop("lon", None)
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [det.lon, det.lat]},
                "properties": props,
            }
        )
    return {"type": "FeatureCollection", "features": features}


def write_geojson(detections: list[Detection], path) -> Path:
    """Serialise detections to ``path`` as GeoJSON; returns the path written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(detections_to_geojson(detections), f, indent=2)
    return path


def write_csv(detections: list[Detection], path) -> Path:
    """Serialise detections as a CSV with Detection fields in declared order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(Detection.__dataclass_fields__)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for det in detections:
            row = asdict(det)
            row["evidence"] = json.dumps(row["evidence"], separators=(",", ":"))
            for name, value in row.items():
                if value is None:
                    row[name] = ""
            writer.writerow(row)
    return path


def write_gpx(detections: list[Detection], path) -> Path:
    """Serialise detections as GPX 1.1 waypoints."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    namespace = "http://www.topografix.com/GPX/1/1"
    ET.register_namespace("", namespace)
    root = ET.Element(f"{{{namespace}}}gpx", version="1.1", creator="sonaris")

    for det in detections:
        waypoint = ET.SubElement(root, f"{{{namespace}}}wpt", lat=str(det.lat), lon=str(det.lon))
        ET.SubElement(waypoint, f"{{{namespace}}}name").text = det.detection_id
        ET.SubElement(waypoint, f"{{{namespace}}}desc").text = (
            f"cls={det.cls} conf={det.confidence:.2f} unc={det.position_uncertainty_m:.2f}m"
        )

    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path


def write_reports(
    detections: list[Detection], out_dir, formats: list[str] | None = None
) -> dict[str, Path]:
    """Write requested report formats and return their paths keyed by format name."""
    out_dir = Path(out_dir)
    selected_formats = formats if formats is not None else load_config()["report"]["formats"]
    writers = {"geojson": write_geojson, "csv": write_csv, "gpx": write_gpx}

    reports = {}
    for report_format in selected_formats:
        try:
            writer = writers[report_format]
        except KeyError as e:
            supported = ", ".join(writers)
            raise ValueError(
                f"Unsupported report format {report_format!r}; expected one of: {supported}."
            ) from e
        reports[report_format] = writer(detections, out_dir / f"detections.{report_format}")
    return reports

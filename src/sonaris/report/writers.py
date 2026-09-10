"""Report writers: list[Detection] -> GeoJSON (plan gate P1.3 minimal; CSV/GPX at P1.8).

Writers are dumb and total (layout section 5): they serialise whatever fields a Detection carries
and never reach back to raw files or recompute geometry. GeoJSON comes first because it is the
one artefact you can eyeball on a map to confirm the whole coordinate chain landed a detection on
the right spot and side -- the visual half of the star gate P1.3.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

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

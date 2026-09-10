"""Minimal operator map: survey track + detection pins (plan gate P1.9; layout section 5).

A single self-contained HTML page (Leaflet) with the track as a polyline and detections as pins
carrying their class/confidence/uncertainty. This is the MVP stand-in for the Phase 3 console: one
place to manually confirm a pin sits on the right feature. If the basemap tiles cannot load
(no internet), the track and pins still render on a blank background -- the plan's offline fallback.
"""
from __future__ import annotations

import json
from pathlib import Path

from .schema import Detection
from .writers import detections_to_geojson

_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>SONARIS detections</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css"/>
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<style>html,body,#map{{height:100%;margin:0}}</style></head>
<body><div id="map"></div><script>
const track = {track};
const detections = {detections};
const map = L.map('map');
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
  {{maxZoom:19, attribution:'(c) OpenStreetMap'}}).addTo(map);
const line = L.polyline(track, {{color:'#3b82f6'}}).addTo(map);
const pins = L.geoJSON(detections, {{onEachFeature:(f,l)=>{{
  const p=f.properties||{{}};
  l.bindPopup(`${{p.cls||'det'}} conf=${{(p.confidence??0).toFixed(2)}} +/-${{(p.position_uncertainty_m??0).toFixed(1)}}m`);
}}}}).addTo(map);
const bounds = track.length ? line.getBounds() : pins.getBounds();
if (bounds.isValid()) map.fitBounds(bounds.pad(0.3)); else map.setView([0,0],2);
</script></body></html>
"""


def build_map_html(detections: list[Detection], track: list[tuple[float, float]]) -> str:
    """Return a self-contained Leaflet page. ``track`` is a list of ``(lat, lon)`` points."""
    return _PAGE.format(
        track=json.dumps([[lat, lon] for lat, lon in track]),
        detections=json.dumps(detections_to_geojson(detections)),
    )


def write_map(detections: list[Detection], track: list[tuple[float, float]], path) -> Path:
    """Write the map page to ``path``; returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_map_html(detections, track), encoding="utf-8")
    return path

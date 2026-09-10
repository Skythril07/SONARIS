"""End-to-end pipeline: survey file -> georeferenced detections + reports (plan gate P1, section 4.2).

Orchestrates the stages in memory -- decode, ground-range correct, preprocess, tile, detect,
filter/NMS, georeference -- then writes the deliverables (GeoJSON/CSV/GPX, annotated waterfalls,
map). Each stage stays in its own module behind its contract; this file only wires them together,
so swapping the blob stub for a trained YOLO-Seg model (P1.6) changes nothing here.
"""
from __future__ import annotations

import json
import time
from itertools import pairwise
from pathlib import Path

import numpy as np
from pyproj import Geod

from .aggregate.cluster import aggregate_detections
from .config import load_config
from .geometry.correct import correct_line
from .io.xtf_reader import read_xtf
from .models.detector import detect_tiles
from .postprocess.filter import filter_confidence, nms
from .postprocess.georeference import raw_to_detection
from .preprocess.enhance import build_stack, normalize_uint8
from .preprocess.tiling import TileRef, iter_tiles
from .report.annotate import annotate_waterfall
from .report.mapview import write_map
from .report.schema import Detection
from .report.writers import write_reports
from .verify.plausibility import apply_plausibility
from .verify.shadow import shadow_consistency

_GEOD = Geod(ellps="WGS84")


class _Progress:
    """Append one JSON line per pipeline stage to ``progress.jsonl`` (drives the web SSE rail, P3.4).

    Stages are emitted once each, in real execution order. The web layer prepends UPLOAD and tails
    this file; when ``path`` is None (plain CLI/test use) emitting is a no-op.
    """

    STAGES = ("DECODE", "CORRECT", "ENHANCE", "DETECT", "LOCATE", "VERIFY", "REPORT", "DONE")

    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self._seen: set[str] = set()

    def emit(self, stage: str, **extra) -> None:
        if self.path is None or stage in self._seen:
            return
        self._seen.add(stage)
        rec = {
            "stage": stage,
            "index": self.STAGES.index(stage),
            "total": len(self.STAGES),
            "ts": round(time.time(), 3),
            **extra,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")


def _along_track_res_m(pings) -> float:
    """Mean geodesic spacing between consecutive pings (along-track metres/row)."""
    if len(pings) < 2:
        return 1.0
    d = [_GEOD.inv(a.lon, a.lat, b.lon, b.lat)[2] for a, b in pairwise(pings)]
    return float(np.mean(d)) if d else 1.0


def run_pipeline(
    input_path, out_dir, cfg: dict | None = None, progress_path: str | Path | None = None
) -> list[Detection]:
    """Run the full slice on one XTF and write all reports into ``out_dir``. Returns detections.

    If ``progress_path`` is given, a JSON line is appended per stage so a caller (the web backend)
    can stream a live progress rail; otherwise progress emission is a no-op.
    """
    cfg = cfg or load_config()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prog = _Progress(progress_path)

    ground_res_m = cfg["geometry"]["ground_res_m"]
    size_px, stride_frac = cfg["tiling"]["size_px"], cfg["tiling"]["stride_frac"]
    norm = cfg["preprocess"]["normalize"]
    shadow_cfg = cfg["verify"]["shadow"]        # P2.2
    plaus_cfg = cfg["verify"]["plausibility"]   # P2.3
    agg_cfg = cfg["aggregate"]                  # P2.1

    line = read_xtf(input_path)
    prog.emit("DECODE")
    detections: list[Detection] = []
    track: list[tuple[float, float]] = []

    for channel, pings in line.channels.items():
        pings = sorted(pings, key=lambda p: p.index)
        if not track:
            track = [(p.lat, p.lon) for p in pings]
        pings_by_index = {p.index: p for p in pings}
        along_res = _along_track_res_m(pings)

        cl = correct_line(line, channel, ground_res_m)
        prog.emit("CORRECT")
        stack = build_stack(cl.image, cfg["preprocess"])
        prog.emit("ENHANCE")

        refs_by_id: dict[str, TileRef] = {}
        tiles: list[tuple[str, np.ndarray]] = []
        for ps, pe, cs, ce, tile in iter_tiles(stack, size_px, stride_frac):
            tid = f"{line.line_id}_{channel}_p{ps}_c{cs}"
            refs_by_id[tid] = TileRef(tid, line.line_id, channel, ps, pe, cs, ce, ground_res_m, "")
            tiles.append((tid, tile))

        raws = detect_tiles(tiles, cfg["detect"])
        raws = nms(filter_confidence(raws, cfg["detect"]["conf_min"]), cfg["detect"]["nms_iou"])
        prog.emit("DETECT")

        boxes = []
        for raw in raws:
            ref = refs_by_id[raw.tile_id]
            det = raw_to_detection(raw, ref, cl.ping_index, pings_by_index, along_res)
            img_box = (ref.col_start + raw.x1, ref.ping_start + raw.y1,
                       ref.col_start + raw.x2, ref.ping_start + raw.y2)
            if shadow_cfg.get("enabled"):  # P2.2: corroborate with the acoustic shadow beyond it
                ok, ev = shadow_consistency(
                    cl.image, img_box,
                    min_shadow_px=shadow_cfg.get("min_shadow_px", 4),
                    min_drop_frac=shadow_cfg.get("min_drop_frac", 0.5),
                )
                det.shadow_consistent = ok
                det.evidence = {**det.evidence, "shadow": ev}
            detections.append(det)
            boxes.append((*img_box, raw.cls))

        annotate_waterfall(
            normalize_uint8(cl.image, norm["p_low"], norm["p_high"]),
            boxes, out_dir / f"{channel}_annotated.png",
        )

    prog.emit("LOCATE")  # every raw detection now carries a georeferenced lat/lon
    # P2.1 world-space aggregation: collapse tile-seam / cross-channel duplicates into one contact
    # each, setting persistence_count and salience-ranking the survivors.
    detections = aggregate_detections(
        detections, agg_cfg["cluster_radius_m"], agg_cfg.get("alpha", 0.9)
    )
    # P2.3 physical plausibility: flag (or drop) detections outside per-class size/aspect gates.
    if plaus_cfg.get("enabled"):
        detections = apply_plausibility(detections, plaus_cfg, drop=plaus_cfg.get("drop", False))
    prog.emit("VERIFY")

    prog.emit("REPORT")
    write_reports(detections, out_dir, cfg["report"]["formats"])
    write_map(detections, track, out_dir / "map.html")
    _write_track(track, out_dir / "track.geojson")
    prog.emit("DONE", detections=len(detections))
    return detections


def _write_track(track: list[tuple[float, float]], path: Path) -> None:
    """Write the survey track as a GeoJSON LineString ([lon, lat] order, RFC 7946)."""
    feature = {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": [[lon, lat] for lat, lon in track]},
        "properties": {"role": "survey_track"},
    }
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": [feature]}), encoding="utf-8"
    )

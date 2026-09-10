"""End-to-end pipeline: survey file -> georeferenced detections + reports (plan gate P1, section 4.2).

Orchestrates the stages in memory -- decode, ground-range correct, preprocess, tile, detect,
filter/NMS, georeference -- then writes the deliverables (GeoJSON/CSV/GPX, annotated waterfalls,
map). Each stage stays in its own module behind its contract; this file only wires them together,
so swapping the blob stub for a trained YOLO-Seg model (P1.6) changes nothing here.
"""
from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import numpy as np
from pyproj import Geod

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

_GEOD = Geod(ellps="WGS84")


def _along_track_res_m(pings) -> float:
    """Mean geodesic spacing between consecutive pings (along-track metres/row)."""
    if len(pings) < 2:
        return 1.0
    d = [_GEOD.inv(a.lon, a.lat, b.lon, b.lat)[2] for a, b in pairwise(pings)]
    return float(np.mean(d)) if d else 1.0


def run_pipeline(input_path, out_dir, cfg: dict | None = None) -> list[Detection]:
    """Run the full slice on one XTF and write all reports into ``out_dir``. Returns detections."""
    cfg = cfg or load_config()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ground_res_m = cfg["geometry"]["ground_res_m"]
    size_px, stride_frac = cfg["tiling"]["size_px"], cfg["tiling"]["stride_frac"]
    norm = cfg["preprocess"]["normalize"]

    line = read_xtf(input_path)
    detections: list[Detection] = []
    track: list[tuple[float, float]] = []

    for channel, pings in line.channels.items():
        pings = sorted(pings, key=lambda p: p.index)
        if not track:
            track = [(p.lat, p.lon) for p in pings]
        pings_by_index = {p.index: p for p in pings}
        along_res = _along_track_res_m(pings)

        cl = correct_line(line, channel, ground_res_m)
        stack = build_stack(cl.image, cfg["preprocess"])

        refs_by_id: dict[str, TileRef] = {}
        tiles: list[tuple[str, np.ndarray]] = []
        for ps, pe, cs, ce, tile in iter_tiles(stack, size_px, stride_frac):
            tid = f"{line.line_id}_{channel}_p{ps}_c{cs}"
            refs_by_id[tid] = TileRef(tid, line.line_id, channel, ps, pe, cs, ce, ground_res_m, "")
            tiles.append((tid, tile))

        raws = detect_tiles(tiles, cfg["detect"])
        raws = nms(filter_confidence(raws, cfg["detect"]["conf_min"]), cfg["detect"]["nms_iou"])

        boxes = []
        for raw in raws:
            ref = refs_by_id[raw.tile_id]
            detections.append(
                raw_to_detection(raw, ref, cl.ping_index, pings_by_index, along_res)
            )
            boxes.append((ref.col_start + raw.x1, ref.ping_start + raw.y1,
                          ref.col_start + raw.x2, ref.ping_start + raw.y2, raw.cls))

        annotate_waterfall(
            normalize_uint8(cl.image, norm["p_low"], norm["p_high"]),
            boxes, out_dir / f"{channel}_annotated.png",
        )

    write_reports(detections, out_dir, cfg["report"]["formats"])
    write_map(detections, track, out_dir / "map.html")
    return detections

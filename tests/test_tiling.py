"""Tile-index world-recovery test (plan gate P1.4; extends the star discipline of P1.3).

P1.4's pass condition is "world coords recoverable from the tile index, never the filename". So:
plant an object, correct + tile the line with deliberately small overlapping tiles, find the tile
and pixel the object fell in, and walk the sanctioned mapping (tile pixel -> image row/col -> ping
-> ground range -> lat/lon) back to a coordinate within a metre of truth. If the tile-index
bookkeeping has an off-by-one, this fails loudly instead of shipping a silently misplaced pin.
"""
from __future__ import annotations

import numpy as np
from pyproj import Geod

from sonaris.config import load_config
from sonaris.geometry.correct import correct_line
from sonaris.geometry.georef import detection_to_latlon
from sonaris.io.xtf_reader import read_xtf
from sonaris.preprocess.enhance import build_stack
from sonaris.preprocess.tiling import (
    TileRef,
    ground_col_to_range_m,
    iter_tiles,
    tile_pixel_to_image,
    tiles_dataframe,
)
from tests.synthetic import write_synthetic_xtf

_GEOD = Geod(ellps="WGS84")


def _build_refs(stack, line_id, channel, ground_res_m, size_px, stride_frac):
    refs = []
    for ping_start, ping_end, col_start, col_end, _ in iter_tiles(stack, size_px, stride_frac):
        refs.append(TileRef(
            tile_id=f"{line_id}_{channel}_p{ping_start}_c{col_start}", line_id=line_id,
            channel=channel, ping_start=ping_start, ping_end=ping_end,
            col_start=col_start, col_end=col_end, ground_res_m=ground_res_m, path="",
        ))
    return refs


def test_world_coords_recoverable_from_tile_index(tmp_path):
    ground_res_m = load_config()["geometry"]["ground_res_m"]
    truth = write_synthetic_xtf(tmp_path / "synthetic.xtf")
    line = read_xtf(truth.path)
    side = truth.object_side

    cl = correct_line(line, side, ground_res_m)
    # Locate the object in the corrected image (brightest pixel) -- our "detection".
    row, col = np.unravel_index(int(np.argmax(cl.image)), cl.image.shape)

    stack = build_stack(cl.image, load_config()["preprocess"])
    # Small, overlapping tiles so the index bookkeeping is genuinely exercised.
    refs = _build_refs(stack, line.line_id, side, ground_res_m, size_px=16, stride_frac=0.5)

    # Find a tile that contains the object and recover its world position through the index alone.
    containing = [
        r for r in refs
        if r.ping_start <= row < r.ping_end and r.col_start <= col < r.col_end
    ]
    assert containing, "no tile contains the planted object"

    ref = containing[0]
    x, y = col - ref.col_start, row - ref.ping_start          # tile-local pixel
    img_row, img_col = tile_pixel_to_image(ref, x, y)         # back to image coords
    assert (img_row, img_col) == (row, col)                   # index round-trips exactly

    ping = line.channels[side][int(cl.ping_index[img_row])]
    r_ground = ground_col_to_range_m(img_col, ref.ground_res_m)
    lat, lon = detection_to_latlon(ping, r_ground, side)

    dist = _GEOD.inv(lon, lat, truth.object_lon, truth.object_lat)[2]
    assert dist < 1.0, f"tile-index recovery off by {dist:.3f} m (>1 m -> index bookkeeping bug)"


def test_tiles_cover_the_whole_image(tmp_path):
    ground_res_m = load_config()["geometry"]["ground_res_m"]
    truth = write_synthetic_xtf(tmp_path / "synthetic.xtf")
    line = read_xtf(truth.path)
    cl = correct_line(line, truth.object_side, ground_res_m)
    stack = build_stack(cl.image, load_config()["preprocess"])

    h, w = stack.shape[:2]
    covered = np.zeros((h, w), dtype=bool)
    refs = _build_refs(stack, line.line_id, truth.object_side, ground_res_m, 16, 0.5)
    for r in refs:
        covered[r.ping_start:r.ping_end, r.col_start:r.col_end] = True
    assert covered.all(), "tiling left gaps -- an object in a gap would never be detected"

    df = tiles_dataframe(refs)
    assert list(df.columns)[:3] == ["tile_id", "line_id", "channel"]
    assert len(df) == len(refs)

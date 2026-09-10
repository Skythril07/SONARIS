"""Georeference raw tile detections into world-space ``Detection``s (Stage 6; plan gate P1.7).

This is the pixel -> lat/lon bridge, and the last place the coordinate chain can go silently wrong,
so it lives under close review. A raw detection's box centre is mapped through the **tile index**
(never the filename, layout section 2.3): tile pixel -> corrected-image (row, col) -> ping (via
``CorrectedLine.ping_index``) -> ground range -> lat/lon (``geometry.georef``).

It consumes only decoded/derived inputs handed to it -- it never re-reads the raw survey file
(layout section 5).
"""
from __future__ import annotations

import numpy as np

from ..geometry.georef import detection_to_latlon, position_uncertainty
from ..io.schema import Ping
from ..models.detector import RawDetection
from ..preprocess.tiling import TileRef, ground_col_to_range_m, tile_pixel_to_image
from ..report.schema import Detection


def raw_to_detection(
    raw: RawDetection,
    ref: TileRef,
    ping_index: np.ndarray,
    pings_by_index: dict[int, Ping],
    along_track_res_m: float,
) -> Detection:
    """Map one ``RawDetection`` to a world-space ``Detection`` through the tile index.

    Sizes: width is across-track (columns x ground_res); length is along-track (rows x mean ping
    spacing). Both are MVP estimates -- honest metric extents, refined by segmentation later.
    """
    cx = (raw.x1 + raw.x2) // 2
    cy = (raw.y1 + raw.y2) // 2
    img_row, img_col = tile_pixel_to_image(ref, cx, cy)
    img_row = int(np.clip(img_row, 0, len(ping_index) - 1))

    ping = pings_by_index[int(ping_index[img_row])]
    r_ground = ground_col_to_range_m(img_col, ref.ground_res_m)
    lat, lon = detection_to_latlon(ping, r_ground, ref.channel)

    width_m = (raw.x2 - raw.x1) * ref.ground_res_m
    length_m = (raw.y2 - raw.y1) * along_track_res_m

    return Detection(
        detection_id=raw.det_id,
        line_id=ref.line_id,
        cls=raw.cls,
        confidence=raw.conf,
        confidence_raw=raw.conf,
        lat=lat,
        lon=lon,
        position_uncertainty_m=position_uncertainty(ping, r_ground),
        length_m=float(length_m),
        width_m=float(width_m),
        ground_range_m=float(r_ground),
        channel=ref.channel,
        ping_index=int(ping.index),
    )

"""Detection stage (Stage 4; plan gates P1.6/P1.7; layout sections 2.4, 5).

Emits ``RawDetection``s in **tile-pixel** coordinates. This module knows nothing about the world:
no lat/lon, no ground range (layout section 5) -- georeferencing is postprocess's job.

Until a YOLO-Seg model is trained (P1.6, needs the ``ml`` extra + a labelled corpus), the standing
detector is a classical bright-blob finder. The plan explicitly allows a dumb detector for the
vertical slice: the point is that the *chain around it* is correct, and the model is a drop-in
upgrade behind this same ``RawDetection`` contract.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class RawDetection:
    """One detection in a single tile's pixel frame (layout section 2.4). Nothing world-aware."""

    det_id: str
    tile_id: str
    cls: str
    conf: float
    x1: int
    y1: int
    x2: int
    y2: int
    mask_rle: str | None = None   # filled by YOLO-Seg / SAM later; None for the blob stub


def detect_blobs(tile: np.ndarray, tile_id: str, detect_cfg: dict) -> list[RawDetection]:
    """Find bright connected blobs in a tile stack and return them as ``RawDetection``s.

    Operates on the first channel (raw-normalised intensity). Confidence is the blob's mean
    intensity scaled to 0..1 -- a crude but honest stand-in for a model score.
    """
    stub = detect_cfg["stub"]
    plane = tile[..., 0] if tile.ndim == 3 else tile
    plane = np.ascontiguousarray(plane, dtype=np.uint8)

    _, mask = cv2.threshold(plane, int(stub["threshold"]), 255, cv2.THRESH_BINARY)
    n, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    dets: list[RawDetection] = []
    for i in range(1, n):  # 0 is the background component
        x, y, w, h, area = stats[i]
        if area < int(stub["min_area_px"]):
            continue
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        conf = float(plane[y1:y2, x1:x2].mean()) / 255.0
        dets.append(RawDetection(
            det_id=f"{tile_id}_d{i}", tile_id=tile_id, cls="stub",
            conf=conf, x1=x1, y1=y1, x2=x2, y2=y2,
        ))
    return dets


def detect_tiles(tiles: list[tuple[str, np.ndarray]], detect_cfg: dict) -> list[RawDetection]:
    """Run the configured detector over ``(tile_id, tile_stack)`` pairs.

    ``detect.model`` selects the backend. The YOLO-Seg path is deferred to P1.6 (needs the ``ml``
    extra + trained weights); until then everything routes to the blob stub.
    """
    dets: list[RawDetection] = []
    for tile_id, tile in tiles:
        dets.extend(detect_blobs(tile, tile_id, detect_cfg))
    return dets

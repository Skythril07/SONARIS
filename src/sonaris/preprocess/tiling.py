"""Overlapping tiling + the tile index (Stage 3; plan gate P1.4; layout sections 2.3, 3).

The single most consequential thing here, after the coordinate chain itself: **the tile index is
the only way back to world coordinates -- never derive it from the filename** (layout section 2.3).
Each tile records the image-row range (pings) and column range (ground bins) it was cut from, so a
detection at tile pixel ``(x, y)`` maps to image ``(row=ping_start+y, col=col_start+x)``, then to a
ping (via ``CorrectedLine.ping_index``) and a ground range (``col * ground_res_m``), then to lat/lon.

Tiles overlap by ``1 - stride_frac`` so one object lands in several tiles; cross-tile dedup is a
world-space job for P2.1, not here. This module preserves rows/cols exactly; it does no resampling.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Column order of processed/tiles.parquet (layout section 2.3). Kept explicit so the schema is
# stable and readable; the parquet is the contract, the filename is not.
TILE_COLUMNS = [
    "tile_id", "line_id", "channel",
    "ping_start", "ping_end", "col_start", "col_end",
    "ground_res_m", "path",
]


@dataclass
class TileRef:
    """One row of the tile index -- geometry only, no pixels."""

    tile_id: str
    line_id: str
    channel: str
    ping_start: int      # inclusive image row (== ping order index) of the tile's top edge
    ping_end: int        # exclusive image row of the bottom edge
    col_start: int       # inclusive ground-bin column of the left edge
    col_end: int         # exclusive ground-bin column of the right edge
    ground_res_m: float
    path: str


def _starts(length: int, size: int, stride: int) -> list[int]:
    """Window start offsets covering ``[0, length)``; a final clamped start guarantees the edge is
    covered without running past it. One window when the image is smaller than a tile."""
    if length <= size:
        return [0]
    starts = list(range(0, length - size + 1, stride))
    if starts[-1] != length - size:
        starts.append(length - size)
    return starts


def iter_tiles(
    stack: np.ndarray, size_px: int, stride_frac: float
) -> Iterator[tuple[int, int, int, int, np.ndarray]]:
    """Yield ``(ping_start, ping_end, col_start, col_end, tile_array)`` over an ``(H, W, C)`` stack.

    Windows are clamped at the edges (never padded here, so pixel offsets stay exact). ``H`` is the
    ping axis, ``W`` the ground-range axis.
    """
    if stack.ndim != 3:
        raise ValueError(f"expected an (H, W, C) stack, got shape {stack.shape}.")
    h, w = stack.shape[:2]
    stride = max(1, round(size_px * stride_frac))
    for ys in _starts(h, size_px, stride):
        ye = min(ys + size_px, h)
        for xs in _starts(w, size_px, stride):
            xe = min(xs + size_px, w)
            yield ys, ye, xs, xe, stack[ys:ye, xs:xe]


def tiles_dataframe(refs: list[TileRef]) -> pd.DataFrame:
    """Build the tiles.parquet frame from tile references, columns in the layout's order."""
    return pd.DataFrame([[getattr(r, c) for c in TILE_COLUMNS] for r in refs], columns=TILE_COLUMNS)


# --- world-recovery helpers: the sanctioned pixel -> line-coords mapping (layout section 3) -----

def tile_pixel_to_image(ref: TileRef, x: int, y: int) -> tuple[int, int]:
    """Tile-local pixel ``(x, y)`` -> corrected-image ``(row, col)``. Row is a ping-order index
    (resolve to a ping via ``CorrectedLine.ping_index[row]``); col is a ground-range bin."""
    return ref.ping_start + int(y), ref.col_start + int(x)


def ground_col_to_range_m(col: int, ground_res_m: float) -> float:
    """Ground-range bin -> across-track metres. Column index is linear in metres by construction."""
    return col * ground_res_m

"""Assemble a ground-range-corrected waterfall (Stage 2 'correct'; plan gate P1.2, feeds P1.4).

Each ping's across-track samples are resampled from slant range onto a uniform ground-range grid
(``slant_to_ground``) and stacked into a 2-D image: rows are pings (in acquisition order),
columns are ground-range bins of ``ground_res_m`` metres each. That per-column-is-metres property
is exactly what lets a later tile-pixel be turned back into a world coordinate -- so this stays in
``geometry/`` (it is coordinate work), and knows nothing about models, tiling, or files.

``CorrectedLine.ping_index[row]`` maps an image row back to the ping it came from; that mapping,
carried in the tile index, is the only sanctioned route from a pixel to a lat/lon (layout section 3).
"""
from __future__ import annotations

import numpy as np

from ..io.schema import CorrectedLine, SurveyLine
from .slant_range import remove_water_column, slant_to_ground

_UINT16_MAX = np.iinfo(np.uint16).max


def correct_line(line: SurveyLine, channel: str, ground_res_m: float) -> CorrectedLine:
    """Build the ground-range ``CorrectedLine`` for one channel of a decoded ``SurveyLine``.

    Rows are pings in ascending ``ping.index`` order; columns are uniform ``ground_res_m`` bins.
    If the vendor already ground-range corrected the file (``line.corrected``, gotcha #2), the
    samples are used as-is -- resampling an already-corrected line would smear it.
    """
    if channel not in line.channels:
        raise KeyError(f"channel {channel!r} not in survey line (have {list(line.channels)}).")
    pings = sorted(line.channels[channel], key=lambda p: p.index)
    if not pings:
        raise ValueError(f"channel {channel!r} has no pings.")

    rows: list[np.ndarray] = []
    bottom_idx: list[int] = []
    altitude: list[float] = []
    ping_index: list[int] = []
    for p in pings:
        if line.corrected:
            row = np.asarray(p.samples, dtype=float)
            b = 0
        else:
            res_slant = p.slant_range_m / p.n_samples
            row = slant_to_ground(p.samples, p.slant_range_m, p.altitude_m, ground_res_m)
            _, b = remove_water_column(p.samples, p.altitude_m, res_slant)
        rows.append(row)
        bottom_idx.append(int(b))
        altitude.append(float(p.altitude_m))
        ping_index.append(int(p.index))

    width = max((r.shape[0] for r in rows), default=0)
    image = np.zeros((len(rows), width), dtype=np.uint16)
    for i, r in enumerate(rows):
        w = r.shape[0]
        if w:
            image[i, :w] = np.clip(np.rint(r), 0, _UINT16_MAX).astype(np.uint16)

    return CorrectedLine(
        line_id=line.line_id,
        channel=channel,
        image=image,
        ground_res_m=float(ground_res_m),
        bottom_idx=np.asarray(bottom_idx, dtype=int),
        altitude_m=np.asarray(altitude, dtype=float),
        ping_index=np.asarray(ping_index, dtype=int),
    )

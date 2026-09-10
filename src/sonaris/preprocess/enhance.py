"""Intensity normalisation and enhancement (Stage 3 pre-tiling; plan gate P1.4; layout section 5).

Turns a ground-range ``CorrectedLine.image`` (uint16, physically meaningful but visually flat)
into the model input stack. This module changes *intensities*, never *geometry or resolution*
(layout section 5): it must not resample columns or shift rows, so the tile index stays a faithful
map back to world coordinates.

Channels come from ``configs/default.yaml`` ``preprocess.channels`` (default ``[raw_norm, clahe]``).
"""
from __future__ import annotations

import cv2
import numpy as np


def normalize_uint8(image: np.ndarray, p_low: float, p_high: float) -> np.ndarray:
    """Percentile-stretch to 8-bit. Robust to a few saturated returns that would crush a min/max
    stretch. ``p_low``/``p_high`` are percentiles (e.g. 1 and 99)."""
    img = np.asarray(image, dtype=np.float32)
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        return np.zeros(img.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, [p_low, p_high])
    if hi <= lo:
        hi = lo + 1.0
    stretched = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
    return (stretched * 255.0).astype(np.uint8)


def apply_clahe(image_u8: np.ndarray, clip_limit: float, tile_grid) -> np.ndarray:
    """Contrast-limited adaptive histogram equalisation on an 8-bit image."""
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tuple(int(t) for t in tile_grid))
    return clahe.apply(np.asarray(image_u8, dtype=np.uint8))


def build_stack(image_u16: np.ndarray, preprocess_cfg: dict) -> np.ndarray:
    """Build an ``(H, W, C)`` uint8 stack for the channels named in ``preprocess_cfg['channels']``.

    Known channels: ``raw_norm`` (percentile-normalised) and ``clahe`` (CLAHE over raw_norm).
    Row/column count is preserved exactly, so pixel (x, y) still means (ground_col, ping_row).
    """
    norm = preprocess_cfg["normalize"]
    raw_norm = normalize_uint8(image_u16, norm["p_low"], norm["p_high"])

    builders = {
        "raw_norm": lambda: raw_norm,
        "clahe": lambda: apply_clahe(
            raw_norm, preprocess_cfg["clahe"]["clip_limit"], preprocess_cfg["clahe"]["tile_grid"]
        ),
    }
    names = preprocess_cfg["channels"]
    planes = []
    for name in names:
        if name not in builders:
            raise ValueError(f"unknown preprocess channel {name!r}; known: {sorted(builders)}.")
        planes.append(builders[name]())
    return np.stack(planes, axis=-1)

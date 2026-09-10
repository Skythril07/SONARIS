"""Slant-range -> ground-range correction (Stage 2, plan gate P1.2; layout section 3.1).

The across-track sample axis of raw side-scan is *slant* range: the straight-line distance from
the towfish to each echo. Directly under the fish the beam has to travel the altitude before it
hits anything, so the first samples image nothing but the water column (the nadir gap). This
module removes that gap and resamples the remaining echoes onto a uniform ground-range grid, so
that a column index becomes linear in metres across the seabed.

This module does the arithmetic in named functions only (layout section 3): nothing here knows
about models, pixels, or files -- just numbers.
"""
from __future__ import annotations

import numpy as np


def remove_water_column(
    samples: np.ndarray, altitude_m: float, res_slant_m: float
) -> tuple[np.ndarray, int]:
    """Drop the nadir water-column samples.

    Sample ``i`` sits at slant range ``i * res_slant_m``. Every sample closer than the towfish
    altitude is water column (its ground range would be imaginary) and is discarded rather than
    interpolated across. Returns ``(samples_beyond_nadir, nadir_idx)`` where ``nadir_idx`` is the
    index of the first kept sample -- the seabed first return.
    """
    samples = np.asarray(samples)
    if res_slant_m <= 0:
        raise ValueError(f"res_slant_m must be positive, got {res_slant_m}.")
    # First index whose slant range reaches the altitude: i*res >= altitude -> i >= alt/res.
    nadir_idx = int(np.ceil(max(0.0, altitude_m) / res_slant_m))
    nadir_idx = min(nadir_idx, samples.shape[0])
    return samples[nadir_idx:], nadir_idx


def slant_to_ground(
    samples: np.ndarray,
    slant_range_m: float,
    altitude_m: float,
    ground_res_m: float,
) -> np.ndarray:
    """Resample one ping's samples from slant range onto a uniform ground-range grid.

    ``res_slant = slant_range_m / n_samples``; each surviving sample's ground range is
    ``sqrt(r_slant**2 - altitude**2)``. Ground range is monotonic in slant range beyond the
    nadir, so we interpolate intensity onto an evenly spaced ``ground_res_m`` grid starting at 0.
    The returned array's column index is therefore linear in metres (column ``c`` = ``c *
    ground_res_m`` metres across-track), which is the invariant the whole coordinate chain relies
    on. Returns an empty array when there is no seabed beyond the nadir.
    """
    samples = np.asarray(samples)
    n = samples.shape[0]
    if n == 0:
        return np.empty(0, dtype=float)
    if slant_range_m <= 0:
        raise ValueError(f"slant_range_m must be positive, got {slant_range_m}.")
    if ground_res_m <= 0:
        raise ValueError(f"ground_res_m must be positive, got {ground_res_m}.")

    res_slant = slant_range_m / n
    beyond, nadir_idx = remove_water_column(samples, altitude_m, res_slant)
    if beyond.shape[0] == 0:
        return np.empty(0, dtype=float)

    r_slant = np.arange(nadir_idx, n, dtype=float) * res_slant
    r_ground = np.sqrt(np.maximum(0.0, r_slant**2 - altitude_m**2))

    # Bins from 0 up to the last full ground_res step at or below the farthest echo -- no bin
    # is placed beyond the real data (no extrapolation past r_ground[-1]).
    n_bins = int(np.floor(r_ground[-1] / ground_res_m)) + 1
    grid = np.arange(n_bins) * ground_res_m
    return np.interp(grid, r_ground, beyond.astype(float))

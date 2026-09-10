"""The synthetic ground-truth geometry test -- the star gate (plan.md section 4.1, P1.3).

Build a valid XTF (straight track, constant heading + altitude, one bright object planted at a
known ground range on a known side), decode it, and assert the coordinate chain recovers the
object's lat/lon to within a metre. This catches every silent geometry gotcha (layout section 6:
nav units, heading source, side flip, per-ping slant range) at once, and runs in milliseconds.
Written for real BEFORE the stub-detector -> GeoJSON gate, not after.
"""
from __future__ import annotations

from math import sqrt

import numpy as np
from pyproj import Geod

from sonaris.geometry.georef import detection_to_latlon, position_uncertainty
from sonaris.geometry.slant_range import remove_water_column, slant_to_ground
from sonaris.io.schema import SIDE_PORT, SIDE_STBD
from sonaris.io.xtf_reader import read_xtf
from tests.synthetic import write_synthetic_xtf

_GEOD = Geod(ellps="WGS84")


def test_synthetic_ground_truth_recovers_latlon(tmp_path):
    truth = write_synthetic_xtf(tmp_path / "synthetic.xtf")
    line = read_xtf(truth.path)

    ping = line.channels[truth.object_side][truth.object_ping]

    # Recover ground range from the planted sample index, exactly as the real chain will.
    r_slant = truth.object_sample_index * truth.res_slant_m
    r_ground = sqrt(r_slant**2 - truth.altitude_m**2)
    assert abs(r_ground - truth.object_ground_range_m) < truth.res_slant_m * 2

    lat, lon = detection_to_latlon(ping, r_ground, truth.object_side)

    dist = _GEOD.inv(lon, lat, truth.object_lon, truth.object_lat)[2]
    assert dist < 1.0, f"recovered position off by {dist:.3f} m (>1 m -> a geometry gotcha bit)"


def test_side_flip_lands_on_the_wrong_side(tmp_path):
    """Guard gotcha #6: georeferencing the wrong side mirrors the detection across the track."""
    truth = write_synthetic_xtf(tmp_path / "synthetic.xtf")
    line = read_xtf(truth.path)
    ping = line.channels[truth.object_side][truth.object_ping]
    r_slant = truth.object_sample_index * truth.res_slant_m
    r_ground = sqrt(r_slant**2 - truth.altitude_m**2)

    other = SIDE_PORT if truth.object_side == SIDE_STBD else SIDE_STBD
    lat_wrong, lon_wrong = detection_to_latlon(ping, r_ground, other)
    dist_wrong = _GEOD.inv(lon_wrong, lat_wrong, truth.object_lon, truth.object_lat)[2]
    # Wrong side is ~2 * ground_range away (mirrored across the track line).
    assert dist_wrong > truth.object_ground_range_m


def test_remove_water_column_drops_the_nadir_gap():
    res_slant = 0.25
    altitude = 10.0
    samples = np.arange(200, dtype=float)
    beyond, nadir_idx = remove_water_column(samples, altitude, res_slant)
    # First kept sample's slant range must reach the altitude; the one before must not.
    assert nadir_idx * res_slant >= altitude
    assert (nadir_idx - 1) * res_slant < altitude
    assert beyond.shape[0] == samples.shape[0] - nadir_idx
    assert beyond[0] == samples[nadir_idx]


def test_slant_to_ground_grid_is_linear_in_metres():
    n, slant_range, altitude, ground_res = 200, 50.0, 10.0, 0.10
    samples = np.full(n, 10.0)
    ground = slant_to_ground(samples, slant_range, altitude, ground_res)
    assert ground.ndim == 1 and ground.shape[0] > 0
    # Farthest ground range must match the geometry of the last sample.
    res_slant = slant_range / n
    r_ground_max = sqrt(((n - 1) * res_slant) ** 2 - altitude**2)
    assert abs((ground.shape[0] - 1) * ground_res - r_ground_max) <= ground_res


def test_position_uncertainty_grows_with_range():
    from sonaris.io.schema import Ping

    ping = Ping(
        index=0, time_utc=0.0, lat=8.76, lon=78.10, heading_deg=90.0, altitude_m=10.0,
        depth_m=100.0, pitch_deg=0.0, roll_deg=0.0, heave_m=0.0, layback_m=0.0,
        slant_range_m=50.0, n_samples=200, samples=np.zeros(200),
    )
    near = position_uncertainty(ping, 5.0)
    far = position_uncertainty(ping, 50.0)
    assert far > near > 0

"""Synthetic XTF builder for tests (not collected as a test module).

Writes a *valid* XTF using pyxtf's own writer, with a known straight-line geometry and a single
bright object planted at a known ground range on a known side. Returns the ground truth so decode
(P1.1) and georeferencing (P1.3, the ★ gate) can be asserted against it.
"""
from __future__ import annotations

import ctypes
import datetime
from dataclasses import dataclass

import numpy as np
import pyxtf
from pyproj import Geod

_GEOD = Geod(ellps="WGS84")


@dataclass
class SyntheticTruth:
    path: str
    n_pings: int
    n_samples: int
    slant_range_m: float
    altitude_m: float
    res_slant_m: float
    heading_deg: float
    start_lat: float
    start_lon: float
    ping_spacing_m: float
    object_ping: int
    object_side: str            # 'port' | 'stbd'
    object_ground_range_m: float
    object_sample_index: int
    object_lat: float
    object_lon: float


def write_synthetic_xtf(
    path,
    *,
    n_pings: int = 30,
    n_samples: int = 200,
    slant_range_m: float = 50.0,
    altitude_m: float = 10.0,
    heading_deg: float = 90.0,
    start_lat: float = 8.76,
    start_lon: float = 78.10,
    ping_spacing_m: float = 1.0,
    object_ping: int | None = None,
    object_side: str = "stbd",
    object_ground_range_m: float = 15.0,
    nav_units: int = int(pyxtf.XTFNavUnits.latlon),
) -> SyntheticTruth:
    if object_ping is None:
        object_ping = n_pings // 2
    res_slant = slant_range_m / n_samples
    r_slant = float(np.hypot(object_ground_range_m, altitude_m))
    obj_idx = round(r_slant / res_slant)

    # Straight track from the start point along a constant heading.
    lats, lons = [], []
    lon, lat = start_lon, start_lat
    for _ in range(n_pings):
        lats.append(lat)
        lons.append(lon)
        lon, lat, _ = _GEOD.fwd(lon, lat, heading_deg, ping_spacing_m)

    # Ground-truth object position: perpendicular to track, on the chosen side.
    side_sign = {"port": -90.0, "stbd": +90.0}[object_side]
    obj_bearing = (heading_deg + side_sign) % 360.0
    obj_lon, obj_lat, _ = _GEOD.fwd(
        lons[object_ping], lats[object_ping], obj_bearing, object_ground_range_m
    )

    fh = pyxtf.XTFFileHeader()
    fh.SonarName = b"Synthetic"
    fh.SonarType = pyxtf.XTFSonarType.unknown1
    fh.NavUnits = nav_units
    fh.NumberOfSonarChannels = 2
    for k, side in ((0, pyxtf.XTFChannelType.port), (1, pyxtf.XTFChannelType.stbd)):
        fh.ChanInfo[k].TypeOfChannel = side.value
        fh.ChanInfo[k].SubChannelNumber = k
        fh.ChanInfo[k].BytesPerSample = 1
        fh.ChanInfo[k].SampleFormat = pyxtf.XTFSampleFormat.byte.value

    base = datetime.datetime(2024, 1, 1, 0, 0, 0)  # noqa: DTZ001 - naive is fine; only fills XTF Y/M/D/H/M/S fields
    pings = []
    for i in range(n_pings):
        t = base + datetime.timedelta(seconds=i)
        p = pyxtf.XTFPingHeader()
        p.HeaderType = pyxtf.XTFHeaderType.sonar.value
        p.NumChansToFollow = 2
        p.Year, p.Month, p.Day = t.year, t.month, t.day
        p.Hour, p.Minute, p.Second, p.HSeconds = t.hour, t.minute, t.second, 0
        p.PingNumber = i
        p.SoundVelocity = 1500
        p.SensorYcoordinate = lats[i]
        p.SensorXcoordinate = lons[i]
        p.SensorPrimaryAltitude = altitude_m
        p.SensorDepth = 100
        p.SensorHeading = heading_deg
        p.SensorPitch = 0
        p.SensorRoll = 0

        c = (pyxtf.XTFPingChanHeader(), pyxtf.XTFPingChanHeader())
        for k in (0, 1):
            c[k].ChannelNumber = k
            c[k].SlantRange = slant_range_m
            c[k].NumSamples = n_samples
            c[k].SampleFormat = 8  # byte / uint8
        p.ping_chan_headers = c

        d_port = np.full(n_samples, 10, dtype=np.uint8)
        d_stbd = np.full(n_samples, 10, dtype=np.uint8)
        if i == object_ping:
            target = d_stbd if object_side == "stbd" else d_port
            lo, hi = max(0, obj_idx - 1), min(n_samples, obj_idx + 2)
            target[lo:hi] = 255
        p.data = [d_port, d_stbd]

        p.NumBytesThisRecord = (
            ctypes.sizeof(pyxtf.XTFPingHeader)
            + 2 * ctypes.sizeof(pyxtf.XTFPingChanHeader)
            + len(d_port)
            + len(d_stbd)
        )
        pings.append(p)

    with open(path, "wb") as f:
        f.write(fh.to_bytes())
        for p in pings:
            f.write(p.to_bytes())

    return SyntheticTruth(
        path=str(path),
        n_pings=n_pings,
        n_samples=n_samples,
        slant_range_m=slant_range_m,
        altitude_m=altitude_m,
        res_slant_m=res_slant,
        heading_deg=heading_deg,
        start_lat=start_lat,
        start_lon=start_lon,
        ping_spacing_m=ping_spacing_m,
        object_ping=object_ping,
        object_side=object_side,
        object_ground_range_m=object_ground_range_m,
        object_sample_index=obj_idx,
        object_lat=obj_lat,
        object_lon=obj_lon,
    )

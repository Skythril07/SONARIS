"""Decode an XTF file into a SurveyLine (Stage 1, plan gate P1.1).

This module's one job is turning a vendor file into a ``SurveyLine``. It does not correct
geometry or normalise intensity (layout §5). Field mappings are verified against the installed
pyxtf 1.5.0 structs (XTF rev 44); the gotcha numbers refer to layout §6.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyxtf

from .schema import NAV_LATLON, NAV_METERS, SIDE_PORT, SIDE_STBD, Ping, SurveyLine

# pyxtf returns these enum fields as plain ints, so compare against the int values.
_SIDE_BY_CHANTYPE = {
    int(pyxtf.XTFChannelType.port): SIDE_PORT,
    int(pyxtf.XTFChannelType.stbd): SIDE_STBD,
}
_GROUND_RANGE = int(pyxtf.XTFCorrectionFlags.ground_range)
_LATLON = int(pyxtf.XTFNavUnits.latlon)
_METERS = int(pyxtf.XTFNavUnits.meters)


class NavUnitsError(ValueError):
    """Raised when a file's nav units are not lat/lon (gotcha #1).

    If NavUnits is 'meters', SensorX/Ycoordinate are projected easting/northing in an unstated
    CRS, not degrees. We refuse rather than silently emit coordinates near (0, 0) off West Africa.
    """


def read_xtf(path, *, require_latlon: bool = True) -> SurveyLine:
    """Decode ``path`` into a SurveyLine. Raises NavUnitsError on non-latlon files."""
    path = str(path)
    fh, packets = pyxtf.xtf_read(path, [pyxtf.XTFHeaderType.sonar])

    nav_units = _nav_units_str(fh.NavUnits)
    if require_latlon and nav_units != NAV_LATLON:
        raise NavUnitsError(
            f"NavUnits is '{nav_units}': SensorX/Ycoordinate are projected in an unstated CRS, "
            f"not degrees. Refusing to produce coordinates (layout section 6, gotcha #1)."
        )

    sonar = packets.get(pyxtf.XTFHeaderType.sonar, [])
    if not sonar:
        raise ValueError(f"No side-scan (sonar) packets found in {path}.")

    # channel index -> side, read from the FILE header (gotcha #6: never infer side from order).
    chan_side: dict[int, str] = {}
    chan_corrected: dict[int, bool] = {}
    for idx in range(int(fh.NumberOfSonarChannels)):
        ci = fh.ChanInfo[idx]
        chan_side[idx] = _SIDE_BY_CHANTYPE.get(int(ci.TypeOfChannel))
        chan_corrected[idx] = int(ci.CorrectionFlags) == _GROUND_RANGE

    # File-level source decisions (gotchas #4, #5), applied uniformly and reported in provenance.
    heading_source = _choose_heading_source(sonar)
    position_source = _choose_position_source(sonar)

    channels: dict[str, list[Ping]] = {}
    corrected_any = False
    for i, pk in enumerate(sonar):
        lat, lon = _position(pk, position_source)
        heading = _heading(pk, heading_source)
        t = _time_utc(pk)
        for k, chdr in enumerate(pk.ping_chan_headers):
            side = chan_side.get(int(chdr.ChannelNumber))
            if side is None:
                continue  # subbottom / bathy channel — not side-scan imagery
            corrected_any |= chan_corrected.get(int(chdr.ChannelNumber), False)
            channels.setdefault(side, []).append(
                Ping(
                    index=i,
                    time_utc=t,
                    lat=lat,
                    lon=lon,
                    heading_deg=heading,
                    altitude_m=float(pk.SensorPrimaryAltitude),
                    depth_m=float(pk.SensorDepth),
                    pitch_deg=float(pk.SensorPitch),
                    roll_deg=float(pk.SensorRoll),
                    heave_m=float(pk.Heave),
                    layback_m=float(pk.Layback),
                    slant_range_m=float(chdr.SlantRange),   # per ping, per channel (gotcha #3)
                    n_samples=int(chdr.NumSamples),
                    samples=np.asarray(pk.data[k]),
                )
            )

    return SurveyLine(
        line_id=Path(path).stem,
        source_file=path,
        sonar_type=_sonar_type_name(fh.SonarType),
        nav_units=nav_units,
        channels=channels,
        corrected=corrected_any,
        provenance={
            "heading_source": heading_source,
            "position_source": position_source,
            "n_pings": len(sonar),
            "sonar_name": _decode_bytes(fh.SonarName),
        },
    )


def nav_dataframe(line: SurveyLine):
    """One row per ping: shared per-ping scalars plus slant range / sample count per side.

    This is the nav table decode prints, and the basis for the interim ``nav.parquet`` later.
    """
    import pandas as pd

    sides = list(line.channels)
    per_side = {s: {p.index: p for p in line.channels[s]} for s in sides}
    indices = sorted({p.index for s in sides for p in line.channels[s]})

    rows = []
    for i in indices:
        rep = next(per_side[s][i] for s in sides if i in per_side[s])
        row = {
            "index": i,
            "time_utc": rep.time_utc,
            "lat": rep.lat,
            "lon": rep.lon,
            "heading_deg": rep.heading_deg,
            "altitude_m": rep.altitude_m,
            "depth_m": rep.depth_m,
            "roll_deg": rep.roll_deg,
        }
        for s in sides:
            p = per_side[s].get(i)
            row[f"slant_{s}_m"] = p.slant_range_m if p else float("nan")
            row[f"nsamp_{s}"] = p.n_samples if p else 0
        rows.append(row)
    return pd.DataFrame(rows)


# --- helpers -------------------------------------------------------------------------------

def _nav_units_str(raw) -> str:
    v = int(raw)
    if v == _LATLON:
        return NAV_LATLON
    if v == _METERS:
        return NAV_METERS
    return f"unknown({v})"


def _sonar_type_name(raw) -> str:
    try:
        return pyxtf.XTFSonarType(int(raw)).name
    except ValueError:
        return f"unknown({int(raw)})"


def _time_utc(pk) -> float:
    try:
        return datetime(
            int(pk.Year), int(pk.Month), int(pk.Day),
            int(pk.Hour), int(pk.Minute), int(pk.Second),
            int(pk.HSeconds) * 10_000, tzinfo=timezone.utc,
        ).timestamp()
    except (ValueError, OverflowError):
        return float("nan")


def _choose_heading_source(sonar) -> str:
    # SensorHeading is the towfish heading — correct for georeferencing the sonar (gotcha #5).
    # Fall back to ShipGyro only when SensorHeading is clearly unrecorded (all zero) but the
    # vessel gyro has signal.
    if all(float(p.SensorHeading) == 0.0 for p in sonar) and any(
        float(p.ShipGyro) != 0.0 for p in sonar
    ):
        return "ship_gyro"
    return "sensor_heading"


def _heading(pk, source: str) -> float:
    return float(pk.ShipGyro if source == "ship_gyro" else pk.SensorHeading)


def _choose_position_source(sonar) -> str:
    # SensorXcoordinate may be zero on some recorders (gotcha #4). Then only ship nav remains;
    # a proper layback back-projection is deferred to the geometry stage (P1.2).
    sensor_all_zero = all(
        float(p.SensorXcoordinate) == 0.0 and float(p.SensorYcoordinate) == 0.0 for p in sonar
    )
    ship_has_signal = any(
        float(p.ShipXcoordinate) != 0.0 or float(p.ShipYcoordinate) != 0.0 for p in sonar
    )
    return "ship" if sensor_all_zero and ship_has_signal else "sensor"


def _position(pk, source: str) -> tuple[float, float]:
    if source == "ship":
        return float(pk.ShipYcoordinate), float(pk.ShipXcoordinate)
    return float(pk.SensorYcoordinate), float(pk.SensorXcoordinate)


def _decode_bytes(b) -> str:
    if isinstance(b, (bytes, bytearray)):
        return b.split(b"\x00", 1)[0].decode("ascii", "replace").strip()
    return str(b)

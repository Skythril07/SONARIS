"""Ground range -> WGS84 lat/lon (Stage 2, plan gate P1.2; layout section 3).

Given a ping's own position and heading, a detection at a known across-track ground range on a
known side becomes a real-world coordinate. This is the step every geometry bug hides behind: it
always produces a confident, well-formatted -- and possibly wrong -- coordinate. It is proven at
gate P1.3 against synthetic ground truth to within a metre.

The arithmetic lives only here (layout section 3): no model, pixel, or file logic.
"""
from __future__ import annotations

from pyproj import Geod

from ..config import load_config
from ..io.schema import SIDE_PORT, SIDE_STBD, Ping

# WGS84 forward/inverse geodesic. fwd takes (lon, lat, az_deg, dist_m) and returns
# (lon, lat, back_az) -- note the lon-before-lat order, a classic silent-bug source.
_GEOD = Geod(ellps="WGS84")

# Across-track look direction relative to the towfish heading (layout section 3).
_SIDE_BEARING_OFFSET = {SIDE_PORT: -90.0, SIDE_STBD: +90.0}


def detection_to_latlon(ping: Ping, ground_range_m: float, side: str) -> tuple[float, float]:
    """Project a detection out from the ping to its WGS84 (lat, lon).

    Bearing is the towfish heading rotated 90 degrees to port or starboard; the detection lies
    ``ground_range_m`` metres along that bearing from the ping's nav position.
    """
    if side not in _SIDE_BEARING_OFFSET:
        raise ValueError(f"side must be '{SIDE_PORT}' or '{SIDE_STBD}', got {side!r}.")
    bearing = (ping.heading_deg + _SIDE_BEARING_OFFSET[side]) % 360.0
    lon, lat, _ = _GEOD.fwd(ping.lon, ping.lat, bearing, ground_range_m)
    return lat, lon


def position_uncertainty(
    ping: Ping,
    ground_range_m: float,
    *,
    gnss_accuracy_m: float | None = None,
    heading_accuracy_deg: float | None = None,
    layback_uncertainty_frac: float | None = None,
) -> float:
    """A 1-sigma horizontal position uncertainty (metres), honestly combining three terms.

    - **GNSS accuracy**: a stated constant, independent of range.
    - **Heading uncertainty * range**: an angular error displaces a target laterally by
      ``range * radians(heading_error)`` -- it grows with across-track distance.
    - **Layback**: when the towfish position is derived from ship nav plus cable layback rather
      than measured directly, a fraction of the layback distance is added.

    Constants default to ``configs/default.yaml`` (``runtime.*``); overridable for tests.
    """
    rt = load_config()["runtime"]
    gnss = rt["gnss_accuracy_m"] if gnss_accuracy_m is None else gnss_accuracy_m
    head_deg = rt["heading_accuracy_deg"] if heading_accuracy_deg is None else heading_accuracy_deg
    lay_frac = (
        rt["layback_uncertainty_frac"]
        if layback_uncertainty_frac is None
        else layback_uncertainty_frac
    )

    from math import radians

    heading_term = abs(ground_range_m) * radians(head_deg)
    layback_term = lay_frac * abs(getattr(ping, "layback_m", 0.0) or 0.0)
    return float(gnss + heading_term + layback_term)

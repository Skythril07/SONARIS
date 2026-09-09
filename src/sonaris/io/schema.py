"""Data contracts for the decode and geometry stages (layout §2.1, §2.2).

These dataclasses are the *only* thing modules share about line-level data. No stage reaches
back past its input. ``CorrectedLine`` lives here too (not in ``geometry/``) so every stage
imports one contract module with no import cycles.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# XTFNavUnits values (layout §6, gotcha #1). If 'meters', SensorX/Ycoordinate are projected
# easting/northing in an unstated CRS — assert and refuse rather than emit coords off (0, 0).
NAV_LATLON = "latlon"   # XTFNavUnits == 3
NAV_METERS = "meters"   # XTFNavUnits == 0

# XTFChannelType values (gotcha #6). Read the enum; never infer side from channel index order,
# or a silent flip mirrors every detection across the track line.
SIDE_PORT = "port"      # XTFChannelType == 1
SIDE_STBD = "stbd"      # XTFChannelType == 2


@dataclass
class Ping:
    """One sonar ping on one channel. Fields map to verified pyxtf struct members (XTF rev 44)."""

    index: int
    time_utc: float            # from Year/Month/Day/Hour/Minute/Second/HSeconds
    lat: float                 # XTFPingHeader.SensorYcoordinate
    lon: float                 # XTFPingHeader.SensorXcoordinate
    heading_deg: float         # SensorHeading (fall back: ShipGyro) — see gotcha #5
    altitude_m: float          # SensorPrimaryAltitude — height above seabed
    depth_m: float             # SensorDepth
    pitch_deg: float           # SensorPitch
    roll_deg: float            # SensorRoll
    heave_m: float             # Heave
    layback_m: float           # Layback
    slant_range_m: float       # XTFPingChanHeader.SlantRange — per ping, per channel (gotcha #3)
    n_samples: int             # XTFPingChanHeader.NumSamples
    samples: np.ndarray        # 1-D intensity for this channel


@dataclass
class SurveyLine:
    """A decoded survey line: nav + per-channel intensity. Output of Stage 1 (decode)."""

    line_id: str
    source_file: str
    sonar_type: str                                   # XTFFileHeader.SonarType
    nav_units: str                                    # NAV_LATLON | NAV_METERS (assert at load)
    channels: dict[str, list[Ping]] = field(default_factory=dict)   # SIDE_PORT | SIDE_STBD
    corrected: bool = False    # XTFChanInfo.CorrectionFlags already ground_range? (gotcha #2)
    provenance: dict = field(default_factory=dict)  # heading/position source etc. (gotchas #4, #5)


@dataclass
class CorrectedLine:
    """A geometry-corrected channel on a uniform ground-range grid. Output of Stage 2 (correct)."""

    line_id: str
    channel: str                    # SIDE_PORT | SIDE_STBD
    image: np.ndarray               # (n_pings, n_ground_bins) uint16
    ground_res_m: float             # metres per column, uniform after resample
    bottom_idx: np.ndarray          # (n_pings,) first-return sample index
    altitude_m: np.ndarray          # (n_pings,) altitude used for the correction
    ping_index: np.ndarray          # (n_pings,) back-reference into nav.parquet

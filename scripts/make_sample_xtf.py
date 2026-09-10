"""Generate a larger, realistic sample XTF for testing the pipeline + web console.

Unlike the tiny single-object test fixture (tests/synthetic.py), this writes a multi-ping survey
with a gently curving track, a textured seabed (range-dependent TVG + speckle + water column), and
several planted targets on both channels -- each a bright return followed by a dark acoustic shadow
at greater range, so the P2.2 shadow-consistency check has something real to find.

Still a *valid* XTF written with pyxtf's own writer, so pyxtf/our reader decode it normally. Output
defaults to ``data/raw/sonar.xtf``. Run: ``.venv/Scripts/python.exe scripts/make_sample_xtf.py``
"""
from __future__ import annotations

import argparse
import ctypes
import datetime
from pathlib import Path

import numpy as np
import pyxtf
from pyproj import Geod

_GEOD = Geod(ellps="WGS84")

# (center_ping_fraction, side, ground_range_m, along_pings, across_samples, intensity)
# fractions place targets along the track; both sides; varied range/size/brightness.
_TARGETS = [
    (0.10, "stbd", 18.0, 7, 6, 250),
    (0.22, "port", 40.0, 5, 4, 230),
    (0.35, "stbd", 62.0, 9, 8, 255),
    (0.48, "port", 12.0, 4, 3, 235),
    (0.60, "stbd", 33.0, 6, 5, 245),
    (0.74, "port", 55.0, 8, 7, 240),
    (0.88, "stbd", 26.0, 5, 5, 252),
]


def _seabed_row(
    n_samples: int, bottom_idx: int, rng: np.random.Generator, clean: bool = False
) -> np.ndarray:
    """A background across-track return: dark water column, then seabed.

    ``clean=True`` returns a flat, low, quiet seabed (no TVG/speckle) so that after normalize+CLAHE
    nothing crosses the stub detector's brightness threshold -- only planted targets do. This keeps
    the classical stand-in detector fast and its output clean (ideal for a live demo). ``clean=False``
    adds TVG shading + speckle for a realistic-looking survey (best used with the trained model).
    """
    if clean:
        row = np.full(n_samples, 12.0)
        row[:bottom_idx] = 6.0
        return row.astype(np.uint8)
    r = np.arange(n_samples, dtype=float)
    tvg = 45.0 * np.exp(-(r - bottom_idx) / (n_samples * 0.55))  # brighter near nadir, fades w/ range
    tvg[r < bottom_idx] = 0.0
    base = 14.0 + np.clip(tvg, 0, 70)
    row = base + rng.normal(0.0, 6.0, size=n_samples)          # speckle
    row[:bottom_idx] = rng.normal(6.0, 2.0, size=bottom_idx)   # water column: low, quiet
    return np.clip(row, 0, 200).astype(np.uint8)               # keep background < detector threshold


def make_sample_xtf(
    path: Path,
    *,
    n_pings: int = 1400,
    n_samples: int = 1200,
    slant_range_m: float = 80.0,
    altitude_m: float = 12.0,
    start_lat: float = 8.76,
    start_lon: float = 78.10,
    heading_deg: float = 90.0,
    ping_spacing_m: float = 0.5,
    seed: int = 7,
    clean: bool = False,
) -> Path:
    rng = np.random.default_rng(seed)
    res_slant = slant_range_m / n_samples
    bottom_idx = round(altitude_m / res_slant)

    # Gently curving track: heading drifts a few degrees over the line (more realistic than dead straight).
    headings = heading_deg + 8.0 * np.sin(np.linspace(0, np.pi, n_pings))
    lats, lons = [], []
    lon, lat = start_lon, start_lat
    for i in range(n_pings):
        lats.append(lat)
        lons.append(lon)
        lon, lat, _ = _GEOD.fwd(lon, lat, float(headings[i]), ping_spacing_m)

    # Resolve targets to concrete (ping, side, slant sample index) with shadows.
    planted = []
    for frac, side, gr, along, across, inten in _TARGETS:
        cp = int(frac * n_pings)
        idx = round(float(np.hypot(gr, altitude_m)) / res_slant)
        planted.append((cp, side, idx, along, across, inten))

    fh = pyxtf.XTFFileHeader()
    fh.SonarName = b"SONARIS-Sample"
    fh.SonarType = pyxtf.XTFSonarType.unknown1
    fh.NavUnits = int(pyxtf.XTFNavUnits.latlon)
    fh.NumberOfSonarChannels = 2
    for k, side in ((0, pyxtf.XTFChannelType.port), (1, pyxtf.XTFChannelType.stbd)):
        fh.ChanInfo[k].TypeOfChannel = side.value
        fh.ChanInfo[k].SubChannelNumber = k
        fh.ChanInfo[k].BytesPerSample = 1
        fh.ChanInfo[k].SampleFormat = pyxtf.XTFSampleFormat.byte.value

    base_t = datetime.datetime(2024, 1, 1, 0, 0, 0)  # noqa: DTZ001 - only fills XTF Y/M/D/H/M/S fields
    chan_hdr_size = ctypes.sizeof(pyxtf.XTFPingChanHeader)
    ping_hdr_size = ctypes.sizeof(pyxtf.XTFPingHeader)

    with open(path, "wb") as f:
        f.write(fh.to_bytes())
        for i in range(n_pings):
            t = base_t + datetime.timedelta(seconds=i)
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
            p.SensorHeading = float(headings[i])
            p.SensorPitch = 0
            p.SensorRoll = 0

            c = (pyxtf.XTFPingChanHeader(), pyxtf.XTFPingChanHeader())
            for k in (0, 1):
                c[k].ChannelNumber = k
                c[k].SlantRange = slant_range_m
                c[k].NumSamples = n_samples
                c[k].SampleFormat = 8
            p.ping_chan_headers = c

            rows = {"port": _seabed_row(n_samples, bottom_idx, rng, clean),
                    "stbd": _seabed_row(n_samples, bottom_idx, rng, clean)}
            for cp, side, idx, along, across, inten in planted:
                if abs(i - cp) <= along // 2:
                    row = rows[side]
                    lo, hi = max(0, idx - across // 2), min(n_samples, idx + across // 2 + 1)
                    row[lo:hi] = inten                                   # bright target return
                    s0, s1 = hi, min(n_samples, hi + max(6, across * 2))
                    row[s0:s1] = 2                                       # dark acoustic shadow beyond
            p.data = [rows["port"], rows["stbd"]]

            p.NumBytesThisRecord = ping_hdr_size + 2 * chan_hdr_size + 2 * n_samples
            f.write(p.to_bytes())

    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a larger sample XTF for testing.")
    ap.add_argument("--out", default="data/raw/sonar.xtf")
    ap.add_argument("--pings", type=int, default=1400)
    ap.add_argument("--samples", type=int, default=1200)
    ap.add_argument("--clean", action="store_true",
                    help="flat low-contrast seabed: fast + clean for the stub detector (demo mode)")
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    make_sample_xtf(out, n_pings=args.pings, n_samples=args.samples, clean=args.clean)
    size_mb = out.stat().st_size / 1e6
    print(f"[make_sample_xtf] wrote {out}  ({size_mb:.1f} MB, {args.pings} pings x {args.samples} samples x 2 chan)")
    print(f"[make_sample_xtf] {len(_TARGETS)} targets planted (both channels, with acoustic shadows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

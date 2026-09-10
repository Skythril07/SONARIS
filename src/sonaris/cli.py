"""sonaris command-line interface.

Subcommands mirror the pipeline stages (layout §1). Phase 0 wires the parser and stubs each
stage; every stub names the plan gate at which it gets implemented. The CLI is the single
source of truth — the web app (Phase 3) shells out to it and never reimplements pipeline logic.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__

# stage -> plan gate where it becomes real. Stages not yet implemented print their gate.
_STAGE_GATE = {
    "decode": "P1.1",
    "correct": "P1.2",
    "tile": "P1.4",
    "train": "P1.6",
    "infer": "P1.7",
    "report": "P1.8",
    "run": "P1 (full vertical slice)",
}


def _stub(args: argparse.Namespace) -> int:
    gate = _STAGE_GATE.get(args.command, "?")
    print(f"[sonaris] '{args.command}' is not implemented yet - scheduled for plan gate {gate}.")
    return 2


def _cmd_decode(args: argparse.Namespace) -> int:
    """[P1.1] Decode an XTF survey file, print its nav table, and run the G1 sanity checks."""
    p = Path(args.input)
    if not p.exists():
        print(f"[sonaris] file not found: {p}")
        return 2
    if p.suffix.lower() != ".xtf":
        print(
            f"[sonaris] decode handles survey files (.xtf); got '{p.suffix}'. "
            f"Bare images are the dataset path (io/image_reader.py), not decode."
        )
        return 2

    from .io.xtf_reader import NavUnitsError, nav_dataframe, read_xtf

    try:
        line = read_xtf(p)
    except NavUnitsError as e:
        print(f"[sonaris] REFUSED: {e}")
        return 1
    except Exception as e:  # noqa: BLE001 - surface a clean message, not a traceback
        print(f"[sonaris] failed to decode {p}: {e.__class__.__name__}: {e}")
        return 1

    import pandas as pd

    df = nav_dataframe(line)
    prov = line.provenance
    print(f"file         : {line.source_file}")
    print(f"line_id      : {line.line_id}")
    print(f"sonar        : {line.sonar_type}  ({prov.get('sonar_name', '')})")
    print(f"nav_units    : {line.nav_units}")
    print("channels     : " + ", ".join(f"{s}={len(ps)}" for s, ps in line.channels.items()))
    print(f"corrected    : {line.corrected} (vendor already ground-range corrected)")
    print(f"heading src  : {prov.get('heading_source')}")
    print(f"position src : {prov.get('position_source')}")
    print()

    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(df.head(8).to_string(index=False))
    if len(df) > 8:
        print(f"... ({len(df)} pings total)")
    print()

    # ---- G1 sanity checks ----
    ok = True

    def check(name: str, cond: bool) -> None:
        nonlocal ok
        ok = ok and cond
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    t = df["time_utc"].dropna()
    check("nav_units is lat/lon", line.nav_units == "latlon")
    check("time monotonic non-decreasing", bool(t.is_monotonic_increasing))
    check("lat within [-90, 90]", bool(df["lat"].between(-90, 90).all()))
    check("lon within [-180, 180]", bool(df["lon"].between(-180, 180).all()))
    check("position not all-zero", bool(((df["lat"] != 0) | (df["lon"] != 0)).any()))
    print(f"\nG1 {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def _cmd_infer(args: argparse.Namespace) -> int:
    """[P1.7 stub, P1.3 gate] Stub detector: plant one detection at the brightest seabed sample,
    push it through the real coordinate chain, and write GeoJSON. No ML -- this proves geometry.
    """
    if not getattr(args, "stub", False):
        return _stub(args)

    p = Path(args.input)
    if not p.exists():
        print(f"[sonaris] file not found: {p}")
        return 2

    import numpy as np

    from .geometry.georef import detection_to_latlon, position_uncertainty
    from .geometry.slant_range import remove_water_column
    from .io.xtf_reader import NavUnitsError, read_xtf
    from .report.schema import Detection
    from .report.writers import write_geojson

    try:
        line = read_xtf(p)
    except NavUnitsError as e:
        print(f"[sonaris] REFUSED: {e}")
        return 1

    # Stub "detector": the single brightest sample beyond the nadir, across all channels/pings.
    best = None  # (intensity, side, ping, sample_idx)
    for side, pings in line.channels.items():
        for ping in pings:
            if ping.n_samples <= 0 or ping.slant_range_m <= 0:
                continue
            res_slant = ping.slant_range_m / ping.n_samples
            beyond, nadir_idx = remove_water_column(ping.samples, ping.altitude_m, res_slant)
            if beyond.shape[0] == 0:
                continue
            k = int(np.argmax(beyond))
            intensity = float(beyond[k])
            if best is None or intensity > best[0]:
                best = (intensity, side, ping, nadir_idx + k)

    if best is None:
        print("[sonaris] no seabed samples beyond the nadir; nothing to detect.")
        return 1

    _, side, ping, sample_idx = best
    res_slant = ping.slant_range_m / ping.n_samples
    r_slant = sample_idx * res_slant
    r_ground = float(np.sqrt(max(0.0, r_slant**2 - ping.altitude_m**2)))
    lat, lon = detection_to_latlon(ping, r_ground, side)
    unc = position_uncertainty(ping, r_ground)

    det = Detection(
        detection_id=f"{line.line_id}-stub-0",
        line_id=line.line_id,
        cls="stub",
        confidence=1.0,
        confidence_raw=1.0,
        lat=lat,
        lon=lon,
        position_uncertainty_m=unc,
        length_m=0.0,
        width_m=0.0,
        ground_range_m=r_ground,
        channel=side,
        ping_index=ping.index,
    )

    out = Path(args.out) if args.out else Path("data/outputs/detections.geojson")
    write_geojson([det], out)

    print(f"line_id      : {line.line_id}")
    print(f"detection    : {side} ping={ping.index} sample={sample_idx} intensity={best[0]:.0f}")
    print(f"ground_range : {r_ground:.2f} m")
    print(f"position     : lat={lat:.6f} lon={lon:.6f}  (+/- {unc:.2f} m)")
    print(f"wrote        : {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sonaris",
        description="Side-scan sonar debris & anomaly detection (SIH26057).",
    )
    p.add_argument("--version", action="version", version=f"sonaris {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("decode", help="[P1.1] vendor file -> SurveyLine; print nav table")
    d.add_argument("input")

    c = sub.add_parser("correct", help="[P1.2] slant->ground range; write corrected waterfall")
    c.add_argument("input")

    t = sub.add_parser("tile", help="[P1.4] preprocess + tile; write tiles.parquet")
    t.add_argument("input")

    tr = sub.add_parser("train", help="[P1.6] train baseline YOLO-Seg")
    tr.add_argument("--data")

    i = sub.add_parser("infer", help="[P1.7] detect + filter + georeference")
    i.add_argument("input")
    i.add_argument("--stub", action="store_true",
                   help="[P1.3] no-ML stub: brightest sample -> coordinate chain -> GeoJSON")
    i.add_argument("--out", help="output GeoJSON path (default data/outputs/detections.geojson)")

    r = sub.add_parser("report", help="[P1.8] write GeoJSON/CSV/GPX")
    r.add_argument("input")

    run = sub.add_parser("run", help="[P1] full vertical slice: file -> detections + reports")
    run.add_argument("--input", required=True)
    run.add_argument("--out", required=True)

    for name in _STAGE_GATE:
        sub.choices[name].set_defaults(func=_stub)
    sub.choices["decode"].set_defaults(func=_cmd_decode)   # P1.1 - implemented
    sub.choices["infer"].set_defaults(func=_cmd_infer)     # P1.3 stub path (--stub)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

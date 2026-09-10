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


def _load_line_or_none(p: Path):
    """Shared front door for correct/tile: validate path, decode, surface refusals cleanly."""
    from .io.xtf_reader import NavUnitsError, read_xtf

    if not p.exists():
        print(f"[sonaris] file not found: {p}")
        return None, 2
    if p.suffix.lower() != ".xtf":
        print(f"[sonaris] expects a survey file (.xtf); got '{p.suffix}'.")
        return None, 2
    try:
        return read_xtf(p), 0
    except NavUnitsError as e:
        print(f"[sonaris] REFUSED: {e}")
        return None, 1


def _cmd_correct(args: argparse.Namespace) -> int:
    """[P1.2] Slant->ground correct each channel into a waterfall; write .npy + sidecar + PNG."""
    import json

    import cv2
    import numpy as np

    from .config import load_config
    from .geometry.correct import correct_line
    from .preprocess.enhance import normalize_uint8

    cfg = load_config()
    ground_res_m = cfg["geometry"]["ground_res_m"]
    norm = cfg["preprocess"]["normalize"]

    p = Path(args.input)
    line, code = _load_line_or_none(p)
    if line is None:
        return code

    out_dir = Path(args.out) if args.out else Path("data/interim") / line.line_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"line_id      : {line.line_id}")
    ok = True
    for channel in line.channels:
        cl = correct_line(line, channel, ground_res_m)
        np.save(out_dir / f"{channel}_ground.npy", cl.image)
        cv2.imwrite(
            str(out_dir / f"{channel}_ground.png"),
            normalize_uint8(cl.image, norm["p_low"], norm["p_high"]),
        )
        with open(out_dir / f"{channel}_ground.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "line_id": cl.line_id,
                    "channel": cl.channel,
                    "source_file": line.source_file,
                    "ground_res_m": cl.ground_res_m,
                    "shape": list(cl.image.shape),
                    "ping_index": cl.ping_index.tolist(),
                    "altitude_m": cl.altitude_m.tolist(),
                    "bottom_idx": cl.bottom_idx.tolist(),
                },
                f,
            )
        h, w = cl.image.shape
        print(f"  {channel:4s}: image {h}x{w}  ground_res={cl.ground_res_m} m  -> {channel}_ground.npy/.png")
        ok = ok and h > 0 and w > 0
    print(f"wrote        : {out_dir}")
    print(f"\nG2 {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def _cmd_tile(args: argparse.Namespace) -> int:
    """[P1.4] Correct + preprocess + tile each channel; write tile .npy stacks + tiles.parquet.

    The tile index (tiles.parquet) is the only route back to world coordinates -- it records the
    ping-row and ground-column origin of every tile, never the filename (layout section 2.3).
    """
    import json

    import numpy as np

    from .config import load_config
    from .geometry.correct import correct_line
    from .preprocess.enhance import build_stack
    from .preprocess.tiling import TileRef, iter_tiles, tiles_dataframe

    cfg = load_config()
    ground_res_m = cfg["geometry"]["ground_res_m"]
    size_px = cfg["tiling"]["size_px"]
    stride_frac = cfg["tiling"]["stride_frac"]

    p = Path(args.input)
    line, code = _load_line_or_none(p)
    if line is None:
        return code

    out_dir = Path(args.out) if args.out else Path("data/processed") / line.line_id
    tiles_dir = out_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    print(f"line_id      : {line.line_id}")
    refs: list[TileRef] = []
    for channel in line.channels:
        cl = correct_line(line, channel, ground_res_m)
        stack = build_stack(cl.image, cfg["preprocess"])
        # Sidecar: image row -> ping mapping, needed to recover world coords from the tile index.
        with open(out_dir / f"{channel}.meta.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "line_id": cl.line_id, "channel": channel, "source_file": line.source_file,
                    "ground_res_m": cl.ground_res_m, "ping_index": cl.ping_index.tolist(),
                },
                f,
            )
        n_before = len(refs)
        for ping_start, ping_end, col_start, col_end, tile in iter_tiles(stack, size_px, stride_frac):
            tile_id = f"{line.line_id}_{channel}_p{ping_start}_c{col_start}"
            rel = f"tiles/{tile_id}.npy"
            np.save(out_dir / rel, tile)
            refs.append(TileRef(
                tile_id=tile_id, line_id=line.line_id, channel=channel,
                ping_start=ping_start, ping_end=ping_end, col_start=col_start, col_end=col_end,
                ground_res_m=cl.ground_res_m, path=rel,
            ))
        print(f"  {channel:4s}: stack {stack.shape} -> {len(refs) - n_before} tiles")

    df = tiles_dataframe(refs)
    df.to_parquet(out_dir / "tiles.parquet", index=False)
    print(f"wrote        : {out_dir / 'tiles.parquet'} ({len(df)} tiles)")
    return 0 if len(df) else 1


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


def _cmd_report(args: argparse.Namespace) -> int:
    """[P1.8] Convert a detection GeoJSON file into configured report formats."""
    p = Path(args.input)
    if not p.exists():
        print(f"[sonaris] file not found: {p}")
        return 2

    import json

    from .report.schema import Detection
    from .report.writers import write_reports

    try:
        with open(p, encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("type") != "FeatureCollection":
            raise ValueError("expected a GeoJSON FeatureCollection")
        detections = []
        for feature in payload["features"]:
            geometry = feature["geometry"]
            if geometry.get("type") != "Point":
                raise ValueError("each GeoJSON feature must be a Point")
            lon, lat = geometry["coordinates"]
            detections.append(Detection(lat=lat, lon=lon, **dict(feature["properties"])))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as e:
        print(f"[sonaris] failed to read detections from {p}: {e}")
        return 1

    formats = args.formats.split(",") if args.formats else None
    try:
        reports = write_reports(detections, args.out, formats)
    except ValueError as e:
        print(f"[sonaris] failed to write reports: {e}")
        return 1
    for report_format, path in reports.items():
        print(f"wrote {report_format}: {path}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    """[P1.6] Train the baseline YOLO-Seg detector (needs the ml extra + a unified corpus)."""
    if not args.data:
        print("[sonaris] train needs --data <data.yaml> from datasets.unify (plan P1.5).")
        return 2
    from .config import load_config
    from .models.train import train

    try:
        result = train(args.data, load_config())
    except (RuntimeError, FileNotFoundError) as e:
        print(f"[sonaris] {e}")
        return 1
    print(f"trained {result['model']} on {result['data']}")
    print(f"mAP@50  : {result.get('map50')}   <- control metric (plan P1.6)")
    print(f"weights : {result.get('save_dir')}")
    print(f"metrics : {result['metrics']}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    """[P1] Full vertical slice: XTF -> georeferenced detections + reports + map."""
    p = Path(args.input)
    if not p.exists():
        print(f"[sonaris] file not found: {p}")
        return 2

    from .io.xtf_reader import NavUnitsError
    from .pipeline import run_pipeline

    cfg = None
    if args.verify:  # opt-in P2.2/P2.3 signals so the evidence panel shows real verdicts
        from .config import load_config
        cfg = load_config()
        cfg["verify"]["shadow"]["enabled"] = True
        cfg["verify"]["plausibility"]["enabled"] = True

    try:
        dets = run_pipeline(p, args.out, cfg=cfg, progress_path=args.progress)
    except NavUnitsError as e:
        print(f"[sonaris] REFUSED: {e}")
        return 1

    print(f"input        : {p}")
    print(f"detections   : {len(dets)}")
    for d in dets[:10]:
        print(f"  {d.channel} ping={d.ping_index} r={d.ground_range_m:.1f}m "
              f"lat={d.lat:.6f} lon={d.lon:.6f} +/-{d.position_uncertainty_m:.1f}m cls={d.cls}")
    if len(dets) > 10:
        print(f"  ... ({len(dets)} total)")
    print(f"wrote        : {args.out} (detections.geojson/csv/gpx, *_annotated.png, map.html)")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    """[P3] Launch the operator-console backend (needs the app extra: fastapi + uvicorn)."""
    try:
        import uvicorn
    except ImportError:
        print("[sonaris] serve needs the app extra. Install: pip install -e '.[app]' (plan P3).")
        return 1
    from .web.app import app
    print(f"[sonaris] operator console API on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
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
    c.add_argument("--out", help="output dir (default data/interim/<line_id>)")

    t = sub.add_parser("tile", help="[P1.4] preprocess + tile; write tiles.parquet")
    t.add_argument("input")
    t.add_argument("--out", help="output dir (default data/processed/<line_id>)")

    tr = sub.add_parser("train", help="[P1.6] train baseline YOLO-Seg")
    tr.add_argument("--data")

    i = sub.add_parser("infer", help="[P1.7] detect + filter + georeference")
    i.add_argument("input")
    i.add_argument("--stub", action="store_true",
                   help="[P1.3] no-ML stub: brightest sample -> coordinate chain -> GeoJSON")
    i.add_argument("--out", help="output GeoJSON path (default data/outputs/detections.geojson)")

    r = sub.add_parser("report", help="[P1.8] write GeoJSON/CSV/GPX")
    r.add_argument("input")
    r.add_argument("--out", required=True, help="output directory")
    r.add_argument("--formats", help="comma-separated formats (default: config report.formats)")

    run = sub.add_parser("run", help="[P1] full vertical slice: file -> detections + reports")
    run.add_argument("--input", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--progress", help="append per-stage JSON lines here (drives the web SSE rail)")
    run.add_argument("--verify", action="store_true",
                     help="enable P2.2 shadow + P2.3 plausibility signals for the evidence panel")

    srv = sub.add_parser("serve", help="[P3] launch the FastAPI operator console backend")
    srv.add_argument("--host", default="127.0.0.1")
    srv.add_argument("--port", type=int, default=8000)

    for name in _STAGE_GATE:
        sub.choices[name].set_defaults(func=_stub)
    sub.choices["decode"].set_defaults(func=_cmd_decode)   # P1.1 - implemented
    sub.choices["correct"].set_defaults(func=_cmd_correct)  # P1.2 - ground-range waterfall
    sub.choices["tile"].set_defaults(func=_cmd_tile)        # P1.4 - tiling + tiles.parquet
    sub.choices["infer"].set_defaults(func=_cmd_infer)     # P1.3 stub path (--stub)
    sub.choices["report"].set_defaults(func=_cmd_report)   # P1.8 - report conversion
    sub.choices["train"].set_defaults(func=_cmd_train)     # P1.6 - YOLO-Seg train scaffold
    sub.choices["run"].set_defaults(func=_cmd_run)         # P1  - full vertical slice
    sub.choices["serve"].set_defaults(func=_cmd_serve)     # P3  - web backend
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

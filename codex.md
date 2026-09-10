# codex.md — SONARIS delegation queue

**Purpose.** A queue of *self-contained* tasks handed to a separate coding agent (Codex / ChatGPT)
so the main session (Claude) spends its tokens only on high-judgment integration. Each task below
is written to be executed with **zero prior context** beyond this file and the three governing docs.

**Who does what.**
- **Delegate (this file):** mechanical, well-specified, contract-fixed work — writers, tests,
  scripts, docs, pure functions with a stated signature and acceptance test.
- **Keep with Claude (do NOT delegate):** the coordinate chain (`geometry/`), the pyxtf real-API
  gotchas (`io/xtf_reader.py`), YOLO-Seg training/domain-shift, and any change to a data contract
  (`io/schema.py`, `report/schema.py`). These are the silent-bug core; they stay under review.

---

## Ground rules for the delegate (read once, obey on every task)

1. **Read first:** `HANDOFF.md` (status), `plan.md` (build order — governs), `SONARIS_Build_Layout.md`
   (contracts §2, coordinate chain §3, module boundaries §5, gotchas §6). Do not contradict them.
2. **Run with the repo venv**, never a system Python:
   - Tests:  `./.venv/Scripts/python.exe -m pytest -q`
   - Lint:   `./.venv/Scripts/ruff.exe check src tests scripts`
   - CLI:    `./.venv/Scripts/sonaris.exe <cmd>`
   Both must be green before a task is "done".
3. **If a number appears in code, it is a bug** (layout §4). Every tunable lives in `configs/*.yaml`;
   read it via `sonaris.config.load_config()`. Add a new key rather than hardcode.
4. **Module boundaries are strict** (layout §5): `io/` only decodes; `geometry/` knows nothing about
   models/pixels/files; `models/` does no coordinate maths; `postprocess/` never re-reads raw files;
   `report/` writers are **dumb and total** (serialise the dataclass, never recompute geometry).
5. **Windows console is cp1252** — every user-facing CLI/print string must be **plain ASCII**
   (no `§ — ° ★ ±`, no emoji). Docstrings/comments may use UTF-8.
6. **Do not commit, push, or touch git.** Leave changes in the working tree; report a summary and
   the test/ruff output. Claude reviews and commits (commits are authored solely by Skythril07).
7. **Do not edit** `io/schema.py`, `report/schema.py`, `io/xtf_reader.py`, or `geometry/*` unless the
   task explicitly says so. Import from them; don't change them.
8. When done, **update this file**: move the task to "Completed", note files touched + test count.

---

## Active delegations

### D1 — `report/writers.py`: CSV + GPX writers (SAFE) — plan gate P1.8
GeoJSON already exists in `src/sonaris/report/writers.py` as the pattern to follow.
- Add `write_csv(detections: list[Detection], path) -> Path`: one row per detection, header =
  the `Detection` dataclass field names in declared order. Use the stdlib `csv` module. Serialise
  `evidence` (a dict) as a compact JSON string; write `None` as empty. Create parent dirs; UTF-8,
  `newline=""`.
- Add `write_gpx(detections, path) -> Path`: GPX 1.1 XML, one `<wpt lat=.. lon=..>` per detection,
  `<name>` = `detection_id`, `<desc>` = `"cls=<cls> conf=<confidence:.2f> unc=<..>m"`. Build with
  `xml.etree.ElementTree` (no new deps). Namespace `http://www.topografix.com/GPX/1/1`.
- Add `write_reports(detections, out_dir, formats: list[str] | None = None) -> dict[str, Path]`:
  dispatch by format; default `formats` from `load_config()["report"]["formats"]`
  (`[geojson, csv, gpx]`). Filenames `detections.<ext>` under `out_dir`.
- **Acceptance:** new `tests/test_writers.py` — build 2–3 `Detection`s, write all three formats to
  `tmp_path`, assert: GeoJSON parses and coords are `[lon, lat]`; CSV re-read with `csv.DictReader`
  has the right rows/lat/lon; GPX parses and each `<wpt>` has correct `lat`/`lon` attributes.
  `pytest` and `ruff` green.

### D2 — Wire `sonaris report` to the writers (SAFE) — plan gate P1.8
`report` is currently a stub in `cli.py`. Make `_cmd_report` read a detections file and write the
configured formats. Input is a GeoJSON produced by `infer --stub` (reuse `detections_to_geojson`
shape). Parse it back into `list[Detection]` (all fields present in `properties` + coords give
lat/lon), call `write_reports`, print the paths written (ASCII only). Add `--out` (dir) and
`--formats` (comma list, optional) args. Register like `infer`/`decode` are registered. Add a CLI
smoke test that runs `infer --stub` then `report` on the result in `tmp_path`.

### D3 — Flesh out `scripts/download_datasets.sh` + `configs/datasets.yaml` doc (SAFE-ISH)
Make the script idempotent and self-documenting: for each dataset named in `configs/datasets.yaml`,
print name/URL/license and download into `data/raw/<name>/` only if absent (`curl -fL --retry`),
guarded so re-runs skip existing files. Do **not** invent URLs — if a dataset entry lacks a URL,
print a clear "manual download required: <where>" line and continue. Keep it POSIX `sh`. No Python.
**Acceptance:** `bash -n scripts/download_datasets.sh` clean; running it with no network prints the
plan without crashing. (Leave `datasets.yaml` values as-is unless a field is obviously malformed.)

### D4 — README quickstart refresh (SAFE) — docs
Update `README.md` so a newcomer can: create the venv, `pip install -e ".[dev]"`, run
`verify_env.py`, `pytest`, and the two working CLI paths (`decode`, `infer --stub`). Mirror the exact
commands from `HANDOFF.md` §4. Keep it short; don't document unbuilt stages as if they work — mark
them "(stub / planned: P1.x)". ASCII in any copy-paste command blocks.

### D5 — Pure-function unit tests for `geometry/slant_range.py` (SAFE — tests only)
Do **not** edit `geometry/`. Add focused tests (extend `tests/test_geometry.py` or a new file) that
pin behaviour of `remove_water_column` and `slant_to_ground`: empty input returns empty; a
`slant_range_m <= 0` or `ground_res_m <= 0` raises `ValueError`; the returned ground grid length
equals `floor(r_ground_max / ground_res_m) + 1`; monotonic ground range. These lock the contract
before P1.4 builds on it.

---

## Delegate with caution (Claude reviews the risky part)

### C1 — P1.4 tiling (`preprocess/` + `tile`) — PARTIAL delegate
Safe to delegate: the **preprocessing** (per layout `preprocess` config — raw normalisation + CLAHE,
producing the model input channels) and the mechanical sliding-window cut (`tiling.size_px`,
`tiling.stride_frac`). **Claude keeps/reviews** the `tiles.parquet` index bookkeeping —
`ping_start`, `col_start`, `channel` per tile — because that mapping (layout §3 top) feeds the
coordinate chain and any off-by-one there is a *silent* geolocation bug. Delegate: produce the
image transforms + tiling with a clear `Tile` dataclass; leave the parquet index columns stubbed
with a `# TODO(claude): verify ping/col origin against geometry §3` marker.

---

## Handback protocol

When a task is finished, the delegate reports back in this shape (keep it terse):

```
TASK: D1
FILES: src/sonaris/report/writers.py, tests/test_writers.py
TESTS: 17 passed, 0 skipped
RUFF:  All checks passed!
NOTES: added report.formats fallback; GPX uses ElementTree, no new deps
```

Claude then reviews, integrates, and commits (Skythril07, no AI trailer). Move the task to
**Completed** below with the commit hash once landed.

---

## Completed

*(none yet — first entries land here after review + commit.)*

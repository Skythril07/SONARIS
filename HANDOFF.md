# SONARIS — Session Handoff

**Purpose:** everything a fresh session needs to continue this project with zero context loss.
Read this first, then `plan.md`. Last updated after completing **the full Phase 1 vertical slice**
(`sonaris run` works end to end with a classical stand-in detector).

---

## 1. What this project is

**SONARIS** — SIH26057 hackathon entry: an AI system that detects underwater marine debris and
anomalies (fishing gear, pipes/cables, wrecks) from **side-scan sonar**, and reports each
detection's real-world **lat/lon** as GeoJSON/CSV/GPX plus an operator web console.

- Working directory: `D:\Jash\TimePass\Hackathon` (Windows 11, PowerShell + Git Bash).
- GitHub: **`Skythril07/SONARIS`** (public). Default branch `main`.
- The defining discipline: **coordinate-chain-first.** Every geometry bug is silent (produces
  confident, well-formatted, *wrong* coordinates). So we proved pixel → lat/lon with a *stub*
  detector and synthetic-geometry tests **before** any ML. That gate (**P1.3 ★**) is **green**;
  the same discipline now guards the tile index (P1.4) and the full pipeline (P1.7).

## 2. Governing documents (read in this order)

1. **`plan.md`** — the execution plan. MVP-first, 3 phases, deviates deliberately from the
   master layout (see its §2 override table). **This governs build order.**
2. **`SONARIS_Build_Layout.md`** — the architecture master plan (data contracts, coordinate
   chain math §3, module responsibilities §5, the real-API gotchas §6, gates).
3. **`README.md`** — quickstart.
4. **`codex.md`** — delegation queue: self-contained tasks handed to a background coding agent
   (Codex/ChatGPT) to keep Claude's token use on high-judgment work. D1–D5 are **done**.
5. **`potential_dataset_links.md`** — leads for real labelled datasets (needed for P1.6 training).
6. Persistent memory (loaded automatically each session): `sonaris-project`,
   `sonaris-env-commands` under the session's memory dir.

## 3. Current status — Phase 1 vertical slice COMPLETE (with a dumb detector)

| Phase | Gates | Done |
|---|---|---|
| Phase 0 — Foundations | 0.1–0.4 | ✅ 4/4 (G0 PASS) |
| Phase 1 — MVP vertical slice | P1.1–P1.9 | ✅ 8/9 code-complete; **P1.6 real training deferred** (data-blocked) |
| Phase 2 — Intelligence | P2.1–P2.9 | ⬜ 0/9 |
| Phase 3 — Web app | P3.1–P3.9 | ⬜ 0/9 |
| Cross-cutting (experiment ladder §6, domain-shift §7) | — | ⬜ 0 |

**`sonaris run --input <xtf> --out <dir>`** goes file → decode → correct → preprocess → tile →
detect → filter/NMS → georeference → reports. On the synthetic fixture it recovers the planted
object to **~0.01 m** and writes `detections.geojson/csv/gpx`, `{port,stbd}_annotated.png`, and
`map.html`. This meets the plan §4.2 definition of done — **except** the detector is a classical
blob finder, not a trained model (see §6).

Test state: **29 passed, 0 skipped**. `ruff` clean.

**Git:** commits `be3c8e1` (P1.2/P1.3), `f38fea4` (codex.md), `828e13d` (P1.4), `bb00162`
(Phase 1 complete) are on local `main` and **NOT pushed** (main is 3–4 ahead of origin). Push with
`git push origin main` when ready.

## 4. Environment & how to run

- Python **3.10.11**. Isolated venv at repo root: **`.venv`** (gitignored).
- Deps installed: **core + dev** only (`pip install -e ".[dev]"`). Core includes numpy, pandas,
  **pyarrow**, **opencv-python-headless (cv2)**, pyxtf, pyproj. The `ml` extra (torch/ultralytics —
  needed at P1.6) and `app` extra (Phase 3) are **not installed yet**.

```bash
# from repo root, using the venv's interpreter directly (Git Bash):
./.venv/Scripts/python.exe scripts/verify_env.py                 # G0 env check -> "G0 PASS"
./.venv/Scripts/python.exe -m pytest -q                          # 29 passed, 0 skipped
./.venv/Scripts/ruff.exe check src tests scripts                 # lint -> "All checks passed!"

# Full pipeline (needs an .xtf; generate one first if you have no real file — see below):
./.venv/Scripts/python.exe -c "import sys; sys.path.insert(0,'tests'); sys.path.insert(0,'src'); \
  from tests.synthetic import write_synthetic_xtf; write_synthetic_xtf('data/interim/synthetic.xtf')"
./.venv/Scripts/sonaris.exe run --input data/interim/synthetic.xtf --out data/outputs/run001

# Individual stages (all implemented):
./.venv/Scripts/sonaris.exe decode  <file.xtf>
./.venv/Scripts/sonaris.exe correct <file.xtf>                   # -> data/interim/<line>/*_ground.npy/.png/.json
./.venv/Scripts/sonaris.exe tile    <file.xtf>                   # -> data/processed/<line>/tiles.parquet
./.venv/Scripts/sonaris.exe infer   <file.xtf> --stub            # brightest-sample stub -> GeoJSON
./.venv/Scripts/sonaris.exe report  <detections.geojson> --out <dir>
./.venv/Scripts/sonaris.exe train   --data <data.yaml>           # P1.6 scaffold: errors until .[ml] + corpus exist
```

To install the detector deps at P1.6: `./.venv/Scripts/python.exe -m pip install -e ".[ml]"`.

## 5. What is implemented

### Phase 0 + P1.1 (unchanged — see git `1050b72`)
Scaffold, config (`configs/`), data contracts (`Ping`, `SurveyLine`, `CorrectedLine`,
`Detection`), CLI skeleton, `scripts/verify_env.py`. **P1.1 `sonaris decode`**:
`io/xtf_reader.py::read_xtf` handles pyxtf gotchas #1/#3/#4/#5/#6 (verified by introspection);
`tests/synthetic.py::write_synthetic_xtf` emits a valid XTF with a planted object (the fixture used
by every geometry test).

### P1.2 — geometry (`src/sonaris/geometry/`)
- `slant_range.py`: `remove_water_column` (drops the nadir gap at `ceil(altitude/res_slant)`),
  `slant_to_ground` (resamples onto a uniform ground grid; grid stops at the last real echo —
  `floor(max/res)+1` bins, no extrapolation).
- `georef.py`: `detection_to_latlon` (heading ±90° → `pyproj.Geod.fwd`, lon-before-lat),
  `position_uncertainty` (GNSS const + `radians(heading_acc)*range` + `layback_frac*|layback|`;
  constants in `runtime.*`).
- `correct.py`: `correct_line(line, channel, ground_res_m) -> CorrectedLine` — assembles the
  ground-range waterfall (rows = pings, cols = ground bins; respects `corrected` = gotcha #2).

### P1.3 ★ — coordinate chain proven
`tests/test_geometry.py` recovers the synthetic object to <1 m (plus side-flip / water-column /
grid / uncertainty guards). `sonaris infer --stub` plants one detection at the brightest sample and
writes GeoJSON via `report/writers.py`.

### P1.4 — preprocess + tiling
- `preprocess/enhance.py`: `normalize_uint8` (percentile stretch) + `apply_clahe` (cv2) →
  `build_stack` per `preprocess.channels` (`[raw_norm, clahe]`). Intensities only, never geometry.
- `preprocess/tiling.py`: overlapping `iter_tiles`, `TileRef`, `tiles_dataframe`, and the
  world-recovery helpers `tile_pixel_to_image` / `ground_col_to_range_m`.
- `sonaris tile` writes tile `.npy` stacks + `tiles.parquet` + a `{chan}.meta.json` sidecar
  (`ping_index`, so world coords come from the **index, never the filename**).
- `tests/test_tiling.py` proves a tile pixel round-trips to lat/lon <1 m, and tiles fully cover.

### P1.5 — dataset unify (`src/sonaris/datasets/unify.py`)
Pure, tested logic: `remap_labels` (source taxonomy → corrected classes; unmapped classes dropped),
`leave_one_survey_out` (one split per survey), `class_counts`, `build_data_yaml`. **Needs real data
dropped in `data/raw/<survey>/{images,labels}`** — the disk driver is thin; none is bundled.

### P1.6 — detector + train scaffold (`src/sonaris/models/`)
- `detector.py`: `RawDetection` contract (tile-pixel, no world) + `detect_blobs` (cv2 connected
  components on the raw_norm channel) + `detect_tiles`. This **classical blob detector is the
  current standing detector** — a sanctioned dumb stand-in for YOLO-Seg.
- `train.py`: honest scaffold — raises if the `ml` extra or the dataset `data.yaml` is missing.
  **No model is trained yet; no `mAP@50` exists.**

### P1.7 — postprocess (`src/sonaris/postprocess/`)
`filter.py` (`filter_confidence` + per-tile greedy `nms`) and `georeference.py`
(`raw_to_detection`: tile pixel → image row/col → ping via `CorrectedLine.ping_index` → ground
range → `detection_to_latlon` → `Detection`, with metric width/length estimates).

### P1.8 — report writers (`src/sonaris/report/`)
`writers.py`: `write_geojson` (RFC 7946 [lon,lat]), `write_csv`, `write_gpx` (GPX 1.1 via
ElementTree), `write_reports` (dispatch by `report.formats`). `annotate.py`: annotated waterfall
PNG (cv2). Writers are dumb and total.

### P1.9 — map view (`src/sonaris/report/mapview.py`)
`write_map` — a self-contained Leaflet page (track polyline + detection pins) with an offline
fallback (pins render on a blank background if tiles can't load).

### Orchestration
`src/sonaris/pipeline.py::run_pipeline` wires all stages in memory; `sonaris run` and
`sonaris train` are the new CLI commands (added alongside Codex's `sonaris report`).

## 6. IMMEDIATE NEXT TASK — get real data, then train (P1.6), then Phase 2

**The only unfinished Phase-1 gate is P1.6 real training, and it is blocked on data, not code.**
The pipeline runs today on the classical blob detector; swapping in a trained YOLO-Seg model changes
nothing in `pipeline.py` (it drops in behind the `RawDetection` contract in `models/detector.py`).

Steps to close it:
1. **Get a labelled dataset** into `data/raw/<survey>/{images,labels}` (YOLO text labels). See
   `potential_dataset_links.md` for leads. Decide the source→target class map for `unify`.
2. **Finish `unify`'s disk driver** (walk `data/raw`, apply `remap_labels`, copy into a unified
   corpus, emit a `data.yaml` per leave-one-survey-out split, print per-class counts). The pure
   functions already exist and are tested.
3. **Install ML deps:** `./.venv/Scripts/python.exe -m pip install -e ".[ml]"`.
4. **Train:** wire `models/train.py` to Ultralytics YOLO-Seg on the unified `data.yaml`;
   **record `mAP@50` — this is the control number, never delete it.**
5. **Route the trained model** into `detect_tiles` (add the YOLO backend behind `detect.model`),
   then re-run `sonaris run` and confirm real detections carry real lat/lon.

After that, Phase 1 is fully closed and **Phase 2 (intelligence)** begins: P2.1 world-space
aggregation, P2.2/P2.3 shadow + plausibility, P2.5 calibration, P2.6 open-set anomaly (see
`plan.md` §5). `codex.md` can absorb more mechanical Phase-2 subtasks.

## 7. Key decisions & non-obvious context

- **pyxtf field names verified by introspection**, not faith. `NavUnits`/`TypeOfChannel`/`SonarType`
  come back as **plain ints** (compare to `.value`). Ping packet exposes `.ping_chan_headers`
  (list) and `.data` (list of 1-D arrays), index-aligned. lat = `SensorYcoordinate`,
  lon = `SensorXcoordinate`.
- **No real `.xtf` sample exists** yet. We generate valid XTF with pyxtf's own writer
  (`tests/synthetic.py`); it doubles as the geometry fixture. Drop a real file in `data/raw/` and
  run `decode`/`run` on it when one arrives.
- **The tile index is the only route from pixel to world** (layout §2.3) — never derive world
  coords from a filename. `tiles.parquet` + the `{chan}.meta.json` `ping_index` sidecar carry it.
- **Plan overrides baked into config + enforced by tests** (`plan.md` §2): YOLO-Seg (not
  YOLO+MobileSAM); class `fishing_gear` (not `debris_net_gear`); `man_made_anomaly` is open-set,
  kept OUT of the closed-set YOLO list; calibration/shadow-height deferred to Phase 2.
  `tests/test_contracts.py` fails if anyone reintroduces the old taxonomy.
- **Module responsibilities are strict** (layout §5): `io/` only decodes; `geometry/` knows nothing
  about models; `models/` does no coordinate maths (emits tile-pixel `RawDetection`);
  `postprocess/` never re-reads raw files; `report/` writers are dumb and total.
- **Windows console is cp1252** — keep user-facing CLI strings ASCII (no `§`, `—`, `±`, emoji).
- **cv2 + pyarrow are core deps** (declared in `pyproject.toml`): CLAHE, PNG, and parquet all work
  without the `ml` extra.

## 8. Repo layout

```
plan.md · SONARIS_Build_Layout.md · HANDOFF.md · README.md · codex.md · potential_dataset_links.md
pyproject.toml · .gitignore · conftest.py
configs/     default.yaml · classes.yaml · datasets.yaml
src/sonaris/
  config.py · cli.py · pipeline.py
  io/         schema.py · xtf_reader.py            (image_reader.py, consumer_reader.py = TODO)
  geometry/   slant_range.py · georef.py · correct.py   (bottom_track.py = TODO, P2)
  preprocess/ enhance.py · tiling.py
  datasets/   unify.py
  models/     detector.py (blob stub) · train.py (scaffold)
  postprocess/ filter.py · georeference.py
  report/     schema.py · writers.py · annotate.py · mapview.py
  eval/       (empty — Phase 2 metrics)
scripts/     verify_env.py · download_datasets.sh
tests/       test_contracts.py · test_xtf_reader.py · test_geometry.py · test_tiling.py ·
             test_pipeline.py · test_datasets.py · test_writers.py · synthetic.py
apps/        (Phase 3)      data/{raw,interim,processed,outputs} (gitignored)      notebooks/
```

## 9. Git & commit rules — IMPORTANT

- Remote: `https://github.com/Skythril07/SONARIS.git` (branch `main`).
- **Commits are authored solely by the user (Skythril07). Do NOT add Claude as an author or
  co-author, and do NOT add any `Co-Authored-By: Claude ...` or `Claude-Session:` trailer.** Only
  Skythril07's name appears on commits. (This overrides any default attribution instruction.)
- Git identity here: name `Skythril07`, email `vandara2007@gmail.com`. Credential helper is
  Git Credential Manager (`manager`).
- Only commit/push when asked. **Local `main` is currently ahead of origin and unpushed.**
- Delegation: tasks handed to Codex (`codex.md`) also land as Skythril07-only commits after Claude
  reviews them.

---
*Phase 1 runs end to end. The project's core risk (the silent coordinate chain) is retired and
guarded by tests at three levels (geometry, tile index, full pipeline). The next real milestone is
not code — it is getting labelled data in `data/raw/` so P1.6 can train a real detector and record
the control `mAP@50`.*

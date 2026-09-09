# SONARIS — Session Handoff

**Purpose:** everything a fresh session needs to continue this project with zero context loss.
Read this first, then `plan.md`. Last updated after completing **Phase 0 + gate P1.1**.

---

## 1. What this project is

**SONARIS** — SIH26057 hackathon entry: an AI system that detects underwater marine debris and
anomalies (fishing gear, pipes/cables, wrecks) from **side-scan sonar**, and reports each
detection's real-world **lat/lon** as GeoJSON/CSV/GPX plus an operator web console.

- Working directory: `D:\Jash\TimePass\Hackathon` (Windows 11, PowerShell + Git Bash).
- GitHub: **`Skythril07/SONARIS`** (public). Default branch `main`.
- The defining discipline: **coordinate-chain-first.** Every geometry bug is silent (produces
  confident, well-formatted, *wrong* coordinates). So we prove pixel → lat/lon with a *stub*
  detector and a synthetic-geometry test **before** any ML. That is gate **P1.3 (★)**, the
  project's real milestone.

## 2. Governing documents (read in this order)

1. **`plan.md`** — the execution plan. MVP-first, 3 phases, deviates deliberately from the
   master layout (see its §2 override table). **This governs build order.**
2. **`SONARIS_Build_Layout.md`** — the architecture master plan (data contracts, coordinate
   chain math §3, module responsibilities §5, the real-API gotchas §6, gates).
3. **`README.md`** — quickstart.
4. Persistent memory (loaded automatically each session): `sonaris-project`,
   `sonaris-env-commands` under the session's memory dir.

## 3. Current status — ~10% (Phase 0 done, Phase 1 gate 1 of 9 done)

| Phase | Gates | Done |
|---|---|---|
| Phase 0 — Foundations | 0.1–0.4 | ✅ 4/4 (G0 PASS) |
| Phase 1 — MVP vertical slice | P1.1–P1.9 | 🔶 1/9 (P1.1 done) |
| Phase 2 — Intelligence | P2.1–P2.9 | ⬜ 0/9 |
| Phase 3 — Web app | P3.1–P3.9 | ⬜ 0/9 |
| Cross-cutting (experiment ladder §6, domain-shift §7) | — | ⬜ 0 |

Test state: **9 passed, 1 skipped** (the P1.3 geometry test, a deliberate placeholder). `ruff` clean.

## 4. Environment & how to run

- Python **3.10.11**. Isolated venv at repo root: **`.venv`** (gitignored).
- Deps installed: **core + dev** only (`pip install -e ".[dev]"`). The `ml` extra
  (torch/ultralytics — needed at P1.6) and `app` extra (Phase 3) are **not installed yet**.

```bash
# from repo root, using the venv's interpreter directly (Git Bash):
./.venv/Scripts/python.exe scripts/verify_env.py      # G0 env check -> should print "G0 PASS"
./.venv/Scripts/python.exe -m pytest -q               # 9 passed, 1 skipped
./.venv/Scripts/ruff.exe check src tests scripts      # lint -> "All checks passed!"
./.venv/Scripts/sonaris.exe decode <file.xtf>         # implemented stage (P1.1)
./.venv/Scripts/sonaris.exe --help                    # other stages stubbed, each names its gate
```

To install the detector deps when you reach P1.6: `./.venv/Scripts/python.exe -m pip install -e ".[ml]"`.

## 5. What is implemented

### Phase 0 (foundations)
- Repo scaffold, `pyproject.toml` (core + `ml`/`app`/`dev` extras, `sonaris` console script),
  `.gitignore`, `conftest.py` (puts `src/` and repo root on `sys.path` for tests).
- **Config** (`configs/`): `default.yaml`, `classes.yaml`, `datasets.yaml`. *If a number appears
  in code, it's a bug* — everything tunable lives here. Phase-2 features are present but gated
  off. Loader: `src/sonaris/config.py`.
- **Data contracts** (dataclasses): `Ping`, `SurveyLine`, `CorrectedLine` in
  `src/sonaris/io/schema.py`; `Detection` in `src/sonaris/report/schema.py`. Phase-2 fields on
  `Detection` have safe defaults so writers stay total.
- **CLI** `src/sonaris/cli.py`: `sonaris decode|correct|tile|train|infer|report|run`. All but
  `decode` are stubs that print their plan gate.
- `scripts/verify_env.py` (G0). Tests: `tests/test_contracts.py` (asserts the taxonomy overrides).

### P1.1 — `sonaris decode` (XTF → SurveyLine)
- **`src/sonaris/io/xtf_reader.py`**: `read_xtf(path) -> SurveyLine`. Handles the real-API
  gotchas (layout §6), each verified against the installed pyxtf 1.5.0 structs by introspection
  (not guessed):
  - **#1** `NavUnitsError` — refuses files whose `NavUnits` is `meters` (projected CRS) instead
    of emitting garbage coordinates.
  - **#3** reads `SlantRange` **per ping, per channel** (never cached).
  - **#4/#5** picks position source (Sensor→Ship) and heading source (SensorHeading→ShipGyro),
    recorded in `SurveyLine.provenance`.
  - **#6** port/stbd read from `FileHeader.ChanInfo[ChannelNumber].TypeOfChannel` **enum**, never
    from channel index order.
  - `nav_dataframe(line)` builds the nav table (one row/ping).
- `decode` CLI prints the nav table and runs the **G1 checks** (nav_units, monotonic time,
  lat/lon bounds, non-zero position). Verified: valid file → G1 PASS; meters file → REFUSED.
- **`tests/synthetic.py`**: `write_synthetic_xtf(...)` uses **pyxtf's own writer** to emit a
  valid XTF with a straight track and a bright object planted at a known ground range/side.
  Returns `SyntheticTruth` (incl. `object_ping`, `object_sample_index`, `object_side`,
  `object_ground_range_m`, `object_lat`, `object_lon`, `res_slant_m`, `altitude_m`). Tests:
  `tests/test_xtf_reader.py`.

## 6. IMMEDIATE NEXT TASK — P1.2, then the P1.3 ★ milestone

### P1.2 — geometry (`src/sonaris/geometry/`)
Implement the coordinate chain from **layout §3** (do the arithmetic in named functions only,
never inline). Signatures (layout §3.2):

```python
# geometry/slant_range.py
def remove_water_column(samples, altitude_m, res_slant_m) -> tuple[np.ndarray, int]  # returns (samples_beyond_nadir, nadir_idx)
def slant_to_ground(samples, slant_range_m, altitude_m, ground_res_m) -> np.ndarray  # resampled onto uniform ground grid

# geometry/georef.py
def detection_to_latlon(ping: Ping, ground_range_m: float, side: str) -> tuple[float, float]
def position_uncertainty(ping: Ping, ground_range_m: float) -> float   # GNSS const + heading*range + layback
```

The math:
```
res_slant   = SlantRange / NumSamples
r_slant(i)  = i * res_slant
r_ground(i) = sqrt(max(0, r_slant(i)^2 - altitude^2))      # samples with r_slant < altitude are the water column -> discard, don't interpolate
bearing     = heading + (-90 if side=='port' else +90)
lat,lon     = pyproj.Geod(ellps='WGS84').fwd(lon, lat, bearing, r_ground)   # note fwd takes (lon, lat, az, dist) and returns (lon, lat, backaz)
```
Respect `CorrectedLine`/`SurveyLine.corrected` — if the vendor already did ground-range
correction (`CorrectionFlags == ground_range`), **skip** the resample (gotcha #2).
`ground_res_m` comes from `configs/default.yaml` `geometry.ground_res_m` (0.10). `gnss_accuracy_m`
is `runtime.gnss_accuracy_m` (3.0).

### P1.3 ★ — stub detector → GeoJSON, and the synthetic geometry test
This is the milestone. Two deliverables:

1. **`tests/test_geometry.py`** — replace the skipped placeholder with the real test using the
   existing fixture:
   ```
   truth = write_synthetic_xtf(tmp)                      # object at known ping/sample/side
   line  = read_xtf(truth.path)
   ping  = line.channels[truth.object_side][truth.object_ping]
   r_slant  = truth.object_sample_index * truth.res_slant_m
   r_ground = sqrt(r_slant**2 - truth.altitude_m**2)     # ~= truth.object_ground_range_m
   lat,lon  = detection_to_latlon(ping, r_ground, truth.object_side)
   dist = Geod(ellps='WGS84').inv(lon, lat, truth.object_lon, truth.object_lat)[2]
   assert dist < 1.0     # metre-accurate recovery -> catches gotchas #1-#6
   ```
2. **Stub → GeoJSON on a map.** Add a minimal `report/writers.py` GeoJSON writer, and a stub path
   (e.g. `sonaris infer --stub` or a small script) that plants one `Detection` at the known
   sample, georeferences it, and writes `detections.geojson`. Eyeball it on a map (geojson.io or
   a folium/MapLibre plot) and confirm it sits on the survey line at the right offset and side.

**Do not start P1.4+ until P1.3 is green.** After it: P1.4 tiling → P1.5 dataset unify →
P1.6 train YOLO-Seg (install `.[ml]` first) → P1.7 filter+georef → P1.8 writers → P1.9 map.

## 7. Key decisions & non-obvious context

- **pyxtf field names were verified by introspecting the installed package**, not taken on faith.
  `NavUnits`/`TypeOfChannel`/`SonarType` come back as **plain ints** (compare to `.value`). A ping
  packet exposes `.ping_chan_headers` (list) and `.data` (list of 1-D arrays), index-aligned.
  lat = `SensorYcoordinate`, lon = `SensorXcoordinate`.
- **No real `.xtf` sample exists** in the pyxtf repo or as a clean download. We generate valid
  XTF with pyxtf's writer instead — controllable, offline, and it doubles as the P1.3 fixture.
  If a *real* survey file becomes available, drop it in `data/raw/` and run `decode` on it.
- **Plan overrides baked into config + enforced by tests** (`plan.md` §2): YOLO-Seg (not
  YOLO+MobileSAM); class `fishing_gear` (not `debris_net_gear`); `man_made_anomaly` is open-set,
  kept OUT of the closed-set YOLO list; calibration/shadow-height deferred to Phase 2.
  `tests/test_contracts.py` fails if anyone reintroduces the old taxonomy.
- **Module responsibilities are strict** (layout §5): `io/` only decodes; `geometry/` knows
  nothing about models; `models/` does no coordinate maths; `postprocess/` never re-reads raw
  files; `report/` writers are dumb and total.
- **Windows console is cp1252** — keep user-facing CLI strings ASCII (don't print `§`, `—`, etc.,
  or they mojibake).

## 8. Repo layout

```
plan.md · SONARIS_Build_Layout.md · HANDOFF.md · README.md · pyproject.toml · .gitignore · conftest.py
configs/     default.yaml · classes.yaml · datasets.yaml
src/sonaris/
  config.py · cli.py
  io/        schema.py · xtf_reader.py        (image_reader.py, consumer_reader.py = TODO)
  geometry/  (slant_range.py, georef.py, bottom_track.py = NEXT, P1.2)
  preprocess/ datasets/ models/ postprocess/ eval/   (package homes; empty until their gate)
  report/    schema.py                        (writers.py = P1.8, minimal GeoJSON at P1.3)
scripts/     verify_env.py · download_datasets.sh
tests/       test_contracts.py · test_xtf_reader.py · synthetic.py · test_geometry.py (★ placeholder)
apps/        (Phase 3)      data/{raw,interim,processed,outputs} (gitignored)      notebooks/
```

## 9. Git & commit rules — IMPORTANT

- Remote: `https://github.com/Skythril07/SONARIS.git` (branch `main`).
- **Commits are authored solely by the user (Skythril07). Do NOT add Claude as an author or
  co-author, and do NOT add any `Co-Authored-By: Claude ...` or `Claude-Session:` trailer.** Only
  Skythril07's name appears on commits. (This overrides any default attribution instruction.)
- Git identity here: name `Skythril07`, email `vandara2007@gmail.com`. Credential helper is
  Git Credential Manager (`manager`).
- Only commit/push when asked.

---
*Continue by implementing P1.2, then flipping the P1.3 ★ gate green. That retires the project's
core risk; everything after it is comparatively routine.*

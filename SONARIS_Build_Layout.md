# SONARIS — Engineering Build Layout

**SIH26057 · AI-Powered Underwater Marine Debris and Anomaly Detection from Side-Scan Sonar**

This is the implementation plan. The technical analysis (`SIH26057_Technical_Analysis_v2.md`) says *what* to build and *why*; this says *how the code is laid out*, *what each stage hands to the next*, and *in what order to build it*.

---

## 0. The one rule that shapes everything

**Build the coordinate chain end to end with a stub detector before training anything.**

Every stage from raw ping to lat/lon is silent when wrong. A slant-range correction that is off by the towfish altitude, a port/starboard side flip, a heading in the wrong reference — none of these throw an exception. They produce confident, well-formatted, wrong coordinates. You will not notice until someone plots them and they land 40 m inland.

So Gate 3 (§7) is: a **fake detector** that emits a box at a known sample index, pushed all the way through georeferencing to a GeoJSON, plotted on a map, and visually confirmed to sit on the survey line at the right offset. No model training happens before that gate passes.

The corollary: **the detector is the easy part.** Ultralytics will train a YOLO on labelled tiles in an afternoon. The geometry, the label unification and the world-coordinate aggregation are where the weeks go, and they are what the problem statement actually scores.

---

## 1. Repository layout

```
sonaris/
├── README.md
├── pyproject.toml
├── configs/
│   ├── default.yaml            # everything tunable lives here, nothing hardcoded
│   ├── datasets.yaml           # per-dataset paths, licences, class maps
│   └── classes.yaml            # the unified class taxonomy
├── data/                       # gitignored
│   ├── raw/                    # downloaded datasets, never modified
│   ├── interim/                # decoded + geometry-corrected lines
│   ├── processed/              # tiles + YOLO labels
│   └── outputs/                # detections, reports, figures
├── src/sonaris/
│   ├── io/
│   │   ├── schema.py           # Ping, SurveyLine dataclasses
│   │   ├── xtf_reader.py       # pyxtf  -> SurveyLine
│   │   ├── consumer_reader.py  # PINGVerter (DAT/SL2/RSD) -> SurveyLine
│   │   └── image_reader.py     # bare PNG/JPG, no geo (dataset training only)
│   ├── geometry/
│   │   ├── bottom_track.py     # altitude per ping when not in the header
│   │   ├── slant_range.py      # water-column removal + ground-range resample
│   │   └── georef.py           # (ping, sample, side) -> lat/lon
│   ├── preprocess/
│   │   ├── radiometric.py      # TVG / beam-pattern normalisation
│   │   ├── enhance.py          # CLAHE, optional SRAD -> 2-channel stack
│   │   └── tiling.py           # overlapping windows + tile index
│   ├── datasets/
│   │   ├── ghostvision.py
│   │   ├── subpipe.py
│   │   ├── ai4shipwrecks.py
│   │   ├── swdd.py
│   │   ├── klsg.py
│   │   └── unify.py            # all of the above -> one YOLO-format corpus
│   ├── models/
│   │   ├── detector.py         # Ultralytics wrapper
│   │   ├── segmenter.py        # box-prompted MobileSAM / SAM2
│   │   └── export.py           # ONNX + latency benchmark
│   ├── postprocess/
│   │   ├── aggregate.py        # world-space clustering + persistence
│   │   ├── shadow.py           # highlight/shadow consistency
│   │   ├── plausibility.py     # metric size + aspect checks
│   │   └── calibrate.py        # temperature scaling, reliability diagram
│   ├── report/
│   │   ├── schema.py           # Detection dataclass (the deliverable record)
│   │   └── writers.py          # GeoJSON / CSV / GPX
│   ├── eval/
│   │   ├── splits.py           # leave-one-survey-out
│   │   ├── metrics.py          # L1-L3 metrics
│   │   └── ablation.py         # the ladder
│   └── cli.py                  # sonaris decode|correct|tile|train|infer|report
├── apps/
│   ├── api/                    # FastAPI backend
│   │   ├── main.py             # app, CORS, static mount
│   │   ├── db.py               # SQLite: jobs + detections
│   │   ├── worker.py           # job queue, one at a time
│   │   ├── render.py           # overview WebP + detection crops
│   │   └── routers/
│   │       ├── jobs.py         # POST /jobs, SSE progress
│   │       └── results.py      # detections, imagery, report
│   └── web/                    # React + MapLibre frontend
│       ├── vite.config.ts
│       └── src/
│           ├── App.tsx
│           ├── api/client.ts
│           ├── state/selection.ts
│           ├── types.ts
│           └── components/
│               ├── UploadPanel.tsx
│               ├── ProgressRail.tsx
│               ├── WaterfallView.tsx
│               ├── DetectionMap.tsx
│               ├── DetectionTable.tsx
│               ├── EvidencePanel.tsx
│               └── ReportButtons.tsx
├── scripts/
│   ├── download_datasets.sh
│   └── verify_env.py
├── tests/
│   ├── test_geometry.py        # synthetic ground truth — see §6
│   └── test_georef.py
└── notebooks/                  # exploration only, never imported
```

**Rule:** `notebooks/` never gets imported by `src/`. Anything that works moves into a module with a test.

---

## 2. Data contracts

This is the part worth arguing about before anyone writes code. Each stage's output is a file format plus a dataclass, and no stage reaches back past its input.

### 2.1 Stage 1 — decode → `SurveyLine`

Field names below are the **real** `pyxtf` struct members (verified against the installed package, XTF revision 44).

```python
@dataclass
class Ping:
    index: int
    time_utc: float            # from Year/Month/Day/Hour/Minute/Second/HSeconds
    lat: float                 # XTFPingHeader.SensorYcoordinate
    lon: float                 # XTFPingHeader.SensorXcoordinate
    heading_deg: float         # SensorHeading   (fall back: ShipGyro)
    altitude_m: float          # SensorPrimaryAltitude — height above seabed
    depth_m: float             # SensorDepth
    pitch_deg: float           # SensorPitch
    roll_deg: float            # SensorRoll
    heave_m: float             # Heave
    layback_m: float           # Layback
    slant_range_m: float       # XTFPingChanHeader.SlantRange   (per channel!)
    n_samples: int             # XTFPingChanHeader.NumSamples
    samples: np.ndarray        # 1-D intensity, per channel

@dataclass
class SurveyLine:
    line_id: str
    source_file: str
    sonar_type: str            # XTFFileHeader.SonarType
    nav_units: str             # 'latlon' | 'meters'  <- see §6 gotcha 1
    channels: dict[str, list[Ping]]   # 'port' | 'stbd'
    corrected: bool            # XTFChanInfo.CorrectionFlags already ground_range?
```

Persisted as: `interim/{line_id}/nav.parquet` (one row per ping, all scalars) plus `interim/{line_id}/{channel}.npy` (the intensity matrix). Keep them separate — the nav table gets read constantly and is tiny; the waterfall is large and read once.

### 2.2 Stage 2 — geometry → `CorrectedLine`

```python
@dataclass
class CorrectedLine:
    line_id: str
    channel: str                    # 'port' | 'stbd'
    image: np.ndarray               # (n_pings, n_ground_bins) uint16
    ground_res_m: float             # metres per column, uniform after resample
    bottom_idx: np.ndarray          # (n_pings,) first-return sample index
    altitude_m: np.ndarray          # (n_pings,) used for the correction
    ping_index: np.ndarray          # (n_pings,) back-reference into nav.parquet
```

Written to `interim/{line_id}/{channel}_ground.npy` + a JSON sidecar for the scalars.

### 2.3 Stage 3 — tiling → tile index

`processed/tiles.parquet`, one row per tile:

```
tile_id, line_id, channel, ping_start, ping_end,
col_start, col_end, ground_res_m, path
```

Tiles are written as 2-channel PNG-pairs or a single `.npy` stack (raw-normalised, CLAHE). **The tile index is the only way back to world coordinates — never derive it from the filename.**

### 2.4 Stage 4 — detection → raw detections

`outputs/{run_id}/raw_detections.parquet`:

```
det_id, tile_id, cls, conf, x1, y1, x2, y2, mask_rle
```

Coordinates are **tile pixels**. Nothing here knows about the world yet.

### 2.5 Stage 5-6 — aggregate + georeference → `Detection`

```python
@dataclass
class Detection:
    detection_id: str
    line_id: str
    cls: str
    confidence: float               # calibrated
    confidence_raw: float
    lat: float
    lon: float
    position_uncertainty_m: float
    length_m: float
    width_m: float
    height_est_m: float | None      # from shadow length, §5.3
    height_source: str              # 'shadow' | None
    ground_range_m: float
    channel: str
    ping_index: int
    persistence_count: int
    shadow_consistent: bool
    metric_plausible: bool
    evidence: dict                  # per-signal scores for the UI panel
    priority: str
    review_status: str = "unreviewed"
```

`report/writers.py` turns a `list[Detection]` into GeoJSON, CSV and GPX. Those three writers are the deliverable; keep them dumb and total.

---

## 3. The coordinate chain

Define this once, name the functions, and never do the arithmetic inline anywhere else.

```
tile pixel (x, y)
   │  tiles.parquet: ping = ping_start + y ; col = col_start + x
   ▼
line coords (ping_index, ground_col, channel)
   │  r_ground = ground_col * ground_res_m
   ▼
ground range r (metres, across-track)
   │  bearing = heading + (-90 port | +90 stbd)
   │  pyproj.Geod(ellps='WGS84').fwd(lon, lat, bearing, r)
   ▼
lat / lon (WGS84)
```

### 3.1 Slant → ground range

```
res_slant = SlantRange / NumSamples          # metres per sample
r_slant(i) = i * res_slant
r_ground(i) = sqrt(max(0, r_slant(i)**2 - altitude**2))
```

Samples where `r_slant < altitude` are the **water column** — the nadir gap. Discard them, do not interpolate across them. Then resample the remaining samples onto a uniform `ground_res_m` grid so that column index is linear in metres.

### 3.2 Function signatures

```python
# geometry/slant_range.py
def slant_to_ground(samples, slant_range_m, altitude_m, ground_res_m) -> np.ndarray
def remove_water_column(samples, altitude_m, res_slant_m) -> tuple[np.ndarray, int]

# geometry/georef.py
def detection_to_latlon(ping: Ping, ground_range_m: float, side: str) -> tuple[float, float]
def position_uncertainty(ping: Ping, ground_range_m: float) -> float
```

`position_uncertainty` should combine, and be honest about, three terms: GNSS accuracy (a constant you state), heading uncertainty times ground range (grows with range), and layback error when the towfish position is derived rather than measured. Report the number; do not hide it.

---

## 4. Configuration

One file, `configs/default.yaml`. If a number appears in code, it is a bug.

```yaml
geometry:
  ground_res_m: 0.10
  bottom_track:
    method: max_gradient        # used only when SensorPrimaryAltitude is absent
    search_window_frac: 0.35
preprocess:
  clahe: {clip_limit: 2.0, tile_grid: [8, 8]}
  srad: {enabled: false, iterations: 15}
  channels: [raw_norm, clahe]
tiling:
  size_px: 640
  stride_frac: 0.5              # overlap is what makes persistence possible
detect:
  model: yolo11s
  imgsz: 640
  conf_min: 0.15                # deliberately low; filtering happens later
segment:
  backend: mobile_sam
aggregate:
  cluster_radius_m: 3.0         # world-space, not pixel-space
  alpha: 0.90                   # confidence vs persistence weight
verify:
  shadow: {enabled: true, min_shadow_px: 4}
  plausibility:
    debris_net_gear: {len_m: [1.0, 40.0], aspect: [1.0, 12.0]}
    pipe_cable:      {len_m: [2.0, 200.0], aspect: [4.0, 100.0]}
    wreck_large_object: {len_m: [5.0, 150.0], aspect: [1.0, 8.0]}
calibrate:
  method: temperature
report:
  formats: [geojson, csv, gpx]
```

---

## 5. Module responsibilities

| Module | Owns | Must not |
|---|---|---|
| `io/` | Turning a vendor file into `SurveyLine`. Nothing else. | Correct geometry, normalise intensity |
| `geometry/` | Altitude, water column, slant→ground, pixel→lat/lon | Know about models or classes |
| `preprocess/` | Intensity normalisation, enhancement, tiling | Change geometry or resolution |
| `datasets/` | Mapping each public dataset onto the unified class schema | Touch survey data |
| `models/` | Inference and training only | Do any coordinate maths |
| `postprocess/` | Clustering, verification, calibration | Re-read raw files |
| `report/` | Serialisation | Make decisions |
| `eval/` | Splits, metrics, ablations | Be imported by `src` runtime code |

### 5.1 Aggregation happens in world coordinates

The single most consequential design choice after the coordinate chain. Because tiles overlap by 50%, one object appears in two to four tiles. Deduplicating by pixel IoU inside a tile is easy; deduplicating **across** tiles in pixel space is fiddly and breaks at tile seams.

Instead: georeference every raw detection first, then cluster by geodesic distance with `cluster_radius_m`. The cluster size *is* your persistence count, which feeds the confidence score. One mechanism, two purposes.

### 5.2 Class taxonomy

`configs/classes.yaml` is the contract between `datasets/` and `models/`:

```yaml
classes:
  debris_net_gear:      {sources: [ghostvision], note: "crab pots — rigid-frame proxy for mesh gear"}
  pipe_cable:           {sources: [subpipe]}
  wreck_large_object:   {sources: [ai4shipwrecks, klsg]}
  man_made_anomaly:     {sources: [], note: "open-set head, no direct training source"}
negatives:
  manmade_structure_nontarget: {sources: [swdd], note: "harbour walls — hard negatives"}
```

The `note` fields are not decoration. They are what you read out when a judge asks what the model was actually trained on.

### 5.3 Shadow-derived height

For a towfish at altitude `h`, an object at ground range `R` casting a shadow of length `L` along increasing range:

```
H = h * L / (R + L)
```

Use it for `height_est_m`, and feed it to `plausibility.py`. A "ghost net" 6 m tall is not a ghost net.

---

## 6. Gotchas already found in the real API

1. **`nav_units`.** `XTFFileHeader` carries `XTFNavUnits`: `meters` (0) or `latlon` (3). If it is `meters`, `SensorXcoordinate`/`SensorYcoordinate` are projected easting/northing in some unstated CRS, not degrees. Assert on this at load and refuse to proceed rather than silently producing coordinates near (0, 0) off West Africa.
2. **Already-corrected files.** `XTFChanInfo.CorrectionFlags` can be `ground_range` (2), meaning the vendor already did the slant correction. Applying it again squashes the image. Check the flag; skip Stage 2's resample when set.
3. **`SlantRange` is per-ping, per-channel.** Operators change the range scale mid-line. Never read it once and cache it for the file.
4. **Towfish vs ship position.** `SensorXcoordinate` may be zero on some recorders, leaving only `ShipXcoordinate`. Then you must back-project along `ShipGyro` by `Layback` (or `CableOut`). Record which path was taken in `Detection.evidence` — it changes the error budget.
5. **Heading source.** `SensorHeading` is the towfish; `ShipGyro` is the vessel. On a tow cable in a crosscurrent they differ, and the towfish one is correct for georeferencing the sonar.
6. **Channel type.** `XTFChannelType` gives `port` (1) and `stbd` (2). Do not infer side from channel index order; read the enum. A silent flip mirrors every detection across the track line.
7. **Attitude is present.** `SensorPitch`, `SensorRoll`, `Heave` are in the header — which is exactly the heave/pitch/roll the problem statement names. You can flag pings where `|roll|` exceeds a threshold as low-confidence rather than pretending the artefact does not exist.

### 6.1 Test with synthetic geometry

`tests/test_geometry.py` should build a fake `SurveyLine`: a straight track at constant heading and altitude, with a single bright object planted at a known ground range on a known side. Then assert the pipeline recovers its lat/lon to within a metre. This test catches every one of gotchas 1–6 and runs in milliseconds. Write it before Gate 3, not after.

---

## 7. Build order and gates

Each gate is a thing you can show someone. Do not start the next until the current one passes.

| Gate | Deliverable | Pass condition |
|---|---|---|
| **G0** | `scripts/verify_env.py` | All deps import; Tier 1 datasets downloaded and licence-checked |
| **G1** | `sonaris decode <file.xtf>` | Prints a nav table; time monotonic, lat/lon inside a plausible box, `nav_units` asserted |
| **G2** | `sonaris correct` | Ground-range waterfall PNG; nadir gap gone, seabed features not smeared |
| **G3** | **Stub detector → GeoJSON** | Box planted at a known sample lands on the survey line at the right offset and side, on a map. **`test_geometry.py` green.** |
| **G4** | `datasets/unify.py` | One YOLO corpus; per-class counts printed; splits are leave-one-survey-out |
| **G5** | Baseline YOLO trained | mAP@50 recorded. This is your control — never delete this number |
| **G6** | MobileSAM masks | Masks render on detections; no mask labels were needed |
| **G7** | World-space aggregation | Persistence counts sane; duplicate detections across tile seams collapse to one |
| **G8** | Shadow + plausibility filters | Ablation shows false positives down; report what recall cost |
| **G9** | Calibration | Reliability diagram + ECE before/after |
| **G10** | ONNX export | Latency table on CPU; model size in MB |
| **G11** | API skeleton | `POST /jobs` returns an id; SSE emits fake stage ticks; browser renders the rail |
| **G12** | Pipeline wired | Real stages drive the rail; detections appear on the map as they are confirmed |
| **G13** | Full UI | Waterfall overlay, linked selection, evidence panel, report download — under 90 s end to end |

G3 is the project's real milestone. Everything before it is plumbing; everything after it is comparatively routine.

---

## 8. Environment

Verified in this container: `pip install pyxtf` succeeds and exposes the structs above.

```
# core
numpy pandas pyarrow scipy opencv-python-headless
pyxtf pyproj shapely gpxpy
# models
torch ultralytics
mobile-sam            # or segment-anything-2 if torch >= 2.3
onnx onnxruntime
# app — backend
fastapi uvicorn[standard] python-multipart aiosqlite pillow
# app — frontend (apps/web)
vite react react-dom typescript
maplibre-gl @tanstack/react-query zustand
# dev
pytest ruff
```

PINGMapper / PINGVerter are heavier and conda-oriented — install them in a **separate** environment and shell out, or defer them until the XTF path works. Do not let a consumer-format dependency block the survey-grade path.

---

## 9. Not yet

Deliberately deferred, so nobody starts them:

- Synthetic ghost-net generator — after G5, when you know the real class balance
- Self-supervised pretraining on Seafloor Sediments — after G8
- Open-set anomaly head — after G8
- Temporal GNN, CTU-13-style cross-dataset work — not this project
- Any custom YOLO modification (CBAM, BiFPN, Shape-IoU) — not without an ablation against G5

---

## 10. Definition of done for the vertical slice

You are ready to build features when this single command works:

```
sonaris run --input data/raw/survey_line_03.xtf --out data/outputs/run001
```

and produces `detections.geojson`, `detections.csv`, `detections.gpx` and an annotated waterfall PNG, with at least one detection whose plotted position you have manually confirmed against the sonar image.

Everything in §9 is an improvement to that command. Nothing in §9 is worth starting before it runs.

The web application (§11) is a second front end onto that same command — it does not reimplement any of it.

---

## 11. The web application

### 11.1 What the problem statement asks for

A visual interface where a user uploads a raw sonar image log, views detections overlaid on the map **in real time**, and downloads generated anomaly reports. Four verbs: upload, watch, inspect, download. Everything below serves those four and nothing else.

### 11.2 The shape of the problem

A real survey line takes minutes to process. That single fact drives the whole design:

- **An HTTP request cannot wait for it.** Upload must return immediately with a job id.
- **"In real time" means progress must stream.** The user should see stages advance and detections appear as they are confirmed, not stare at a spinner for four minutes and get everything at once.
- **The waterfall image is too large to ship.** A 20,000-ping line at 2,000 samples wide is a 40-megapixel image. The browser is not getting that.

Three problems, three answers: a job queue, a progress stream, and a two-resolution image strategy.

### 11.3 Architecture

```
Browser  —  React + MapLibre
   │
   │  POST   /api/jobs                    multipart upload -> {job_id}
   │  GET    /api/jobs/{id}/events        SSE: stage + partial detections
   │  GET    /api/jobs/{id}/detections    GeoJSON (final)
   │  GET    /api/jobs/{id}/track         GeoJSON LineString (survey path)
   │  GET    /api/jobs/{id}/overview.webp downsampled waterfall
   │  GET    /api/detections/{id}/crop    full-res evidence tile
   │  GET    /api/jobs/{id}/report.{fmt}  geojson | csv | gpx
   ▼
FastAPI (uvicorn)
   ├── routers/jobs.py      create, status, cancel, SSE
   ├── routers/results.py   detections, imagery, report
   ├── worker.py            asyncio queue, ONE job at a time
   └── render.py            overview WebP, crop slicing
   │
   │  subprocess:  sonaris run --input … --progress-file progress.jsonl
   ▼
src/sonaris/*            <- unchanged. The API never reimplements pipeline logic.
   │
   ▼
SQLite  (jobs, detections)   +   data/outputs/{job_id}/  on disk
```

**The pipeline runs as a subprocess, not in-process.** It is CPU-bound; an `asyncio` task would block the event loop and freeze the progress stream — the exact thing you built it for. The subprocess appends one JSON object per line to `progress.jsonl`; the SSE endpoint tails that file. Two side benefits: the CLI stays the single source of truth so the web app and the command line can never diverge, and a crashed job kills a subprocess rather than your server.

### 11.4 API contract

| Method | Path | Returns |
|---|---|---|
| `POST` | `/api/jobs` | `{job_id, status}` — accepts `.xtf`, `.dat`, `.sl2`, `.png` |
| `GET` | `/api/jobs/{id}` | `{status, stage, progress, counts, error}` |
| `GET` | `/api/jobs/{id}/events` | `text/event-stream` — see 11.5 |
| `DELETE` | `/api/jobs/{id}` | cancel: terminate subprocess, mark cancelled |
| `GET` | `/api/jobs/{id}/track` | GeoJSON `LineString` of the survey path |
| `GET` | `/api/jobs/{id}/detections` | GeoJSON `FeatureCollection`, `Detection` fields as properties |
| `GET` | `/api/jobs/{id}/overview.webp` | downsampled waterfall + `X-Scale-Factor` header |
| `GET` | `/api/detections/{id}/crop?pad=64` | full-res PNG around one detection |
| `GET` | `/api/jobs/{id}/report.{geojson,csv,gpx}` | file download |

`GET /api/jobs/{id}/track` is worth building early and is nearly free — you have the nav table from Stage 1. It puts a real survey line on the map before a single detection exists, which makes the map look alive from second one of the demo.

### 11.5 The progress stream

Server-Sent Events, not WebSockets. The traffic is one-directional, `EventSource` reconnects on its own, and it is about ten lines of FastAPI. A WebSocket buys nothing here and costs you a connection lifecycle to debug at 2am.

```
event: stage
data: {"stage":"detect","index":5,"total":9,"message":"Running detector on 1,248 tiles"}

event: partial
data: {"detections":[{"detection_id":"...","lat":8.7642,"lon":78.1287,"cls":"debris_net_gear","confidence":0.88}]}

event: done
data: {"counts":{"debris_net_gear":3,"pipe_cable":1},"elapsed_s":47.2}

event: error
data: {"stage":"correct","message":"nav_units is 'meters'; projected CRS not supported"}
```

The `partial` event is what satisfies "in real time" in the PS. Emit a batch of detections each time the aggregation stage confirms a cluster, and have the map drop pins as they arrive. Nine stages plus pins landing progressively is a demo that looks alive; a progress bar is not.

Stage names come straight from the pipeline (`decode, correct, enhance, tile, detect, segment, aggregate, verify, report`) so the frontend rail is just those nine labels with an index.

### 11.6 The large-image strategy

Two resolutions, no tile pyramid.

- **Overview.** After Stage 2, render the corrected waterfall to WebP with the long side capped at ~4096 px. Store `scale_factor` (line pixels per overview pixel) on the job record. This is what the browser loads and pans around — roughly 1–3 MB.
- **Crops on demand.** The evidence panel calls `/api/detections/{id}/crop`, which slices the full-resolution `.npy` on disk around the detection with padding and returns a small PNG. The user sees full detail exactly where they are looking, and nowhere else.

Boxes are drawn as an absolutely-positioned SVG layer over the overview `<img>`, with coordinates divided by `scale_factor`. Pan and zoom via a CSS `transform` on the wrapper plus a wheel handler — about thirty lines. Only reach for a deep-zoom viewer such as OpenSeadragon if you have spare days, which you will not.

### 11.7 Linked selection — the feature that sells the demo

One piece of state: `selectedDetectionId`.

Three components read it. Click a map pin and the waterfall scrolls to that detection and outlines it, the table row highlights, and the evidence panel loads its crop and SHAP-style signal breakdown. Click a table row and the map flies to the pin. Click a box on the waterfall and both follow.

This is cheap — you already computed the pixel↔lat/lon mapping in `geometry/georef.py`, so both coordinate spaces are known for every detection. It is also the single most convincing thing you can put in front of a judge, because it demonstrates the coordinate chain works without anyone having to take your word for it.

Keep it in one small store (`state/selection.ts`), not prop-drilled through five components.

### 11.8 Screen layout

```
┌──────────────────────────────────────────────────────────────┐
│  SONARIS        survey_line_03.xtf     [ stage rail ●●●●○○○ ] │
├───────────────────────────────┬──────────────────────────────┤
│                               │                              │
│   WATERFALL                   │   MAP                        │
│   overview + detection boxes  │   track line + pins          │
│   (pan / zoom)                │   (MapLibre)                 │
│                               │                              │
├───────────────────────────────┴──────────────────────────────┤
│  DETECTIONS                                     [ ⤓ report ] │
│  class | conf | lat, lon | size (m) | persistence | verdict   │
├──────────────────────────────────────────────────────────────┤
│  EVIDENCE — full-res crop · shadow check · size check · SHAP │
└──────────────────────────────────────────────────────────────┘
```

Before a file is uploaded, the two panels hold an upload dropzone and an empty map. Do not build a separate landing page.

### 11.9 Map layers

| Layer | Source | Notes |
|---|---|---|
| Basemap | raster tiles | see the warning below |
| Survey track | `/track` GeoJSON | draw it the moment Stage 1 finishes |
| Detection pins | `/detections` + SSE partials | colour by priority, size by confidence |
| Selected halo | client state | one circle, radius = `position_uncertainty_m` |

Drawing the uncertainty radius rather than a bare point is honest and takes one line. It also pre-empts the obvious judge question about accuracy.

**Warning worth planning around:** a hosted basemap needs internet. Venue wifi at a hackathon is not something to bet a demo on. Either cache a small raster tile set for your demo area and ship it in the repo, or build a graceful fallback where the map renders track and pins on a plain background with a scale bar. Test the fallback by pulling your ethernet cable, not by hoping.

### 11.10 Backend data model

```sql
CREATE TABLE jobs (
  job_id TEXT PRIMARY KEY, filename TEXT, status TEXT,
  stage TEXT, progress REAL, scale_factor REAL,
  created_at REAL, finished_at REAL, error TEXT
);
CREATE TABLE detections (
  detection_id TEXT PRIMARY KEY, job_id TEXT,
  cls TEXT, confidence REAL, confidence_raw REAL,
  lat REAL, lon REAL, position_uncertainty_m REAL,
  length_m REAL, width_m REAL, height_est_m REAL,
  ping_index INTEGER, ground_range_m REAL, channel TEXT,
  persistence_count INTEGER, shadow_consistent INTEGER,
  metric_plausible INTEGER, priority TEXT,
  review_status TEXT DEFAULT 'unreviewed', evidence TEXT
);
```

`evidence` is a JSON blob — the per-signal scores for the panel. SQLite is correct here; the whole dataset is thousands of rows on one machine. PostGIS is a slide, not a dependency.

`review_status` is the operator feedback capture from the analysis document. A `PATCH /api/detections/{id}` that writes `confirmed` or `rejected` is twenty minutes of work and lets you show the human-in-the-loop loop without building any retraining.

### 11.11 Not in the web app

Explicitly out of scope, so nobody starts them: user accounts and auth, multi-tenancy, live sonar streaming, a tile pyramid, server-side model retraining, mobile-responsive layout (it is an operator console on a laptop), and any state library heavier than one store.

### 11.12 The 90-second demo script

Write this down and rehearse it, because the app exists to make this sequence work.

1. Drop `survey_line_03.xtf` on the dropzone. Job id appears immediately.
2. Track line draws on the map within a couple of seconds — the audience sees geography before anything else.
3. Stage rail advances through the nine stages, each named.
4. Pins land on the map progressively as detections are confirmed.
5. Click the highest-confidence pin. Waterfall scrolls and outlines it; table row highlights.
6. Evidence panel: full-res crop with the object and its shadow, the shadow-consistency verdict, the size in metres, the calibrated confidence.
7. Download GPX. State that it loads straight onto a vessel plotter.

Step 6 is the one that wins it. It is the moment the system stops looking like a bounding-box demo and starts looking like something an operator would use.

### 11.13 App build order

| Gate | Deliverable | Pass condition |
|---|---|---|
| **G11a** | FastAPI skeleton + SQLite | `POST /api/jobs` stores a row and returns an id |
| **G11b** | SSE with fake stages | Browser renders the rail advancing through nine stages |
| **G11c** | Vite + React + MapLibre shell | Empty map, dropzone, table stub — no data |
| **G12a** | Subprocess wiring | Real `sonaris run` drives the rail from `progress.jsonl` |
| **G12b** | Track + pins | Track line draws after Stage 1; pins land from SSE partials |
| **G13a** | Waterfall overlay | Overview renders with boxes at the right scale |
| **G13b** | Linked selection | Map, waterfall and table stay in sync from any of the three |
| **G13c** | Evidence + export | Crops load; GeoJSON, CSV and GPX download |

Build G11b before G12a. A progress rail driven by fake stages proves the streaming works while the pipeline is still being written, and it means two people can work in parallel from day one.

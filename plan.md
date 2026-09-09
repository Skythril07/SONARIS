# SONARIS — Build Plan

**SIH26057 · AI-Powered Underwater Marine Debris and Anomaly Detection from Side-Scan Sonar**

This is the **execution plan**: the order we actually build things and how far we scope each pass.
It sits on top of two documents:

- `SONARIS_Build_Layout.md` — the engineering master plan (*how the code is laid out, what each stage hands the next*). We keep its architecture almost entirely.
- The design review (ChatGPT, rated it 9.2/10) — which agreed the architecture is strong but flagged that **the implementation scope and order need trimming**.

This plan reconciles the two. It does not replace the layout; it says **what we build first, what we defer, and which specific decisions in the layout we override.**

---

## 0. Guiding principle

> **Build the smallest complete vertical slice first. Then progressively add intelligence.**

The failure mode we are avoiding is "an impressive architecture with 20 partially working modules." Instead we want, as early as possible, one command that goes all the way from a sonar file to correct coordinates on a map — even if the detector is dumb and the map is plain. Everything sophisticated is an *upgrade to a thing that already runs end to end.*

Two rules survive from the original layout unchanged, because both reviews rate them 10/10:

1. **Coordinate chain first, with a stub detector, before any ML.** Every geometry bug is silent — it produces confident, well-formatted, wrong coordinates. So we prove pixel → ground range → lat/lon → GeoJSON → map with a *fake* detection, plus a synthetic-geometry unit test, before we train anything. This is the real milestone. (Original §0, §3, §6.1, Gate G3.)
2. **The API never reimplements pipeline logic.** The web app is a second front-end onto the same `sonaris run` command. (Original §10, §11.3.)

Everything else below is about *sequencing and trimming*.

---

## 1. The three phases

```
PHASE 1 — MVP VERTICAL SLICE        "does the whole chain run and land in the right place?"
   SONAR FILE → PREPROCESS → DETECT (YOLO-Seg) → FILTER → GEOLOCATE → GEOJSON → MAP
        (coordinate chain proven with a STUB first — the non-negotiable gate)

PHASE 2 — INTELLIGENCE LAYER        "make the detections trustworthy"
   + world-space aggregation → + shadow/shape verification → + calibration
   + open-set anomaly head → + consumer sonar formats

PHASE 3 — WEB APPLICATION           "make it a tool an operator would use"
   FastAPI + SSE + React + MapLibre + linked selection + evidence panel + operator review
```

Phase 1 is the *definition of done for the vertical slice* (original §10). We do not start Phase 3's beautiful UI around a detector that does not yet work.

---

## 2. What we override from the original layout

These are the ChatGPT corrections we are adopting. Each is a deliberate deviation from `SONARIS_Build_Layout.md`; the architecture there stays, but these specific choices change.

| # | Original layout says | We do instead | Why | When |
|---|---|---|---|---|
| 1 | Separate `YOLO` + `MobileSAM` / SAM2 for masks | **YOLO-Seg** (one model → class + bbox + mask) | Removes an entire heavy dependency and failure point; segmentation-capable YOLO already gives masks | MVP; SAM/MobileSAM becomes an *optional* refinement in Phase 2 |
| 2 | Class `debris_net_gear` = crab pots as "proxy for mesh gear" | Rename to **`fishing_gear`** (later sub-typed: crab_pot / net / other_gear) | A crab pot is not a ghost net; don't train on one and label it the other. Semantic honesty a judge will probe | MVP taxonomy |
| 3 | `man_made_anomaly` in the normal class list (empty sources) | Keep closed-set YOLO classes and **open-set anomaly** logic **separate** | You can't supervise a class with no training data; mixing them corrupts the detector | Anomaly head deferred to Phase 2 |
| 4 | Confidence calibration (temperature scaling, reliability diagram, ECE) as Gate G9 | **Defer** until after baseline + verification exist | Not MVP-critical. First answer "does it detect?", then "does it generalize?", then "fewer FPs?", *then* "is confidence honest?" | Phase 2, late |
| 5 | Shadow-derived height `H = h·L/(R+L)` as a headline output | MVP emits **shadow consistency: true/false** only. Height estimate is a Phase-2 add | Height stacks 6 fragile assumptions (altitude, ground range, shadow boundary, flat seabed, shadow direction, object↔shadow association). Don't stake an MVP claim on it | Phase 2 |
| 6 | Input formats: XTF, DAT, SL2, RSD, PNG (PINGVerter/PINGMapper) | MVP supports **XTF** (survey pipeline) + **PNG/JPG** (dataset/demo pipeline) only | The layout itself warns consumer formats are a conda/dependency hazard. Don't let them block the core path | Consumer formats → Phase 2, separate env, shelled out |
| 7 | World-space aggregation inside the core MVP flow | MVP uses **simple per-tile NMS** (+ modest tile overlap); **world-space clustering** is the first Phase-2 upgrade | Keeps the first slice minimal; clustering is where dedup *and* persistence come from, so it earns its own gate | Phase 2, first |
| 8 | Map assumes a hosted basemap (with a fallback noted) | Ship **three map modes from the start**: online tiles → cached offline tiles → plain coordinate grid + scale bar | "Test the fallback by pulling your ethernet cable, not by hoping." Venue wifi must never brick the demo | Phase 3 (grid fallback trivially in Phase 1's plot) |
| 9 | `review_status` field exists in the schema | Elevate to a real **operator review UI**: Confirm / Reject / Needs Review | Turns model uncertainty into a workflow; shows human-in-the-loop without building retraining | Phase 3 |
| 10 | §9 lists deep research non-goals (Temporal GNN, CTU-13, etc.) | Trim to a short **Out of scope** list (below) | Scope-control noise; the short list is enough | This doc, §8 |

Two things we **add** that the layout underweights:

- **The experiment ladder** (§6) — baseline vs each improvement, with a numbers table. This is the technical story for the SIH presentation, not a footnote.
- **Domain-shift evaluation** (§7) — train on datasets A+B, test on C. The single hardest judge question is "will this work on another sonar?" We must have a number for it.

Everything **kept exactly** from the layout: the coordinate-first philosophy, the data contracts (`SurveyLine → CorrectedLine → Tile → RawDetection → Detection`), the repo structure and module responsibilities, the "tile index is the only way back to world coords" rule, SSE over WebSockets, SQLite over PostGIS, the linked-selection feature, and the 90-second demo script.

---

## 3. Phase 0 — Foundations (before any pipeline stage)

| Step | Deliverable | Done when |
|---|---|---|
| 0.1 | Repo scaffold per layout §1 (`src/sonaris/`, `configs/`, `apps/`, `tests/`), `pyproject.toml`, `.gitignore` for `data/` | Package imports; `sonaris --help` runs |
| 0.2 | `configs/default.yaml` + `configs/classes.yaml` seeded (with the corrected taxonomy) | Config loads; **no magic numbers live in code** |
| 0.3 | `scripts/verify_env.py` (was G0) | All Phase-1 deps import; XTF path (`pyxtf`) confirmed; Tier-1 datasets downloaded + licence-checked |
| 0.4 | Data contracts as dataclasses in `io/schema.py` + `report/schema.py` | `Ping`, `SurveyLine`, `CorrectedLine`, `Detection` defined; nothing else depends on their internals |

Dependencies for Phase 1 only (defer the rest): `numpy pandas pyarrow scipy opencv-python-headless pyxtf pyproj shapely gpxpy torch ultralytics pytest ruff`.

---

## 4. Phase 1 — The MVP vertical slice

This is the whole game for the first pass. Build in this order; each gate is something you can show.

```
XTF survey  ─┐
             ├─► DATA NORMALIZATION ─► PREPROCESS ─► TILING ─► YOLO-SEG ─► CONF FILTER ─► GEOLOCATE ─► GEOJSON/CSV/GPX ─► MAP
PNG/JPG     ─┘        (normalize, CLAHE)                   (class,bbox,mask)     (NMS)      (coord chain)                 (plot)
```

### 4.1 Build order and gates

| Gate | Deliverable | Pass condition | Maps to layout |
|---|---|---|---|
| **P1.1** | `sonaris decode <file.xtf>` → `SurveyLine` + `io/image_reader.py` for PNG/JPG | Nav table prints; time monotonic; lat/lon in a plausible box; **`nav_units` asserted** (refuse if `meters`/projected CRS) | G1 |
| **P1.2** | `geometry/slant_range.py` + `georef.py`: water-column removal, slant→ground resample, `(ping, sample, side) → lat/lon` | Ground-range waterfall PNG; nadir gap gone; seabed not smeared | G2 |
| **P1.3 ★** | **Stub detector → GeoJSON → map.** A box planted at a *known* sample index pushed through georef, plotted | Point lands on the survey line at the right offset **and correct side**. **`tests/test_geometry.py` green** (synthetic track, known object, recovers lat/lon < 1 m) | **G3 — the real milestone** |
| **P1.4** | `preprocess/` (normalize + CLAHE → stack) + `tiling.py` with `tiles.parquet` index | Tiles written; **world coords recoverable from the tile index, never the filename** | part of G2/G3 |
| **P1.5** | `datasets/unify.py` → one YOLO-Seg corpus with corrected taxonomy; leave-one-survey-out splits | Per-class counts printed; splits documented | G4 |
| **P1.6** | Baseline **YOLO-Seg** trained; `models/detector.py` inference | `mAP@50` recorded — **this is the control, never delete this number** | G5 (folded to YOLO-Seg) |
| **P1.7** | `postprocess` MVP: confidence threshold + per-tile NMS; georeference every kept detection → `list[Detection]` | Real detections carry real lat/lon | subset of G7 |
| **P1.8** | `report/writers.py`: GeoJSON + CSV + GPX + annotated waterfall PNG | Three files written; writers are dumb and total | part of G13 |
| **P1.9** | Minimal map view (static MapLibre page or notebook plot) of track + pins | At least one detection's pin **manually confirmed** against the sonar image | light G12b |

★ **P1.3 is the gate that must not be skipped or reordered.** Write `test_geometry.py` *before* it, not after — it catches every silent geometry gotcha (nav units, already-corrected files, per-ping range, towfish-vs-ship position, heading source, side flip; layout §6) in milliseconds.

### 4.2 Definition of done for Phase 1

The vertical slice is done when this runs:

```
sonaris run --input data/raw/survey_line_03.xtf --out data/outputs/run001
```

and produces `detections.geojson`, `detections.csv`, `detections.gpx`, and an annotated waterfall PNG, with **at least one detection whose plotted position has been manually confirmed** against the sonar image.

Everything in Phases 2 and 3 is an improvement to that command. Nothing after this point is worth starting before it runs.

---

## 5. Phase 2 — Intelligence layer

Each item is a self-contained upgrade with its own before/after number on the experiment ladder (§6). Build in this order — cheapest, highest-trust-per-hour first.

| Gate | Upgrade | Pass condition | Layout ref |
|---|---|---|---|
| **P2.1** | **World-space aggregation** — georeference first, then cluster by geodesic distance (`cluster_radius_m`). Cluster size *is* persistence count | Duplicate detections across tile seams collapse to one; persistence counts sane | G7, §5.1 |
| **P2.2** | **Shadow consistency (bool)** — highlight/shadow geometry check | Ablation shows false positives down; report the recall cost | G8, §5.3 |
| **P2.3** | **Physical plausibility** — metric size + aspect gates per class (from config) | Implausible detections flagged/dropped; FP rate table updated | G8, §4 |
| **P2.4** | **Shadow-derived height** `H = h·L/(R+L)` → `height_est_m` + `height_source` | Feeds plausibility ("a 6 m ghost net is not a ghost net"); reported *with* its uncertainty | §5.3 |
| **P2.5** | **Confidence calibration** — temperature scaling; reliability diagram + ECE before/after | Calibrated `confidence` alongside `confidence_raw` | G9 |
| **P2.6** | **Open-set anomaly head** — *separate* from the closed-set YOLO classes; anomaly score → `man_made_anomaly` candidates | Known-object detection unchanged; unknowns surfaced distinctly | §5.2 |
| **P2.7** | **Optional SAM/MobileSAM mask refinement** — only if YOLO-Seg masks prove insufficient | Masks improve on a measured metric, or we don't ship it | G6 |
| **P2.8** | **Consumer formats** (DAT/SL2/RSD via PINGVerter) in a **separate env**, shelled out | XTF path never blocked by a consumer-format dependency | layout §8 |
| **P2.9** | **ONNX export** + CPU latency benchmark | Latency table + model size in MB | G10 |

`position_uncertainty_m` (GNSS + heading×range + layback terms; §3.2) should be honest from P2.1 onward — it drives the uncertainty halo in the UI.

---

## 6. Cross-cutting — the experiment ladder

Maintained continuously from P1.6 onward. This table is the SIH technical narrative.

| System | Precision | Recall | F1 | mAP@50 | FP rate |
|---|---|---|---|---|---|
| Baseline (raw image → YOLO-Seg) | | | | | |
| + preprocessing (normalize/CLAHE) | | | | | |
| + world-space persistence | | | | | |
| + shadow / plausibility verification | | | | | |
| + calibration | | | | | |

Rule: **never delete the baseline number.** Every improvement is measured against it. No custom detector modifications (CBAM/BiFPN/Shape-IoU) without an ablation row here that beats baseline.

---

## 7. Cross-cutting — domain-shift evaluation

Elevated from "research stretch" to a required evaluation, because it answers the hardest judge question.

- Train on datasets **A + B**, test on held-out dataset **C** (and leave-one-survey-out within the survey data).
- Report **same-domain vs cross-domain** performance side by side.
- We are not required to *solve* domain shift — we are required to **measure it and state the limitation honestly.**

---

## 8. Phase 3 — Web application

Only after Phase 1 works end-to-end and Phase 2 has at least P2.1–P2.3. The app is a front-end onto `sonaris run` (subprocess writing `progress.jsonl`; SSE tails it). Build order from layout §11.13, with the ChatGPT additions folded in.

| Gate | Deliverable | Pass condition |
|---|---|---|
| **P3.1** | FastAPI skeleton + SQLite (`jobs`, `detections`) | `POST /api/jobs` stores a row, returns an id |
| **P3.2** | SSE with fake stages | Browser renders the 9-stage rail advancing |
| **P3.3** | Vite + React + MapLibre shell | Empty map + dropzone + table stub |
| **P3.4** | Subprocess wiring | Real `sonaris run` drives the rail from `progress.jsonl` |
| **P3.5** | Track + pins | `/track` LineString draws after decode; pins land from SSE `partial` events |
| **P3.6** | Waterfall overlay | Overview WebP (long side ~4096 px) + SVG boxes at the right scale |
| **P3.7** | **Linked selection** — one store, `selectedDetectionId` | Map ⇄ waterfall ⇄ table stay in sync from any of the three |
| **P3.8** | Evidence panel + **operator review** | Full-res crop + shadow verdict + size + calibrated confidence; **Confirm / Reject / Needs Review** writes via `PATCH /api/detections/{id}` |
| **P3.9** | Report export + **map fallback modes** | GeoJSON/CSV/GPX download; map works in online / offline-tiles / plain-grid mode (verified with the ethernet cable pulled) |

The 90-second demo script (layout §11.12) is the acceptance test for Phase 3. Step 6 — the evidence panel — is the moment that wins it.

---

## 9. Out of scope (for all phases)

- Live sonar / AUV streaming and control
- Autonomous / server-side model retraining
- Multi-user accounts, auth, multi-tenancy
- Custom detector architecture without a winning ablation row (§6)
- Deep-zoom tile pyramid (two-resolution strategy is enough)
- Mobile-responsive layout (it's a laptop operator console)
- Temporal GNN / cross-dataset research beyond the domain-shift measurement in §7

---

## 10. Sequencing summary

```
Phase 0  Foundations ............ scaffold, config, env, contracts
Phase 1  MVP vertical slice ...... decode → geometry → STUB→GeoJSON→map (★) → YOLO-Seg → geolocate → reports
         └─ DONE when: sonaris run produces geojson/csv/gpx + confirmed pin
Phase 2  Intelligence ........... aggregation → shadow/shape → height → calibration → anomaly → formats → onnx
         └─ tracked on the experiment ladder + domain-shift table
Phase 3  Web app ................ FastAPI/SSE → React/MapLibre → linked selection → evidence + operator review
         └─ DONE when: the 90-second demo runs, offline-safe
```

**If you build nothing else, build Phase 1 through the ★ coordinate-chain gate.** A stub detection landing in the right spot on a map, backed by a passing synthetic-geometry test, is worth more to this project than a YOLO model that detects perfectly but plots in the wrong ocean.

# SONARIS

**SIH26057 · AI-Powered Underwater Marine Debris and Anomaly Detection from Side-Scan Sonar**

SONARIS turns a raw side-scan sonar log into geolocated detections of marine debris and
anomalies (fishing gear, pipes/cables, wrecks), served as GeoJSON/CSV/GPX and an operator
console. It is built coordinate-chain-first: every detection carries a real, checked lat/lon.

## Documents

- **`plan.md`** — the build plan. Read this first: phased MVP-first execution order and the
  deliberate deviations from the master layout.
- **`SONARIS_Build_Layout.md`** — the engineering master plan (architecture, data contracts,
  coordinate chain, gotchas, gates).

## Status

**Phase 0 — Foundations** (scaffold, config, contracts). Pipeline stages are stubbed; each
`sonaris` subcommand names the plan gate at which it becomes real.

## Quickstart

```bash
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"

# From the repo root (works without activating the environment too)
.\.venv\Scripts\python.exe scripts\verify_env.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\sonaris.exe decode <survey.xtf>
.\.venv\Scripts\sonaris.exe infer <survey.xtf> --stub --out data\outputs\detections.geojson
```

`decode` reads an XTF survey and checks its navigation. `infer --stub` is the verified
no-model coordinate-chain demo: it finds a bright seabed sample and writes a GeoJSON point.

`correct`, `tile`, `train`, and `run` are stub/planned commands (P1.x). `report` converts a
GeoJSON detection file into configured reports (P1.8). Install `.[ml]` before model work and
`.[app]` before the web app.

## Layout

```
src/sonaris/   io · geometry · preprocess · datasets · models · postprocess · report · eval
configs/       default.yaml · classes.yaml · datasets.yaml   (nothing tunable in code)
scripts/       verify_env.py · download_datasets.sh
tests/         test_contracts.py · test_geometry.py (the ★ coordinate gate, P1.3)
apps/          FastAPI + React/MapLibre front-end (Phase 3)
data/          raw · interim · processed · outputs   (gitignored)
```

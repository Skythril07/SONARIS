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
python -m venv .venv && . .venv/Scripts/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"          # core + dev; add ".[ml]" for the detector, ".[app]" for the web app

python scripts/verify_env.py     # G0 environment check
sonaris --help                   # stage CLI (stages stubbed in Phase 0)
pytest                           # contracts + config green; geometry test skipped until P1.3
```

## Layout

```
src/sonaris/   io · geometry · preprocess · datasets · models · postprocess · report · eval
configs/       default.yaml · classes.yaml · datasets.yaml   (nothing tunable in code)
scripts/       verify_env.py · download_datasets.sh
tests/         test_contracts.py · test_geometry.py (the ★ coordinate gate, P1.3)
apps/          FastAPI + React/MapLibre front-end (Phase 3)
data/          raw · interim · processed · outputs   (gitignored)
```

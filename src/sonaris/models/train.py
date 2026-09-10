"""Train the baseline YOLO-Seg detector (plan gate P1.6; layout section 5).

This is the *control* run: its ``mAP@50`` is the number every later upgrade is measured against, so
it must be produced honestly on real, held-out data -- never faked. Training therefore requires two
things this repo does not ship: the ``ml`` extra (``pip install -e '.[ml]'`` -> torch + ultralytics)
and a unified corpus from :mod:`sonaris.datasets.unify` (plan gate P1.5). Without them this raises a
clear, actionable error instead of pretending to have trained.
"""
from __future__ import annotations

from pathlib import Path


def train(data_yaml, cfg: dict, out_dir="data/outputs/train") -> dict:
    """Train YOLO-Seg on ``data_yaml`` and return the recorded metrics (incl. mAP@50).

    Raises ``RuntimeError`` if the ``ml`` extra is not installed, or ``FileNotFoundError`` if the
    dataset config is missing -- the two real prerequisites, surfaced explicitly.
    """
    data_yaml = Path(data_yaml)
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"dataset config {data_yaml} not found. Build it first with datasets.unify (plan P1.5)."
        )
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise RuntimeError(
            "YOLO-Seg training needs the 'ml' extra. Install it with: "
            "pip install -e '.[ml]'  (plan gate P1.6)."
        ) from e

    detect = cfg["detect"]
    model = YOLO(f"{detect['model']}.pt")
    results = model.train(data=str(data_yaml), imgsz=detect["imgsz"], project=str(out_dir))
    # results.box.map50 is the control metric; keep it -- never delete this number.
    metrics = getattr(results, "results_dict", {}) or {}
    return {"data": str(data_yaml), "model": detect["model"], "metrics": metrics}

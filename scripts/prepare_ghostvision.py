"""Prepare the GhostVision crab-pot side-scan dataset for YOLO-detect training (plan P1.5 -> P1.6).

Input: the raw HuggingFace snapshot of ``PINGEcosystem/sss-crab-pot-detection-ds`` under
``data/raw/ghostvision/{train,valid,test}/*.jpg`` plus a ``metadata.jsonl`` per split. Each JSONL
record is Roboflow COCO-style::

    {"file_name": "...jpg", "objects": {"bbox": [[x, y, w, h], ...], "category": ["Crab-Pot", ...]}}

Output (written in place, idempotent) is a canonical Ultralytics layout::

    data/raw/ghostvision/<split>/images/*.jpg
    data/raw/ghostvision/<split>/labels/*.txt     # one YOLO line per box; EMPTY file = background
    data/raw/ghostvision/data.yaml                # single class: fishing_gear

Why single-class detect and not YOLO-Seg (plan override): the source ships bounding boxes only, no
masks, and only the ``Crab-Pot`` class -> this is the honest ``fishing_gear`` detection control.
The box math lives in :func:`sonaris.datasets.unify.coco_bbox_to_yolo` (unit-tested); this script is
just the disk driver. Run with the project venv:  ``.venv/Scripts/python.exe scripts/prepare_ghostvision.py``
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from sonaris.datasets.unify import coco_bbox_to_yolo

ROOT = Path("data/raw/ghostvision")
# HF split folder -> Ultralytics split name.
SPLITS = {"train": "train", "valid": "val", "test": "test"}

# This control's class list. Only crab pots are labelled in the source; per configs/classes.yaml +
# configs/datasets.yaml (ghostvision.class_map: crab_pot -> fishing_gear) they are SONARIS fishing_gear.
CLASS_NAMES = ["fishing_gear"]
CATEGORY_TO_CLASS = {"crab_pot": 0}


def _norm_category(name: str) -> str:
    """Normalise a source category label ('Crab-Pot') to a class-map key ('crab_pot')."""
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def _prepare_split(hf_split: str) -> tuple[int, int, int, int]:
    """Convert one split in place. Returns (images, boxes, background_images, dropped_boxes)."""
    split_dir = ROOT / hf_split
    meta = split_dir / "metadata.jsonl"
    if not meta.exists():
        raise FileNotFoundError(f"{meta} not found -- download the dataset snapshot first.")
    img_dir = split_dir / "images"
    lbl_dir = split_dir / "labels"
    img_dir.mkdir(exist_ok=True)
    lbl_dir.mkdir(exist_ok=True)

    n_img = n_box = n_bg = n_drop = 0
    for line in meta.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        fname = rec["file_name"]
        # The snapshot lands images flat in the split dir; move each into images/ (idempotent).
        flat = split_dir / fname
        dest = img_dir / fname
        if flat.exists() and not dest.exists():
            flat.replace(dest)
        if not dest.exists():
            raise FileNotFoundError(f"image {fname} listed in {meta} but not found on disk.")

        with Image.open(dest) as im:
            img_w, img_h = im.size

        objs = rec.get("objects", {}) or {}
        boxes = objs.get("bbox", []) or []
        cats = objs.get("category", []) or []
        lines: list[str] = []
        for bbox, cat in zip(boxes, cats):
            cls = CATEGORY_TO_CLASS.get(_norm_category(cat))
            if cls is None:
                n_drop += 1  # category outside this control's taxonomy -> intentionally excluded
                continue
            yolo = coco_bbox_to_yolo(bbox, img_w, img_h)
            if yolo is None:
                n_drop += 1  # degenerate / out-of-frame box
                continue
            cx, cy, w, h = yolo
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        (lbl_dir / f"{Path(fname).stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        n_img += 1
        n_box += len(lines)
        if not lines:
            n_bg += 1
    return n_img, n_box, n_bg, n_drop


def _write_data_yaml() -> Path:
    """Write an Ultralytics data.yaml pointing at the prepared split image dirs."""
    lines = [
        f"path: {ROOT.resolve().as_posix()}",
        "train: train/images",
        "val: valid/images",
        "test: test/images",
        f"nc: {len(CLASS_NAMES)}",
        "names:",
        *[f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES)],
        "",
    ]
    out = ROOT / "data.yaml"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> int:
    if not ROOT.exists():
        print(f"[prepare] {ROOT} not found -- download the HF snapshot into it first.")
        return 2
    totals = [0, 0, 0, 0]
    for hf_split, yolo_name in SPLITS.items():
        img, box, bg, drop = _prepare_split(hf_split)
        for i, v in enumerate((img, box, bg, drop)):
            totals[i] += v
        print(f"[{hf_split:>5} -> {yolo_name:<5}] images={img} boxes={box} background={bg} dropped={drop}")
    data_yaml = _write_data_yaml()
    print(f"[prepare] TOTAL images={totals[0]} boxes={totals[1]} background={totals[2]} dropped={totals[3]}")
    print(f"[prepare] wrote {data_yaml}")
    print(f"[prepare] classes: {CLASS_NAMES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Unify labelled datasets into one YOLO-Seg corpus (plan gate P1.5; layout section 5).

Real datasets are dropped under ``data/raw/<survey>/{images,labels}`` (YOLO text labels). This
module remaps each source taxonomy onto the *corrected* SONARIS classes (plan override: e.g.
``fishing_gear`` not ``debris_net_gear``; ``man_made_anomaly`` stays open-set, out of the closed
YOLO list), then builds **leave-one-survey-out** splits so the reported metric measures transfer to
an unseen survey, not memorisation.

The remap / split / counting logic is pure and unit-tested; the disk driver just walks folders.
No real data is bundled -- with an empty ``data/raw`` the driver reports "nothing to unify".
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SplitPlan:
    """One leave-one-survey-out split."""

    held_out: str
    train_surveys: list[str]
    val_surveys: list[str]


def remap_label_line(line: str, class_map: dict[int, int]) -> str | None:
    """Remap a YOLO label line's class id via ``class_map``; drop it (None) if unmapped.

    An unmapped source class (e.g. one folded into the open-set anomaly path) is intentionally
    excluded from the closed-set corpus rather than silently mislabelled.
    """
    parts = line.split()
    if not parts:
        return None
    src = int(parts[0])
    if src not in class_map:
        return None
    parts[0] = str(class_map[src])
    return " ".join(parts)


def remap_labels(lines: list[str], class_map: dict[int, int]) -> list[str]:
    """Remap a list of label lines, dropping unmapped classes and blanks."""
    out = [remap_label_line(ln, class_map) for ln in lines if ln.strip()]
    return [ln for ln in out if ln is not None]


def leave_one_survey_out(survey_ids: list[str]) -> list[SplitPlan]:
    """One split per survey: that survey is validation, the rest are training."""
    surveys = list(dict.fromkeys(survey_ids))  # de-dup, keep order
    if len(surveys) < 2:
        raise ValueError("need at least 2 surveys for leave-one-survey-out splits.")
    return [SplitPlan(s, [t for t in surveys if t != s], [s]) for s in surveys]


def class_counts(labels_by_survey: dict[str, list[str]]) -> dict[int, int]:
    """Count remapped instances per target class id across all surveys."""
    counts: dict[int, int] = {}
    for lines in labels_by_survey.values():
        for ln in lines:
            cid = int(ln.split()[0])
            counts[cid] = counts.get(cid, 0) + 1
    return dict(sorted(counts.items()))


def coco_bbox_to_yolo(
    bbox: list[float], img_w: int, img_h: int
) -> tuple[float, float, float, float] | None:
    """Convert a COCO-style pixel bbox ``[x, y, w, h]`` (top-left origin) to a normalised YOLO
    ``(cx, cy, w, h)`` tuple in ``[0, 1]``.

    The box is clamped to the image extent before normalising; a degenerate box (non-positive
    width/height, or one that lands fully outside the image) returns ``None`` so the caller can
    drop it rather than emit a garbage label. A silent error here would plant *every* training
    box at the wrong pixel -- the same class of quiet, well-formatted-but-wrong failure the
    coordinate-chain gate guards against downstream -- so this is unit-tested.
    """
    if img_w <= 0 or img_h <= 0:
        raise ValueError("image dimensions must be positive")
    x, y, w, h = (float(v) for v in bbox)
    if w <= 0 or h <= 0:
        return None
    x0 = min(max(x, 0.0), float(img_w))
    y0 = min(max(y, 0.0), float(img_h))
    x1 = min(max(x + w, 0.0), float(img_w))
    y1 = min(max(y + h, 0.0), float(img_h))
    bw = x1 - x0
    bh = y1 - y0
    if bw <= 0 or bh <= 0:
        return None
    cx = (x0 + bw / 2.0) / img_w
    cy = (y0 + bh / 2.0) / img_h
    return (cx, cy, bw / img_w, bh / img_h)


def build_data_yaml(class_names: list[str], train_dirs: list[str], val_dirs: list[str]) -> dict:
    """Assemble an Ultralytics-style ``data.yaml`` dict for one split."""
    return {
        "names": {i: n for i, n in enumerate(class_names)},
        "nc": len(class_names),
        "train": train_dirs,
        "val": val_dirs,
    }

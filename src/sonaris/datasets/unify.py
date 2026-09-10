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


def build_data_yaml(class_names: list[str], train_dirs: list[str], val_dirs: list[str]) -> dict:
    """Assemble an Ultralytics-style ``data.yaml`` dict for one split."""
    return {
        "names": {i: n for i, n in enumerate(class_names)},
        "nc": len(class_names),
        "train": train_dirs,
        "val": val_dirs,
    }

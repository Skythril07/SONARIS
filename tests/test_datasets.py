"""Dataset unification logic (plan gate P1.5).

Pure remap / split / count logic, tested without any real data on disk -- the disk driver just
walks folders over these functions.
"""
from __future__ import annotations

import pytest

from sonaris.datasets.unify import (
    class_counts,
    coco_bbox_to_yolo,
    leave_one_survey_out,
    remap_labels,
)


def test_remap_drops_unmapped_classes():
    lines = ["0 0.5 0.5 0.1 0.1", "2 0.1 0.1 0.2 0.2", "3 0.9 0.9 0.05 0.05"]
    # source 0->0, 2->1; source 3 (e.g. folded to open-set anomaly) is intentionally unmapped.
    out = remap_labels(lines, {0: 0, 2: 1})
    assert out == ["0 0.5 0.5 0.1 0.1", "1 0.1 0.1 0.2 0.2"]


def test_leave_one_survey_out_holds_each_survey_once():
    plans = leave_one_survey_out(["surveyA", "surveyB", "surveyC"])
    assert len(plans) == 3
    assert [p.held_out for p in plans] == ["surveyA", "surveyB", "surveyC"]
    for p in plans:
        assert p.val_surveys == [p.held_out]
        assert p.held_out not in p.train_surveys
        assert len(p.train_surveys) == 2


def test_leave_one_survey_out_needs_at_least_two():
    with pytest.raises(ValueError):
        leave_one_survey_out(["only_one"])


def test_class_counts_after_remap():
    counts = class_counts({"a": ["0 x", "0 x", "1 x"], "b": ["1 x"]})
    assert counts == {0: 2, 1: 2}


def test_coco_bbox_to_yolo_centre_and_norm():
    # a real GhostVision box [x=315, y=213, w=49, h=62.5] in a 640x640 tile
    cx, cy, w, h = coco_bbox_to_yolo([315, 213, 49, 62.5], 640, 640)
    assert cx == pytest.approx((315 + 24.5) / 640)
    assert cy == pytest.approx((213 + 31.25) / 640)
    assert w == pytest.approx(49 / 640)
    assert h == pytest.approx(62.5 / 640)


def test_coco_bbox_to_yolo_clamps_overflow():
    # box runs off the right/bottom edge: only 10px of its 30px width is inside a 100px image
    cx, cy, w, h = coco_bbox_to_yolo([90, 90, 30, 30], 100, 100)
    assert w == pytest.approx(10 / 100)
    assert h == pytest.approx(10 / 100)
    assert cx == pytest.approx((90 + 5) / 100)
    assert cy == pytest.approx((90 + 5) / 100)


def test_coco_bbox_to_yolo_drops_degenerate():
    assert coco_bbox_to_yolo([10, 10, 0, 5], 100, 100) is None      # zero width
    assert coco_bbox_to_yolo([200, 200, 10, 10], 100, 100) is None  # fully outside


def test_coco_bbox_to_yolo_rejects_bad_image_dims():
    with pytest.raises(ValueError):
        coco_bbox_to_yolo([1, 1, 2, 2], 0, 100)

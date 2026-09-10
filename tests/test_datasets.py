"""Dataset unification logic (plan gate P1.5).

Pure remap / split / count logic, tested without any real data on disk -- the disk driver just
walks folders over these functions.
"""
from __future__ import annotations

import pytest

from sonaris.datasets.unify import (
    class_counts,
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

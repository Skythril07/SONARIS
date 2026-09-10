"""Verification signals: physical plausibility (P2.3) and shadow consistency (P2.2)."""
from __future__ import annotations

import numpy as np

from sonaris.report.schema import Detection
from sonaris.verify.plausibility import apply_plausibility, is_plausible, size_and_aspect
from sonaris.verify.shadow import shadow_consistency

FISHING_GATE = {"len_m": [1.0, 40.0], "aspect": [1.0, 12.0]}


def _det(cls: str, length: float, width: float) -> Detection:
    return Detection(
        detection_id="d", line_id="L1", cls=cls, confidence=0.8, confidence_raw=0.8,
        lat=8.7, lon=78.1, position_uncertainty_m=3.0, length_m=length, width_m=width,
        ground_range_m=25.0, channel="port", ping_index=0,
    )


# --- plausibility (P2.3) ---

def test_size_and_aspect_orders_sides():
    long_side, aspect = size_and_aspect(1.0, 4.0)
    assert long_side == 4.0
    assert aspect == 4.0


def test_is_plausible_pass_and_fail():
    ok, reasons = is_plausible(2.0, 1.0, FISHING_GATE)
    assert ok and reasons == {}
    ok, reasons = is_plausible(60.0, 1.0, FISHING_GATE)   # too long
    assert not ok and "length_m" in reasons
    ok, reasons = is_plausible(5.0, 0.2, FISHING_GATE)    # aspect 25 -> too skinny
    assert not ok and "aspect" in reasons


def test_is_plausible_zero_short_side_is_infinite_aspect():
    ok, reasons = is_plausible(5.0, 0.0, FISHING_GATE)
    assert not ok and "aspect" in reasons


def test_apply_plausibility_flags_without_dropping():
    dets = [_det("fishing_gear", 2.0, 1.0), _det("fishing_gear", 60.0, 1.0)]
    out = apply_plausibility(dets, {"fishing_gear": FISHING_GATE})
    assert len(out) == 2
    assert out[0].metric_plausible is True
    assert out[1].metric_plausible is False
    assert out[1].evidence["implausible"]["length_m"] == 60.0


def test_apply_plausibility_drops_when_asked():
    dets = [_det("fishing_gear", 2.0, 1.0), _det("fishing_gear", 60.0, 1.0)]
    out = apply_plausibility(dets, {"fishing_gear": FISHING_GATE}, drop=True)
    assert [d.length_m for d in out] == [2.0]


def test_apply_plausibility_leaves_ungated_class_unevaluated():
    out = apply_plausibility([_det("man_made_anomaly", 999.0, 1.0)], {"fishing_gear": FISHING_GATE})
    assert out[0].metric_plausible is None


# --- shadow consistency (P2.2) ---

def test_shadow_consistency_true_when_dark_region_follows():
    img = np.full((20, 40), 50, dtype=np.uint16)
    img[5:15, 10:16] = 220   # bright highlight (the object)
    img[5:15, 16:24] = 5     # dark acoustic shadow just beyond, at higher columns
    ok, ev = shadow_consistency(img, (10, 5, 16, 15), min_shadow_px=4)
    assert ok
    assert ev["shadow_drop_frac"] > 0.5


def test_shadow_consistency_false_when_uniform_bright():
    img = np.full((20, 40), 200, dtype=np.uint16)   # no shadow anywhere
    ok, _ = shadow_consistency(img, (10, 5, 16, 15), min_shadow_px=4)
    assert not ok


def test_shadow_consistency_false_with_no_room_beyond():
    img = np.full((20, 40), 200, dtype=np.uint16)
    ok, ev = shadow_consistency(img, (36, 5, 40, 15), min_shadow_px=4)
    assert not ok
    assert "reason" in ev

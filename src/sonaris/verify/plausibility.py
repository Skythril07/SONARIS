"""Physical plausibility gate (plan gate P2.3; plan section 4, layout section 5.3).

A detection is plausible only if its metric size and aspect ratio fall inside the per-class limits
in ``configs/default.yaml`` (``verify.plausibility.<class>``). "A 60 m crab pot is not a crab pot":
these gates suppress detections the model is confident about but physics rules out. The gate FLAGS
by default (sets ``metric_plausible`` + evidence) and can optionally DROP -- dropping trades recall
for precision, so it is a deliberate, measured choice recorded on the experiment ladder, never the
silent default.

Size/aspect logic is pure and unit-tested; applying it to a detection list is a thin loop.
"""
from __future__ import annotations

from ..report.schema import Detection


def size_and_aspect(length_m: float, width_m: float) -> tuple[float, float]:
    """Return (longer extent in metres, aspect = long/short >= 1). Guards a zero short side."""
    long_side = max(abs(length_m), abs(width_m))
    short_side = min(abs(length_m), abs(width_m))
    aspect = long_side / short_side if short_side > 0 else float("inf")
    return long_side, aspect


def is_plausible(length_m: float, width_m: float, gate: dict) -> tuple[bool, dict]:
    """True if size and aspect are within ``gate`` = {len_m: [lo, hi], aspect: [lo, hi]}.

    Returns (plausible, reasons); ``reasons`` names each failing check (with its value) for the UI
    evidence panel, and is empty when the detection passes.
    """
    long_side, aspect = size_and_aspect(length_m, width_m)
    lo_len, hi_len = gate["len_m"]
    lo_asp, hi_asp = gate["aspect"]
    reasons: dict = {}
    if not (lo_len <= long_side <= hi_len):
        reasons["length_m"] = round(long_side, 3)
    if not (lo_asp <= aspect <= hi_asp):
        reasons["aspect"] = round(aspect, 3)
    return (not reasons), reasons


def apply_plausibility(
    dets: list[Detection], cfg_plausibility: dict, *, drop: bool = False
) -> list[Detection]:
    """Set ``metric_plausible`` (and evidence) on each detection from its class gate.

    Classes with no configured gate are left unevaluated (``metric_plausible`` stays ``None``). With
    ``drop=True`` implausible detections are removed rather than merely flagged.
    """
    gates = {k: v for k, v in cfg_plausibility.items() if isinstance(v, dict) and "len_m" in v}
    out: list[Detection] = []
    for d in dets:
        gate = gates.get(d.cls)
        if gate is None:
            out.append(d)
            continue
        ok, reasons = is_plausible(d.length_m, d.width_m, gate)
        d.metric_plausible = ok
        if reasons:
            d.evidence = {**d.evidence, "implausible": reasons}
        if drop and not ok:
            continue
        out.append(d)
    return out

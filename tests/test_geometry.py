"""The synthetic ground-truth geometry test — the ★ gate (plan.md §4.1, P1.3).

Build a fake SurveyLine (straight track, constant heading + altitude, one bright object planted
at a known ground range on a known side) and assert the pipeline recovers its lat/lon to within
a metre. This catches every silent geometry gotcha (layout §6) and runs in milliseconds.
Written for real BEFORE the stub-detector -> GeoJSON gate, not after.
"""
import pytest


@pytest.mark.skip(reason="Implemented at plan gate P1.3, once geometry/georef.py exists.")
def test_synthetic_ground_truth_recovers_latlon():
    ...

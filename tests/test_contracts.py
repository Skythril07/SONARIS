"""Phase 0.4 done-check: the data contracts instantiate and config loads correctly,
including the taxonomy corrections from plan.md §2 (overrides #1, #2, #3)."""
import numpy as np

from sonaris.config import load_classes, load_config
from sonaris.io.schema import NAV_LATLON, SIDE_PORT, CorrectedLine, Ping, SurveyLine
from sonaris.report.schema import Detection


def _make_ping() -> Ping:
    return Ping(
        index=0, time_utc=0.0, lat=8.7, lon=78.1, heading_deg=90.0,
        altitude_m=10.0, depth_m=20.0, pitch_deg=0.0, roll_deg=0.0,
        heave_m=0.0, layback_m=0.0, slant_range_m=50.0, n_samples=4,
        samples=np.zeros(4),
    )


def test_surveyline_instantiates():
    line = SurveyLine(
        line_id="L1", source_file="x.xtf", sonar_type="test",
        nav_units=NAV_LATLON, channels={SIDE_PORT: [_make_ping()]},
    )
    assert line.channels[SIDE_PORT][0].n_samples == 4
    assert line.corrected is False


def test_correctedline_instantiates():
    cl = CorrectedLine(
        line_id="L1", channel=SIDE_PORT, image=np.zeros((3, 5), np.uint16),
        ground_res_m=0.1, bottom_idx=np.zeros(3, int),
        altitude_m=np.full(3, 10.0), ping_index=np.arange(3),
    )
    assert cl.image.shape == (3, 5)


def test_detection_defaults_are_total():
    d = Detection(
        detection_id="d1", line_id="L1", cls="fishing_gear",
        confidence=0.9, confidence_raw=0.9, lat=8.7, lon=78.1,
        position_uncertainty_m=3.0, length_m=2.0, width_m=1.0,
        ground_range_m=25.0, channel="port", ping_index=0,
    )
    assert d.persistence_count == 1
    assert d.height_est_m is None
    assert d.evidence == {}
    assert d.review_status == "unreviewed"


def test_detect_model_is_bbox_control():
    # P1.6 control is YOLO-detect, not Seg: the only real labelled source (ghostvision crab pots)
    # ships bounding boxes, not masks (see configs/default.yaml note). Seg (override #1) is deferred
    # until a mask-labelled source exists -- guard that we use a NON-seg model until then.
    model = load_config()["detect"]["model"]
    assert not model.endswith("-seg"), f"detect.model {model!r} is a seg model but P1.6 trains detect"


def test_taxonomy_is_corrected():
    classes = load_classes()
    assert "fishing_gear" in classes["classes"]            # override #2
    assert "debris_net_gear" not in classes["classes"]
    # open-set anomaly kept separate from the closed-set YOLO list (override #3)
    assert "man_made_anomaly" in classes["open_set"]
    assert "man_made_anomaly" not in classes["classes"]

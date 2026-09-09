"""P1.1 — the XTF decoder. Round-trips a synthetic XTF (written by pyxtf's own writer) and
asserts the reader recovers fields, sides, and nav table correctly, including the gotchas."""
import pytest

from sonaris.io.schema import NAV_LATLON, SIDE_PORT, SIDE_STBD
from sonaris.io.xtf_reader import NavUnitsError, nav_dataframe, read_xtf
from tests.synthetic import write_synthetic_xtf

try:
    import pyxtf
    _METERS = int(pyxtf.XTFNavUnits.meters)
except Exception:  # noqa: BLE001 # pragma: no cover - pyxtf absent
    pyxtf = None


def test_reads_channels_and_fields(tmp_path):
    truth = write_synthetic_xtf(tmp_path / "line.xtf", n_pings=10, object_ping=5)
    line = read_xtf(truth.path)

    assert line.nav_units == NAV_LATLON
    assert set(line.channels) == {SIDE_PORT, SIDE_STBD}
    assert len(line.channels[SIDE_PORT]) == 10
    assert len(line.channels[SIDE_STBD]) == 10

    p0 = line.channels[SIDE_PORT][0]
    assert p0.slant_range_m == truth.slant_range_m          # per-channel SlantRange (gotcha #3)
    assert p0.n_samples == truth.n_samples
    assert abs(p0.altitude_m - truth.altitude_m) < 1e-6
    assert abs(p0.heading_deg - truth.heading_deg) < 1e-6
    assert -90 <= p0.lat <= 90 and -180 <= p0.lon <= 180
    assert line.provenance["heading_source"] == "sensor_heading"
    assert line.provenance["position_source"] == "sensor"


def test_side_read_from_enum_not_index_order(tmp_path):
    # The bright object is planted on stbd; it must surface on the stbd channel, not port
    # (gotcha #6 — a silent flip would mirror it across the track line).
    truth = write_synthetic_xtf(tmp_path / "line.xtf", n_pings=8, object_ping=4, object_side="stbd")
    line = read_xtf(truth.path)

    stbd_ping = line.channels[SIDE_STBD][truth.object_ping]
    port_ping = line.channels[SIDE_PORT][truth.object_ping]
    assert stbd_ping.samples.max() == 255
    assert port_ping.samples.max() < 255
    # planted sample lands where the slant->sample maths predicts
    assert stbd_ping.samples[truth.object_sample_index] == 255


def test_nav_dataframe_shape_and_monotonic_time(tmp_path):
    truth = write_synthetic_xtf(tmp_path / "line.xtf", n_pings=12)
    df = nav_dataframe(read_xtf(truth.path))
    assert len(df) == 12
    assert df["time_utc"].is_monotonic_increasing
    assert {"lat", "lon", "heading_deg", "slant_port_m", "slant_stbd_m"} <= set(df.columns)


@pytest.mark.skipif(pyxtf is None, reason="pyxtf not installed")
def test_meters_navunits_is_refused(tmp_path):
    truth = write_synthetic_xtf(tmp_path / "meters.xtf", n_pings=5, nav_units=_METERS)
    with pytest.raises(NavUnitsError):
        read_xtf(truth.path)

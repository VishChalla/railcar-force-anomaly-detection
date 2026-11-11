import numpy as np
import pandas as pd
import pytest

from railforce.detect import DetectConfig, evaluate, score, truck_features, wheel_features
from railforce.simulate import SimConfig, simulate

SMALL = SimConfig(n_cars=300, seed=1)


@pytest.fixture(scope="module")
def sim():
    return simulate(SMALL)


def test_simulation_is_deterministic(sim):
    again = simulate(SMALL)
    pd.testing.assert_frame_equal(sim[0], again[0])


def test_every_pass_has_eight_wheels(sim):
    readings = sim[0]
    assert (readings.groupby("pass_id").size() == 8).all()
    assert (readings["vertical_kips"] > 0).all()
    assert readings["site"].nunique() == 6


def test_default_run_is_about_230k_readings():
    readings, _, _ = simulate()
    assert 220_000 < len(readings) < 240_000


def test_ratio_cancels_load_speed_and_site(sim):
    """Healthy wheels should sit at 1.0 whatever the car weighs or wherever it was measured."""
    readings, wheel_truth, _ = sim
    w = wheel_features(readings)
    bad = set(map(tuple, wheel_truth[["car_id", "wheel"]].to_numpy()))
    healthy = w[[key not in bad for key in zip(w["car_id"], w["wheel"])]]
    by_group = healthy.groupby(["site", "loaded"])["vertical_ratio"].median()
    assert np.allclose(by_group, 1.0, atol=0.02)
    assert healthy["vertical_kips"].groupby(healthy["loaded"]).median().diff().iloc[-1] > 15


def test_features_only_use_the_past(sim):
    """Removing later readings must not change a wheel's features on an earlier day."""
    readings = sim[0]
    cols = ["car_id", "wheel", "pass_id", "vertical_ratio", "ratio_recent", "ratio_trend"]
    full = wheel_features(readings)
    early = wheel_features(readings[readings["day"] <= 60])
    merged = early[cols].merge(full[cols], on=["car_id", "wheel", "pass_id"], suffixes=("", "_full"))
    for c in ("vertical_ratio", "ratio_recent", "ratio_trend"):
        np.testing.assert_allclose(merged[c], merged[f"{c}_full"], equal_nan=True)


def test_score():
    flagged = pd.DataFrame({"a": [1, 2, 3]})
    truth = pd.DataFrame({"a": [2, 3, 4, 5]})
    s = score(flagged, truth, ["a"])
    assert s["precision"] == pytest.approx(2 / 3)
    assert s["recall"] == pytest.approx(0.5)


def test_trend_beats_a_fixed_limit_at_equal_false_alarms(sim):
    res = evaluate(*sim)
    assert res["ratio_trend"]["recall"] > res["fixed_limit"]["recall"]
    assert res["ratio_trend"]["precision"] > 0.8
    assert res["lead_time_days_median"] > 0


def test_hunting_trucks_are_found(sim):
    res = evaluate(*sim)
    assert res["hunting"]["recall"] > 0.8
    assert res["hunting"]["precision"] > 0.8


def test_no_faults_means_almost_no_flags():
    readings, _, _ = simulate(SimConfig(n_cars=300, wheel_defect_rate=0, hunting_truck_rate=0, seed=2))
    w = wheel_features(readings, DetectConfig())
    t = truck_features(readings, DetectConfig())
    assert w.loc[w["wheel_flag"], ["car_id", "wheel"]].drop_duplicates().shape[0] <= 300 * 8 * 0.005
    assert t.loc[t["truck_flag"], ["car_id", "truck"]].drop_duplicates().shape[0] <= 300 * 2 * 0.02

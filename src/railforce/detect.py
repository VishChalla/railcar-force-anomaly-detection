"""Anomaly detection on wheel force readings.

Raw force is a poor health signal: a loaded hopper at 55 mph legitimately
hits the rail five times harder than an empty boxcar at 30 mph, and each
detector site reads a little differently. The approach here is to build
metrics that cancel those effects out, then watch how each wheel and truck
*changes* over successive passes.

1. Vertical ratio      a wheel's vertical force divided by the median of the
                       eight wheels on the same car in the same pass. Load,
                       speed and site calibration divide out; a healthy wheel
                       sits near 1.0 everywhere.
2. Ratio trend         the slope of that ratio over a wheel's recent passes.
                       A growing tread defect shows up as a steady climb long
                       before any single reading looks alarming.
3. Lateral variability how erratic a truck's lateral-to-vertical ratio is
                       across recent passes, relative to its site. Hunting
                       trucks are not always high, but they are inconsistent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DetectConfig:
    window: int = 6  # passes of history used for trend and variability
    min_passes: int = 4
    ratio_z: float = 5.0  # robust z-score a wheel's recent ratio must exceed
    trend_z: float = 3.0  # ... and its trend, to be flagged as a developing defect
    variability_z: float = 5.0


def _robust_z(x: pd.Series) -> pd.Series:
    """Z-score from median and MAD, so the anomalies do not set their own scale."""
    med = x.median()
    mad = (x - med).abs().median() * 1.4826
    return (x - med) / max(mad, 1e-9)


def _rolling_slope(df: pd.DataFrame, keys: list[str], x: str, y: str, window: int, min_n: int):
    """Least-squares slope of y on x over each group's trailing window."""
    g = df.groupby(keys, sort=False)
    roll = lambda s: g[s].rolling(window, min_periods=min_n).mean().reset_index(  # noqa: E731
        level=list(range(len(keys))), drop=True)
    df = df.assign(_xy=df[x] * df[y], _xx=df[x] ** 2)
    g = df.groupby(keys, sort=False)
    mx, my, mxy, mxx = roll(x), roll(y), roll("_xy"), roll("_xx")
    return (mxy - mx * my) / (mxx - mx**2).clip(lower=1e-9)


def wheel_features(readings: pd.DataFrame, cfg: DetectConfig = DetectConfig()) -> pd.DataFrame:
    """Per-reading wheel health features, using only current and past passes."""
    df = readings.sort_values(["car_id", "wheel", "day"], ignore_index=True)
    car_median = df.groupby("pass_id")["vertical_kips"].transform("median")
    df["vertical_ratio"] = df["vertical_kips"] / car_median
    keys = ["car_id", "wheel"]
    g = df.groupby(keys, sort=False)
    df["pass_number"] = g.cumcount() + 1
    df["ratio_recent"] = g["vertical_ratio"].rolling(3, min_periods=2).median().reset_index(
        level=[0, 1], drop=True)
    df["ratio_trend"] = _rolling_slope(df, keys, "day", "vertical_ratio", cfg.window, cfg.min_passes)
    df["ratio_z"] = _robust_z(df["ratio_recent"])
    df["trend_z"] = _robust_z(df["ratio_trend"])
    df["wheel_flag"] = (df["ratio_z"] > cfg.ratio_z) & (df["trend_z"] > cfg.trend_z)
    return df


def truck_features(readings: pd.DataFrame, cfg: DetectConfig = DetectConfig()) -> pd.DataFrame:
    """Per-pass truck stability features."""
    df = readings.assign(lv=readings["lateral_kips"] / readings["vertical_kips"])
    t = df.groupby(["car_id", "truck", "pass_id"], as_index=False).agg(
        day=("day", "first"), site=("site", "first"), speed_mph=("speed_mph", "first"),
        car_type=("car_type", "first"), loaded=("loaded", "first"), lv=("lv", "mean"))
    # Curved sites have higher lateral load for every truck: compare within site.
    t["lv_rel"] = t["lv"] / t.groupby("site")["lv"].transform("median")
    t = t.sort_values(["car_id", "truck", "day"], ignore_index=True)
    g = t.groupby(["car_id", "truck"], sort=False)
    t["pass_number"] = g.cumcount() + 1
    t["lv_variability"] = g["lv_rel"].rolling(cfg.window, min_periods=cfg.min_passes).std().reset_index(
        level=[0, 1], drop=True)
    t["variability_z"] = _robust_z(t["lv_variability"])
    t["truck_flag"] = t["variability_z"] > cfg.variability_z
    return t


def first_flags(features: pd.DataFrame, keys: list[str], flag: str) -> pd.DataFrame:
    """The first day each unit was flagged."""
    hit = features[features[flag]]
    return hit.groupby(keys, as_index=False)["day"].min().rename(columns={"day": f"{flag}_day"})


def score(flagged: pd.DataFrame, truth: pd.DataFrame, keys: list[str]) -> dict:
    """Precision and recall of a flagged set against the planted faults."""
    f = set(map(tuple, flagged[keys].to_numpy()))
    t = set(map(tuple, truth[keys].to_numpy()))
    tp = len(f & t)
    return {"flagged": len(f), "true_faults": len(t), "true_positives": tp,
            "precision": tp / len(f) if f else float("nan"),
            "recall": tp / len(t) if t else float("nan")}


def evaluate(readings: pd.DataFrame, wheel_truth: pd.DataFrame, truck_truth: pd.DataFrame,
             cfg: DetectConfig = DetectConfig()) -> dict:
    """Run detection and compare it with a fixed force limit."""
    w = wheel_features(readings, cfg)
    t = truck_features(readings, cfg)
    keys = ["car_id", "wheel"]

    # A defect can only be caught if the wheel was seen often enough after it
    # began; the rest are reported separately rather than counted as misses.
    after = readings.merge(wheel_truth[[*keys, "onset_day"]], on=keys)
    seen_after = after[after["day"] > after["onset_day"]].groupby(keys).size().rename("seen_after")
    observable = wheel_truth.merge(seen_after.reset_index(), on=keys, how="left")
    observable = observable[observable["seen_after"].fillna(0) >= cfg.min_passes]

    trend_first = first_flags(w, keys, "wheel_flag")
    is_planted = trend_first.merge(wheel_truth, on=keys, how="left", indicator=True)["_merge"].eq("both")
    false_alarms = int((~is_planted).sum())

    # Baseline: one absolute force limit for every wheel. To give it the best
    # possible case, the limit is set as low as it can go while raising no
    # more false alarms than the ratio-trend method did.
    peak = w.groupby(keys, as_index=False)["vertical_kips"].max().merge(
        wheel_truth[keys], on=keys, how="left", indicator=True)
    healthy_peaks = np.sort(peak.loc[peak["_merge"] == "left_only", "vertical_kips"].to_numpy())[::-1]
    fixed_limit = float(healthy_peaks[false_alarms])
    w["fixed_alarm"] = w["vertical_kips"] > fixed_limit
    fixed_first = first_flags(w, keys, "fixed_alarm")
    both = trend_first.merge(fixed_first, on=keys, how="outer").merge(
        wheel_truth[keys + ["onset_day"]], on=keys, how="inner")
    caught_both = both.dropna(subset=["wheel_flag_day", "fixed_alarm_day"])
    lead = caught_both["fixed_alarm_day"] - caught_both["wheel_flag_day"]

    return {
        "readings": int(len(readings)),
        "car_passes": int(readings["pass_id"].nunique()),
        "wheels": int(readings.groupby(keys).ngroups),
        "sites": int(readings["site"].nunique()),
        "wheel_defects_planted": int(len(wheel_truth)),
        "wheel_defects_observable": int(len(observable)),
        "ratio_trend": score(trend_first, observable, keys) | {"false_alarms": false_alarms},
        "fixed_limit": score(fixed_first, observable, keys) | {"limit_kips": round(fixed_limit, 1)},
        "lead_time_days_median": float(lead.median()),
        "lead_time_share_earlier": float((lead > 0).mean()),
        "caught_only_by_trend": int(both["fixed_alarm_day"].isna().sum()),
        "hunting": score(first_flags(t, ["car_id", "truck"], "truck_flag"), truck_truth,
                         ["car_id", "truck"]),
        "_wheel_features": w, "_truck_features": t,
    }

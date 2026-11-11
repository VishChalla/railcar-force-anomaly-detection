"""Synthetic wayside force-detector data.

Wayside detectors measure the vertical and lateral force each wheel puts on
the rail as a train rolls over an instrumented section of track. This module
simulates a fleet passing six such sites and plants two kinds of developing
fault, with the ground truth kept so detection can be scored:

* wheel tread defects: a flat spot that grows, so one wheel's vertical
  impact climbs pass after pass
* truck hunting: an unstable truck that oscillates side to side at speed,
  so its lateral forces become erratic

All values are invented for illustration. No real railway data, thresholds or
results are used anywhere in this project.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

CAR_TYPES = {
    # type: (share of fleet, empty weight lb, loaded weight lb, share of passes loaded)
    "Covered hopper": (0.30, 62_000, 286_000, 0.55),
    "Tank": (0.22, 68_000, 263_000, 0.50),
    "Intermodal": (0.20, 58_000, 220_000, 0.80),
    "Boxcar": (0.16, 70_000, 240_000, 0.60),
    "Gondola": (0.12, 60_000, 275_000, 0.50),
}
SITES = {
    # site: (typical speed mph, calibration bias, curvature factor for lateral load)
    "Site A": (48, 1.00, 0.06), "Site B": (42, 1.04, 0.10), "Site C": (52, 0.98, 0.05),
    "Site D": (38, 1.00, 0.14), "Site E": (50, 1.02, 0.07), "Site F": (45, 0.97, 0.09),
}


@dataclass(frozen=True)
class SimConfig:
    n_cars: int = 1500
    n_days: int = 120
    passes_per_car: float = 19.2  # about 230,000 wheel readings in total
    wheel_defect_rate: float = 0.02
    hunting_truck_rate: float = 0.015
    seed: int = 42


def simulate(cfg: SimConfig = SimConfig()) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return ``(readings, wheel_truth, truck_truth)``."""
    rng = np.random.default_rng(cfg.seed)
    types = list(CAR_TYPES)
    car_type = rng.choice(types, cfg.n_cars, p=[CAR_TYPES[t][0] for t in types])

    # --- car passes --------------------------------------------------------
    n_pass = rng.poisson(cfg.passes_per_car, cfg.n_cars).clip(min=4)
    car = np.repeat(np.arange(cfg.n_cars), n_pass)
    day = rng.uniform(0, cfg.n_days, len(car))
    site = rng.choice(list(SITES), len(car))
    speed = np.clip(rng.normal([SITES[s][0] for s in site], 6.0), 15, 70)
    loaded = rng.random(len(car)) < np.array([CAR_TYPES[t][3] for t in car_type[car]])
    weight = np.where(loaded, [CAR_TYPES[t][2] for t in car_type[car]],
                      [CAR_TYPES[t][1] for t in car_type[car]]) * rng.normal(1, 0.04, len(car))
    passes = pd.DataFrame({
        "pass_id": np.arange(len(car)), "car_id": car, "car_type": car_type[car],
        "day": day, "site": site, "direction": rng.choice(["Eastbound", "Westbound"], len(car)),
        "speed_mph": speed.round(1), "loaded": loaded, "static_kips": weight / 8 / 1000,
    })

    # --- planted faults ----------------------------------------------------
    n_wheels = cfg.n_cars * 8
    bad_wheels = rng.choice(n_wheels, int(n_wheels * cfg.wheel_defect_rate), replace=False)
    wheel_truth = pd.DataFrame({
        "car_id": bad_wheels // 8, "wheel": bad_wheels % 8,
        "onset_day": rng.uniform(0, cfg.n_days * 0.75, len(bad_wheels)).round(1),
        "growth_kips_per_day": rng.uniform(0.04, 0.45, len(bad_wheels)).round(3),
    })
    n_trucks = cfg.n_cars * 2
    bad_trucks = rng.choice(n_trucks, int(n_trucks * cfg.hunting_truck_rate), replace=False)
    truck_truth = pd.DataFrame({"car_id": bad_trucks // 2, "truck": bad_trucks % 2})

    # --- one reading per wheel per pass -----------------------------------
    r = passes.loc[passes.index.repeat(8)].reset_index(drop=True)
    r["wheel"] = np.tile(np.arange(8), len(passes))
    r["truck"] = r["wheel"] // 4  # wheels 0-3 on the lead truck, 4-7 on the trailing truck
    r["axle"] = r["wheel"] // 2 + 1
    r["side"] = np.where(r["wheel"] % 2 == 0, "Left", "Right")
    n = len(r)

    bias = r["site"].map({s: v[1] for s, v in SITES.items()}).to_numpy()
    curve = r["site"].map({s: v[2] for s, v in SITES.items()}).to_numpy()
    spd = r["speed_mph"].to_numpy()
    static = r["static_kips"].to_numpy()

    # Healthy wheel: static load amplified a little by speed, plus noise.
    vertical = static * (1.12 + 0.004 * (spd - 40)) * rng.lognormal(0, 0.07, n)

    # Tread defect: an impact that grows from its onset day and hits harder at speed.
    r = r.merge(wheel_truth, on=["car_id", "wheel"], how="left")
    age = (r["day"] - r["onset_day"]).clip(lower=0).fillna(0).to_numpy()
    impact = age * r["growth_kips_per_day"].fillna(0).to_numpy() * (spd / 45) ** 1.5
    vertical = (vertical + impact * rng.lognormal(0, 0.15, n)) * bias

    # Lateral: a share of vertical set by curvature; hunting makes it erratic at speed.
    lateral = vertical * (0.05 + curve) * rng.lognormal(0, 0.20, n)
    hunting = r.merge(truck_truth, on=["car_id", "truck"], how="left",
                      indicator=True)["_merge"].eq("both").to_numpy()
    unstable = hunting & (spd > 44)
    lateral = np.where(unstable, lateral * rng.lognormal(0.55, 0.65, n) * np.where(r["loaded"], 1, 1.3),
                       lateral)

    r["vertical_kips"] = vertical.round(2)
    r["lateral_kips"] = lateral.round(2)
    cols = ["pass_id", "day", "site", "direction", "speed_mph", "car_id", "car_type", "loaded",
            "axle", "side", "wheel", "truck", "vertical_kips", "lateral_kips"]
    readings = r[cols].sort_values(["day", "pass_id", "wheel"], ignore_index=True)
    return readings, wheel_truth, truck_truth

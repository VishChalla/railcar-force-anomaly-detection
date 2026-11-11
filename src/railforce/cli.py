"""Simulate, detect, score and plot.

    python -m railforce            # writes outputs/ and docs/ figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .detect import evaluate  # noqa: E402
from .simulate import simulate  # noqa: E402

INK, ACCENT, MUTED, WARN = "#14213d", "#0f766e", "#b6bcc6", "#b45309"


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#d1d5db")
    ax.tick_params(colors="#4b5563", labelsize=9)
    ax.grid(axis="y", color="#eef0f3", linewidth=0.8)
    ax.set_axisbelow(True)


def _title(ax, text):
    ax.set_title(text, loc="left", color=INK, fontweight="bold", fontsize=11)


def plot_one_wheel(w: pd.DataFrame, wheel_truth: pd.DataFrame, limit: float, path: Path) -> None:
    """Raw force hides a growing defect; the ratio exposes it."""
    flagged = w[w["wheel_flag"]].merge(wheel_truth, on=["car_id", "wheel"])
    alarmed = w[w["fixed_alarm"]].merge(wheel_truth, on=["car_id", "wheel"])
    both = flagged.groupby(["car_id", "wheel"])["day"].min().to_frame("flag").join(
        alarmed.groupby(["car_id", "wheel"])["day"].min().rename("alarm"), how="inner")
    car, wheel = (both["alarm"] - both["flag"]).sort_values().index[len(both) * 3 // 4]
    d = w[(w["car_id"] == car) & (w["wheel"] == wheel)]
    onset = wheel_truth.query("car_id == @car and wheel == @wheel")["onset_day"].iloc[0]

    fig, (a, b) = plt.subplots(2, 1, figsize=(7.5, 5.8), dpi=160, sharex=True)
    colors = d["loaded"].map({True: INK, False: MUTED})
    a.scatter(d["day"], d["vertical_kips"], c=colors, s=28, zorder=3)
    a.axhline(limit, color=WARN, ls="--", lw=1)
    a.text(1, limit + 2, f"fixed limit {limit:.0f} kips", color=WARN, fontsize=8)
    a.set_ylabel("Vertical force (kips)", color="#4b5563")
    _title(a, f"Raw force, car {car} wheel {wheel}: load swamps the signal (dark = loaded)")
    b.plot(d["day"], d["vertical_ratio"], "-o", color=ACCENT, ms=4, lw=1.5, zorder=3)
    b.axhline(1.0, color=MUTED, lw=1)
    b.set_ylabel("Ratio to car median", color="#4b5563")
    b.set_xlabel("Day", color="#4b5563")
    _title(b, "Same wheel as a ratio: it leaves 1.0 soon after the defect starts")
    for ax in (a, b):
        ax.axvline(onset, color="#6b7280", ls=":", lw=1)
        ax.axvline(both.loc[(car, wheel), "flag"], color=ACCENT, lw=1.2)
        ax.axvline(both.loc[(car, wheel), "alarm"], color=WARN, lw=1.2)
        _style(ax)
    top = b.get_ylim()[1]
    b.text(onset, top, " defect starts", fontsize=8, color="#6b7280", va="top")
    b.text(both.loc[(car, wheel), "flag"], top, " trend flag", fontsize=8, color=ACCENT, va="top")
    b.text(both.loc[(car, wheel), "alarm"], top, " fixed-limit alarm", fontsize=8, color=WARN,
           va="top")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_comparison(res: dict, lead: pd.Series, path: Path) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.5, 3.4), dpi=160)
    vals = [res["fixed_limit"]["recall"] * 100, res["ratio_trend"]["recall"] * 100]
    bars = a.bar(["Fixed force limit", "Ratio trend"], vals, color=[MUTED, ACCENT], width=0.55)
    a.bar_label(bars, fmt="%.0f%%", color=INK, fontsize=10, padding=3)
    a.set_ylim(0, 105)
    a.set_ylabel("Defective wheels caught (%)", color="#4b5563")
    _title(a, "Recall at equal false alarms")
    b.hist(lead, bins=20, color=ACCENT, alpha=0.85)
    b.axvline(0, color=INK, lw=1)
    b.set_xlabel("Days of extra warning from the trend", color="#4b5563")
    b.set_ylabel("Wheels", color="#4b5563")
    _title(b, "How much earlier (when both fire)")
    for ax in (a, b):
        _style(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_force_patterns(readings: pd.DataFrame, path: Path) -> None:
    """Loading by car type, load state and direction."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.5, 3.6), dpi=160, gridspec_kw={"width_ratios": [3, 2]})
    types = sorted(readings["car_type"].unique())
    for i, state in enumerate((False, True)):
        data = [readings[(readings["car_type"] == t) & (readings["loaded"] == state)]["vertical_kips"]
                for t in types]
        box = a.boxplot(data, positions=[j + (i - 0.5) * 0.36 for j in range(len(types))],
                        widths=0.3, patch_artist=True, showfliers=False,
                        medianprops={"color": "white"})
        for patch in box["boxes"]:
            patch.set(facecolor=INK if state else MUTED, edgecolor="none")
    a.set_xticks(range(len(types)), [t.replace(" ", "\n") for t in types], fontsize=8)
    a.set_ylabel("Vertical force (kips)", color="#4b5563")
    _title(a, "Force by car type (dark = loaded)")
    lv = readings.assign(lv=readings["lateral_kips"] / readings["vertical_kips"])
    piv = lv.groupby(["site", "direction"])["lv"].median().unstack()
    x = range(len(piv))
    b.bar([i - 0.18 for i in x], piv["Eastbound"], width=0.34, color=ACCENT, label="Eastbound")
    b.bar([i + 0.18 for i in x], piv["Westbound"], width=0.34, color=MUTED, label="Westbound")
    b.set_xticks(list(x), [s.split()[-1] for s in piv.index])
    b.set_xlabel("Site", color="#4b5563")
    b.set_ylabel("Median lateral / vertical", color="#4b5563")
    b.legend(frameon=False, fontsize=8)
    _title(b, "Lateral load by site")
    for ax in (a, b):
        _style(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_hunting(t: pd.DataFrame, truck_truth: pd.DataFrame, path: Path) -> None:
    t = t.merge(truck_truth.assign(hunting=True), on=["car_id", "truck"], how="left")
    bad = t["hunting"].eq(True)
    fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=160)
    ok = t[~bad].sample(6000, random_state=0)
    ax.scatter(ok["speed_mph"], ok["lv_rel"], s=4, color=MUTED, alpha=0.5, label="Healthy trucks")
    ax.scatter(t.loc[bad, "speed_mph"], t.loc[bad, "lv_rel"], s=9, color=WARN, alpha=0.8,
               label="Hunting trucks")
    ax.set_xlabel("Speed (mph)", color="#4b5563")
    ax.set_ylabel("Lateral / vertical, relative to site", color="#4b5563")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _title(ax, "Hunting trucks look normal at low speed and erratic above about 44 mph")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Railcar force anomaly detection on synthetic data.")
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    parser.add_argument("--docs", type=Path, default=Path("docs"))
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True)
    args.docs.mkdir(exist_ok=True)

    readings, wheel_truth, truck_truth = simulate()
    res = evaluate(readings, wheel_truth, truck_truth)
    w, t = res.pop("_wheel_features"), res.pop("_truck_features")

    keys = ["car_id", "wheel"]
    first = w[w["wheel_flag"]].groupby(keys)["day"].min().to_frame("flag").join(
        w[w["fixed_alarm"]].groupby(keys)["day"].min().rename("alarm"), how="inner")
    first = first.reset_index().merge(wheel_truth[keys], on=keys)
    lead = first["alarm"] - first["flag"]

    plot_one_wheel(w, wheel_truth, res["fixed_limit"]["limit_kips"], args.docs / "one_wheel.png")
    plot_comparison(res, lead, args.docs / "comparison.png")
    plot_force_patterns(readings, args.docs / "force_patterns.png")
    plot_hunting(t, truck_truth, args.docs / "hunting.png")

    flagged = w[w["wheel_flag"]].sort_values("day").groupby(keys, as_index=False).first()
    flagged[["car_id", "wheel", "car_type", "day", "site", "vertical_ratio", "ratio_trend"]].to_csv(
        args.out / "flagged_wheels.csv", index=False)
    readings.head(2000).to_csv(args.out / "sample_readings.csv", index=False)
    (args.out / "metrics.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

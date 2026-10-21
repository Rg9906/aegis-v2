"""
Publication figures from a campaign directory.

Writes to results/<name>/figures/:
  accuracy_over_time.png  - seed-mean accuracy per strategy vs mission day, SAA steps marked
  pareto_bandwidth.png    - time-averaged accuracy vs golden-store traffic per day
  fault_timeline.png      - bit flips per step (all bits vs exponent MSB)
  summary_table.csv       - the numbers behind the figures (table view)

Example:
    python experiments/make_figures.py --name main
"""

import argparse
import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aegis.reporting import STRATEGIES, load_seed_runs, pareto_points  # noqa: E402

INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def seed_mean(runs, attr):
    series = np.array([[t[attr] for t in r["telemetry"]] for r in runs.values()], dtype=np.float64)
    return series.mean(axis=0), series.min(axis=0), series.max(axis=0)


def fig_accuracy(runs, out_path):
    first = next(iter(runs.values()))["telemetry"]
    days = np.array([t["elapsed_days"] for t in first])
    fig, ax = plt.subplots(figsize=(9.5, 4.2), facecolor=SURFACE)
    style_axes(ax)
    for t in first:
        if t["in_saa"]:
            ax.axvspan(t["elapsed_days"] - 1 / 24, t["elapsed_days"], color="#efeeea", lw=0, zorder=0)
    for attr, label, color, _ in STRATEGIES:
        mean, lo, hi = seed_mean(runs, attr)
        ax.fill_between(days, lo, hi, color=color, alpha=0.12, lw=0)
        ax.plot(days, mean, color=color, lw=2, label=label)
    ax.set_xlabel("Mission day", color=INK_2)
    ax.set_ylabel("Top-1 accuracy (EuroSAT)", color=INK_2)
    ax.set_ylim(0, 1.0)
    ax.set_xlim(days[0], days[-1])
    ax.set_title(f"Accuracy under accumulated LEO upsets (mean of {len(runs)} seeds, band = min-max; grey = SAA)",
                 color=INK, fontsize=10, loc="left")
    # Five series: legend (outside the plot) + summary_table.csv as the table view
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def fig_pareto(points, out_path):
    fig, ax = plt.subplots(figsize=(8.5, 4.2), facecolor=SURFACE)
    style_axes(ax)
    scrub = sorted((p for p in points if p["family"] == "scrub"), key=lambda p: p["mb_per_day"])
    color_of = {label: c for _, label, c, _ in STRATEGIES}
    ax.plot([p["mb_per_day"] for p in scrub], [p["mean_accuracy"] for p in scrub],
            color=color_of["Periodic scrubbing (12 h)"], lw=2, marker="o", ms=7, label="Periodic scrubbing")
    for p in scrub:
        ax.annotate(f"{p['interval_h']:g} h", (p["mb_per_day"], p["mean_accuracy"]), xytext=(6, -10),
                    textcoords="offset points", fontsize=8, color=INK_2)
    for p in points:
        if p["family"] == "scrub":
            continue
        ax.plot(p["mb_per_day"], p["mean_accuracy"], marker="o" if p["family"] == "aegis" else "s", ms=9,
                color=color_of[p["strategy"]], ls="none", label=p["strategy"], markeredgecolor=SURFACE, mew=2)
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlabel("Golden-store traffic (MB / day, symlog)", color=INK_2)
    ax.set_ylabel("Time-averaged accuracy", color=INK_2)
    ax.set_title("Accuracy vs. repair bandwidth", color=INK, fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def fig_faults(runs, out_path):
    first = next(iter(runs.values()))["telemetry"]
    days = np.array([t["elapsed_days"] for t in first])
    fig, axes = plt.subplots(2, 1, figsize=(8, 3.8), sharex=True, facecolor=SURFACE)
    for ax, attr, title in ((axes[0], "bit_flips_step", "All bit flips per step"),
                            (axes[1], "exponent_msb_flips_step", "Exponent-MSB (bit 30) flips per step")):
        style_axes(ax)
        mean, _, _ = seed_mean(runs, attr)
        ax.bar(days, mean, width=1 / 24 * 0.8, color="#2a78d6", align="edge")
        ax.set_title(title + f" (mean of {len(runs)} seeds)", color=INK, fontsize=9, loc="left")
    axes[1].set_xlabel("Mission day", color=INK_2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def write_table(runs, points, out_path):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["strategy", "mean_accuracy", "mb_per_day"])
        for p in points:
            w.writerow([p["strategy"], round(p["mean_accuracy"], 4), round(p["mb_per_day"], 2)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="main")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    campaign_dir = os.path.join(args.results_dir, args.name)
    runs = load_seed_runs(campaign_dir)
    if not runs:
        raise SystemExit(f"No seed_*.json files in {campaign_dir}")
    fig_dir = os.path.join(campaign_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    points = pareto_points(runs)
    fig_accuracy(runs, os.path.join(fig_dir, "accuracy_over_time.png"))
    fig_pareto(points, os.path.join(fig_dir, "pareto_bandwidth.png"))
    fig_faults(runs, os.path.join(fig_dir, "fault_timeline.png"))
    write_table(runs, points, os.path.join(fig_dir, "summary_table.csv"))
    print(f"Wrote figures to {fig_dir}")


if __name__ == "__main__":
    main()

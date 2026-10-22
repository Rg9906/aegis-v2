"""
Vector figures for the Aegis-v2 paper, generated from the final campaigns.

Reads results/main_final (ResNet-18) and results/main_final_mobilenet (MobileNetV2) seed files and the
orbit/field export in frontend/public/data/main_final.json; writes PDFs to researchpaper/figures/.

    python researchpaper/make_paper_figures.py
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from aegis.reporting import STRATEGIES, SUMMARY_KEYS, load_seed_runs  # noqa: E402

OUT = os.path.join(ROOT, "researchpaper", "figures")
CAMPAIGNS = {"ResNet-18": "main_final", "MobileNetV2": "main_final_mobilenet"}
COLOR = {attr: light for attr, _, light, _ in STRATEGIES}
LABEL = {attr: label for attr, label, _, _ in STRATEGIES}
SHOWN = ["acc_aegis", "acc_aegis_audit", "acc_periodic_scrub", "acc_uplink", "acc_ranger", "acc_blink", "acc_unmitigated"]
SCRUB = "#eb6834"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "legend.fontsize": 6.8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.35,
        "lines.linewidth": 1.2,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)


def runs(name):
    r = load_seed_runs(os.path.join(ROOT, "results", name))
    if not r:
        sys.exit(f"no seed files in results/{name}")
    return r


def series(rs, key):
    """Mean, min and max of a telemetry column across seeds."""
    arr = np.array([[t[key] for t in r["telemetry"]] for r in rs.values()], dtype=float)
    return arr.mean(0), arr.min(0), arr.max(0)


def days(rs):
    return np.array([t["elapsed_days"] for t in next(iter(rs.values()))["telemetry"]])


def save(fig, name):
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)
    print("wrote", name)


def fig_accuracy(all_runs):
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.35), sharey=True)
    for ax, (arch, rs) in zip(axes, all_runs.items()):
        d = days(rs)
        for attr in SHOWN:
            m, lo, hi = series(rs, attr)
            z = 4 if attr in ("acc_aegis", "acc_aegis_audit") else 2
            ax.step(d, 100 * m, where="post", color=COLOR[attr], label=LABEL[attr], zorder=z,
                    lw=1.6 if attr == "acc_aegis" else 1.0, ls="--" if attr == "acc_aegis_audit" else "-")
            if attr == "acc_aegis":
                ax.fill_between(d, 100 * lo, 100 * hi, step="post", color=COLOR[attr], alpha=0.12, lw=0)
        clean = next(iter(rs.values()))["summary"]["clean_accuracy"]
        ax.axhline(100 * clean, color="0.35", lw=0.6, ls=":")
        ax.text(0.02, 100 * clean + 1.5, f"clean {100 * clean:.1f}%", fontsize=6.5, color="0.35")
        ax.set_title(f"{arch} (mean of {len(rs)} seeds)")
        ax.set_xlabel("Mission time (days)")
        ax.set_xlim(0, d[-1])
        ax.set_ylim(0, 105)
        ax.grid(True, axis="y")
    axes[0].set_ylabel("Top-1 accuracy (%)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.17))
    save(fig, "fig_accuracy.pdf")


def fig_pareto(all_runs):
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.3))
    for ax, (arch, rs) in zip(axes, all_runs.items()):
        n_days = next(iter(rs.values()))["summary"]["total_simulated_days"]
        sweep = {}
        for r in rs.values():
            for h, v in r["summary"]["scrub_sweep"].items():
                sweep.setdefault(float(h), []).append((v["mean_accuracy"], v["bytes_transferred"]))
        sweep[12.0] = [(r["summary"]["mean_accuracy"]["periodic_scrubbing"], r["summary"]["bytes_transferred"]["periodic_scrubbing"]) for r in rs.values()]
        hs = sorted(sweep)
        xs = [np.mean([b for _, b in sweep[h]]) / 1e6 / n_days for h in hs]
        ys = [100 * np.mean([a for a, _ in sweep[h]]) for h in hs]
        ax.plot(xs, ys, "-o", color=SCRUB, ms=3, lw=1.0, label="Periodic scrubbing (interval sweep)", zorder=2)
        for h, x, y in zip(hs, xs, ys):
            ax.annotate(f"{h:g} h", (x, y), textcoords="offset points", xytext=(3, -8), fontsize=6, color=SCRUB)
        for attr, key in (("acc_aegis", "aegis_v2"), ("acc_ranger_aegis", "ranger_aegis"), ("acc_aegis_audit", "aegis_audit"), ("acc_uplink", "ground_uplink")):
            acc = 100 * np.mean([r["summary"]["mean_accuracy"][key] for r in rs.values()])
            mb = np.mean([r["summary"]["bytes_transferred"][key] for r in rs.values()]) / 1e6 / n_days
            ax.scatter([max(mb, 0.5)], [acc], s=34 if attr == "acc_aegis" else 22, color=COLOR[attr], edgecolor="white",
                       lw=0.6, zorder=4, marker="D" if attr == "acc_uplink" else "o", label=LABEL[attr])
        for attr in ("acc_ranger", "acc_blink", "acc_unmitigated"):
            acc = 100 * np.mean([r["summary"]["mean_accuracy"][SUMMARY_KEYS[attr]] for r in rs.values()])
            ax.scatter([0.5], [acc], s=18, color=COLOR[attr], marker="s", zorder=3, label=f"{LABEL[attr]} (0 B)")
        ax.set_xscale("log")
        ax.set_xlim(0.35, 1500)
        ax.set_ylim(0, 105)
        ax.set_title(arch)
        ax.set_xlabel("Repair traffic (MB/day, log; 0 B plotted at 0.5)")
        ax.grid(True, which="major")
    axes[0].set_ylabel("Mean accuracy (%)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.2))
    save(fig, "fig_pareto.pdf")


def fig_costs(all_runs):
    keys = [("aegis_v2", "acc_aegis", "Aegis-v2"), ("ranger_aegis", "acc_ranger_aegis", "Ranger+Aegis"),
            ("aegis_audit", "acc_aegis_audit", "Aegis+audit"), ("periodic_scrubbing", "acc_periodic_scrub", "Scrub 12 h"),
            ("ground_uplink", "acc_uplink", "Uplink")]
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.05), sharey=True)
    y = np.arange(len(keys))
    w = 0.38
    for ax, metric, unit in ((axes[0], "bytes_transferred", "MB/day"), (axes[1], "energy_consumed_uj", "mJ/day")):
        for j, (arch, rs) in enumerate(all_runs.items()):
            n_days = next(iter(rs.values()))["summary"]["total_simulated_days"]
            scale = 1e6 if metric == "bytes_transferred" else 1e3
            vals = [np.mean([r["summary"][metric][k] for r in rs.values()]) / scale / n_days for k, _, _ in keys]
            ax.barh(y + (j - 0.5) * w, vals, height=w, color=[COLOR[a] for _, a, _ in keys],
                    alpha=1.0 if j == 0 else 0.5, hatch=None if j == 0 else "////", edgecolor="white", lw=0.3,
                    label=arch)
        ax.set_xlabel(unit)
        ax.grid(True, axis="x")
    axes[0].set_yticks(y, [lab for _, _, lab in keys])
    axes[0].invert_yaxis()
    axes[0].set_title("Repair traffic")
    axes[1].set_title("Energy")
    from matplotlib.patches import Patch

    fig.legend(handles=[Patch(facecolor="0.4", label="ResNet-18"), Patch(facecolor="0.75", hatch="////", label="MobileNetV2")],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.55, -0.19))
    save(fig, "fig_costs.pdf")


def fig_faults(rs):
    seed = sorted(rs)[1] if len(rs) > 1 else sorted(rs)[0]
    tel = rs[seed]["telemetry"]
    d = np.array([t["elapsed_days"] for t in tel])
    flips = np.array([t["bit_flips_step"] for t in tel])
    msb = np.array([t["exponent_msb_flips_step"] for t in tel])
    saa = np.array([t["saa_minutes_in_step"] for t in tel])
    fig, ax = plt.subplots(figsize=(3.5, 1.75))
    ax2 = ax.twinx()
    ax2.bar(d, saa, width=d[1] - d[0], color="#e2601e", alpha=0.16, align="edge", lw=0, label="SAA-core minutes in step")
    ax2.set_ylabel("SAA-core min/step", color="#c4521a")
    ax2.spines["right"].set_visible(True)
    ax2.tick_params(axis="y", colors="#c4521a")
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.bar(d, flips, width=(d[1] - d[0]) * 0.8, align="edge", color="0.35", label="bit flips per step")
    for x in d[msb > 0]:
        ax.plot([x + 0.02], [flips[d == x][0] + 1.5], marker="v", color="#d93a3a", ms=4)
    restores = [t["elapsed_days"] for t in tel if t["aegis_restored_blocks"]]
    for x in restores:
        ax.axvline(x + 0.02, color=COLOR["acc_aegis"], lw=0.6, ls=(0, (2, 2)), alpha=0.7)
    ax.set_xlabel(f"Mission time (days), ResNet-18 seed {seed}")
    ax.set_ylabel("Bit flips / step")
    ax.set_xlim(0, d[-1] + d[1] - d[0])
    ax.plot([], [], "v", color="#d93a3a", ms=4, label="exponent-MSB hit")
    ax.plot([], [], "--", color=COLOR["acc_aegis"], lw=0.7, label="Aegis restore")
    ax.plot([], [], "s", color="#e2601e", alpha=0.3, ms=5, label="SAA-core minutes")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, ncol=4, fontsize=6, handletextpad=0.3, columnspacing=0.8)
    ax.set_ylim(0, max(flips) * 1.15)
    save(fig, "fig_faults.pdf")


def fig_orbit():
    path = os.path.join(ROOT, "frontend", "public", "data", "main_final.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    track, field = data.get("track"), data.get("field")
    if not track or not field:
        print("skip fig_orbit: no track/field in export")
        return
    fig, ax = plt.subplots(figsize=(3.5, 1.95))
    lons, lats = np.array(field["lons"]), np.array(field["lats"])
    b = np.array(field["b_nt"]) / 1000.0
    cs = ax.contourf(lons, lats, b, levels=np.arange(18, 62, 4), cmap="Blues_r", alpha=0.85, extend="both")
    ax.contour(lons, lats, b, levels=[field["core_nt"] / 1000, field["edge_nt"] / 1000], colors=["#c4521a", "#e2601e"],
               linewidths=[1.0, 0.7], linestyles=["-", "--"])
    lon = np.array(track["lon"])
    lat = np.array(track["lat"])
    n = min(len(lon), int(1440 / max(1, np.median(np.diff(track["minutes"])))))
    lon, lat = lon[:n], lat[:n]
    breaks = np.where(np.abs(np.diff(lon)) > 180)[0] + 1
    for seg_lon, seg_lat in zip(np.split(lon, breaks), np.split(lat, breaks)):
        ax.plot(seg_lon, seg_lat, color="0.15", lw=0.45, alpha=0.8)
    cb = fig.colorbar(cs, ax=ax, pad=0.015, fraction=0.04)
    cb.set_label("|B| at altitude ($\\mu$T)", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xticks(range(-180, 181, 60))
    ax.set_yticks(range(-90, 91, 45))
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    save(fig, "fig_orbit.pdf")


def main():
    os.makedirs(OUT, exist_ok=True)
    all_runs = {arch: runs(name) for arch, name in CAMPAIGNS.items()}
    fig_accuracy(all_runs)
    fig_pareto(all_runs)
    fig_costs(all_runs)
    fig_faults(all_runs["ResNet-18"])
    fig_orbit()


if __name__ == "__main__":
    main()

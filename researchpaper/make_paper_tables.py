"""
Tables (LaTeX fragments) and the secondary figures for the Aegis-v2 paper.

Every number in researchpaper/tables/*.tex is computed here from the result files, so the paper never
carries a hand-copied value. Inputs:
  results/main_final, results/main_final_mobilenet   final campaigns (seed_*.json)
  results/main_gpu                                    pre-fix campaign on the same seeds (ablation)
  frontend/public/data/main_final*.json               per-step fault records and the orbit/field export
  results/detection_study_*.json                      single-fault detection study (optional)
  results/transient_faults.json                       transient activation-fault study (optional)

    python researchpaper/make_paper_tables.py
"""

import json
import os
import sys
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "researchpaper"))
from aegis.reporting import STRATEGIES, load_seed_runs  # noqa: E402
from make_paper_figures import COLOR, LABEL, save  # noqa: E402  (also applies the shared rcParams)

TAB = os.path.join(ROOT, "researchpaper", "tables")
ARCHES = {"ResNet-18": "main_final", "MobileNetV2": "main_final_mobilenet"}
KEYS = [
    ("aegis_v2", "acc_aegis", r"\aegis{}"),
    ("aegis_audit", "acc_aegis_audit", r"\aegis{} + audit"),
    ("ranger_aegis", "acc_ranger_aegis", r"Ranger + \aegis{}"),
    ("periodic_scrubbing", "acc_periodic_scrub", "Scrubbing (12 h)"),
    ("ground_uplink", "acc_uplink", "Ground uplink"),
    ("ranger_clipping", "acc_ranger", "Ranger clipping"),
    ("blink_recentering", "acc_blink", "BLINK re-centering"),
    ("unmitigated", "acc_unmitigated", "Unmitigated"),
]
FIELD_OF_BIT = lambda b: "sign" if b == 31 else ("exponent" if b >= 23 else "mantissa")  # noqa: E731
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}


def mean_ci(v):
    v = np.asarray(v, dtype=float)
    if len(v) < 2:
        return float(v.mean()), 0.0
    return float(v.mean()), float(T975.get(len(v) - 1, 1.96) * v.std(ddof=1) / np.sqrt(len(v)))


def write(name, body):
    with open(os.path.join(TAB, name), "w", encoding="utf-8") as f:
        f.write(body.strip() + "\n")
    print("wrote tables/" + name)


def load_export(name):
    with open(os.path.join(ROOT, "frontend", "public", "data", f"{name}.json"), encoding="utf-8") as f:
        return json.load(f)


def severity(h):
    """Same rule as the dossier (frontend/src/lib/data.ts)."""
    to = h["to"]
    if isinstance(to, str):
        return "critical"
    if h["p"].endswith("running_var") and to < 0:
        return "critical"
    ratio = abs(to) / max(abs(h["from"]), 1e-12)
    if h["bit"] == 30 or ratio > 1e3:
        return "critical"
    if ratio > 3 or ratio < 1 / 3:
        return "moderate"
    return "minor"


# ------------------------------------------------------------------ tables from the campaigns


def tab_perseed(runs):
    lines = []
    for arch, rs in runs.items():
        seeds = sorted(rs)
        lines.append(r"\multicolumn{%d}{@{}l}{\textit{%s}} \\" % (len(seeds) + 1, arch))
        for key, _, label in KEYS:
            vals = [100 * rs[s]["summary"]["mean_accuracy"][key] for s in seeds]
            fin = [100 * rs[s]["summary"]["final_accuracy"][key] for s in seeds]
            cells = " & ".join(f"{v:.1f} ({f:.0f})" for v, f in zip(vals, fin))
            lines.append(f"{label} & {cells} \\\\")
        lines.append(r"\addlinespace")
    write("tab_perseed.tex", "\n".join(lines[:-1]))


def tab_availability(runs):
    """Fraction of mission steps within 5 points of clean accuracy, and accuracy-loss area."""
    lines = []
    for key, attr, label in KEYS:
        cells = []
        for arch, rs in runs.items():
            avail, loss = [], []
            for r in rs.values():
                clean = r["summary"]["clean_accuracy"]
                acc = np.array([t[attr] for t in r["telemetry"]])
                avail.append(100 * np.mean(acc >= clean - 0.05))
                loss.append(100 * np.mean(np.clip(clean - acc, 0, None)) * r["summary"]["total_simulated_days"] * 24)
            m, c = mean_ci(avail)
            lm, _ = mean_ci(loss)
            cells.append(f"{m:.0f}\\pmci{{{c:.0f}}} & {lm:.0f}")
        lines.append(f"{label} & {' & '.join(cells)} \\\\")
    write("tab_availability.tex", "\n".join(lines))


def tab_repairs(runs):
    rows = []
    for arch, rs in runs.items():
        s = [r["summary"] for r in rs.values()]
        tel = [r["telemetry"] for r in rs.values()]
        days = s[0]["total_simulated_days"]
        steps = len(tel[0])
        ev = [x["aegis_restorations_count"] for x in s]
        blocks = [sum(len(t["aegis_restored_blocks"]) for t in tt) for tt in tel]
        mb = [x["bytes_transferred"]["aegis_v2"] / 1e6 for x in s]
        rows.append((arch, {
            "Restore events / seed": f"{np.mean(ev):.1f}",
            "Block restores / seed": f"{np.mean(blocks):.1f}",
            "Steps with a restore (\\%)": f"{100 * np.mean(ev) / steps:.0f}",
            "Mean MB per restore event": f"{np.sum(mb) / max(np.sum(ev), 1):.1f}",
            "Ranger + \\aegis{} restore events / seed": f"{np.mean([x['ranger_aegis_restorations_count'] for x in s]):.1f}",
            "Audits / seed (6 h)": f"{np.mean([x['aegis_audits_executed'] for x in s]):.0f}",
            "Full scrubs / seed (12 h)": f"{np.mean([x['scrubs_executed'] for x in s]):.0f}",
            "Uplink contacts used / seed": f"{np.mean([x['ground_uplink_contacts_used'] for x in s]):.1f}",
            "Golden store (MB, zstd)": f"{s[0]['golden_store_mb']['aegis']:.2f}",
            "Restore events per day": f"{np.mean(ev) / days:.1f}",
        }))
    keys = list(rows[0][1])
    write("tab_repairs.tex", "\n".join(f"{k} & {rows[0][1][k]} & {rows[1][1][k]} \\\\" for k in keys))


def tab_faults(runs, exports):
    rows = []
    for (arch, rs), ex in zip(runs.items(), exports):
        s = [r["summary"] for r in rs.values()]
        tel = [r["telemetry"] for r in rs.values()]
        days = s[0]["total_simulated_days"]
        flips = [x["total_faults_accumulated"] / days for x in s]
        msb = [x["exponent_msb_flips_total"] / days for x in s]
        mbu = [sum(t["mbus_injected_step"] for t in tt) for tt in tel]
        events = [sum(t["sbus_injected_step"] + t["mbus_injected_step"] for t in tt) for tt in tel]
        in_saa = [sum(t["bit_flips_step"] for t in tt if t["saa_minutes_in_step"] > 0) /
                  max(sum(t["bit_flips_step"] for t in tt), 1) for tt in tel]
        hits = [h for k in ex["seeds"] for step in ex["seeds"][k]["faults"] for h in step]
        fields = Counter(FIELD_OF_BIT(h["bit"]) for h in hits)
        sev = Counter(severity(h) for h in hits)
        bits = 8 * sum(b["bytes"] for b in ex["blocks"])
        n = len(hits)
        m, c = mean_ci(flips)
        mm, mc = mean_ci(msb)
        rows.append({
            "Stored bits (Mbit)": f"{bits / 1e6:.1f}",
            "Predicted flips/day ($B\\bar\\lambda$)": f"{bits * 4.04e-7:.1f}",
            "Recorded flips/day": f"{m:.1f}\\pmci{{{c:.1f}}}",
            "Exponent-MSB flips/day": f"{mm:.2f}\\pmci{{{mc:.2f}}}",
            "MBU events (\\% of events)": f"{100 * np.sum(mbu) / max(np.sum(events), 1):.1f}",
            "Flips in steps with SAA exposure (\\%)": f"{100 * np.mean(in_saa):.0f}",
            "Mantissa / exponent / sign (\\%)": f"{100 * fields['mantissa'] / n:.1f} / {100 * fields['exponent'] / n:.1f} / {100 * fields['sign'] / n:.1f}",
            "Destructive / moderate / negligible (\\%)": f"{100 * sev['critical'] / n:.1f} / {100 * sev['moderate'] / n:.1f} / {100 * sev['minor'] / n:.1f}",
            "Total recorded hits (3 seeds)": f"{n:,}",
        })
    write("tab_faults.tex", "\n".join(f"{k} & {rows[0][k]} & {rows[1][k]} \\\\" for k in rows[0]))


def tab_blocks(exports):
    ex = exports[0]
    hits = [h for k in ex["seeds"] for step in ex["seeds"][k]["faults"] for h in step]
    hit_count = Counter(h["b"] for h in hits)
    crit_count = Counter(h["b"] for h in hits if severity(h) == "critical")
    restores = Counter(b for k in ex["seeds"] for t in ex["seeds"][k]["telemetry"] for b in t["aegis_restored_blocks"])
    total = sum(b["bytes"] for b in ex["blocks"])
    names = {"stem": "stem", "classifier": "fc"}
    lines = []
    for b in ex["blocks"]:
        bid = b["id"]
        name = names.get(bid, bid.replace("layer", "layer").replace("_block", "."))
        lines.append(f"\\texttt{{{name}}} & {b['bytes'] / 1e6:.2f} & {100 * b['bytes'] / total:.1f} & "
                     f"{hit_count[bid]} & {crit_count[bid]} & {restores[bid]} \\\\")
    write("tab_blocks.tex", "\n".join(lines))


def tab_ablation():
    pre = load_seed_runs(os.path.join(ROOT, "results", "main_gpu"))
    post = load_seed_runs(os.path.join(ROOT, "results", "main_final"))
    if not pre:
        print("skip ablation: results/main_gpu missing")
        return
    lines = []
    for s in sorted(post):
        a, b = pre[s]["summary"], post[s]["summary"]
        same = a["total_faults_accumulated"] == b["total_faults_accumulated"]
        lines.append(
            f"{s} & {a['total_faults_accumulated']}{'' if same else '*'} & "
            f"{100 * a['mean_accuracy']['aegis_v2']:.1f} & {100 * b['mean_accuracy']['aegis_v2']:.1f} & "
            f"{100 * a['mean_accuracy']['ranger_aegis']:.1f} & {100 * b['mean_accuracy']['ranger_aegis']:.1f} & "
            f"{100 * a['final_accuracy']['ranger_aegis']:.1f} & {100 * b['final_accuracy']['ranger_aegis']:.1f} \\\\")
    ma = mean_ci([100 * pre[s]["summary"]["mean_accuracy"]["ranger_aegis"] for s in sorted(post)])
    mb = mean_ci([100 * post[s]["summary"]["mean_accuracy"]["ranger_aegis"] for s in sorted(post)])
    aa = mean_ci([100 * pre[s]["summary"]["mean_accuracy"]["aegis_v2"] for s in sorted(post)])
    ab = mean_ci([100 * post[s]["summary"]["mean_accuracy"]["aegis_v2"] for s in sorted(post)])
    lines.append(r"\midrule")
    lines.append(f"mean & -- & {aa[0]:.1f} & {ab[0]:.1f} & {ma[0]:.1f} & {mb[0]:.1f} & -- & -- \\\\")
    write("tab_ablation.tex", "\n".join(lines))


def tab_energy(runs):
    lines = []
    for arch, rs in runs.items():
        s = [r["summary"]["energy_consumed_uj"] for r in rs.values()]
        days = next(iter(rs.values()))["summary"]["total_simulated_days"]
        tot = np.mean([x["aegis_v2"] for x in s]) / 1e3 / days
        mon = np.mean([x["aegis_v2_monitor_share"] for x in s]) / 1e3 / days
        hook = np.mean([x["aegis_v2_monitor_compute_share"] for x in s]) / 1e3 / days
        crc = mon - hook
        rest = tot - mon
        lines.append(f"{arch} & {rest:.1f} ({100 * rest / tot:.0f}\\%) & {crc:.1f} ({100 * crc / tot:.0f}\\%) & "
                     f"{hook:.1f} ({100 * hook / tot:.0f}\\%) & {tot:.1f} \\\\")
    write("tab_energy.tex", "\n".join(lines))


# ------------------------------------------------------------------ figures from the campaigns


def fig_traffic_time(runs):
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.0))
    cols = [("bytes_aegis", "acc_aegis"), ("bytes_aegis_audit", "acc_aegis_audit"),
            ("bytes_ranger_aegis", "acc_ranger_aegis"), ("bytes_scrub", "acc_periodic_scrub"), ("bytes_uplink", "acc_uplink")]
    for ax, (arch, rs) in zip(axes, runs.items()):
        d = np.array([t["elapsed_days"] for t in next(iter(rs.values()))["telemetry"]])
        for col, attr in cols:
            arr = np.array([[t[col] for t in r["telemetry"]] for r in rs.values()], dtype=float) / 1e6
            ax.step(d, arr.mean(0), where="post", color=COLOR[attr], label=LABEL[attr],
                    ls="--" if attr == "acc_aegis_audit" else "-", lw=1.4 if attr == "acc_aegis" else 1.0)
        ax.set_title(arch)
        ax.set_xlabel("Mission time (days)")
        ax.set_xlim(0, d[-1])
        ax.grid(True, axis="y")
    axes[0].set_ylabel("Cumulative repair traffic (MB)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.2))
    save(fig, "fig_traffic_time.pdf")


def fig_seeds(runs):
    rs = runs["ResNet-18"]
    seeds = sorted(rs)
    fig, axes = plt.subplots(1, len(seeds), figsize=(7.16, 1.9), sharey=True)
    attrs = ["acc_aegis", "acc_aegis_audit", "acc_periodic_scrub", "acc_uplink", "acc_ranger", "acc_unmitigated"]
    for ax, s in zip(axes, seeds):
        tel = rs[s]["telemetry"]
        d = np.array([t["elapsed_days"] for t in tel])
        for attr in attrs:
            ax.step(d, [100 * t[attr] for t in tel], where="post", color=COLOR[attr], label=LABEL[attr],
                    lw=1.5 if attr == "acc_aegis" else 0.9, ls="--" if attr == "acc_aegis_audit" else "-",
                    zorder=4 if attr.startswith("acc_aegis") else 2)
        for t in tel:
            if t["aegis_restored_blocks"]:
                ax.axvline(t["elapsed_days"], color=COLOR["acc_aegis"], lw=0.35, alpha=0.35, zorder=1)
        ax.set_title(f"ResNet-18, seed {s}")
        ax.set_xlabel("Days")
        ax.set_xlim(0, d[-1])
        ax.set_ylim(0, 100)
        ax.grid(True, axis="y")
    axes[0].set_ylabel("Accuracy (%)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.22))
    save(fig, "fig_seeds.pdf")


def fig_rate(export):
    """Upset rate along one day of the real track: |B| along track mapped through the ramp of Eq. (1)."""
    tr, fld = export["track"], export["field"]
    lons, lats, b = np.array(fld["lons"]), np.array(fld["lats"]), np.array(fld["b_nt"])
    tmin = np.array(tr["minutes"])
    sel = tmin < 1440
    lo, la = np.array(tr["lon"])[sel], np.array(tr["lat"])[sel]
    ix = np.clip(np.searchsorted(lons, lo), 1, len(lons) - 1)
    iy = np.clip(np.searchsorted(lats, la), 1, len(lats) - 1)
    # bilinear interpolation on the regular grid
    x0, x1, y0, y1 = lons[ix - 1], lons[ix], lats[iy - 1], lats[iy]
    wx, wy = (lo - x0) / (x1 - x0), (la - y0) / (y1 - y0)
    bb = (b[iy - 1, ix - 1] * (1 - wx) * (1 - wy) + b[iy - 1, ix] * wx * (1 - wy) +
          b[iy, ix - 1] * (1 - wx) * wy + b[iy, ix] * wx * wy)
    bg, core = export["setup"]["rates"]["background"], export["setup"]["rates"]["saa_core"]
    frac = np.clip((fld["edge_nt"] - bb) / (fld["edge_nt"] - fld["core_nt"]), 0, 1)
    lam = bg + (core - bg) * frac
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.16, 1.85), gridspec_kw={"width_ratios": [1, 2.6]})
    bx = np.linspace(16000, 22000, 200)
    ax0.plot(bx / 1000, np.where(bx <= fld["core_nt"], core, np.where(bx >= fld["edge_nt"], bg,
             bg + (core - bg) * (fld["edge_nt"] - bx) / (fld["edge_nt"] - fld["core_nt"]))), color="#c4521a")
    ax0.set_yscale("log")
    ax0.set_xlabel("|B| ($\\mu$T)")
    ax0.set_ylabel("Upsets / bit / day")
    ax0.set_title("Rate ramp, Eq. (1)")
    ax0.grid(True)
    ax1.fill_between(tmin[sel] / 60, lam, bg, color="#e2601e", alpha=0.35, lw=0)
    ax1.plot(tmin[sel] / 60, lam, color="#c4521a", lw=0.9)
    ax1.axhline(4.04e-7, color="0.3", ls=":", lw=0.8)
    ax1.text(0.3, 4.6e-7, "orbit mean $4.04\\times10^{-7}$ (Alsat-1)", fontsize=6.5, color="0.3")
    ax1.set_yscale("log")
    ax1.set_xlim(0, 24)
    ax1.set_xticks(range(0, 25, 3))
    ax1.set_xlabel("Hour of mission day 1 (Sentinel-2A, SGP4 + IGRF)")
    ax1.set_title("Upset rate along the track")
    ax1.grid(True, axis="y")
    save(fig, "fig_rate.pdf")


def fig_energy_breakdown(runs):
    fig, ax = plt.subplots(figsize=(3.5, 1.45))
    labels, parts = [], []
    for arch, rs in runs.items():
        s = [r["summary"]["energy_consumed_uj"] for r in rs.values()]
        days = next(iter(rs.values()))["summary"]["total_simulated_days"]
        tot = np.mean([x["aegis_v2"] for x in s]) / 1e3 / days
        mon = np.mean([x["aegis_v2_monitor_share"] for x in s]) / 1e3 / days
        hook = np.mean([x["aegis_v2_monitor_compute_share"] for x in s]) / 1e3 / days
        scrub = np.mean([x["periodic_scrubbing"] for x in s]) / 1e3 / days
        labels += [f"{arch}\nAegis-v2", f"{arch}\nscrub 12 h"]
        parts += [(tot - mon, mon - hook, hook), (scrub, 0, 0)]
    y = np.arange(len(labels))
    left = np.zeros(len(labels))
    for i, (name, col) in enumerate([("restore (flash read + DRAM write)", "#2a78d6"),
                                     ("T3 CRC reads", "#9085e9"), ("T1/T2 hook compute", "#e34948")]):
        vals = np.array([p[i] for p in parts])
        ax.barh(y, vals, left=left, color=col, label=name, height=0.6)
        left += vals
    ax.set_yticks(y, labels, fontsize=6.3)
    ax.invert_yaxis()
    ax.set_xlabel("Energy (mJ/day)")
    ax.grid(True, axis="x")
    ax.legend(loc="lower right", frameon=False, fontsize=6)
    save(fig, "fig_energy_breakdown.pdf")


def fig_ablation():
    pre = load_seed_runs(os.path.join(ROOT, "results", "main_gpu"))
    post = load_seed_runs(os.path.join(ROOT, "results", "main_final"))
    if not pre:
        return
    fig, ax = plt.subplots(figsize=(3.5, 1.7))
    tel_pre, tel_post = pre[1]["telemetry"], post[1]["telemetry"]
    d = np.array([t["elapsed_days"] for t in tel_post]) * 24
    for tel, attr, ls, lab in ((tel_pre, "acc_ranger_aegis", "-", "Ranger + Aegis-v2, no upstream rule"),
                               (tel_pre, "acc_aegis", "--", "Aegis-v2, no upstream rule"),
                               (tel_post, "acc_ranger_aegis", "-", "with upstream rule (both)")):
        col = "#d93a3a" if tel is tel_pre else COLOR["acc_aegis"]
        ax.step(d, [100 * t[attr] for t in tel], where="post", color=col, ls=ls, lw=1.6 if tel is tel_post else 1.1, label=lab,
                zorder=3 if tel is tel_post else 2)
    ax.set_xlabel("Mission time (hours), ResNet-18 seed 1")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlim(0, d[-1])
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=6)
    save(fig, "fig_ablation.pdf")


# ------------------------------------------------------------------ detection study


def detection(arch_key):
    path = os.path.join(ROOT, "results", f"detection_study_{arch_key}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


FIELD_LABEL = {"exp_msb": "Exponent MSB (30)", "exponent": "Exponent (23--29)", "sign": "Sign (31)",
               "mantissa_hi": "Mantissa high (16--22)", "mantissa_lo": "Mantissa low (0--15)"}


def tab_detection(studies):
    lines = []
    for arch, st in studies.items():
        if st is None:
            continue
        chip = 1.0 / st["summary"]["eval_chips"]
        lines.append(r"\multicolumn{8}{@{}l}{\textit{%s} (clean %.1f\%%, %d trials)} \\" % (
            arch, 100 * st["summary"]["clean_accuracy"], len(st["trials"])))
        for field, lab in FIELD_LABEL.items():
            r = [x for x in st["trials"] if x["field"] == field]
            if not r:
                continue
            dmg = [x for x in r if x["drop"] > chip + 1e-9]
            alarm = [x for x in r if x["t1"] or x["t2"]]
            rec = f"{sum(x['t1'] or x['t2'] for x in dmg)}/{len(dmg)}" if dmg else "--"
            loc = f"{100 * sum(x['localized'] for x in alarm) / len(alarm):.0f}" if alarm else "--"
            lines.append(
                f"{lab} & {len(dmg)}/{len(r)} & {100 * np.mean([x['drop'] for x in r]):.1f} & "
                f"{100 * sum(x['t1'] for x in r) / len(r):.0f} & {100 * sum(x['t2'] for x in r) / len(r):.0f} & "
                f"{rec} & {loc} & {sum(len(set(x['confirmed']) - {x['block']}) for x in r)} \\\\")
        lines.append(r"\addlinespace")
    if lines:
        write("tab_detection.tex", "\n".join(lines[:-1]))


def fig_detection(studies):
    avail = {a: s for a, s in studies.items() if s}
    if not avail:
        return
    fig, axes = plt.subplots(1, len(avail), figsize=(3.5 * len(avail) if len(avail) > 1 else 3.5, 2.0), squeeze=False)
    colors = {"exp_msb": "#d93a3a", "exponent": "#2a78d6", "sign": "#e87ba4", "mantissa_hi": "#1baf7a", "mantissa_lo": "#8a8a8a"}
    for ax, (arch, st) in zip(axes[0], avail.items()):
        rng = np.random.default_rng(0)
        for field in FIELD_LABEL:
            r = [x for x in st["trials"] if x["field"] == field]
            if not r:
                continue
            for x in r:
                y = 100 * x["drop"] + rng.uniform(-0.6, 0.6)
                det = x["t1"] or x["t2"]
                ax.scatter([list(FIELD_LABEL).index(field) + rng.uniform(-0.22, 0.22)], [y], s=12,
                           marker="o" if det else "x", color=colors[field],
                           facecolor=colors[field] if det else colors[field], lw=0.8, alpha=0.85)
        ax.set_xticks(range(len(FIELD_LABEL)), ["exp\nMSB", "exp\n23-29", "sign", "mant\n16-22", "mant\n0-15"], fontsize=6.5)
        ax.set_ylabel("Accuracy loss (pts)")
        ax.set_title(f"{arch}: one persistent upset")
        ax.grid(True, axis="y")
    axes[0][0].scatter([], [], marker="o", color="0.3", s=12, label="behavioral alarm")
    axes[0][0].scatter([], [], marker="x", color="0.3", s=12, label="no alarm")
    axes[0][0].legend(loc="upper right", frameon=False, fontsize=6)
    save(fig, "fig_detection.pdf")


def fig_confusion(studies):
    st = studies.get("ResNet-18")
    if not st:
        return
    classes = st["classes"]
    n = len(classes)
    chip = 1.0 / st["summary"]["eval_chips"]
    dmg = [x for x in st["trials"] if x["drop"] > 0.2]
    if not dmg:
        dmg = [x for x in st["trials"] if x["drop"] > chip]
    if not dmg:
        return
    cm = np.sum([np.array(x["confusion"]).reshape(n, n) for x in dmg], axis=0).astype(float)
    cm = cm / cm.sum(1, keepdims=True)
    collapse = Counter(classes[int(np.argmax(x["pred_hist"]))] for x in dmg)
    # Abbreviate the checkpoint's own label order (not alphabetical: see the label-order pitfall)
    abbrev = {"AnnualCrop": "Ann.Crop", "HerbaceousVegetation": "Herb.Veg", "Industrial": "Industr.",
              "PermanentCrop": "Perm.Crop", "Residential": "Resid."}
    short = [abbrev.get(c, c) for c in classes]
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.16, 2.55), gridspec_kw={"width_ratios": [1.3, 1], "wspace": 0.55})
    im = ax0.imshow(100 * cm, cmap="Reds", vmin=0, vmax=100)
    ax0.set_xticks(range(n), short, rotation=55, ha="right", fontsize=6)
    ax0.set_yticks(range(n), short, fontsize=6)
    ax0.set_xlabel("Predicted")
    ax0.set_ylabel("True")
    ax0.set_title(f"Confusion under damaging faults ({len(dmg)} trials, loss > 20 pts)")
    cb = fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.02)
    cb.set_label("% of true class", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    names = [c for c, _ in collapse.most_common()]
    short_of = dict(zip(classes, short))
    ax1.barh(range(len(names)), [collapse[c] for c in names], color="#d93a3a")
    ax1.set_yticks(range(len(names)), [short_of[c] for c in names], fontsize=6.5)
    ax1.xaxis.get_major_locator().set_params(integer=True)
    ax1.invert_yaxis()
    ax1.set_xlabel("Damaging trials")
    ax1.set_title("Most-predicted class after the fault")
    ax1.grid(True, axis="x")
    save(fig, "fig_confusion.pdf")
    with open(os.path.join(TAB, "confusion_stats.json"), "w", encoding="utf-8") as f:
        json.dump({"damaging_trials": len(dmg), "collapse": collapse.most_common(),
                   "diag_mean": float(np.mean(np.diag(cm)))}, f, indent=1)


def tab_transient():
    path = os.path.join(ROOT, "results", "transient_faults.json")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        s = json.load(f)["summary"]
    write("tab_transient.tex", (
        f"Trials (bit {s['bit']} flipped in one activation) & {s['trials']} \\\\\n"
        f"Behavioral alarm (T1 or T2) & {100 * s['behavioral_alarm_rate']:.1f}\\% \\\\\n"
        f"Spurious restores (T3 confirmed a block) & {100 * s['spurious_restore_rate']:.1f}\\% \\\\\n"
        f"Batch accuracy: clean / faulty / faulty + Ranger & {100 * s['mean_acc_clean']:.1f} / "
        f"{100 * s['mean_acc_faulty']:.1f} / {100 * s['mean_acc_faulty_ranger']:.1f}\\% \\\\"))


def main():
    os.makedirs(TAB, exist_ok=True)
    runs = {arch: load_seed_runs(os.path.join(ROOT, "results", name)) for arch, name in ARCHES.items()}
    exports = [load_export(name) for name in ARCHES.values()]
    tab_perseed(runs)
    tab_availability(runs)
    tab_repairs(runs)
    tab_faults(runs, exports)
    tab_blocks(exports)
    tab_ablation()
    tab_energy(runs)
    tab_transient()
    fig_traffic_time(runs)
    fig_seeds(runs)
    fig_rate(exports[0])
    fig_energy_breakdown(runs)
    fig_ablation()
    studies = {"ResNet-18": detection("resnet18"), "MobileNetV2": detection("mobilenet_v2")}
    tab_detection(studies)
    fig_detection(studies)
    fig_confusion(studies)


if __name__ == "__main__":
    main()

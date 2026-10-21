"""
Statistical results report from a campaign directory.

Writes results/<name>/report.md with:
  - setup (orbit, days, seeds, faults, SAA exposure, clean accuracy),
  - per-strategy accuracy as seed mean with a 95% t-interval, traffic and energy per day,
  - paired per-seed accuracy differences Aegis-v2 minus each baseline,
  - the bandwidth verdict: is Aegis-v2 Pareto-dominated by any scrubbing interval, and how much
    traffic scrubbing needs (interpolated along the sweep) to match Aegis-v2 accuracy.

Campaigns produced before T1/T2 hook compute was charged get it added analytically (the per-step
op count is a deterministic function of the architecture, input size and monitor batch size).

Example:
    python experiments/write_report.py --name main
"""

import argparse
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aegis.config import DEFAULT_BUDGET_CONFIG, MODEL_INPUT_SIZE  # noqa: E402
from aegis.reporting import load_seed_runs, pareto_points  # noqa: E402

# Two-sided 95% Student-t critical values by degrees of freedom
T_CRIT_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}

ACC_ROWS: Tuple[Tuple[str, str], ...] = (
    ("aegis_v2", "Aegis-v2 selective restore"),
    ("aegis_audit", "Aegis-v2 + CRC audit"),
    ("ranger_aegis", "Ranger + Aegis-v2"),
    ("periodic_scrubbing", "Periodic scrubbing (12 h)"),
    ("ground_uplink", "Ground uplink reload"),
    ("ranger_clipping", "Ranger clipping"),
    ("blink_recentering", "BLINK re-centering"),
    ("unmitigated", "Unmitigated"),
)
MONITORED = ("aegis_v2", "ranger_aegis", "ground_uplink", "aegis_audit")


def mean_ci(values: Sequence[float]) -> Tuple[float, float]:
    """Seed mean and 95% t-interval half-width (0 for a single seed)."""
    v = np.asarray(values, dtype=np.float64)
    if v.size < 2:
        return float(v.mean()), 0.0
    t = T_CRIT_95.get(v.size - 1, 1.96)
    return float(v.mean()), float(t * v.std(ddof=1) / np.sqrt(v.size))


def monitor_ops_per_batch(arch: str = "resnet18", batch_size: int = 16, input_size: int = MODEL_INPUT_SIZE) -> int:
    """T1 (4 passes per Conv/Linear output) + T2 (3 passes per BN input) ops for one monitored batch."""
    import torch

    from aegis.models.factory import build_mobilenet_v2

    if arch == "mobilenet_v2":
        net = build_mobilenet_v2().eval()
    else:
        import torchvision.models as models

        net = models.resnet18(weights=None, num_classes=10).eval()
    ops = {"n": 0}

    def hook(module, inp, out):
        ops["n"] += 3 * inp[0].numel() if isinstance(module, torch.nn.BatchNorm2d) else 4 * out.numel()

    for m in net.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.BatchNorm2d)):
            m.register_forward_hook(hook)
    with torch.no_grad():
        net(torch.zeros(batch_size, 3, input_size, input_size))
    return ops["n"]


def energy_per_day_mj(runs: Dict[int, dict]) -> Tuple[Dict[str, Tuple[float, float]], bool]:
    """Energy per simulated day per repairing strategy; adds hook compute if the run predates it."""
    summaries = [r["summary"] for r in runs.values()]
    added = "aegis_v2_monitor_compute_share" not in summaries[0]["energy_consumed_uj"]
    per_step_uj = 0.0
    if added:
        arch = summaries[0].get("arch", "resnet18")
        per_step_uj = monitor_ops_per_batch(arch) * DEFAULT_BUDGET_CONFIG.monitor_energy_pj_per_op / 1e6
    out = {}
    keys = ["aegis_v2", "ranger_aegis", "ground_uplink", "periodic_scrubbing", "aegis_audit"]
    for key in keys:
        if key not in summaries[0]["energy_consumed_uj"]:
            continue
        vals = []
        for run in runs.values():
            s = run["summary"]
            uj = s["energy_consumed_uj"][key]
            if added and key in MONITORED:
                uj += per_step_uj * len(run["telemetry"])
            vals.append(uj / 1000.0 / s["total_simulated_days"])
        out[key] = mean_ci(vals)
    for hours in summaries[0].get("scrub_sweep", {}):
        out[f"scrub_{hours}h"] = mean_ci(
            [s["scrub_sweep"][hours]["energy_consumed_uj"] / 1000.0 / s["total_simulated_days"] for s in summaries])
    return out, added


def scrub_bandwidth_to_match(points: List[Dict[str, object]], target_acc: float) -> Optional[float]:
    """
    MB/day scrubbing needs to reach `target_acc`, by linear interpolation along the scrub sweep
    (sorted by traffic). None if no interval reaches it.
    """
    scrub = sorted((p for p in points if p["family"] == "scrub"), key=lambda p: p["mb_per_day"])
    if not scrub:
        return None
    if scrub[0]["mean_accuracy"] >= target_acc:
        return float(scrub[0]["mb_per_day"])
    for lo, hi in zip(scrub, scrub[1:]):
        if lo["mean_accuracy"] < target_acc <= hi["mean_accuracy"]:
            frac = (target_acc - lo["mean_accuracy"]) / (hi["mean_accuracy"] - lo["mean_accuracy"])
            return float(lo["mb_per_day"] + frac * (hi["mb_per_day"] - lo["mb_per_day"]))
    return None


def dominating_scrubs(points: List[Dict[str, object]], name: str) -> List[Dict[str, object]]:
    """Scrub points with accuracy >= and traffic <= the named strategy (at least one strict)."""
    ref = next(p for p in points if p["strategy"] == name)
    return [
        p for p in points
        if p["family"] == "scrub"
        and p["mean_accuracy"] >= ref["mean_accuracy"] and p["mb_per_day"] <= ref["mb_per_day"]
        and (p["mean_accuracy"] > ref["mean_accuracy"] or p["mb_per_day"] < ref["mb_per_day"])
    ]


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def build_report(name: str, runs: Dict[int, dict]) -> str:
    summaries = [r["summary"] for r in runs.values()]
    seeds = sorted(runs)
    days = summaries[0]["total_simulated_days"]
    n = len(seeds)
    lines = [f"# Campaign report: `{name}`", ""]

    faults_day = mean_ci([s["total_faults_accumulated"] / s["total_simulated_days"] for s in summaries])
    msb_day = mean_ci([s["exponent_msb_flips_total"] / s["total_simulated_days"] for s in summaries])
    saa = [s.get("saa_minutes_total", float("nan")) / s["total_simulated_days"] for s in summaries]
    lines += [
        "## Setup",
        "",
        f"- Architecture: `{summaries[0].get('arch', 'resnet18')}`",
        f"- Seeds: {', '.join(map(str, seeds))} ({n}); {days:g} simulated days each; "
        f"{len(runs[seeds[0]]['telemetry'])} steps per seed",
        f"- Orbit: `{summaries[0].get('orbit', 'periodic')}`; SAA-core-equivalent minutes/day: "
        f"{np.nanmean(saa):.1f}",
        f"- Fault field: `{summaries[0]['fault_field']}`; bit flips/day {faults_day[0]:.1f} "
        f"(± {faults_day[1]:.1f}); exponent-MSB flips/day {msb_day[0]:.2f} (± {msb_day[1]:.2f})",
        f"- Clean accuracy: {pct(summaries[0]['clean_accuracy'])}%",
        f"- Golden store: lossy INT8 = {summaries[0].get('golden_int8', False)}; "
        f"BN recalibration = {summaries[0].get('aegis_recalibrate', False)}",
        "",
        f"Intervals are 95% Student-t over {n} seeds." if n > 1 else "Single seed: no intervals.",
        "",
    ]

    points = pareto_points(runs)
    by_name = {p["strategy"]: p for p in points}
    energy, energy_added = energy_per_day_mj(runs)
    energy_key = {k: k for k in ("aegis_v2", "ranger_aegis", "ground_uplink", "periodic_scrubbing", "aegis_audit")}

    lines += ["## Accuracy, traffic and energy", "",
              "| Strategy | Mean acc. (%) | Final acc. (%) | MB/day | Energy (mJ/day) |",
              "|---|---|---|---|---|"]
    for key, label in ACC_ROWS:
        if key not in summaries[0]["mean_accuracy"]:
            continue
        m, h = mean_ci([s["mean_accuracy"][key] for s in summaries])
        f, fh = mean_ci([s["final_accuracy"][key] for s in summaries])
        mb = by_name[label]["mb_per_day"] if label in by_name else 0.0
        e = energy.get(energy_key.get(key, ""), None)
        e_txt = f"{e[0]:.1f} ± {e[1]:.1f}" if e else "not charged"
        lines.append(f"| {label} | {pct(m)} ± {pct(h)} | {pct(f)} ± {pct(fh)} | {mb:.1f} | {e_txt} |")
    for hours in summaries[0].get("scrub_sweep", {}):
        label = f"Periodic scrubbing ({hours} h)"
        m, h = mean_ci([s["scrub_sweep"][hours]["mean_accuracy"] for s in summaries])
        e = energy[f"scrub_{hours}h"]
        lines.append(f"| {label} | {pct(m)} ± {pct(h)} | n/a | {by_name[label]['mb_per_day']:.1f} | "
                     f"{e[0]:.1f} ± {e[1]:.1f} |")
    lines += ["", "Ranger and BLINK move no bytes; their per-inference compute is not charged."]
    if energy_added:
        lines.append("Monitored strategies include T1/T2 hook compute added analytically "
                     "(campaign predates its in-simulation accounting).")
    lines.append("")

    lines += ["## Paired accuracy difference: Aegis-v2 minus baseline", "",
              "Same fault stream per seed, so differences are paired.", "",
              "| Baseline | Δ mean acc. (pts) | Seeds where Aegis-v2 higher |", "|---|---|---|"]
    for key, label in ACC_ROWS[1:]:
        if key not in summaries[0]["mean_accuracy"]:
            continue
        diffs = [s["mean_accuracy"]["aegis_v2"] - s["mean_accuracy"][key] for s in summaries]
        m, h = mean_ci(diffs)
        lines.append(f"| {label} | {100 * m:+.2f} ± {100 * h:.2f} | {sum(d > 0 for d in diffs)}/{n} |")
    lines.append("")

    lines += ["## Bandwidth verdict", ""]
    for label in ("Aegis-v2 selective restore", "Aegis-v2 + CRC audit", "Ranger + Aegis-v2"):
        if label not in by_name:
            continue
        ref = by_name[label]
        dom = dominating_scrubs(points, label)
        match = scrub_bandwidth_to_match(points, ref["mean_accuracy"])
        lines.append(f"- **{label}**: {pct(ref['mean_accuracy'])}% at {ref['mb_per_day']:.1f} MB/day.")
        if dom:
            names = ", ".join(f"{p['strategy']} ({pct(p['mean_accuracy'])}%, {p['mb_per_day']:.1f} MB/day)"
                              for p in dom)
            lines.append(f"  Pareto-dominated by: {names}.")
        else:
            lines.append("  Not dominated by any scrubbing interval.")
        if match is None:
            lines.append("  No scrubbing interval in the sweep reaches this accuracy.")
        elif ref["mb_per_day"] > 0:
            lines.append(f"  Scrubbing needs ~{match:.1f} MB/day to match it "
                         f"({match / ref['mb_per_day']:.2f}x the traffic).")
    lines += ["", "Scrub-matching traffic is interpolated linearly between swept intervals.", ""]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="main")
    p.add_argument("--results-dir", default="results")
    args = p.parse_args()
    # The report uses Greek/plus-minus symbols; Windows consoles default to cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    campaign_dir = os.path.join(args.results_dir, args.name)
    runs = load_seed_runs(campaign_dir)
    if not runs:
        raise SystemExit(f"No seed_*.json in {campaign_dir}")
    report = build_report(args.name, runs)
    out = os.path.join(campaign_dir, "report.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()

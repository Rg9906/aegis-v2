"""
Exports a campaign for the React mission dossier (frontend/).

Writes frontend/public/data/<name>.json and refreshes frontend/public/data/index.json:
  - setup + report statistics (same numbers as write_report.py),
  - per-seed telemetry trimmed to what the UI plots,
  - per-seed fault hits (block, parameter, bit, value change) replayed from the seed: the fault
    stream depends only on the seeded radiation environment and injector, not on model weights,
    and the replay is checked against the recorded per-step flip counts,
  - the satellite ground track (1 point / 5 min) and a |B| grid (IGRF) for drawing the SAA.

Example:
    python experiments/export_frontend_data.py --name main_gpu
"""

import argparse
import datetime
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aegis.config import DEFAULT_BUDGET_CONFIG, DEFAULT_RAD_CONFIG  # noqa: E402
from aegis.reporting import STRATEGIES, load_seed_runs, pareto_points, strategies_present  # noqa: E402
from experiments.write_report import (  # noqa: E402
    ACC_ROWS,
    dominating_scrubs,
    energy_per_day_mj,
    mean_ci,
    scrub_bandwidth_to_match,
)

OUT_DIR = os.path.join("frontend", "public", "data")
TELEMETRY_KEYS = ("step_index", "elapsed_days", "in_saa", "saa_minutes_in_step", "bit_flips_step",
                  "exponent_msb_flips_step", "aegis_alarm", "aegis_restored_blocks", "scrub_executed")


def replay_faults(seed: int, steps: int, step_minutes: float, arch: str, orbit: str, fault_field: str):
    """Per-step fault hits for one seed, replayed with the campaign's seeded generators."""
    import torch

    from aegis.fault_engine.injector import ModelFaultInjector
    from aegis.fault_engine.radiation_model import LEORadiationEnvironment
    from aegis.models.layer_registry import LayerRegistry

    if arch == "mobilenet_v2":
        from aegis.models.factory import build_mobilenet_v2

        model = build_mobilenet_v2()
    else:
        import torchvision.models as models

        model = models.resnet18(weights=None, num_classes=10)
    # Real weights only matter for the before/after values shown in the UI
    try:
        from aegis.models.factory import load_backbone

        model = load_backbone(arch)
    except Exception as exc:  # noqa: BLE001 - values are cosmetic; the fault stream is unaffected
        print(f"  (using untrained weights for value display: {exc})")
    model.eval()
    registry = LayerRegistry(model)
    injector = ModelFaultInjector(model, registry, generator=torch.Generator().manual_seed(seed),
                                  rng=np.random.default_rng(seed))
    if orbit == "tle":
        from aegis.fault_engine.orbit_model import OrbitTrackRadiationEnvironment

        env = OrbitTrackRadiationEnvironment(DEFAULT_RAD_CONFIG, random_seed=seed)
    else:
        env = LEORadiationEnvironment(DEFAULT_RAD_CONFIG, random_seed=seed)
    bits = sum(b.total_bits_fp32 for b in registry._blocks.values())
    per_step = []
    with torch.no_grad():
        for _ in range(steps):
            records = injector.inject_orbital_flux_step(env.step(step_minutes, bits), target_field=fault_field)
            per_step.append([
                {"b": r.block_id, "p": r.param_name, "bit": r.bit_position,
                 "from": float(f"{r.pre_value:.4g}"), "to": float(f"{r.post_value:.4g}") if np.isfinite(r.post_value)
                 else str(r.post_value), "mbu": r.is_mbu}
                for r in records
            ])
    return registry.block_ids, [registry.get_block_metadata(b).size_bytes_fp32 for b in registry.block_ids], per_step


def ground_track(days: float, step_minutes: float = 5.0):
    from aegis.fault_engine.orbit_model import track_payload

    return track_payload(days * 1440.0, step_minutes)


def field_grid(alt_km: float, step_deg: float = 3.0):
    from aegis.fault_engine.orbit_model import field_grid as grid

    return grid(alt_km, step_deg)


def stats(runs):
    summaries = [r["summary"] for r in runs.values()]
    n = len(summaries)
    points = pareto_points(runs)
    by_name = {p["strategy"]: p for p in points}
    energy, energy_added = energy_per_day_mj(runs)
    rows = []
    for key, label in ACC_ROWS:
        if key not in summaries[0]["mean_accuracy"]:
            continue
        m, h = mean_ci([s["mean_accuracy"][key] for s in summaries])
        f, fh = mean_ci([s["final_accuracy"][key] for s in summaries])
        e = energy.get(key)
        diffs = [s["mean_accuracy"]["aegis_v2"] - s["mean_accuracy"][key] for s in summaries]
        dm, dh = mean_ci(diffs)
        rows.append({"key": key, "label": label, "mean": m, "ci": h, "final": f, "final_ci": fh,
                     "mb_per_day": by_name[label]["mb_per_day"] if label in by_name else 0.0,
                     "energy_mj_day": e[0] if e else None, "energy_ci": e[1] if e else None,
                     "per_seed": [s["mean_accuracy"][key] for s in summaries],
                     "delta_vs_aegis": None if key == "aegis_v2" else {"mean": dm, "ci": dh,
                                                                       "wins": sum(d > 0 for d in diffs)}})
    for hours in summaries[0].get("scrub_sweep", {}):
        label = f"Periodic scrubbing ({hours} h)"
        m, h = mean_ci([s["scrub_sweep"][hours]["mean_accuracy"] for s in summaries])
        e = energy[f"scrub_{hours}h"]
        rows.append({"key": f"scrub_{hours}h", "label": label, "mean": m, "ci": h, "final": None,
                     "final_ci": None, "mb_per_day": by_name[label]["mb_per_day"], "energy_mj_day": e[0],
                     "energy_ci": e[1], "per_seed": [s["scrub_sweep"][hours]["mean_accuracy"] for s in summaries],
                     "delta_vs_aegis": None, "interval_h": float(hours)})
    verdicts = []
    for label in ("Aegis-v2 selective restore", "Aegis-v2 + CRC audit", "Ranger + Aegis-v2"):
        if label not in by_name:
            continue
        ref = by_name[label]
        match = scrub_bandwidth_to_match(points, ref["mean_accuracy"])
        verdicts.append({"label": label, "mean_accuracy": ref["mean_accuracy"], "mb_per_day": ref["mb_per_day"],
                         "dominated_by": [p["strategy"] for p in dominating_scrubs(points, label)],
                         "scrub_mb_to_match": match,
                         "traffic_ratio": (match / ref["mb_per_day"]) if match and ref["mb_per_day"] else None})
    return {"seeds": n, "rows": rows, "pareto": points, "verdicts": verdicts,
            "energy_hook_compute_added": energy_added}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="main")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--out-dir", default=OUT_DIR)
    args = p.parse_args()
    runs = load_seed_runs(os.path.join(args.results_dir, args.name))
    if not runs:
        raise SystemExit(f"No seed_*.json in {os.path.join(args.results_dir, args.name)}")
    first = next(iter(runs.values()))["summary"]
    arch, orbit = first.get("arch", "resnet18"), first.get("orbit", "periodic")
    days = first["total_simulated_days"]
    steps = len(next(iter(runs.values()))["telemetry"])
    step_minutes = days * 1440.0 / steps

    seeds = {}
    block_ids, block_bytes = None, None
    for seed, run in sorted(runs.items()):
        print(f"seed {seed}: replaying {steps} steps of faults")
        block_ids, block_bytes, faults = replay_faults(seed, steps, step_minutes, arch, orbit, first["fault_field"])
        recorded = [t["bit_flips_step"] for t in run["telemetry"]]
        if [len(f) for f in faults] != recorded:
            raise SystemExit(f"Fault replay for seed {seed} does not match recorded flip counts")
        present = strategies_present(runs)
        seeds[seed] = {
            "summary": {k: run["summary"][k] for k in ("mean_accuracy", "final_accuracy", "bytes_transferred",
                                                     "energy_consumed_uj", "total_faults_accumulated",
                                                     "exponent_msb_flips_total") if k in run["summary"]},
            "telemetry": [{**{k: t[k] for k in TELEMETRY_KEYS},
                           "acc": {a: t[a] for a, *_ in present},
                           "acc_scrub_sweep": t.get("acc_scrub_sweep", {})}
                          for t in run["telemetry"]],
            "faults": faults,
        }

    track = ground_track(days) if orbit == "tle" else None
    payload = {
        "name": args.name,
        "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "setup": {"arch": arch, "orbit": orbit, "days": days, "steps": steps, "step_minutes": step_minutes,
                  "fault_field": first["fault_field"], "clean_accuracy": first["clean_accuracy"],
                  "golden_int8": first.get("golden_int8", False), "audit_hours": first.get("aegis_audit_hours"),
                  "rates": {"background": DEFAULT_RAD_CONFIG.background_rate_per_bit_day,
                            "saa_core": DEFAULT_RAD_CONFIG.saa_peak_rate_per_bit_day,
                            "measured_mean": DEFAULT_RAD_CONFIG.measured_mean_rate_per_bit_day,
                            "sbu_fraction": DEFAULT_RAD_CONFIG.sbu_fraction},
                  "energy": {"monitor_pj_per_op": DEFAULT_BUDGET_CONFIG.monitor_energy_pj_per_op,
                             "dram_read_pj_per_bit": DEFAULT_BUDGET_CONFIG.dram_read_energy_pj_per_bit,
                             "inference_mj_per_image": DEFAULT_BUDGET_CONFIG.inference_energy_mj_per_image}},
        "strategies": [{"attr": a, "label": label, "light": lc, "dark": dc} for a, label, lc, dc in STRATEGIES
                       if a in next(iter(seeds.values()))["telemetry"][0]["acc"]],
        "blocks": [{"id": b, "bytes": n} for b, n in zip(block_ids, block_bytes)],
        "stats": stats(runs),
        "seeds": seeds,
        "track": track,
        "field": field_grid(track["alt_km"]) if track else None,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"{args.name}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))
    index_path = os.path.join(args.out_dir, "index.json")
    names = sorted({os.path.splitext(n)[0] for n in os.listdir(args.out_dir) if n.endswith(".json") and n != "index.json"})
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump({"campaigns": names}, f)
    print(f"Wrote {out} ({os.path.getsize(out) / 1e6:.2f} MB); index: {names}")


if __name__ == "__main__":
    main()

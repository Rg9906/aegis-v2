"""
Reproducible LEO comparative campaign on real EuroSAT imagery.

Runs the five-strategy MissionSimulationHarness for one or more seeds and writes
results/<name>/seed_<k>.json (per-step telemetry + summary) and results/<name>/campaign.json
(config + per-seed summaries + across-seed mean/std).

Example:
    python experiments/run_campaign.py --days 3 --step-minutes 60 --seeds 0 1 2 --name main
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aegis.data.eurosat import load_eurosat_splits  # noqa: E402
from aegis.models.eurosat_classifier import EuroSATResNet18  # noqa: E402
from aegis.simulation.mission_sim import MissionSimulationHarness  # noqa: E402

STRATEGIES = ("unmitigated", "ranger_clipping", "blink_recentering", "periodic_scrubbing", "aegis_v2",
              "ranger_aegis", "ground_uplink")
REPAIRING = ("periodic_scrubbing", "aegis_v2", "ranger_aegis", "ground_uplink")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--days", type=float, default=3.0)
    p.add_argument("--step-minutes", type=float, default=60.0)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--eval-per-class", type=int, default=20)
    p.add_argument("--calib-per-class", type=int, default=10)
    p.add_argument("--fault-field", choices=["any", "exponent", "mantissa", "sign"], default="any")
    p.add_argument("--scrub-interval-hours", type=float, default=12.0)
    p.add_argument("--scrub-sweep-hours", type=float, nargs="*", default=[],
                   help="Extra periodic-scrub intervals to simulate for the Pareto sweep")
    p.add_argument("--orbit", choices=["tle", "periodic"], default="tle",
                   help="tle: real Sentinel-2A track + IGRF SAA; periodic: synthetic fixed passes")
    p.add_argument("--tle", default=None, help="TLE file (default orbits/sentinel2a.tle)")
    p.add_argument("--golden-int8", action="store_true",
                   help="Aegis golden copy stored as per-channel INT8 (lossy restores)")
    p.add_argument("--recalibrate", action="store_true", help="Enable post-restore BN recalibration")
    p.add_argument("--name", default="main")
    p.add_argument("--out-dir", default="results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = os.path.join(args.out_dir, args.name)
    os.makedirs(out_dir, exist_ok=True)
    torch.set_grad_enabled(False)

    classifier = EuroSATResNet18(pretrained=True)
    # Data split is fixed across seeds so seeds vary only the radiation process
    splits = load_eurosat_splits(calib_per_class=args.calib_per_class, eval_per_class=args.eval_per_class, seed=0)
    print(f"Eval chips: {splits.eval_labels.numel()}  calib chips: {splits.calib_images.size(0)}")

    seed_summaries = []
    for seed in args.seeds:
        torch.manual_seed(seed)
        t0 = time.time()
        sim = MissionSimulationHarness(
            base_model=classifier.backbone,
            val_images=splits.eval_images,
            val_labels=splits.eval_labels,
            calib_images=splits.calib_images,
            random_seed=seed,
            fault_field=args.fault_field,
            scrub_interval_hours=args.scrub_interval_hours,
            aegis_recalibrate=args.recalibrate,
            scrub_sweep_hours=args.scrub_sweep_hours,
            golden_int8=args.golden_int8,
            orbit=args.orbit,
            tle_path=args.tle,
        )
        print(f"\n[seed {seed}] clean accuracy {sim.clean_accuracy:.4f}")
        sim.run_simulation(total_days=args.days, step_minutes=args.step_minutes, progress=True)
        summary = sim.get_campaign_summary()
        summary["seed"] = seed
        summary["wall_seconds"] = round(time.time() - t0, 1)
        seed_summaries.append(summary)
        with open(os.path.join(out_dir, f"seed_{seed}.json"), "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "telemetry": [asdict(t) for t in sim.telemetry_history]}, f, indent=1)
        print(json.dumps(summary, indent=1))

    def agg(path):
        vals = np.array([s[path[0]][path[1]] for s in seed_summaries], dtype=np.float64)
        return {"mean": round(float(vals.mean()), 4), "std": round(float(vals.std(ddof=1)) if len(vals) > 1 else 0.0, 4)}

    aggregate = {
        "mean_accuracy": {k: agg(("mean_accuracy", k)) for k in STRATEGIES},
        "final_accuracy": {k: agg(("final_accuracy", k)) for k in STRATEGIES},
        "energy_uj": {k: agg(("energy_consumed_uj", k)) for k in REPAIRING},
        "bytes": {k: agg(("bytes_transferred", k)) for k in REPAIRING},
    }
    campaign = {"config": vars(args), "per_seed": seed_summaries, "aggregate": aggregate}
    with open(os.path.join(out_dir, "campaign.json"), "w", encoding="utf-8") as f:
        json.dump(campaign, f, indent=1)
    print("\nAGGREGATE:", json.dumps(aggregate, indent=1))


if __name__ == "__main__":
    main()

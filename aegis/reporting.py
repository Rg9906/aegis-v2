"""
Campaign Results Loading and Shared Presentation Constants.

Used by experiments/make_figures.py (static publication figures) and
dashboard/app.py (Streamlit mission console) so both read results identically.
"""

import glob
import json
import os
from typing import Dict, List, Tuple

# Telemetry accuracy attribute, display label, light-mode color, dark-mode color.
# Fixed slot order (validated categorical palette: CVD-adjacent >= 9.1 dE light / 8.4 dark).
STRATEGIES: Tuple[Tuple[str, str, str, str], ...] = (
    ("acc_aegis", "Aegis-v2 selective restore", "#2a78d6", "#3987e5"),
    ("acc_periodic_scrub", "Periodic scrubbing (12 h)", "#eb6834", "#d95926"),
    ("acc_ranger", "Ranger clipping", "#1baf7a", "#199e70"),
    ("acc_blink", "BLINK re-centering", "#eda100", "#c98500"),
    ("acc_unmitigated", "Unmitigated", "#e87ba4", "#d55181"),
    ("acc_uplink", "Ground uplink reload", "#008300", "#008300"),
    ("acc_ranger_aegis", "Ranger + Aegis-v2", "#6250d6", "#9085e9"),
    ("acc_aegis_audit", "Aegis-v2 + CRC audit", "#e34948", "#e66767"),
)

SUMMARY_KEYS: Dict[str, str] = {
    "acc_aegis": "aegis_v2",
    "acc_periodic_scrub": "periodic_scrubbing",
    "acc_ranger": "ranger_clipping",
    "acc_blink": "blink_recentering",
    "acc_unmitigated": "unmitigated",
    "acc_uplink": "ground_uplink",
    "acc_ranger_aegis": "ranger_aegis",
    "acc_aegis_audit": "aegis_audit",
}


def strategies_present(runs: Dict[int, dict]) -> Tuple[Tuple[str, str, str, str], ...]:
    """STRATEGIES recorded in these runs (older campaigns predate some strategies)."""
    if not runs:
        return STRATEGIES
    first = next(iter(runs.values()))["telemetry"][0]
    return tuple(s for s in STRATEGIES if s[0] in first)


def list_campaigns(results_dir: str = "results") -> List[str]:
    """Campaign names that have at least one seed file."""
    paths = glob.glob(os.path.join(results_dir, "*", "seed_*.json"))
    return sorted({os.path.basename(os.path.dirname(p)) for p in paths})


def load_seed_runs(campaign_dir: str) -> Dict[int, dict]:
    """Returns {seed: {"summary": ..., "telemetry": [...]}} for every seed file in the campaign."""
    runs = {}
    for path in sorted(glob.glob(os.path.join(campaign_dir, "seed_*.json"))):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        runs[int(data["summary"]["seed"])] = data
    return runs


def pareto_points(runs: Dict[int, dict]) -> List[Dict[str, object]]:
    """
    Accuracy-vs-bandwidth points averaged across seeds: every scrubbing interval (main 12 h +
    sweep), Aegis-v2, and the zero-traffic baselines. Bytes are per simulated day.
    """
    if not runs:
        return []
    summaries = [r["summary"] for r in runs.values()]
    days = summaries[0]["total_simulated_days"] or 1.0

    def mean(values):
        values = list(values)
        return sum(values) / len(values)

    points = [
        {
            "strategy": "Aegis-v2 selective restore",
            "mean_accuracy": mean(s["mean_accuracy"]["aegis_v2"] for s in summaries),
            "mb_per_day": mean(s["bytes_transferred"]["aegis_v2"] for s in summaries) / days / 1e6,
            "family": "aegis",
        },
        {
            "strategy": "Periodic scrubbing (12 h)",
            "mean_accuracy": mean(s["mean_accuracy"]["periodic_scrubbing"] for s in summaries),
            "mb_per_day": mean(s["bytes_transferred"]["periodic_scrubbing"] for s in summaries) / days / 1e6,
            "family": "scrub",
            "interval_h": 12.0,
        },
    ]
    for label, key, family in (("Ranger + Aegis-v2", "ranger_aegis", "aegis"),
                               ("Aegis-v2 + CRC audit", "aegis_audit", "aegis"),
                               ("Ground uplink reload", "ground_uplink", "uplink")):
        if key in summaries[0]["mean_accuracy"]:
            points.append({
                "strategy": label,
                "mean_accuracy": mean(s["mean_accuracy"][key] for s in summaries),
                "mb_per_day": mean(s["bytes_transferred"][key] for s in summaries) / days / 1e6,
                "family": family,
            })
    for key in summaries[0].get("scrub_sweep", {}):
        points.append({
            "strategy": f"Periodic scrubbing ({key} h)",
            "mean_accuracy": mean(s["scrub_sweep"][key]["mean_accuracy"] for s in summaries),
            "mb_per_day": mean(s["scrub_sweep"][key]["bytes_transferred"] for s in summaries) / days / 1e6,
            "family": "scrub",
            "interval_h": float(key),
        })
    for label, key in (("Ranger clipping", "ranger_clipping"), ("BLINK re-centering", "blink_recentering"),
                       ("Unmitigated", "unmitigated")):
        points.append({
            "strategy": label,
            "mean_accuracy": mean(s["mean_accuracy"][key] for s in summaries),
            "mb_per_day": 0.0,
            "family": "baseline",
        })
    return points

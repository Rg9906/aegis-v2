"""Tests for the statistical campaign report (experiments/write_report.py)."""

import json
import os
import sys
from dataclasses import asdict

import pytest
import torch
import torchvision.models as models

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

from write_report import build_report, dominating_scrubs, mean_ci, scrub_bandwidth_to_match  # noqa: E402

from aegis.config import EUROSAT_NUM_CLASSES  # noqa: E402
from aegis.reporting import load_seed_runs  # noqa: E402
from aegis.simulation.mission_sim import MissionSimulationHarness  # noqa: E402


def _scrub(h, acc, mb):
    return {"strategy": f"Periodic scrubbing ({h} h)", "family": "scrub", "mean_accuracy": acc, "mb_per_day": mb}


def test_mean_ci_uses_student_t():
    m, h = mean_ci([0.8, 0.9, 1.0])
    assert m == pytest.approx(0.9)
    assert h == pytest.approx(4.303 * 0.1 / 3 ** 0.5, rel=1e-3)
    assert mean_ci([0.5]) == (0.5, 0.0)


def test_bandwidth_match_and_dominance():
    points = [_scrub(24, 0.70, 40.0), _scrub(6, 0.80, 160.0), _scrub(1, 0.90, 960.0),
              {"strategy": "A", "family": "aegis", "mean_accuracy": 0.85, "mb_per_day": 100.0},
              {"strategy": "B", "family": "aegis", "mean_accuracy": 0.75, "mb_per_day": 200.0}]
    assert scrub_bandwidth_to_match(points, 0.85) == pytest.approx(560.0)
    assert scrub_bandwidth_to_match(points, 0.95) is None
    assert dominating_scrubs(points, "A") == []
    assert [p["strategy"] for p in dominating_scrubs(points, "B")] == ["Periodic scrubbing (6 h)"]


def test_report_from_real_harness_runs(tmp_path):
    torch.manual_seed(0)
    model = models.resnet18(weights=None, num_classes=EUROSAT_NUM_CLASSES)
    images = torch.randn(8, 3, 64, 64)  # plumbing only; never used for reported numbers
    labels = torch.randint(0, EUROSAT_NUM_CLASSES, (8,))
    for seed in (0, 1):
        sim = MissionSimulationHarness(model, images, labels, random_seed=seed, scrub_sweep_hours=(1.0,),
                                       monitor_batch_size=4)
        sim.run_simulation(total_days=2 / 24, step_minutes=60.0)
        summary = sim.get_campaign_summary()
        summary["seed"] = seed
        with open(tmp_path / f"seed_{seed}.json", "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "telemetry": [asdict(t) for t in sim.telemetry_history]}, f)

    report = build_report("t", load_seed_runs(str(tmp_path)))
    assert "Intervals are 95% Student-t over 2 seeds." in report
    assert "| Periodic scrubbing (1 h) |" in report
    assert "## Bandwidth verdict" in report
    assert "added analytically" not in report  # this run already charges hook compute

    # A campaign from before hook compute was charged gets it added analytically
    runs = load_seed_runs(str(tmp_path))
    for run in runs.values():
        del run["summary"]["energy_consumed_uj"]["aegis_v2_monitor_compute_share"]
    assert "added analytically" in build_report("t", runs)

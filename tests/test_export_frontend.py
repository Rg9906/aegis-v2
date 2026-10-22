"""The frontend exporter replays each seed's fault stream and must match the recorded telemetry."""

import json
import subprocess
import sys
from dataclasses import asdict

import torch
import torchvision.models as models

from aegis.config import EUROSAT_NUM_CLASSES
from aegis.simulation.mission_sim import MissionSimulationHarness


def test_export_replays_faults_and_writes_index(tmp_path):
    torch.manual_seed(0)
    model = models.resnet18(weights=None, num_classes=EUROSAT_NUM_CLASSES)
    images = torch.randn(8, 3, 64, 64)  # plumbing only
    labels = torch.randint(0, EUROSAT_NUM_CLASSES, (8,))
    camp = tmp_path / "results" / "t"
    camp.mkdir(parents=True)
    for seed in (0, 1):
        sim = MissionSimulationHarness(model, images, labels, random_seed=seed, monitor_batch_size=4, orbit="tle")
        sim.run_simulation(total_days=3 / 24, step_minutes=60.0)
        summary = sim.get_campaign_summary()
        summary["seed"] = seed
        (camp / f"seed_{seed}.json").write_text(
            json.dumps({"summary": summary, "telemetry": [asdict(t) for t in sim.telemetry_history]}))

    out = tmp_path / "data"
    proc = subprocess.run(
        [sys.executable, "experiments/export_frontend_data.py", "--name", "t",
         "--results-dir", str(tmp_path / "results"), "--out-dir", str(out)],
        capture_output=True, text=True, env={**__import__("os").environ, "HF_HUB_OFFLINE": "1"},
    )
    assert proc.returncode == 0, proc.stderr[-2000:]

    data = json.loads((out / "t.json").read_text())
    assert json.loads((out / "index.json").read_text()) == {"campaigns": ["t"]}
    for seed, run in data["seeds"].items():
        assert [len(f) for f in run["faults"]] == [t["bit_flips_step"] for t in run["telemetry"]]
    assert {s["attr"] for s in data["strategies"]} >= {"acc_aegis", "acc_aegis_audit"}
    assert len(data["track"]["lat"]) == len(data["track"]["lon"]) > 0
    assert data["field"]["edge_nt"] == 20000

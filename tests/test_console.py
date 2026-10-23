"""
Live console: helpers, calibration copy, the streaming mission protocol, and (with real data)
the HTTP API end to end. Random tensors below are plumbing only, never accuracy claims.
"""

import json
import os
import time

import pytest
import torch
import torchvision.models as models

from aegis.config import DEFAULT_EUROSAT_DATA_DIR
from aegis.console.common import bits32, block_crcs, finite, severity
from aegis.console.mission import LiveMission, MissionConfig
from aegis.console.server import jsonable
from aegis.models.layer_registry import LayerRegistry
from aegis.monitor.orchestrator import AegisHealthMonitor


def test_bits32_and_severity():
    assert bits32(1.0) == "0" + "01111111" + "0" * 23
    assert bits32(-2.0)[0] == "1"
    assert severity("layer1.0.conv1.weight", 30, 0.02, 6.8e36) == "critical"
    assert severity("layer1.0.bn1.running_var", 31, 0.5, -0.5) == "critical"
    assert severity("layer1.0.conv1.weight", 22, 0.1, 0.15) == "minor"
    assert severity("layer1.0.conv1.weight", 23, 0.1, 0.4) == "moderate"
    assert severity("fc.weight", 4, 0.1, float("nan")) == "critical"


def test_non_finite_values_are_json_safe():
    assert finite(float("nan")) == "NaN" and finite(float("-inf")) == "-Infinity"
    payload = jsonable({"a": [float("inf"), 1.5], "b": {"c": float("nan")}})
    json.dumps(payload, allow_nan=False)  # must not raise
    assert payload == {"a": ["Infinity", 1.5], "b": {"c": "NaN"}}


def _small_model():
    torch.manual_seed(0)
    return models.resnet18(num_classes=10).eval()


def test_copied_calibration_matches_fresh_calibration():
    model = _small_model()
    calib = torch.randn(32, 3, 64, 64)
    src = AegisHealthMonitor(model, LayerRegistry(model))
    src.calibrate(calib)
    replica = _small_model()
    dst = AegisHealthMonitor(replica, LayerRegistry(replica))
    dst.copy_calibration_from(src)
    assert dst.tier1_envelope.bounds == src.tier1_envelope.bounds
    assert dst.tier2_bn_drift._calibrated_tolerances == src.tier2_bn_drift._calibrated_tolerances
    assert dst.tier3_checksum.scan_all_blocks() == []


def test_live_mission_streams_steps_with_identical_fault_stream():
    model = _small_model()
    calib = torch.randn(32, 3, 64, 64)
    monitor = AegisHealthMonitor(model, LayerRegistry(model))
    monitor.calibrate(calib)
    events = []
    mission = LiveMission(
        base_model=model, calibrated_monitor=monitor, golden_crcs=block_crcs(LayerRegistry(model)),
        probe_images=torch.randn(10, 3, 64, 64), probe_labels=torch.arange(10),
        incoming_images=torch.randn(32, 3, 64, 64),
        config=MissionConfig(hours=2, step_minutes=60, flux=100, scrub_hours=1, probe_per_class=1, seed=3),
        emit=lambda kind, payload: events.append((kind, payload)),
    )
    summary = mission.run()
    steps = [p for k, p in events if k == "mission.step"]
    assert len(steps) == 2 and summary["steps"] == 2
    assert all(s["flips"] > 0 for s in steps)  # 100x flux over an hour always strikes
    # Every flip gets a strike minute inside its own step (drives where the map bursts)
    for s in steps:
        assert len(s["hit_minutes"]) == s["flips"]
        assert all(s["t_min"] - 60 <= m <= s["t_min"] for m in s["hit_minutes"])
        assert all("t_min" in h for h in s["hits"])
    # Scrubbing reloads every hour, so its memory is clean after each step...
    assert all(s["strategies"]["scrub"]["scrubbed"] and not s["strategies"]["scrub"]["corrupted"] for s in steps)
    # ...while the unprotected replica keeps every flip (same stream as everyone else)
    assert steps[-1]["strategies"]["unmitigated"]["corrupted"]
    json.dumps(jsonable(steps), allow_nan=False)


def test_mission_config_validation():
    with pytest.raises(ValueError):
        MissionConfig(hours=200).validate()
    with pytest.raises(ValueError):
        MissionConfig(flux=0.5).validate()


@pytest.mark.skipif(not os.path.isdir(DEFAULT_EUROSAT_DATA_DIR), reason="EuroSAT data not downloaded")
def test_console_api_end_to_end_on_real_data():
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from aegis.console.server import create_app

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    with TestClient(create_app(serve_frontend=False)) as client:
        deadline = time.time() + 600
        while client.get("/api/health").json()["status"] == "booting" and time.time() < deadline:
            time.sleep(0.5)
        state = client.get("/api/state").json()
        assert state["status"] == "ready" and state["corrupted"] == []

        r = client.post("/api/strike", json={"block": "layer3_block1", "field": "exponent_msb"}).json()
        assert r["hits"][0]["bit"] == 30
        assert r["impact"]["accuracy"] < state["clean_acc"]
        assert r["diagnosis"]["confirmed"] == ["layer3_block1"]
        assert r["repair"]["verified"] and r["repair"]["bytes"] < r["repair"]["full_bytes"]
        assert r["recovered"]["accuracy"] == pytest.approx(state["clean_acc"])
        assert r["state"]["corrupted"] == []

        assert client.post("/api/strike", json={"block": "nope"}).status_code == 400
        assert client.post("/api/strike", json={"count": 999}).status_code == 422
        assert client.get("/api/chip/0.png").headers["content-type"] == "image/png"

        # The last state broadcast after a mission must already report it finished (it is sent
        # from the mission thread while that thread is still alive; a stale "running" locked the UI)
        engine = client.app.state.engine
        states = []
        publish = engine._emit_fn
        engine._emit_fn = lambda kind, payload: (states.append(payload) if kind == "state" else None, publish(kind, payload))
        assert client.post("/api/mission/start", json={"hours": 1, "step_minutes": 60, "probe_per_class": 1}).status_code == 200
        assert client.post("/api/strike", json={}).status_code == 409
        engine.wait_for_mission(timeout=300)
        assert states and states[-1]["mission_running"] is False
        mission = client.get("/api/mission").json()
        assert not mission["running"] and len(mission["steps"]) == 1
        assert client.post("/api/diagnose", json={}).status_code == 200

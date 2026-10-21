"""Second architecture: MobileNetV2 block partitioning, alarm-to-block mapping, leakage guard."""

import json
import os

import pytest
import torch

from aegis.fault_engine.ieee754 import FLOAT32_EXPONENT_MSB
from aegis.fault_engine.injector import ModelFaultInjector
from aegis.models.factory import assert_held_out, build_mobilenet_v2
from aegis.models.layer_registry import LayerRegistry
from aegis.monitor.orchestrator import AegisHealthMonitor
from aegis.repair.golden_store import CompressedGoldenStore
from aegis.repair.selective_restore import DetectionGatedRestorer


def test_mobilenet_blocks_cover_every_parameter():
    model = build_mobilenet_v2()
    registry = LayerRegistry(model)
    assert registry.block_ids[0] == "features_block0"
    assert registry.block_ids[-1] == "classifier"
    assert "other" not in registry.block_ids
    assert len(registry.block_ids) == 20  # 19 feature stages + classifier
    covered = sum(len(registry.get_block_metadata(b).state_dict_keys) for b in registry.block_ids)
    assert covered == len(model.state_dict())


def test_mobilenet_alarm_maps_to_struck_block_and_restores_it():
    torch.manual_seed(0)
    model = build_mobilenet_v2().eval()
    registry = LayerRegistry(model)
    calib = torch.randn(32, 3, 64, 64)  # plumbing only
    monitor = AegisHealthMonitor(model, registry)
    monitor.calibrate(calib)
    monitor.attach()
    assert monitor._map_layer_name_to_block_id("features.7.conv.1.0") == "features_block7"
    assert monitor._map_layer_name_to_block_id("classifier.1") == "classifier"

    clean = registry.get_block_state_dict("features_block7")
    store = CompressedGoldenStore(registry)  # golden copy taken while the model is clean
    ModelFaultInjector(model, registry).inject_targeted_bit_flip(
        "features_block7", "features.7.conv.1.0.weight", 0, FLOAT32_EXPONENT_MSB)
    monitor.clear()
    with torch.no_grad():
        model(calib[:16])
    report = monitor.diagnose_current_state()
    assert not report.is_healthy
    assert "features_block7" in report.flagged_blocks

    DetectionGatedRestorer(registry, store).process_health_report(report, auto_recalibrate=False)
    restored = registry.get_block_state_dict("features_block7")
    assert all(torch.equal(restored[k], clean[k]) for k in clean)


def test_assert_held_out_rejects_training_chips(tmp_path):
    ckpt = tmp_path / "m.safetensors"
    with open(tmp_path / "m.json", "w", encoding="utf-8") as f:
        json.dump({"held_out_chips": ["Forest/a.jpg", "River/b.jpg"]}, f)
    assert_held_out(str(ckpt), ["Forest/a.jpg"])
    with pytest.raises(ValueError, match="training set"):
        assert_held_out(str(ckpt), ["Forest/a.jpg", "Forest/c.jpg"])


@pytest.mark.skipif(not os.path.isdir("data/eurosat/2750"), reason="EuroSAT not downloaded")
def test_held_out_covers_campaign_splits():
    from aegis.data.eurosat import held_out_paths, split_eurosat_paths

    held = held_out_paths([(10, 15), (4, 20)])
    calib, evals = split_eurosat_paths(calib_per_class=10, eval_per_class=15)
    assert set(calib) | {p for p, _ in evals} <= held
    assert len(held) <= 10 * (25 + 24)

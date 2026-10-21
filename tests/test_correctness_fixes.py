"""
Regression tests for evaluation-validity fixes:
SAA flux integration, forward-order suspect ranking, CRC-confirmed restore,
bit-uniform fault placement, recalibration energy accounting, and real-data accuracy.
"""

import os

import numpy as np
import pytest
import torch
import torchvision.models as models

from aegis.config import DEFAULT_EUROSAT_DATA_DIR, DEFAULT_RAD_CONFIG
from aegis.fault_engine.ieee754 import FLOAT32_EXPONENT_MSB
from aegis.fault_engine.injector import ModelFaultInjector
from aegis.fault_engine.radiation_model import LEORadiationEnvironment
from aegis.models.layer_registry import LayerRegistry
from aegis.monitor.orchestrator import AegisHealthMonitor
from aegis.repair.golden_store import CompressedGoldenStore
from aegis.repair.recalibrate import BNRecalibrator
from aegis.repair.selective_restore import DetectionGatedRestorer


def test_saa_overlap_is_integrated_over_step():
    env = LEORadiationEnvironment(random_seed=0)
    # Windows are [0,12), [240,252), ... -> a 240-minute step starting at 0 sees 12 SAA minutes
    assert env.saa_minutes_in_interval(0.0, 240.0) == pytest.approx(12.0)
    assert env.saa_minutes_in_interval(12.0, 240.0) == pytest.approx(0.0)
    assert env.saa_minutes_in_interval(0.0, 1440.0) == pytest.approx(72.0)
    assert env.saa_minutes_in_interval(235.0, 245.0) == pytest.approx(5.0)


def test_step_rate_not_classified_by_endpoint():
    """A 240-min step ending exactly on a pass boundary must not be billed at the SAA peak rate."""
    env = LEORadiationEnvironment(random_seed=0)
    state = env.step(delta_minutes=240.0, total_model_bits=1_000_000)
    cfg = DEFAULT_RAD_CONFIG
    expected = (cfg.saa_peak_rate_per_bit_day * 12.0 + cfg.background_rate_per_bit_day * 228.0) / 240.0
    assert state.expected_seu_rate_per_bit_day == pytest.approx(expected)
    assert state.expected_seu_rate_per_bit_day < cfg.saa_peak_rate_per_bit_day / 10


@pytest.fixture
def small_setup():
    torch.manual_seed(0)
    model = models.resnet18(weights=None, num_classes=10).eval()
    registry = LayerRegistry(model)
    batch = torch.randn(4, 3, 64, 64)
    return model, registry, batch


def test_primary_suspect_follows_forward_order(small_setup):
    model, registry, batch = small_setup
    monitor = AegisHealthMonitor(model, registry)
    monitor.calibrate(batch)
    monitor.attach()
    injector = ModelFaultInjector(model, registry)
    # Corrupt both an early block and the classifier; alphabetical order would pick "classifier"
    injector.inject_targeted_bit_flip("layer1_block0", bit_position=FLOAT32_EXPONENT_MSB)
    injector.inject_targeted_bit_flip("classifier", bit_position=FLOAT32_EXPONENT_MSB)
    monitor.clear()
    with torch.no_grad():
        model(batch)
    report = monitor.diagnose_current_state(run_tier3_scan=True)
    assert report.primary_suspect_block == "layer1_block0"
    assert report.flagged_blocks[0] == "layer1_block0"
    monitor.detach()


def test_restore_covers_all_crc_confirmed_blocks(small_setup):
    model, registry, batch = small_setup
    with torch.no_grad():
        clean_out = model(batch)
    store = CompressedGoldenStore(registry)
    monitor = AegisHealthMonitor(model, registry)
    monitor.calibrate(batch)
    monitor.attach()
    injector = ModelFaultInjector(model, registry)
    injector.inject_targeted_bit_flip("layer2_block0", bit_position=FLOAT32_EXPONENT_MSB)
    injector.inject_targeted_bit_flip("layer4_block1", bit_position=FLOAT32_EXPONENT_MSB)
    monitor.clear()
    with torch.no_grad():
        model(batch)
    report = monitor.diagnose_current_state(run_tier3_scan=True)
    assert set(report.tier3_corrupted_blocks) == {"layer2_block0", "layer4_block1"}

    event = DetectionGatedRestorer(registry, store).process_health_report(report, auto_recalibrate=False)
    assert event.restored_blocks == ["layer2_block0", "layer4_block1"]
    assert monitor.tier3_checksum.scan_all_blocks() == []
    with torch.no_grad():
        assert torch.equal(model(batch), clean_out)
    monitor.detach()


def test_recalibration_energy_is_charged(small_setup):
    model, registry, batch = small_setup
    store = CompressedGoldenStore(registry)
    recal = BNRecalibrator()
    recal.ingest_unlabeled_imagery(batch)
    monitor = AegisHealthMonitor(model, registry)
    monitor.calibrate(batch)
    ModelFaultInjector(model, registry).inject_targeted_bit_flip("layer3_block0", bit_position=FLOAT32_EXPONENT_MSB)
    report = monitor.diagnose_current_state(run_tier3_scan=True)
    event = DetectionGatedRestorer(registry, store, recal).process_health_report(report, auto_recalibrate=True)
    per_image_uj = store.budget_config.inference_energy_mj_per_image * 1000.0
    assert event.recalibration_energy_uj == pytest.approx(batch.size(0) * per_image_uj)
    assert event.total_energy_uj == pytest.approx(event.memory_energy_uj + event.recalibration_energy_uj)


def test_orbital_faults_are_uniform_over_bits(small_setup):
    """Large conv weights must absorb (nearly) all strikes; tiny BN vectors almost none."""
    model, registry, _ = small_setup
    injector = ModelFaultInjector(model, registry, generator=torch.Generator().manual_seed(1),
                                  rng=np.random.default_rng(1))
    env = LEORadiationEnvironment(random_seed=1)
    state = env.step(delta_minutes=1440.0, total_model_bits=sum(b.total_bits_fp32 for b in registry._blocks.values()))
    records = injector.inject_orbital_flux_step(state, target_field="any")
    assert len(records) > 50
    bn_hits = sum(1 for r in records if "bn" in r.param_name or "downsample.1" in r.param_name)
    # BN tensors hold ~0.1% of ResNet-18 bits
    assert bn_hits / len(records) < 0.05


@pytest.mark.skipif(not os.path.isdir(DEFAULT_EUROSAT_DATA_DIR), reason="EuroSAT data not downloaded")
def test_pretrained_model_accuracy_on_real_eurosat():
    from aegis.data.eurosat import load_eurosat_splits
    from aegis.models.eurosat_classifier import EuroSATResNet18

    try:
        classifier = EuroSATResNet18(pretrained=True)
    except Exception as exc:  # weights not cached and no network
        pytest.skip(f"pretrained weights unavailable: {exc}")
    splits = load_eurosat_splits(calib_per_class=1, eval_per_class=5, seed=3)
    metrics = classifier.evaluate(splits.eval_images, splits.eval_labels)
    assert metrics["accuracy"] >= 0.8
    # Calibration and evaluation splits are disjoint tensors of the right shape
    assert splits.calib_images.shape[1:] == (3, 224, 224)


def test_monitor_sees_raw_activations_when_ranger_attached_after(small_setup):
    """Ranger+Aegis: monitor hooks registered first must still catch the explosion Ranger clamps."""
    from aegis.baselines.ranger_clipper import RangerActivationClipper

    model, registry, batch = small_setup
    clipper = RangerActivationClipper()
    clipper.calibrate(model, batch)
    monitor = AegisHealthMonitor(model, registry)
    monitor.calibrate(batch)
    monitor.attach()
    clipper.attach(model)
    ModelFaultInjector(model, registry).inject_targeted_bit_flip("layer2_block1", bit_position=FLOAT32_EXPONENT_MSB)
    monitor.clear()
    with torch.no_grad():
        out = model(batch)
    assert torch.isfinite(out).all()  # Ranger contained it
    report = monitor.diagnose_current_state()
    assert report.tier3_corrupted_blocks == ["layer2_block1"]  # ...and Aegis still localized it
    clipper.detach()
    monitor.detach()


def test_ground_uplink_repairs_only_at_contact(small_setup):
    from aegis.baselines.ground_uplink import GroundUplinkReloader

    model, registry, batch = small_setup
    with torch.no_grad():
        clean_out = model(batch)
    uplink = GroundUplinkReloader(registry, CompressedGoldenStore(registry), first_contact_offset_minutes=90.0)
    monitor = AegisHealthMonitor(model, registry)
    monitor.calibrate(batch)
    ModelFaultInjector(model, registry).inject_targeted_bit_flip("layer3_block1", bit_position=FLOAT32_EXPONENT_MSB)
    uplink.report(monitor.diagnose_current_state(run_tier3_scan=True))
    assert uplink.pending == {"layer3_block1"}

    assert uplink.step(0.0, 60.0) is None  # no contact before minute 90
    event = uplink.step(60.0, 120.0)
    assert event is not None and event.restored_blocks == ["layer3_block1"]
    assert event.link_seconds == pytest.approx(event.bytes_uplinked * 8 / 1e6)
    with torch.no_grad():
        assert torch.equal(model(batch), clean_out)
    assert uplink.step(120.0, 1440.0) is None  # nothing pending


def test_int8_golden_store_is_smaller_and_rebaseline_clears_alarm(small_setup):
    model, registry, batch = small_setup
    fp32_store = CompressedGoldenStore(registry)
    int8_store = CompressedGoldenStore(registry, quantize_int8=True)
    assert int8_store.total_compressed_bytes < fp32_store.total_compressed_bytes / 3

    monitor = AegisHealthMonitor(model, registry)
    monitor.calibrate(batch)
    clean_w = registry.get_block_state_dict("layer3_block0")
    ModelFaultInjector(model, registry).inject_targeted_bit_flip("layer3_block0", bit_position=FLOAT32_EXPONENT_MSB)
    report = monitor.diagnose_current_state(run_tier3_scan=True)
    event = DetectionGatedRestorer(registry, int8_store).process_health_report(report, auto_recalibrate=False)

    restored = registry.get_block_state_dict("layer3_block0")
    for key, tensor in clean_w.items():
        # Per-channel INT8: error bounded by half a quantization step
        step = tensor.reshape(tensor.size(0), -1).abs().amax(dim=1) / 127.0 if tensor.dim() >= 2 else None
        if step is not None:
            err = (restored[key] - tensor).reshape(tensor.size(0), -1).abs().amax(dim=1)
            assert torch.all(err <= step / 2 + 1e-7)
    # Lossy restore still mismatches the original CRC until the block is re-baselined
    assert "layer3_block0" in monitor.tier3_checksum.scan_all_blocks()
    monitor.rebaseline_blocks(event.restored_blocks)
    assert monitor.tier3_checksum.scan_all_blocks() == []

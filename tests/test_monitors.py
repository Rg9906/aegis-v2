"""
Unit tests for Aegis Multi-Tier Health Monitoring Substructure.
"""

import pytest
import torch
import torchvision.models as models

from aegis.fault_engine.ieee754 import FLOAT32_EXPONENT_MSB
from aegis.fault_engine.injector import ModelFaultInjector
from aegis.models.layer_registry import LayerRegistry
from aegis.monitor.activation_envelope import ActivationEnvelopeMonitor
from aegis.monitor.bn_drift import BNDriftMonitor
from aegis.monitor.checksum import ChecksumMonitor
from aegis.monitor.orchestrator import AegisHealthMonitor


@pytest.fixture
def test_setup():
    """Initializes a small ResNet-18 model, registry, and calibration batch."""
    torch.manual_seed(42)
    model = models.resnet18(weights=None, num_classes=10)
    registry = LayerRegistry(model)
    calib_batch = torch.randn(4, 3, 64, 64)
    injector = ModelFaultInjector(model, registry)
    return model, registry, calib_batch, injector


def test_activation_envelope_detection(test_setup):
    """Verifies that Tier 1 cleanly passes on healthy inputs and catches exponent-MSB flips."""
    model, registry, calib_batch, injector = test_setup

    monitor = ActivationEnvelopeMonitor(margin=0.5)
    monitor.profile_clean_bounds(model, calib_batch)
    monitor.attach(model)

    # 1. Clean forward pass
    with torch.no_grad():
        model(calib_batch)
    assert not monitor.has_violations

    # 2. Inject an exponent-MSB flip into layer2_block0
    target_block = "layer2_block0"
    injector.inject_targeted_bit_flip(block_id=target_block, bit_position=FLOAT32_EXPONENT_MSB)

    # Corrupted forward pass
    monitor.clear_violations()
    with torch.no_grad():
        model(calib_batch)

    # Tier 1 should flag the layer
    assert monitor.has_violations
    violation_layers = list(monitor.latest_violations.keys())
    assert any("layer2" in name or "layer3" in name for name in violation_layers)

    monitor.detach()
    injector.rollback_all_faults()


def test_bn_drift_detection(test_setup):
    """Verifies that Tier 2 BLINK monitor calculates batch normalization drift."""
    model, registry, calib_batch, injector = test_setup

    bn_monitor = BNDriftMonitor(drift_threshold=3.5)
    bn_monitor.calibrate_clean_reference(model, calib_batch)
    bn_monitor.attach(model)

    # Clean pass
    with torch.no_grad():
        model(calib_batch)
    assert not bn_monitor.has_anomalies

    # Inject bit flip into layer1_block1
    injector.inject_targeted_bit_flip(block_id="layer1_block1", bit_position=FLOAT32_EXPONENT_MSB)
    bn_monitor.clear_drifts()

    with torch.no_grad():
        model(calib_batch)

    assert bn_monitor.has_anomalies

    bn_monitor.detach()
    injector.rollback_all_faults()


def test_checksum_monitor(test_setup):
    """Verifies that Tier 3 CRC-32 checksum deterministically detects corrupted blocks."""
    model, registry, _, injector = test_setup
    checksum = ChecksumMonitor(registry)

    # Clean state
    assert len(checksum.scan_all_blocks()) == 0

    # Corrupt layer3_block1
    target_block = "layer3_block1"
    injector.inject_targeted_bit_flip(block_id=target_block, bit_position=15)

    corrupted = checksum.scan_all_blocks()
    assert corrupted == [target_block]

    # Verify single block check
    assert checksum.verify_block_integrity(target_block) is False
    assert checksum.verify_block_integrity("layer1_block0") is True

    injector.rollback_all_faults()
    assert checksum.verify_block_integrity(target_block) is True


def test_orchestrator_diagnose(test_setup):
    """Verifies that AegisHealthMonitor integrates all tiers into a unified diagnostic report."""
    model, registry, calib_batch, injector = test_setup

    orchestrator = AegisHealthMonitor(model, registry)
    orchestrator.calibrate(calib_batch)
    orchestrator.attach()

    # Clean diagnosis
    with torch.no_grad():
        model(calib_batch)
    report = orchestrator.diagnose_current_state(run_tier3_scan=False)
    assert report.is_healthy is True
    assert len(report.flagged_blocks) == 0

    # Corrupt layer4_block0
    target_block = "layer4_block0"
    injector.inject_targeted_bit_flip(block_id=target_block, bit_position=FLOAT32_EXPONENT_MSB)
    orchestrator.clear()

    with torch.no_grad():
        model(calib_batch)

    # Diagnosed report
    corrupt_report = orchestrator.diagnose_current_state(run_tier3_scan=True)
    assert corrupt_report.is_healthy is False
    assert target_block in corrupt_report.flagged_blocks

    orchestrator.detach()
    injector.rollback_all_faults()

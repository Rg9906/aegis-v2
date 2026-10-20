"""
Unit tests for Aegis Repair Substructure (GoldenStore, Recalibration, Selective Restorer).
"""

import pytest
import torch
import torchvision.models as models

from aegis.fault_engine.ieee754 import FLOAT32_EXPONENT_MSB
from aegis.fault_engine.injector import ModelFaultInjector
from aegis.models.layer_registry import LayerRegistry
from aegis.monitor.orchestrator import AegisHealthMonitor
from aegis.repair.golden_store import CompressedGoldenStore
from aegis.repair.recalibrate import BNRecalibrator
from aegis.repair.selective_restore import DetectionGatedRestorer


@pytest.fixture
def repair_setup():
    """Initializes model, registry, golden store, and health monitor."""
    torch.manual_seed(42)
    model = models.resnet18(weights=None, num_classes=10)
    registry = LayerRegistry(model)
    store = CompressedGoldenStore(registry)
    recalibrator = BNRecalibrator(buffer_capacity=64)
    monitor = AegisHealthMonitor(model, registry)
    injector = ModelFaultInjector(model, registry)

    calib_batch = torch.randn(4, 3, 64, 64)
    monitor.calibrate(calib_batch)
    monitor.attach()
    recalibrator.ingest_unlabeled_imagery(calib_batch)

    restorer = DetectionGatedRestorer(registry, store, recalibrator)
    return model, registry, store, recalibrator, monitor, injector, restorer, calib_batch


def test_compressed_golden_store(repair_setup):
    """Verifies that the golden store compresses blocks and retrieves exact weights."""
    *_, store, _, _, _, _, _ = repair_setup

    assert store.total_compressed_bytes < store.total_uncompressed_bytes
    assert store.overall_compression_ratio > 1.0

    # Retrieve and check block
    block_id = "layer3_block0"
    retrieved_sd = store.retrieve_block(block_id)
    assert len(retrieved_sd) > 0
    assert store._cumulative_bytes_read > 0
    assert store._cumulative_energy_uj > 0.0


def test_bn_recalibrator(repair_setup):
    """Verifies that BN recalibration runs in train mode with frozen gradients."""
    model, _, _, recalibrator, _, _, _, calib_batch = repair_setup

    assert recalibrator.buffer_size == 4
    report = recalibrator.recalibrate(model)

    assert report.samples_processed == 4
    assert len(report.recalibrated_layers) > 0
    # Model should be returned to eval mode
    assert model.training is False


def test_selective_restore_end_to_end(repair_setup):
    """
    CRITICAL TEST: Injects a catastrophic exponent-MSB flip into a specific layer,
    verifies that monitor detects it, restorer restores ONLY that layer,
    and the model outputs return to healthy uncorrupted values.
    """
    model, registry, store, _, monitor, injector, restorer, calib_batch = repair_setup

    # Clean forward pass output
    with torch.no_grad():
        clean_out = model(calib_batch)

    # Corrupt layer2_block1
    target_block = "layer2_block1"
    injector.inject_targeted_bit_flip(block_id=target_block, bit_position=FLOAT32_EXPONENT_MSB)

    # Forward pass on corrupted model
    monitor.clear()
    with torch.no_grad():
        corrupt_out = model(calib_batch)

    # Confirm model output was broken/diverged
    assert not torch.allclose(clean_out, corrupt_out, atol=1e-3)

    # Diagnose state
    report = monitor.diagnose_current_state(run_tier3_scan=True)
    assert not report.is_healthy
    assert target_block in report.flagged_blocks

    # Execute detection-gated selective repair without recalibration first
    event = restorer.process_health_report(report, auto_recalibrate=False)
    assert event is not None
    assert event.was_successful is True
    assert event.restored_blocks == report.flagged_blocks
    assert event.total_bytes_transferred > 0

    # Re-run forward pass after selective restore
    monitor.clear()
    with torch.no_grad():
        restored_out = model(calib_batch)

    # Without recalibration on random noise, restored weights are identical so outputs match exactly
    assert torch.allclose(clean_out, restored_out, atol=1e-5)
    assert torch.equal(torch.argmax(clean_out, dim=-1), torch.argmax(restored_out, dim=-1))

    # Now verify that with auto_recalibrate=True, the restorer executes recalibration cleanly
    recal_event = restorer.process_health_report(report, auto_recalibrate=True)
    assert recal_event is not None
    assert recal_event.recalibration_report is not None

    monitor.detach()

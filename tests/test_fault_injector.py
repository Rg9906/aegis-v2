"""
Unit tests for ModelFaultInjector.
"""

import pytest
import torch
import torchvision.models as models

from aegis.fault_engine.ieee754 import FLOAT32_EXPONENT_MSB
from aegis.fault_engine.injector import ModelFaultInjector
from aegis.models.layer_registry import LayerRegistry


def test_model_fault_injector_targeted():
    """Verifies targeted bit flip into specific layer block."""
    model = models.resnet18(weights=None, num_classes=10)
    registry = LayerRegistry(model)
    injector = ModelFaultInjector(model, registry)

    target_block = "layer2_block0"
    record = injector.inject_targeted_bit_flip(
        block_id=target_block,
        bit_position=FLOAT32_EXPONENT_MSB,
    )

    assert record.block_id == target_block
    assert record.is_exponent_msb is True
    assert injector.total_faults_injected == 1
    assert len(injector.fault_history) == 1

    # Verify rollback
    injector.rollback_all_faults()
    assert injector.total_faults_injected == 0

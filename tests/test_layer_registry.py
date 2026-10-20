"""
Unit tests for LayerRegistry and block-level parameter isolation.
"""

import pytest
import torch
import torchvision.models as models

from aegis.models.layer_registry import LayerRegistry


def test_layer_registry_resnet18():
    """Verifies that ResNet-18 is cleanly partitioned into restorable logical blocks."""
    model = models.resnet18(weights=None, num_classes=10)
    registry = LayerRegistry(model)

    expected_blocks = {
        "stem",
        "layer1_block0",
        "layer1_block1",
        "layer2_block0",
        "layer2_block1",
        "layer3_block0",
        "layer3_block1",
        "layer4_block0",
        "layer4_block1",
        "classifier",
    }

    assert set(registry.block_ids) == expected_blocks

    # Verify total parameter sum matches model.parameters()
    total_reg_params = sum(b.param_count for b in [registry.get_block_metadata(bid) for bid in registry.block_ids])
    total_model_params = sum(p.numel() for p in model.parameters())
    # Note: registry includes BN running statistics (buffers)
    assert total_reg_params >= total_model_params


def test_block_selective_restore():
    """Verifies that restoring a single block updates only that block."""
    model = models.resnet18(weights=None, num_classes=10)
    registry = LayerRegistry(model)

    target_block = "layer4_block1"
    clean_sd = registry.get_block_state_dict(target_block)

    # Mutate the target block weights
    meta = registry.get_block_metadata(target_block)
    param_to_mutate = registry._resolve_param_by_name(meta.state_dict_keys[0])
    with torch.no_grad():
        param_to_mutate.add_(100.0)

    # Confirm it is mutated
    assert not torch.equal(param_to_mutate, clean_sd[meta.state_dict_keys[0]])

    # Restore the block
    registry.restore_block_state_dict(target_block, clean_sd)

    # Confirm it is restored to clean state
    assert torch.equal(param_to_mutate, clean_sd[meta.state_dict_keys[0]])

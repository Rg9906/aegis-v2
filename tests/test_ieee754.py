"""
Unit tests for IEEE 754 Binary Floating-Point Fault Injection Engine.
"""

import math
import pytest
import torch
import numpy as np

from aegis.fault_engine.ieee754 import (
    FLOAT32_EXPONENT_MSB,
    FLOAT32_SIGN_BIT,
    flip_bits_in_tensor,
    inject_mbu_cluster,
    inject_random_seus,
    inject_single_bit_flip,
    inspect_float32_bits,
    sample_bit_positions,
)


def test_inspect_float32_bits():
    """Verifies binary deconstruction of float32 values."""
    info = inspect_float32_bits(1.0)
    # IEEE-754 for 1.0f: Sign=0, Exp=01111111 (127), Mantissa=00000000000000000000000
    assert info["sign_bit"] == 0
    assert info["exponent_decimal"] == 127
    assert info["unbiased_exponent"] == 0
    assert not info["is_nan"]
    assert not info["is_inf"]


def test_exponent_msb_explosion():
    """
    CRITICAL TEST: Proves that flipping the exponent MSB of a typical weight (~0.75)
    produces an astronomical activation (~2.55e38) rather than a dead/zero neuron.
    """
    weight = torch.tensor([0.75], dtype=torch.float32)
    # Flip bit 30 (Exponent MSB)
    corrupted = inject_single_bit_flip(weight, flat_index=0, bit_position=FLOAT32_EXPONENT_MSB)

    corrupted_val = corrupted.item()
    # 0.75 with bit 30 flipped should be > 1e37
    assert corrupted_val > 1.0e37, f"Expected astronomical explosion > 1e37, got {corrupted_val}"
    assert not math.isnan(corrupted_val)


def test_sign_bit_flip():
    """Verifies that flipping bit 31 cleanly changes the numerical sign."""
    weight = torch.tensor([3.14159], dtype=torch.float32)
    corrupted = inject_single_bit_flip(weight, flat_index=0, bit_position=FLOAT32_SIGN_BIT)
    assert np.isclose(corrupted.item(), -3.14159, atol=1e-5)


def test_vectorized_bit_flip():
    """Verifies batched tensor bit flipping across multiple indices."""
    weights = torch.tensor([1.0, 2.0, 4.0, 8.0], dtype=torch.float32)
    indices = torch.tensor([0, 2], dtype=torch.long)
    bits = torch.tensor([FLOAT32_SIGN_BIT, FLOAT32_SIGN_BIT], dtype=torch.int32)

    flip_bits_in_tensor(weights, indices, bits)
    assert weights[0].item() == -1.0
    assert weights[1].item() == 2.0  # Untouched
    assert weights[2].item() == -4.0
    assert weights[3].item() == 8.0  # Untouched


def test_inject_random_seus():
    """Verifies random SEU generation and audit logging."""
    weights = torch.randn(10, 10, dtype=torch.float32)
    orig_clone = weights.clone()

    logs = inject_random_seus(weights, num_flips=5, target_field="mantissa")
    assert len(logs) == 5
    # Ensure at least some elements were mutated
    assert not torch.equal(weights, orig_clone)


def test_inject_mbu_cluster():
    """Verifies multi-bit upset injection into adjacent memory cells."""
    weights = torch.zeros(16, dtype=torch.float32)
    logs = inject_mbu_cluster(weights, center_index=4, cluster_size=3)
    assert len(logs) == 3
    # Center indices should be 4, 5, 6
    assert [log["flat_index"] for log in logs] == [4, 5, 6]
    assert all(log["is_mbu"] for log in logs)

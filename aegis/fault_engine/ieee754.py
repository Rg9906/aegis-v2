"""
IEEE 754 Binary Floating-Point Fault Injection Engine.

Provides high-speed, fully vectorized bit-manipulation of PyTorch weight tensors
for simulating radiation-induced Single Event Upsets (SEU) and Multiple Bit Upsets (MBU).
"""

from typing import Dict, List, Literal, Optional, Tuple, Union
import torch
import numpy as np


# IEEE 754 Single Precision (float32) Field Bit Indices
FLOAT32_SIGN_BIT = 31
FLOAT32_EXPONENT_MSB = 30
FLOAT32_EXPONENT_RANGE = range(23, 31)  # Bits 23 through 30
FLOAT32_MANTISSA_RANGE = range(0, 23)   # Bits 0 through 22

TargetFieldType = Literal["any", "exponent_msb", "exponent", "mantissa", "sign"]


def inspect_float32_bits(value: float) -> Dict[str, Union[str, int, float, bool]]:
    """
    Deconstructs a Python float into its exact 32-bit IEEE 754 bitfields.
    """
    tensor_val = torch.tensor(value, dtype=torch.float32)
    int_val = tensor_val.view(torch.int32).item()
    # Unpack into 32-bit unsigned representation
    uint32_val = int_val & 0xFFFFFFFF
    binary_str = f"{uint32_val:032b}"

    sign_bit = int(binary_str[0])
    exponent_bits = binary_str[1:9]
    mantissa_bits = binary_str[9:32]

    exponent_int = int(exponent_bits, 2)
    biased_exponent = exponent_int - 127

    return {
        "float_value": float(value),
        "binary_representation": f"{sign_bit}_{exponent_bits}_{mantissa_bits}",
        "sign_bit": sign_bit,
        "exponent_binary": exponent_bits,
        "exponent_decimal": exponent_int,
        "unbiased_exponent": biased_exponent,
        "mantissa_binary": mantissa_bits,
        "is_nan": bool(np.isnan(value)),
        "is_inf": bool(np.isinf(value)),
    }


def flip_bits_in_tensor(
    tensor: torch.Tensor,
    flat_indices: torch.Tensor,
    bit_positions: torch.Tensor,
) -> torch.Tensor:
    """
    Vectorized IEEE 754 bit-flip operation on arbitrary PyTorch tensor.
    Modifies weights in-place using integer reinterpretation.

    Args:
        tensor: float32 PyTorch tensor (parameter or activation).
        flat_indices: 1D integer tensor of flattened indices to corrupt.
        bit_positions: 1D integer tensor of bit positions (0-31) to flip.

    Returns:
        The mutated tensor.
    """
    if tensor.dtype != torch.float32:
        raise ValueError(f"Expected torch.float32 tensor, got {tensor.dtype}")

    orig_shape = tensor.shape
    # Reinterpret underlying raw memory as 32-bit signed integers
    flat_int_view = tensor.view(-1).view(torch.int32)

    # Compute bit masks: mask = 1 << bit_position
    masks = torch.bitwise_left_shift(
        torch.tensor(1, dtype=torch.int32, device=tensor.device),
        bit_positions.to(dtype=torch.int32, device=tensor.device),
    )

    # Apply XOR bit-flip
    flat_int_view[flat_indices] ^= masks

    # Reinterpret back to float32
    return flat_int_view.view(torch.float32).view(orig_shape)


def inject_single_bit_flip(
    tensor: torch.Tensor,
    flat_index: int,
    bit_position: int,
) -> torch.Tensor:
    """
    Injects a single bit flip at a specified flattened index and bit position.
    """
    indices = torch.tensor([flat_index], dtype=torch.long)
    bits = torch.tensor([bit_position], dtype=torch.int32)
    return flip_bits_in_tensor(tensor, indices, bits)


def sample_bit_positions(
    num_samples: int,
    target_field: TargetFieldType = "any",
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Samples bit positions (0-31) conditioned on the targeted IEEE 754 field.
    """
    if target_field == "exponent_msb":
        return torch.full((num_samples,), FLOAT32_EXPONENT_MSB, dtype=torch.int32)
    elif target_field == "exponent":
        return torch.randint(23, 31, (num_samples,), generator=generator, dtype=torch.int32)
    elif target_field == "mantissa":
        return torch.randint(0, 23, (num_samples,), generator=generator, dtype=torch.int32)
    elif target_field == "sign":
        return torch.full((num_samples,), FLOAT32_SIGN_BIT, dtype=torch.int32)
    elif target_field == "any":
        return torch.randint(0, 32, (num_samples,), generator=generator, dtype=torch.int32)
    else:
        raise ValueError(f"Unknown target_field: {target_field}")


def inject_random_seus(
    tensor: torch.Tensor,
    num_flips: int,
    target_field: TargetFieldType = "any",
    generator: Optional[torch.Generator] = None,
) -> List[Dict[str, Union[int, float]]]:
    """
    Injects random Single Event Upsets (SEUs) into a float32 tensor.

    Returns:
        List of event logs with pre- and post-flip values.
    """
    total_elements = tensor.numel()
    if total_elements == 0 or num_flips <= 0:
        return []

    # Sample random element indices
    flat_indices = torch.randint(0, total_elements, (num_flips,), generator=generator)
    bit_positions = sample_bit_positions(num_flips, target_field, generator)

    logs = []
    flat_tensor = tensor.view(-1)
    for idx, bit in zip(flat_indices.tolist(), bit_positions.tolist()):
        pre_val = float(flat_tensor[idx].item())
        # Apply flip
        inject_single_bit_flip(tensor, idx, bit)
        post_val = float(flat_tensor[idx].item())
        logs.append({
            "flat_index": idx,
            "bit_position": bit,
            "pre_value": pre_val,
            "post_value": post_val,
            "is_exponent_msb": (bit == FLOAT32_EXPONENT_MSB),
        })

    return logs


def inject_mbu_cluster(
    tensor: torch.Tensor,
    center_index: int,
    cluster_size: int = 2,
    generator: Optional[torch.Generator] = None,
) -> List[Dict[str, Union[int, float]]]:
    """
    Simulates a Multiple Bit Upset (MBU) striking adjacent memory bits/words.
    Typically accounts for ~20% of upsets measured in LEO SRAM (Alsat-1).
    """
    total_elements = tensor.numel()
    cluster_indices = [
        min(max(0, center_index + offset), total_elements - 1)
        for offset in range(cluster_size)
    ]
    # In MBUs, ionizing track often upsets bits at the same or adjacent bitlines
    base_bit = torch.randint(0, 32, (1,), generator=generator).item()
    bit_positions = [(base_bit + i) % 32 for i in range(cluster_size)]

    indices_tensor = torch.tensor(cluster_indices, dtype=torch.long)
    bits_tensor = torch.tensor(bit_positions, dtype=torch.int32)

    logs = []
    flat_tensor = tensor.view(-1)
    for idx, bit in zip(cluster_indices, bit_positions):
        pre_val = float(flat_tensor[idx].item())
        inject_single_bit_flip(tensor, idx, bit)
        post_val = float(flat_tensor[idx].item())
        logs.append({
            "flat_index": idx,
            "bit_position": bit,
            "pre_value": pre_val,
            "post_value": post_val,
            "is_mbu": True,
        })

    return logs

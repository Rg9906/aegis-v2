"""
Layer Registry and Introspection Substructure for Selective Model Management.

Provides structured metadata, parameter counting, and targeted state-dict isolation
for each logical layer block in deep convolutional networks (ResNet-18 and MobileNetV2).
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import torch
import torch.nn as nn


@dataclass
class LayerBlockMetadata:
    """Detailed specifications of an isolated, restorable network layer block."""
    block_id: str
    display_name: str
    state_dict_keys: List[str]
    param_count: int
    size_bytes_fp32: int
    size_bytes_int8: int
    total_bits_fp32: int

    @property
    def size_mb_fp32(self) -> float:
        return self.size_bytes_fp32 / (1024 * 1024)

    @property
    def size_mb_int8(self) -> float:
        return self.size_bytes_int8 / (1024 * 1024)


class LayerRegistry:
    """
    Registry that partitions a PyTorch model into discrete, restorable blocks.
    Enables layer-selective corruption, selective reloading, and energy accounting.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._blocks: Dict[str, LayerBlockMetadata] = {}
        self._key_to_block: Dict[str, str] = {}
        self._build_registry()

    def _build_registry(self) -> None:
        """Inspects model structure and groups parameters into logical restorable units."""
        raw_state_dict = self.model.state_dict()
        block_keys: Dict[str, List[str]] = {}

        for key in raw_state_dict.keys():
            # Classify key into a logical block
            block_id = self.classify_key(key)
            if block_id not in block_keys:
                block_keys[block_id] = []
            block_keys[block_id].append(key)
            self._key_to_block[key] = block_id

        # Calculate metadata for each block
        for block_id, keys in block_keys.items():
            param_count = sum(raw_state_dict[k].numel() for k in keys)
            size_fp32 = param_count * 4  # 4 bytes per float32
            size_int8 = param_count * 1  # 1 byte per int8
            total_bits = size_fp32 * 8

            self._blocks[block_id] = LayerBlockMetadata(
                block_id=block_id,
                display_name=block_id.replace("_", " ").title(),
                state_dict_keys=keys,
                param_count=param_count,
                size_bytes_fp32=size_fp32,
                size_bytes_int8=size_int8,
                total_bits_fp32=total_bits,
            )

    @staticmethod
    def classify_key(key: str) -> str:
        """
        Groups parameter keys (or module names) into logical restoration units.
        ResNet: conv1/bn1 -> stem, layerX.Y -> layerX_blockY, fc -> classifier.
        MobileNetV2: features.N -> features_blockN (N=0 is the stem conv), classifier -> classifier.
        """
        parts = key.split(".")
        if parts[0] in ("conv1", "bn1"):
            return "stem"
        if parts[0] in ("fc", "classifier"):
            return "classifier"
        if parts[0] == "features" and len(parts) >= 2:
            return f"features_block{parts[1]}"
        if parts[0].startswith("layer"):
            # e.g., layer1.0 -> layer1_block0, layer2.1 -> layer2_block1
            if len(parts) >= 2:
                return f"{parts[0]}_block{parts[1]}"
            return parts[0]
        return "other"

    @property
    def block_ids(self) -> List[str]:
        return list(self._blocks.keys())

    def get_block_metadata(self, block_id: str) -> LayerBlockMetadata:
        if block_id not in self._blocks:
            raise KeyError(f"Block '{block_id}' not found in registry. Valid: {self.block_ids}")
        return self._blocks[block_id]

    def get_block_for_param(self, param_key: str) -> str:
        return self._key_to_block.get(param_key, "unknown")

    def get_block_state_dict(self, block_id: str) -> Dict[str, torch.Tensor]:
        """Extracts cloned state-dict tensors for a specific block."""
        meta = self.get_block_metadata(block_id)
        current_sd = self.model.state_dict()
        return {k: current_sd[k].clone() for k in meta.state_dict_keys}

    def restore_block_state_dict(
        self, block_id: str, clean_block_sd: Dict[str, torch.Tensor]
    ) -> None:
        """
        Hot-patches parameters for the specified block in-place.
        Leaves all other layers untouched.
        """
        meta = self.get_block_metadata(block_id)
        with torch.no_grad():
            for key in meta.state_dict_keys:
                if key in clean_block_sd:
                    # Target parameter in model
                    target_param = self._resolve_param_by_name(key)
                    target_param.copy_(clean_block_sd[key])

    def _resolve_param_by_name(self, name: str) -> torch.Tensor:
        """Traverses named parameters or buffers to locate target tensor."""
        names = name.split(".")
        submod = self.model
        for part in names[:-1]:
            submod = getattr(submod, part)
        return getattr(submod, names[-1])

    def summary(self) -> List[Dict[str, object]]:
        """Returns structured tabular overview of all blocks and memory footprints."""
        return [
            {
                "block_id": b.block_id,
                "display_name": b.display_name,
                "params": b.param_count,
                "fp32_mb": round(b.size_mb_fp32, 3),
                "int8_mb": round(b.size_mb_int8, 3),
                "total_bits": b.total_bits_fp32,
            }
            for b in self._blocks.values()
        ]

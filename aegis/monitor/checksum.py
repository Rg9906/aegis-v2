"""
Tier 3: Deterministic Layer Block Integrity Checksum Monitor (RADAR Style).

Calculates fast CRC-32 cyclic redundancy checksums over model parameter memory blocks.
Runs asynchronously or on-demand when Tier 1 or Tier 2 flags suspicious activity.
"""

from typing import Dict, List
import zlib
import torch
import torch.nn as nn

from aegis.models.layer_registry import LayerRegistry


class ChecksumMonitor:
    """
    Computes and verifies CRC-32 checksums over physical parameter buffers.
    Provides deterministic confirmation of bit-level weight corruption.
    """

    def __init__(self, registry: LayerRegistry):
        self.registry = registry
        self._golden_checksums: Dict[str, int] = {}
        # Bytes read by integrity verifications (golden-reference computation excluded)
        self.bytes_scanned: int = 0
        self.compute_golden_checksums()

    def _hash_tensor(self, tensor: torch.Tensor) -> int:
        """Computes CRC-32 hash of contiguous raw bytes of a tensor."""
        # Ensure contiguous memory layout and get byte representation
        cpu_bytes = tensor.detach().cpu().contiguous().numpy().tobytes()
        return zlib.crc32(cpu_bytes)

    def compute_golden_checksums(self) -> None:
        """Stores baseline golden checksums for all registered layer blocks."""
        self._golden_checksums.clear()
        for block_id in self.registry.block_ids:
            block_sd = self.registry.get_block_state_dict(block_id)
            # Combine hashes of all parameters in this block
            block_crc = 0
            for name, tensor in sorted(block_sd.items()):
                tensor_crc = self._hash_tensor(tensor)
                # Combine CRC-32 values using XOR
                block_crc ^= tensor_crc
            self._golden_checksums[block_id] = block_crc

    def rebaseline_block(self, block_id: str) -> None:
        """Sets one block's reference CRC to its current contents (use only right after restoring it)."""
        block_crc = 0
        for name, tensor in sorted(self.registry.get_block_state_dict(block_id).items()):
            block_crc ^= self._hash_tensor(tensor)
        self._golden_checksums[block_id] = block_crc

    def verify_block_integrity(self, block_id: str) -> bool:
        """
        Verifies if current block parameters match their golden CRC-32 checksum.
        Returns True if intact, False if any bit has flipped.
        """
        if block_id not in self._golden_checksums:
            raise KeyError(f"Block '{block_id}' not initialized in checksum table.")

        block_sd = self.registry.get_block_state_dict(block_id)
        current_crc = 0
        for name, tensor in sorted(block_sd.items()):
            tensor_crc = self._hash_tensor(tensor)
            current_crc ^= tensor_crc
            self.bytes_scanned += tensor.numel() * tensor.element_size()

        return current_crc == self._golden_checksums[block_id]

    def scan_all_blocks(self) -> List[str]:
        """
        Scans all registered layer blocks and returns a list of block_ids with corrupt checksums.
        """
        corrupted_blocks = []
        for block_id in self.registry.block_ids:
            if not self.verify_block_integrity(block_id):
                corrupted_blocks.append(block_id)
        return corrupted_blocks

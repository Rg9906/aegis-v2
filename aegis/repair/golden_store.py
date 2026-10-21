"""
Rad-Tolerant Compressed Golden Store Substructure.

Simulates onboard radiation-hardened companion non-volatile memory (e.g. SPI NOR Flash / MRAM).
Stores layer block parameters compressed with zstandard (zstd) and serialized via safetensors.
Enables sub-millisecond, granular block retrieval with exact energy and byte accounting.
"""

import io
from typing import Dict, List, Optional
import torch
import zstandard as zstd
from safetensors.torch import load, save

from aegis.config import DEFAULT_BUDGET_CONFIG, SatellitePayloadBudgetConfig
from aegis.models.layer_registry import LayerRegistry


class CompressedGoldenStore:
    """
    Onboard compressed golden copy repository.
    Stores partitioned layer blocks independently to support selective, low-energy reloading.
    """

    def __init__(
        self,
        registry: LayerRegistry,
        compression_level: int = 3,
        budget_config: SatellitePayloadBudgetConfig = DEFAULT_BUDGET_CONFIG,
        quantize_int8: bool = False,
    ):
        """
        quantize_int8: store conv/linear weights (dim >= 2) as per-output-channel symmetric INT8
        plus fp32 scales; 1-D tensors (BN, biases) stay fp32. Restores are then LOSSY: the
        restored block differs from the original, so callers must re-baseline that block's
        Tier 3 checksum (AegisHealthMonitor.rebaseline_blocks) and may want BN recalibration.
        """
        self.registry = registry
        self.compression_level = compression_level
        self.quantize_int8 = quantize_int8
        self.budget_config = budget_config
        self.cctx = zstd.ZstdCompressor(level=compression_level)
        self.dctx = zstd.ZstdDecompressor()

        # Storage mapping: block_id -> compressed bytes
        self._compressed_blocks: Dict[str, bytes] = {}
        self._uncompressed_sizes: Dict[str, int] = {}
        self._cumulative_bytes_read: int = 0
        self._cumulative_energy_uj: float = 0.0

        self.initialize_store()

    def initialize_store(self) -> None:
        """Serializes and compresses all model blocks into onboard protected storage."""
        self._compressed_blocks.clear()
        self._uncompressed_sizes.clear()

        for block_id in self.registry.block_ids:
            block_sd = self.registry.get_block_state_dict(block_id)
            if self.quantize_int8:
                block_sd = self._quantize(block_sd)
            # Serialize tensors using safetensors binary format
            raw_safetensors_bytes = save(block_sd)
            self._uncompressed_sizes[block_id] = len(raw_safetensors_bytes)

            # Compress using zstandard
            compressed_data = self.cctx.compress(raw_safetensors_bytes)
            self._compressed_blocks[block_id] = compressed_data

    @property
    def total_compressed_bytes(self) -> int:
        return sum(len(b) for b in self._compressed_blocks.values())

    @property
    def total_uncompressed_bytes(self) -> int:
        return sum(self._uncompressed_sizes.values())

    @property
    def overall_compression_ratio(self) -> float:
        return (
            self.total_uncompressed_bytes / self.total_compressed_bytes
            if self.total_compressed_bytes > 0
            else 1.0
        )

    def retrieve_block(self, block_id: str) -> Dict[str, torch.Tensor]:
        """
        Retrieves, decompresses, and deserializes a single layer block from protected storage.
        Tracks memory bus traffic and energy cost.
        """
        if block_id not in self._compressed_blocks:
            raise KeyError(f"Block '{block_id}' not found in golden store.")

        compressed_data = self._compressed_blocks[block_id]
        decompressed_bytes = self.dctx.decompress(compressed_data)

        # Deserialize safetensors into PyTorch tensors
        block_sd = load(decompressed_bytes)
        if self.quantize_int8:
            block_sd = self._dequantize(block_sd)

        # Accounting: Bytes transferred from SPI Flash + Written to DRAM (as restored fp32 tensors)
        compressed_len = len(compressed_data)
        uncompressed_len = sum(t.numel() * t.element_size() for t in block_sd.values())
        self._cumulative_bytes_read += compressed_len

        # Energy calculation:
        # Read from SPI flash (pj/bit) + write decompressed to DRAM (pj/bit)
        read_energy_pj = compressed_len * 8 * self.budget_config.spi_flash_read_energy_pj_per_bit
        write_energy_pj = uncompressed_len * 8 * self.budget_config.dram_write_energy_pj_per_bit
        total_energy_uj = (read_energy_pj + write_energy_pj) / 1e6  # Convert pJ to microjoules (µJ)
        self._cumulative_energy_uj += total_energy_uj

        return block_sd

    @staticmethod
    def _quantize(block_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for key, t in block_sd.items():
            if t.dtype == torch.float32 and t.dim() >= 2:
                flat = t.reshape(t.size(0), -1)
                scale = flat.abs().amax(dim=1).clamp_min(1e-12) / 127.0
                q = torch.round(flat / scale[:, None]).clamp(-127, 127).to(torch.int8)
                out[key + ".__q"] = q.reshape(t.shape).contiguous()
                out[key + ".__scale"] = scale.contiguous()
            else:
                out[key] = t
        return out

    @staticmethod
    def _dequantize(stored: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for key, t in stored.items():
            if key.endswith(".__scale"):
                continue
            if key.endswith(".__q"):
                base = key[: -len(".__q")]
                scale = stored[base + ".__scale"]
                shape = [-1] + [1] * (t.dim() - 1)
                out[base] = t.to(torch.float32) * scale.view(shape)
            else:
                out[key] = t
        return out

    def get_block_storage_info(self, block_id: str) -> Dict[str, float]:
        """Returns size and compression statistics for a specific block."""
        comp_size = len(self._compressed_blocks[block_id])
        uncomp_size = self._uncompressed_sizes[block_id]
        return {
            "compressed_bytes": comp_size,
            "uncompressed_bytes": uncomp_size,
            "compression_ratio": round(uncomp_size / comp_size, 2) if comp_size > 0 else 1.0,
            "compressed_kb": round(comp_size / 1024.0, 2),
        }

    def reset_telemetry(self) -> None:
        """Resets cumulative byte and energy meters."""
        self._cumulative_bytes_read = 0
        self._cumulative_energy_uj = 0.0

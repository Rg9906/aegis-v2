"""
Model-Level Fault Injection Substructure.

Orchestrates bit-level radiation strikes across the neural network parameter space,
maintains audit logs, and handles clean rollback to baseline golden weights.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn

from aegis.fault_engine.ieee754 import (
    FLOAT32_EXPONENT_MSB,
    TargetFieldType,
    inject_mbu_cluster,
    inject_random_seus,
    inject_single_bit_flip,
)
from aegis.fault_engine.radiation_model import OrbitalState
from aegis.models.layer_registry import LayerRegistry


@dataclass
class InjectedFaultRecord:
    """Record of a physical bit-flip injected into model memory."""
    fault_id: int
    block_id: str
    param_name: str
    flat_index: int
    bit_position: int
    pre_value: float
    post_value: float
    is_exponent_msb: bool
    is_mbu: bool = False


class ModelFaultInjector:
    """
    High-level fault injector that applies physical bit-level damage to a PyTorch model.
    Tracks all mutations and coordinates with the LayerRegistry.
    """

    def __init__(
        self,
        model: nn.Module,
        registry: LayerRegistry,
        generator: Optional[torch.Generator] = None,
    ):
        self.model = model
        self.registry = registry
        self.generator = generator or torch.Generator()
        self._fault_history: List[InjectedFaultRecord] = []
        self._golden_weights_backup: Dict[str, torch.Tensor] = {
            k: v.clone() for k, v in self.model.state_dict().items()
        }

    @property
    def fault_history(self) -> List[InjectedFaultRecord]:
        return self._fault_history

    @property
    def total_faults_injected(self) -> int:
        return len(self._fault_history)

    def inject_targeted_bit_flip(
        self,
        block_id: str,
        param_name: Optional[str] = None,
        flat_index: Optional[int] = None,
        bit_position: int = FLOAT32_EXPONENT_MSB,
    ) -> InjectedFaultRecord:
        """
        Injects a single bit-flip into a targeted layer block and parameter.
        """
        meta = self.registry.get_block_metadata(block_id)
        # Select parameter tensor
        if param_name is None:
            # Pick the primary weight tensor in this block (skip 1D biases/running stats if possible)
            weight_keys = [k for k in meta.state_dict_keys if "weight" in k and "bn" not in k]
            param_name = weight_keys[0] if weight_keys else meta.state_dict_keys[0]

        target_tensor = self.registry._resolve_param_by_name(param_name)
        total_elements = target_tensor.numel()

        if flat_index is None:
            flat_index = torch.randint(0, total_elements, (1,), generator=self.generator).item()

        flat_tensor = target_tensor.view(-1)
        pre_val = float(flat_tensor[flat_index].item())

        # Perform bit-flip
        inject_single_bit_flip(target_tensor, flat_index, bit_position)
        post_val = float(flat_tensor[flat_index].item())

        record = InjectedFaultRecord(
            fault_id=len(self._fault_history) + 1,
            block_id=block_id,
            param_name=param_name,
            flat_index=flat_index,
            bit_position=bit_position,
            pre_value=pre_val,
            post_value=post_val,
            is_exponent_msb=(bit_position == FLOAT32_EXPONENT_MSB),
            is_mbu=False,
        )
        self._fault_history.append(record)
        return record

    def inject_orbital_flux_step(
        self, orbital_state: OrbitalState, target_field: TargetFieldType = "any"
    ) -> List[InjectedFaultRecord]:
        """
        Applies the SBUs and MBUs generated during an orbital time interval.
        Distributes flips across model blocks weighted proportionally to their bit footprint.
        """
        records: List[InjectedFaultRecord] = []
        blocks = [self.registry.get_block_metadata(bid) for bid in self.registry.block_ids]
        total_model_bits = sum(b.total_bits_fp32 for b in blocks)
        block_probs = [b.total_bits_fp32 / total_model_bits for b in blocks]

        # 1. Inject Single Bit Upsets (SBUs)
        for _ in range(orbital_state.total_sbus_generated):
            # Select block weighted by bit count
            chosen_block_meta = np.random.choice(blocks, p=block_probs)
            weight_keys = [k for k in chosen_block_meta.state_dict_keys if "weight" in k]
            param_name = np.random.choice(weight_keys if weight_keys else chosen_block_meta.state_dict_keys)
            target_tensor = self.registry._resolve_param_by_name(param_name)

            sbu_logs = inject_random_seus(target_tensor, num_flips=1, target_field=target_field, generator=self.generator)
            if sbu_logs:
                log = sbu_logs[0]
                rec = InjectedFaultRecord(
                    fault_id=len(self._fault_history) + 1,
                    block_id=chosen_block_meta.block_id,
                    param_name=param_name,
                    flat_index=int(log["flat_index"]),
                    bit_position=int(log["bit_position"]),
                    pre_value=float(log["pre_value"]),
                    post_value=float(log["post_value"]),
                    is_exponent_msb=bool(log["is_exponent_msb"]),
                    is_mbu=False,
                )
                self._fault_history.append(rec)
                records.append(rec)

        # 2. Inject Multiple Bit Upsets (MBUs)
        for _ in range(orbital_state.total_mbus_generated):
            chosen_block_meta = np.random.choice(blocks, p=block_probs)
            weight_keys = [k for k in chosen_block_meta.state_dict_keys if "weight" in k]
            param_name = np.random.choice(weight_keys if weight_keys else chosen_block_meta.state_dict_keys)
            target_tensor = self.registry._resolve_param_by_name(param_name)

            center_idx = torch.randint(0, target_tensor.numel(), (1,), generator=self.generator).item()
            mbu_logs = inject_mbu_cluster(target_tensor, center_index=center_idx, cluster_size=2, generator=self.generator)

            for log in mbu_logs:
                rec = InjectedFaultRecord(
                    fault_id=len(self._fault_history) + 1,
                    block_id=chosen_block_meta.block_id,
                    param_name=param_name,
                    flat_index=int(log["flat_index"]),
                    bit_position=int(log["bit_position"]),
                    pre_value=float(log["pre_value"]),
                    post_value=float(log["post_value"]),
                    is_exponent_msb=(log["bit_position"] == FLOAT32_EXPONENT_MSB),
                    is_mbu=True,
                )
                self._fault_history.append(rec)
                records.append(rec)

        return records

    def rollback_all_faults(self) -> None:
        """Restores model parameters to pristine baseline state and clears logs."""
        with torch.no_grad():
            for key, clean_tensor in self._golden_weights_backup.items():
                param = self.registry._resolve_param_by_name(key)
                param.copy_(clean_tensor)
        self._fault_history.clear()

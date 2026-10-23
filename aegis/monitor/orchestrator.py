"""
Unified Multi-Tier Health Monitor Orchestrator.

Combines:
- Tier 1: Activation Magnitude Envelope (Ranger Bounds)
- Tier 2: Batch Normalization Feature Drift (BLINK Drift)
- Tier 3: Deterministic CRC-32 Checksums (RADAR Checksums)

Maps neural network layer anomalies directly to restorable LayerRegistry blocks.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import torch
import torch.nn as nn

from aegis.models.layer_registry import LayerRegistry
from aegis.monitor.activation_envelope import ActivationEnvelopeMonitor
from aegis.monitor.bn_drift import BNDriftMonitor
from aegis.monitor.checksum import ChecksumMonitor


@dataclass
class HealthReport:
    """Consolidated diagnostic report emitted after an inference step or diagnostic scan."""
    is_healthy: bool
    flagged_blocks: List[str] = field(default_factory=list)
    primary_suspect_block: Optional[str] = None
    tier1_violations: Dict[str, Dict[str, object]] = field(default_factory=dict)
    tier2_drifts: Dict[str, Dict[str, float]] = field(default_factory=dict)
    tier3_corrupted_blocks: List[str] = field(default_factory=list)
    # Blocks T1/T2 implicated (plus the upstream block) before T3 confirmation, forward order
    candidate_blocks: List[str] = field(default_factory=list)

    @property
    def has_catastrophic_anomaly(self) -> bool:
        """Indicates presence of NaNs, Infs, or massive exponent-MSB activation spikes."""
        return any(
            v.get("has_nan", False) or v.get("has_inf", False) or abs(float(v.get("max_val", 0.0))) > 1e10
            for v in self.tier1_violations.values()
        )


class AegisHealthMonitor:
    """
    Hierarchical runtime monitoring system for satellite neural network payloads.
    Provides multi-tier fault detection with minimal latency and zero ground contact.
    """

    def __init__(self, model: nn.Module, registry: LayerRegistry):
        self.model = model
        self.registry = registry

        self.tier1_envelope = ActivationEnvelopeMonitor()
        self.tier2_bn_drift = BNDriftMonitor()
        self.tier3_checksum = ChecksumMonitor(registry)

    def calibrate(self, calibration_batch: torch.Tensor) -> None:
        """
        Profiles baseline behavior on clean calibration imagery.
        Sets up Tier 1 envelopes, Tier 2 BN tolerances, and Tier 3 golden checksums.
        """
        self.tier1_envelope.profile_clean_bounds(self.model, calibration_batch)
        self.tier2_bn_drift.calibrate_clean_reference(self.model, calibration_batch)
        self.tier3_checksum.compute_golden_checksums()

    def copy_calibration_from(self, other: "AegisHealthMonitor") -> None:
        """
        Reuses another monitor's T1 bounds and T2 tolerances instead of re-profiling, and takes
        T3 references from this monitor's own model. Valid only when both models are clean
        copies of the same weights (calibration is a deterministic function of the weights and
        the calibration imagery), e.g. replicas deep-copied from a calibrated golden model.
        """
        self.tier1_envelope._bounds = dict(other.tier1_envelope._bounds)
        self.tier2_bn_drift._calibrated_tolerances = dict(other.tier2_bn_drift._calibrated_tolerances)
        self.tier3_checksum.compute_golden_checksums()

    def attach(self) -> None:
        """Attaches runtime hooks to active model."""
        self.tier1_envelope.attach(self.model)
        self.tier2_bn_drift.attach(self.model)

    def detach(self) -> None:
        """Detaches runtime monitoring hooks."""
        self.tier1_envelope.detach()
        self.tier2_bn_drift.detach()

    def refresh_golden_checksums(self) -> None:
        """
        Recomputes the Tier 3 golden CRC-32 checksums from the *current* model state.

        WARNING: any corruption present at call time (e.g. in blocks that were not restored)
        becomes part of the new reference and can never again be confirmed by Tier 3.
        Only call this when the model is known to be fully clean (e.g. right after a full
        reload, or after deliberately re-baselining BN statistics following a lossy restore).
        The lossless golden-store pipeline does not need it.
        """
        self.tier3_checksum.compute_golden_checksums()

    def rebaseline_blocks(self, block_ids: List[str]) -> None:
        """
        Re-baselines Tier 3 for blocks that were just restored from a LOSSY (INT8) golden copy.
        Unlike refresh_golden_checksums(), this only touches blocks whose contents are known-good.
        """
        for block_id in block_ids:
            self.tier3_checksum.rebaseline_block(block_id)

    def _map_layer_name_to_block_id(self, layer_name: str) -> str:
        """Maps PyTorch module name (e.g. 'backbone.layer2.0.conv1') to LayerRegistry block_id."""
        # Same grouping as the registry's parameter keys, so alarms map to restorable blocks
        block_id = LayerRegistry.classify_key(layer_name.replace("backbone.", ""))
        return block_id if block_id in self.registry.block_ids else self.registry.block_ids[0]

    def diagnose_current_state(self, run_tier3_scan: bool = False) -> HealthReport:
        """
        Synthesizes active violations into a structured HealthReport.
        Maps any detected layer faults back to restorable LayerRegistry blocks.
        """
        flagged_blocks = set()

        # 1. Process Tier 1 (Envelope violations)
        t1_violations = dict(self.tier1_envelope.latest_violations)
        candidate_blocks = set()
        for layer_name in t1_violations.keys():
            block_id = self._map_layer_name_to_block_id(layer_name)
            candidate_blocks.add(block_id)

        # 2. Process Tier 2 (BLINK BN Drifts)
        t2_drifts = dict(self.tier2_bn_drift.latest_drifts)
        for layer_name, info in t2_drifts.items():
            if info.get("is_anomalous", False):
                block_id = self._map_layer_name_to_block_id(layer_name)
                candidate_blocks.add(block_id)

        # T1 hooks conv outputs and T2 hooks BN inputs, so a fault in a block's BN parameters or
        # output path (e.g. a sign flip on running_var -> NaN) first shows in the NEXT block.
        # Also check the block just upstream of the earliest flagged one.
        forward_order = {bid: i for i, bid in enumerate(self.registry.block_ids)}
        if candidate_blocks:
            earliest = min(forward_order.get(b, 0) for b in candidate_blocks)
            if earliest > 0:
                candidate_blocks.add(self.registry.block_ids[earliest - 1])

        # 3. Tier 3 (CRC Checksums): Deterministic confirmation
        # Fast filter-and-confirm: only confirm blocks that fail CRC-32 checksum
        t3_corrupted = []
        confirmed_blocks = set()

        if run_tier3_scan or len(candidate_blocks) > 0:
            blocks_to_check = list(candidate_blocks) if candidate_blocks else (self.registry.block_ids if run_tier3_scan else [])
            for bid in blocks_to_check:
                if not self.tier3_checksum.verify_block_integrity(bid):
                    t3_corrupted.append(bid)
                    confirmed_blocks.add(bid)
        else:
            confirmed_blocks = candidate_blocks

        is_healthy = (len(confirmed_blocks) == 0)
        # Order by forward-pass position (registry order: stem -> layer1..4 -> classifier),
        # so the primary suspect is the most upstream confirmed block, not the alphabetical first
        sorted_flagged = sorted(confirmed_blocks, key=lambda b: forward_order.get(b, len(forward_order)))
        primary_suspect = sorted_flagged[0] if sorted_flagged else None

        return HealthReport(
            is_healthy=is_healthy,
            flagged_blocks=sorted_flagged,
            primary_suspect_block=primary_suspect,
            tier1_violations=t1_violations,
            tier2_drifts=t2_drifts,
            tier3_corrupted_blocks=t3_corrupted,
            candidate_blocks=sorted(candidate_blocks, key=lambda b: forward_order.get(b, len(forward_order))),
        )

    def clear(self) -> None:
        """Resets violation and drift tracking states for the next inference batch."""
        self.tier1_envelope.clear_violations()
        self.tier2_bn_drift.clear_drifts()

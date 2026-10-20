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
        Recomputes the Tier 3 golden CRC-32 checksums to reflect the current model state.
        Must be called after any selective restore + BN recalibration to prevent
        false-positive checksum alarms from updated BN running statistics.
        """
        self.tier3_checksum.compute_golden_checksums()

    def _map_layer_name_to_block_id(self, layer_name: str) -> str:
        """Maps PyTorch module name (e.g. 'backbone.layer2.0.conv1') to LayerRegistry block_id."""
        clean_name = layer_name.replace("backbone.", "")
        parts = clean_name.split(".")
        if parts[0] in ("conv1", "bn1"):
            return "stem"
        if parts[0] == "fc":
            return "classifier"
        if parts[0].startswith("layer") and len(parts) >= 2:
            return f"{parts[0]}_block{parts[1]}"
        return "stem"

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
        sorted_flagged = sorted(list(confirmed_blocks))
        primary_suspect = sorted_flagged[0] if sorted_flagged else None

        return HealthReport(
            is_healthy=is_healthy,
            flagged_blocks=sorted_flagged,
            primary_suspect_block=primary_suspect,
            tier1_violations=t1_violations,
            tier2_drifts=t2_drifts,
            tier3_corrupted_blocks=t3_corrupted,
        )

    def clear(self) -> None:
        """Resets violation and drift tracking states for the next inference batch."""
        self.tier1_envelope.clear_violations()
        self.tier2_bn_drift.clear_drifts()

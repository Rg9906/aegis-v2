"""
Detection-Gated Layer-Selective Restore Substructure.

Consumes diagnostic health reports from the AegisHealthMonitor,
fetches compressed slices from the CompressedGoldenStore,
and hot-patches afflicted model blocks with zero payload reboot or ground contact.
"""

from dataclasses import dataclass, field
import time
from typing import Dict, List, Optional
import torch
import torch.nn as nn

from aegis.models.layer_registry import LayerRegistry
from aegis.monitor.orchestrator import HealthReport
from aegis.repair.golden_store import CompressedGoldenStore
from aegis.repair.recalibrate import BNRecalibrator, RecalibrationReport


@dataclass
class RepairEvent:
    """Telemetry record generated upon executing a layer restoration action."""
    restored_blocks: List[str]
    total_bytes_transferred: int
    total_energy_uj: float
    downtime_ms: float
    recalibration_report: Optional[RecalibrationReport] = None
    was_successful: bool = True
    # Breakdown of total_energy_uj: golden-store read + DRAM write vs. recalibration inference
    memory_energy_uj: float = 0.0
    recalibration_energy_uj: float = 0.0


class DetectionGatedRestorer:
    """
    Onboard autonomous repair controller.
    Gates restoration strictly behind behavior-derived monitor alerts.
    """

    def __init__(
        self,
        registry: LayerRegistry,
        golden_store: CompressedGoldenStore,
        recalibrator: Optional[BNRecalibrator] = None,
    ):
        self.registry = registry
        self.golden_store = golden_store
        self.recalibrator = recalibrator
        self._repair_history: List[RepairEvent] = []

    @property
    def repair_history(self) -> List[RepairEvent]:
        return self._repair_history

    @property
    def total_restorations_performed(self) -> int:
        return len(self._repair_history)

    @property
    def cumulative_energy_uj(self) -> float:
        """Total repair energy including recalibration inference (not just golden-store reads)."""
        return sum(e.total_energy_uj for e in self._repair_history)

    @property
    def cumulative_bytes_transferred(self) -> int:
        return sum(e.total_bytes_transferred for e in self._repair_history)

    def process_health_report(
        self,
        report: HealthReport,
        auto_recalibrate: bool = True,
    ) -> Optional[RepairEvent]:
        """
        Evaluates health report. If anomalies are detected, restores the confirmed blocks.

        Architecture note: an exponent-field SEU in one block makes activations explode in every
        downstream layer, so Tier 1/2 flag many *candidate* blocks. The monitor filters these
        through Tier 3 CRC-32, so report.tier3_corrupted_blocks contains only blocks whose stored
        bits actually differ from the golden reference. Those are all restored (cascade victims
        pass CRC and are left alone). Without Tier 3 confirmation, only the most upstream
        candidate (primary suspect) is restored.

        auto_recalibrate: the golden store is lossless, so a restored block is bit-exact and
        needs no recalibration. Recalibration mutates BN running statistics in *every* block
        (invalidating Tier 3 references) and costs one inference per buffered image; enable it
        only for lossy (e.g. INT8) golden copies.
        """
        if report.is_healthy or not report.flagged_blocks:
            return None  # No damage detected, skip repair

        start_time = time.perf_counter()
        initial_bytes = self.golden_store._cumulative_bytes_read
        initial_energy = self.golden_store._cumulative_energy_uj

        # Restore only the primary suspect block (root cause), not all downstream cascade victims
        # primary_suspect_block is the earliest flagged block in forward-pass order
        if report.tier3_corrupted_blocks:
            forward_order = {bid: i for i, bid in enumerate(self.registry.block_ids)}
            blocks_to_restore = sorted(set(report.tier3_corrupted_blocks), key=forward_order.get)
        elif report.primary_suspect_block:
            blocks_to_restore = [report.primary_suspect_block]
        else:
            blocks_to_restore = list(report.flagged_blocks)

        for block_id in blocks_to_restore:
            clean_block_sd = self.golden_store.retrieve_block(block_id)
            self.registry.restore_block_state_dict(block_id, clean_block_sd)

        # Optional post-repair label-free Batch Normalization recalibration
        recal_report = None
        if auto_recalibrate and self.recalibrator is not None and self.recalibrator.buffer_size > 0:
            recal_report = self.recalibrator.recalibrate(self.registry.model)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        bytes_transferred = self.golden_store._cumulative_bytes_read - initial_bytes
        memory_energy_uj = self.golden_store._cumulative_energy_uj - initial_energy
        # Each recalibration image costs one full forward pass on the payload accelerator
        recal_energy_uj = (
            recal_report.samples_processed
            * self.golden_store.budget_config.inference_energy_mj_per_image
            * 1000.0
            if recal_report is not None
            else 0.0
        )

        event = RepairEvent(
            restored_blocks=list(blocks_to_restore),
            total_bytes_transferred=bytes_transferred,
            total_energy_uj=memory_energy_uj + recal_energy_uj,
            downtime_ms=round(elapsed_ms, 2),
            recalibration_report=recal_report,
            was_successful=True,
            memory_energy_uj=memory_energy_uj,
            recalibration_energy_uj=recal_energy_uj,
        )
        self._repair_history.append(event)
        return event

    def restore_full_model(self) -> RepairEvent:
        """
        Executes a blind full reload of ALL blocks.
        Used as the comparative baseline (Periodic Blind Scrubbing).
        """
        start_time = time.perf_counter()
        initial_bytes = self.golden_store._cumulative_bytes_read
        initial_energy = self.golden_store._cumulative_energy_uj

        for block_id in self.registry.block_ids:
            clean_block_sd = self.golden_store.retrieve_block(block_id)
            self.registry.restore_block_state_dict(block_id, clean_block_sd)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        bytes_transferred = self.golden_store._cumulative_bytes_read - initial_bytes
        energy_uj = self.golden_store._cumulative_energy_uj - initial_energy

        event = RepairEvent(
            restored_blocks=list(self.registry.block_ids),
            total_bytes_transferred=bytes_transferred,
            total_energy_uj=energy_uj,
            downtime_ms=round(elapsed_ms, 2),
            was_successful=True,
            memory_energy_uj=energy_uj,
        )
        self._repair_history.append(event)
        return event

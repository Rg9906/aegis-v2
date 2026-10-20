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

    def process_health_report(
        self,
        report: HealthReport,
        auto_recalibrate: bool = True,
    ) -> Optional[RepairEvent]:
        """
        Evaluates health report. If anomalies are detected, restores afflicted blocks.
        """
        if report.is_healthy or not report.flagged_blocks:
            return None  # No damage detected, skip repair

        start_time = time.perf_counter()
        initial_bytes = self.golden_store._cumulative_bytes_read
        initial_energy = self.golden_store._cumulative_energy_uj

        # Restore each flagged block selectively
        for block_id in report.flagged_blocks:
            clean_block_sd = self.golden_store.retrieve_block(block_id)
            self.registry.restore_block_state_dict(block_id, clean_block_sd)

        # Optional post-repair label-free Batch Normalization recalibration
        recal_report = None
        if auto_recalibrate and self.recalibrator is not None and self.recalibrator.buffer_size > 0:
            recal_report = self.recalibrator.recalibrate(self.registry.model)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        bytes_transferred = self.golden_store._cumulative_bytes_read - initial_bytes
        energy_uj = self.golden_store._cumulative_energy_uj - initial_energy

        event = RepairEvent(
            restored_blocks=list(report.flagged_blocks),
            total_bytes_transferred=bytes_transferred,
            total_energy_uj=energy_uj,
            downtime_ms=round(elapsed_ms, 2),
            recalibration_report=recal_report,
            was_successful=True,
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
        )
        self._repair_history.append(event)
        return event

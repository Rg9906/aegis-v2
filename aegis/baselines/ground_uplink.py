"""
Ground-Uplink Reload Baseline Substructure.

The satellite carries no onboard golden copy. Faults are detected onboard with the same
label-free monitor as Aegis-v2, but replacement weights must be uplinked from the ground
station, which is only possible during contact windows and over a narrow link
(Chaohu-1 S-band ~1 Mbps). Damage therefore persists until the next contact.
"""

from dataclasses import dataclass
from typing import List, Optional, Set

from safetensors.torch import load

from aegis.config import DEFAULT_BUDGET_CONFIG, SatellitePayloadBudgetConfig
from aegis.models.layer_registry import LayerRegistry
from aegis.monitor.orchestrator import HealthReport
from aegis.repair.golden_store import CompressedGoldenStore


@dataclass
class UplinkEvent:
    """A contact window in which corrupted blocks were (partially) re-uplinked."""
    elapsed_minutes: float
    restored_blocks: List[str]
    bytes_uplinked: int
    energy_uj: float
    link_seconds: float


class GroundUplinkReloader:
    """
    Queues CRC-confirmed corrupted blocks and reloads them during ground contacts.

    Contact windows are evenly spaced: start at offset + k * (1440 / passes_per_day) minutes,
    lasting ground_pass_duration_seconds. Each window can carry at most
    duration * uplink_bandwidth_bps bits; blocks that do not fit wait for the next window.
    The ground segment keeps the golden weights; `ground_copy` stands in for it (its zstd-compressed
    size is what crosses the link) and its onboard read energy is NOT charged.
    """

    def __init__(
        self,
        registry: LayerRegistry,
        ground_copy: CompressedGoldenStore,
        budget: SatellitePayloadBudgetConfig = DEFAULT_BUDGET_CONFIG,
        first_contact_offset_minutes: float = 90.0,
    ):
        self.registry = registry
        self.ground_copy = ground_copy
        self.budget = budget
        self.offset = first_contact_offset_minutes
        self.interval = 1440.0 / budget.ground_passes_per_day
        self.pending: Set[str] = set()
        self.events: List[UplinkEvent] = []

    @property
    def cumulative_bytes(self) -> int:
        return sum(e.bytes_uplinked for e in self.events)

    @property
    def cumulative_energy_uj(self) -> float:
        return sum(e.energy_uj for e in self.events)

    def report(self, health: HealthReport) -> None:
        """Queue blocks the onboard monitor confirmed as corrupted (telemetry to the ground)."""
        if not health.is_healthy:
            self.pending.update(health.tier3_corrupted_blocks or health.flagged_blocks)

    def _contact_in(self, start_minutes: float, end_minutes: float) -> bool:
        k = max(0, int((start_minutes - self.offset) // self.interval))
        while True:
            window_start = self.offset + k * self.interval
            if window_start >= end_minutes:
                return False
            if window_start >= start_minutes:
                return True
            k += 1

    def step(self, start_minutes: float, end_minutes: float) -> Optional[UplinkEvent]:
        """If a contact window opens in [start, end), uplink as many pending blocks as fit."""
        if not self.pending or not self._contact_in(start_minutes, end_minutes):
            return None

        capacity_bytes = self.budget.ground_pass_duration_seconds * self.budget.uplink_bandwidth_bps / 8.0
        forward_order = {bid: i for i, bid in enumerate(self.registry.block_ids)}
        restored, sent, energy_pj = [], 0, 0.0
        for block_id in sorted(self.pending, key=forward_order.get):
            info = self.ground_copy.get_block_storage_info(block_id)
            if sent + info["compressed_bytes"] > capacity_bytes:
                break
            # Decompress the ground copy without touching its onboard-energy meters
            clean = self._ground_block(block_id)
            self.registry.restore_block_state_dict(block_id, clean)
            sent += int(info["compressed_bytes"])
            energy_pj += info["compressed_bytes"] * 8 * self.budget.radio_rx_energy_pj_per_bit
            energy_pj += info["uncompressed_bytes"] * 8 * self.budget.dram_write_energy_pj_per_bit
            restored.append(block_id)

        if not restored:
            return None
        self.pending.difference_update(restored)
        event = UplinkEvent(
            elapsed_minutes=end_minutes,
            restored_blocks=restored,
            bytes_uplinked=sent,
            energy_uj=energy_pj / 1e6,
            link_seconds=sent * 8 / self.budget.uplink_bandwidth_bps,
        )
        self.events.append(event)
        return event

    def _ground_block(self, block_id: str):
        return load(self.ground_copy.dctx.decompress(self.ground_copy._compressed_blocks[block_id]))

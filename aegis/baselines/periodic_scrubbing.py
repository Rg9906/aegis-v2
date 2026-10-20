"""
Periodic Blind Scrubbing Baseline Substructure.

Executes a full reload of all model parameters on a rigid, fixed timer
regardless of whether faults have occurred, modeling traditional spacecraft memory scrubbing.
"""

from typing import Optional
from aegis.repair.selective_restore import DetectionGatedRestorer, RepairEvent


class PeriodicScrubbingController:
    """
    Simulates fixed-interval blind memory scrubbing.
    Reloads the full model every `scrub_interval_hours`.
    """

    def __init__(self, restorer: DetectionGatedRestorer, scrub_interval_hours: float = 12.0):
        self.restorer = restorer
        self.scrub_interval_hours = scrub_interval_hours
        self.scrub_interval_minutes = scrub_interval_hours * 60.0
        self.last_scrub_minute: float = 0.0
        self.total_scrubs_executed: int = 0

    def step(self, current_sim_minutes: float) -> Optional[RepairEvent]:
        """
        Checks if elapsed time exceeds scrub interval. If so, triggers full reload.
        """
        if current_sim_minutes - self.last_scrub_minute >= self.scrub_interval_minutes:
            self.last_scrub_minute = current_sim_minutes
            self.total_scrubs_executed += 1
            return self.restorer.restore_full_model()
        return None

    def reset(self) -> None:
        self.last_scrub_minute = 0.0
        self.total_scrubs_executed = 0

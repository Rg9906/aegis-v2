"""
Low Earth Orbit (LEO) Physical Radiation Environment Simulator.

Implements orbit propagation, South Atlantic Anomaly (SAA) transit modeling,
and Poisson-distributed Single-Event Upset (SEU) and Multi-Bit Upset (MBU) generation.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np

from aegis.config import DEFAULT_RAD_CONFIG, RadiationEnvironmentConfig


@dataclass
class OrbitalState:
    """Telemetry of the simulated satellite orbital position and radiation flux."""
    elapsed_minutes: float
    elapsed_days: float
    orbit_index: int
    in_saa: bool
    instantaneous_flux_multiplier: float
    expected_seu_rate_per_bit_day: float
    total_sbus_generated: int
    total_mbus_generated: int
    saa_minutes_in_step: float = 0.0


class LEORadiationEnvironment:
    """
    Simulates physical radiation exposure for a satellite payload in Low Earth Orbit (~600 km).
    Incorporates the South Atlantic Anomaly (SAA) where >73% of soft errors accumulate.
    """

    def __init__(
        self,
        config: RadiationEnvironmentConfig = DEFAULT_RAD_CONFIG,
        random_seed: Optional[int] = 42,
    ):
        self.config = config
        self.rng = np.random.default_rng(random_seed)
        self.elapsed_minutes: float = 0.0
        self.cumulative_sbus: int = 0
        self.cumulative_mbus: int = 0

    @property
    def elapsed_days(self) -> float:
        return self.elapsed_minutes / 1440.0

    @property
    def current_orbit_index(self) -> int:
        return int(self.elapsed_minutes // self.config.orbit_period_minutes)

    def is_inside_saa(self, current_minutes: float) -> bool:
        """
        Determines if the satellite is within the South Atlantic Anomaly boundary.
        Modeled as periodic passes (~6 passes/day, each lasting ~12 minutes).
        """
        # Position within the current day (0 to 1440 minutes)
        minute_of_day = current_minutes % 1440.0
        # Distribute SAA encounters across ascending/descending node clusters
        saa_interval = 1440.0 / self.config.saa_passes_per_day
        pass_offset = minute_of_day % saa_interval
        return pass_offset < self.config.saa_pass_duration_minutes

    def saa_minutes_in_interval(self, start_minutes: float, end_minutes: float) -> float:
        """
        Exact overlap (in minutes) between [start, end) and the periodic SAA pass windows.
        Windows start at k * (1440 / passes_per_day) and last saa_pass_duration_minutes.
        """
        interval = 1440.0 / self.config.saa_passes_per_day
        duration = self.config.saa_pass_duration_minutes
        overlap = 0.0
        k = int(start_minutes // interval)
        while k * interval < end_minutes:
            win_start = k * interval
            win_end = win_start + duration
            overlap += max(0.0, min(end_minutes, win_end) - max(start_minutes, win_start))
            k += 1
        return overlap

    def step(self, delta_minutes: float, total_model_bits: int) -> OrbitalState:
        """
        Advances orbital simulation time by delta_minutes and generates upset counts.

        Args:
            delta_minutes: Time step in orbital minutes.
            total_model_bits: Total number of bits in protected/active model memory.

        Returns:
            OrbitalState containing upset counts for this time interval.
        """
        start_minutes = self.elapsed_minutes
        self.elapsed_minutes += delta_minutes

        # Integrate flux over the step: SAA-core rate for the minutes spent inside SAA
        # windows, galactic background rate for the remainder. (Classifying the whole step
        # by its endpoint would put every 240-minute step entirely inside the SAA.)
        saa_minutes = self.saa_minutes_in_interval(start_minutes, self.elapsed_minutes)
        background_minutes = delta_minutes - saa_minutes
        in_saa = saa_minutes > 0.0

        rate_per_bit_day = (
            self.config.saa_peak_rate_per_bit_day * saa_minutes
            + self.config.background_rate_per_bit_day * background_minutes
        ) / delta_minutes if delta_minutes > 0 else 0.0
        flux_mult = rate_per_bit_day / self.config.background_rate_per_bit_day

        # Calculate expected number of events over delta_minutes
        delta_days = delta_minutes / 1440.0
        lambda_events = rate_per_bit_day * total_model_bits * delta_days

        # Sample Poisson arrival of ionizing radiation strikes
        num_strikes = int(self.rng.poisson(lambda_events))

        # Split strikes into Single Bit Upsets (SBUs) vs Multiple Bit Upsets (MBUs)
        # Using binomial distribution for realistic stochasticity
        num_mbus = int(self.rng.binomial(num_strikes, self.config.mbu_fraction))
        num_sbus = num_strikes - num_mbus

        self.cumulative_sbus += num_sbus
        self.cumulative_mbus += num_mbus

        return OrbitalState(
            elapsed_minutes=self.elapsed_minutes,
            elapsed_days=self.elapsed_days,
            orbit_index=self.current_orbit_index,
            in_saa=in_saa,
            instantaneous_flux_multiplier=flux_mult,
            expected_seu_rate_per_bit_day=rate_per_bit_day,
            total_sbus_generated=num_sbus,
            total_mbus_generated=num_mbus,
            saa_minutes_in_step=saa_minutes,
        )

    def reset(self) -> None:
        """Resets the orbital clock and event counters."""
        self.elapsed_minutes = 0.0
        self.cumulative_sbus = 0
        self.cumulative_mbus = 0

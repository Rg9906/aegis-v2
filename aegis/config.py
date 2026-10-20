"""
Project Aegis Configuration and LEO Physical Radiation Constants.

Ground-truth parameters derived from spaceflight heritage:
- Alsat-1 (686 km LEO, SEU in random access memory)
- Flying Laptop (600 km LEO, LEON3FT SRAM EDAC measurements)
- Chaohu-1 SAR satellite (NVIDIA Jetson Xavier NX payload)
"""

from dataclasses import dataclass
from typing import Tuple


# EuroSAT Land Use / Land Cover 10 Classes
EUROSAT_CLASSES: Tuple[str, ...] = (
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
)

EUROSAT_NUM_CLASSES: int = len(EUROSAT_CLASSES)
EUROSAT_IMAGE_SIZE: Tuple[int, int, int] = (3, 64, 64)

# Pretrained Checkpoint on Hugging Face Hub
DEFAULT_EUROSAT_MODEL_REPO: str = "cm93/resnet18-eurosat"
DEFAULT_EUROSAT_WEIGHTS_FILE: str = "model.safetensors"


@dataclass(frozen=True)
class RadiationEnvironmentConfig:
    """
    Physical radiation environment specifications for Low Earth Orbit (LEO).
    Calibrated against Alsat-1 and Flying Laptop telemetry data.
    """
    # Background upset rate outside South Atlantic Anomaly (upsets / bit / day)
    background_rate_per_bit_day: float = 1.66e-7

    # Peak upset rate inside SAA core (upsets / bit / day)
    saa_peak_rate_per_bit_day: float = 1.25e-5

    # Percentage of total daily upsets occurring within SAA (~73.41% per Flying Laptop)
    saa_upset_fraction: float = 0.7341

    # Ratio of Single Bit Upsets (SBU) vs Multiple Bit Upsets (MBU)
    sbu_fraction: float = 0.80
    mbu_fraction: float = 0.20
    mbu_max_cluster_size: int = 4

    # Orbit parameters (~600 km altitude, 97° inclination Sun-Synchronous LEO)
    orbit_period_minutes: float = 96.0
    orbits_per_day: float = 1440.0 / 96.0  # 15 orbits/day
    saa_passes_per_day: int = 6
    saa_pass_duration_minutes: float = 12.0


@dataclass(frozen=True)
class SatellitePayloadBudgetConfig:
    """
    COTS AI accelerator and communication payload budget parameters.
    Calibrated against Chaohu-1 and NVIDIA Jetson edge satellite payloads.
    """
    # Ground station uplink bandwidth (Chaohu-1 S-band = ~1.0 Mbps)
    uplink_bandwidth_bps: float = 1_000_000.0

    # Number of ground station contact windows per day (typically 3-5 passes)
    ground_passes_per_day: int = 4
    ground_pass_duration_seconds: float = 480.0  # 8 minutes per pass

    # Energy modeling constants (picojoules per bit transferred / written)
    spi_flash_read_energy_pj_per_bit: float = 15.0
    dram_write_energy_pj_per_bit: float = 25.0
    radio_rx_energy_pj_per_bit: float = 500.0  # Satellite RF receiver power proxy

    # Inference energy proxy (Jetson Xavier NX running ResNet-18 at 15W TDP)
    inference_energy_mj_per_image: float = 12.5


# Global default instances
DEFAULT_RAD_CONFIG = RadiationEnvironmentConfig()
DEFAULT_BUDGET_CONFIG = SatellitePayloadBudgetConfig()

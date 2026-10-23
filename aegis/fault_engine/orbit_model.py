"""
Real-Orbit LEO Radiation Environment.

Replaces the fixed periodic SAA windows of LEORadiationEnvironment with:
  1. A real satellite ground track: SGP4 propagation of a CelesTrak TLE (default Sentinel-2A,
     NORAD 40697, ~790 km sun-synchronous), sampled every minute.
  2. A physical SAA definition: the SAA is the region of weakest geomagnetic field. The total
     field |B| at the satellite position is computed from the IGRF model (ppigrf), and the upset
     rate ramps linearly from the background rate at |B| >= b_edge_nt to the SAA-core rate at
     |B| <= b_core_nt.

Upset counts per step are Poisson with the rate integrated minute by minute along the track, so
SAA crossings fall where (and as long as) the real orbit crosses the low-field region.

Assumptions to disclose: the rate *constants* (background / SAA-core upsets per bit per day) come
from RadiationEnvironmentConfig and are not derived from AP8/AE9 + device cross-sections; the
|B| ramp (default 20,000 -> 18,000 nT at ~790 km) is a proxy for trapped-proton flux.
"""

import datetime
from typing import Optional

import numpy as np
import ppigrf
from sgp4.api import Satrec

from aegis.config import DEFAULT_RAD_CONFIG, RadiationEnvironmentConfig
from aegis.fault_engine.radiation_model import OrbitalState

EARTH_RADIUS_KM = 6371.2
DEFAULT_TLE_PATH = "orbits/sentinel2a.tle"  # CelesTrak, fetched 2026-10-21 (epoch 2026-10-20)


def load_tle(path: str = DEFAULT_TLE_PATH) -> Satrec:
    """Reads a 2- or 3-line TLE file (as served by CelesTrak) into an SGP4 satellite record."""
    lines = [ln.strip() for ln in open(path, encoding="utf-8").read().splitlines() if ln.strip()]
    line1, line2 = [ln for ln in lines if ln.startswith(("1 ", "2 "))][:2]
    return Satrec.twoline2rv(line1, line2)


def ground_track(sat: Satrec, minutes: np.ndarray):
    """Geocentric latitude/longitude (deg) and altitude (km) at `minutes` after the TLE epoch."""
    days = minutes / 1440.0
    err, r, _ = sat.sgp4_array(np.full_like(days, sat.jdsatepoch), sat.jdsatepochF + days)
    if np.any(err):
        raise RuntimeError(f"SGP4 propagation error codes: {set(err.tolist())}")
    r = np.asarray(r)
    jd = sat.jdsatepoch + sat.jdsatepochF + days
    # TEME -> Earth-fixed via Greenwich mean sidereal time (polar motion / equation of equinoxes
    # are sub-km effects, irrelevant for a region hundreds of km across)
    gmst = np.deg2rad((280.46061837 + 360.98564736629 * (jd - 2451545.0)) % 360.0)
    x = r[:, 0] * np.cos(gmst) + r[:, 1] * np.sin(gmst)
    y = -r[:, 0] * np.sin(gmst) + r[:, 1] * np.cos(gmst)
    radius = np.linalg.norm(r, axis=1)
    lat = np.degrees(np.arcsin(r[:, 2] / radius))
    lon = np.degrees(np.arctan2(y, x))
    return lat, lon, radius - EARTH_RADIUS_KM


def tle_epoch(sat: Satrec) -> datetime.datetime:
    """UTC epoch of a TLE (used as the IGRF evaluation date)."""
    return datetime.datetime(2000, 1, 1, 12) + datetime.timedelta(
        days=sat.jdsatepoch + sat.jdsatepochF - 2451545.0)


def track_payload(total_minutes: float, step_minutes: float = 5.0, start_minutes: float = 0.0,
                  tle_path: str = DEFAULT_TLE_PATH) -> dict:
    """Ground track sampled every `step_minutes` over [start, start + total], as plain JSON lists."""
    sat = load_tle(tle_path)
    minutes = np.arange(start_minutes, start_minutes + total_minutes + step_minutes, step_minutes)
    lat, lon, alt = ground_track(sat, minutes)
    return {"minutes": (minutes - start_minutes).round(1).tolist(), "lat": lat.round(2).tolist(),
            "lon": lon.round(2).tolist(), "alt_km": round(float(np.mean(alt)), 1),
            "period_min": round(float(2 * np.pi / sat.no_kozai), 2)}


def field_grid(alt_km: float, step_deg: float = 3.0, tle_path: str = DEFAULT_TLE_PATH) -> dict:
    """Total field |B| (nT) on a lon/lat grid at orbit altitude (IGRF), for the SAA contour."""
    epoch = tle_epoch(load_tle(tle_path))
    lons = np.arange(-180.0, 180.0 + step_deg, step_deg)
    lats = np.arange(-84.0, 84.0 + step_deg, step_deg)
    lon_g, lat_g = np.meshgrid(lons, lats)
    be, bn, bu = ppigrf.igrf(lon_g.ravel(), lat_g.ravel(), np.full(lon_g.size, alt_km), epoch)
    b = np.sqrt(be ** 2 + bn ** 2 + bu ** 2).reshape(lat_g.shape)
    return {"lons": lons.tolist(), "lats": lats.tolist(), "b_nt": b.round(0).astype(int).tolist(),
            "edge_nt": 20000, "core_nt": 18000}


class OrbitTrackRadiationEnvironment:
    """Drop-in replacement for LEORadiationEnvironment driven by a real TLE and the IGRF field."""

    def __init__(
        self,
        config: RadiationEnvironmentConfig = DEFAULT_RAD_CONFIG,
        random_seed: Optional[int] = 42,
        tle_path: str = DEFAULT_TLE_PATH,
        b_edge_nt: float = 20000.0,
        b_core_nt: float = 18000.0,
    ):
        self.config = config
        self.rng = np.random.default_rng(random_seed)
        self.sat = load_tle(tle_path)
        self.b_edge_nt = b_edge_nt
        self.b_core_nt = b_core_nt
        self.orbit_period_minutes = 2 * np.pi / self.sat.no_kozai  # no_kozai is rad/min
        self.epoch = tle_epoch(self.sat)
        self.elapsed_minutes: float = 0.0
        self.cumulative_sbus: int = 0
        self.cumulative_mbus: int = 0
        self._weights = np.zeros(0)  # SAA weight per propagated minute, extended on demand
        self._lat = np.zeros(0)
        self._lon = np.zeros(0)

    @property
    def elapsed_days(self) -> float:
        return self.elapsed_minutes / 1440.0

    @property
    def current_orbit_index(self) -> int:
        return int(self.elapsed_minutes // self.orbit_period_minutes)

    def _ensure_propagated(self, upto_minute: int) -> None:
        """Propagates the track (1-minute samples) and evaluates IGRF out to `upto_minute`."""
        have = self._weights.size
        if upto_minute < have:
            return
        new_end = max(upto_minute + 1, have + 1440)
        minutes = np.arange(have, new_end, dtype=np.float64) + 0.5  # minute midpoints
        lat, lon, alt = ground_track(self.sat, minutes)
        be, bn, bu = ppigrf.igrf(lon, lat, alt, self.epoch)
        b_total = np.sqrt(be ** 2 + bn ** 2 + bu ** 2).reshape(-1)
        weights = np.clip((self.b_edge_nt - b_total) / (self.b_edge_nt - self.b_core_nt), 0.0, 1.0)
        self._weights = np.concatenate([self._weights, weights])
        self._lat = np.concatenate([self._lat, lat])
        self._lon = np.concatenate([self._lon, lon])

    def saa_weight_in_interval(self, start_minutes: float, end_minutes: float) -> float:
        """SAA-core-equivalent minutes in [start, end): sum of per-minute SAA weights."""
        first, last = int(np.floor(start_minutes)), int(np.ceil(end_minutes))
        self._ensure_propagated(last)
        return float(self._weights[first:last].sum())

    def minute_rates(self, start_minutes: float, end_minutes: float) -> np.ndarray:
        """Upset rate (per bit per day) of each propagated minute in [start, end)."""
        first, last = int(np.floor(start_minutes)), int(np.ceil(end_minutes))
        self._ensure_propagated(last)
        w = self._weights[first:last]
        return self.config.saa_peak_rate_per_bit_day * w + self.config.background_rate_per_bit_day * (1.0 - w)

    def is_inside_saa(self, current_minutes: float) -> bool:
        self._ensure_propagated(int(current_minutes))
        return bool(self._weights[int(current_minutes)] > 0.0)

    def step(self, delta_minutes: float, total_model_bits: int) -> OrbitalState:
        start_minutes = self.elapsed_minutes
        self.elapsed_minutes += delta_minutes

        saa_minutes = self.saa_weight_in_interval(start_minutes, self.elapsed_minutes)
        background_minutes = delta_minutes - saa_minutes
        rate_per_bit_day = (
            self.config.saa_peak_rate_per_bit_day * saa_minutes
            + self.config.background_rate_per_bit_day * background_minutes
        ) / delta_minutes if delta_minutes > 0 else 0.0

        lambda_events = rate_per_bit_day * total_model_bits * (delta_minutes / 1440.0)
        num_strikes = int(self.rng.poisson(lambda_events))
        num_mbus = int(self.rng.binomial(num_strikes, self.config.mbu_fraction))
        num_sbus = num_strikes - num_mbus
        self.cumulative_sbus += num_sbus
        self.cumulative_mbus += num_mbus

        return OrbitalState(
            elapsed_minutes=self.elapsed_minutes,
            elapsed_days=self.elapsed_days,
            orbit_index=self.current_orbit_index,
            in_saa=saa_minutes > 0.0,
            instantaneous_flux_multiplier=rate_per_bit_day / self.config.background_rate_per_bit_day,
            expected_seu_rate_per_bit_day=rate_per_bit_day,
            total_sbus_generated=num_sbus,
            total_mbus_generated=num_mbus,
            saa_minutes_in_step=saa_minutes,
        )

    def reset(self) -> None:
        self.elapsed_minutes = 0.0
        self.cumulative_sbus = 0
        self.cumulative_mbus = 0

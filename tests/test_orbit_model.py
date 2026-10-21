"""
Tests for the real-orbit (SGP4 + IGRF) radiation environment.
"""

import os

import numpy as np
import pytest

pytest.importorskip("sgp4")
pytest.importorskip("ppigrf")

from aegis.fault_engine.orbit_model import DEFAULT_TLE_PATH, OrbitTrackRadiationEnvironment  # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.exists(DEFAULT_TLE_PATH), reason="TLE file missing")


def test_sentinel2a_orbit_geometry():
    env = OrbitTrackRadiationEnvironment(random_seed=0)
    assert env.orbit_period_minutes == pytest.approx(100.6, abs=0.3)  # published S2A period
    env._ensure_propagated(1440)
    # Sun-synchronous ~98.6 deg inclination -> track reaches ~81.4 deg latitude
    assert np.abs(env._lat).max() == pytest.approx(81.4, abs=1.0)


def test_saa_is_over_south_atlantic_and_dominates_upsets():
    env = OrbitTrackRadiationEnvironment(random_seed=0)
    env._ensure_propagated(3 * 1440)
    in_saa = env._weights[: 3 * 1440] > 0
    assert 0.02 < in_saa.mean() < 0.15
    assert np.median(env._lat[: 3 * 1440][in_saa]) < 0          # southern hemisphere
    assert -90 < np.median(env._lon[: 3 * 1440][in_saa]) < 0   # Atlantic / South America

    cfg = env.config
    saa_minutes = env._weights[:1440].sum()
    saa_share = cfg.saa_peak_rate_per_bit_day * saa_minutes / (
        cfg.saa_peak_rate_per_bit_day * saa_minutes + cfg.background_rate_per_bit_day * (1440 - saa_minutes))
    assert 0.6 < saa_share < 0.95  # flight data cited in config: ~73%


def test_step_interface_matches_periodic_model():
    env = OrbitTrackRadiationEnvironment(random_seed=1)
    states = [env.step(60.0, 300_000_000) for _ in range(24)]
    assert env.elapsed_days == pytest.approx(1.0)
    assert any(s.in_saa for s in states) and not all(s.in_saa for s in states)
    saa_rates = [s.expected_seu_rate_per_bit_day for s in states if s.in_saa]
    quiet_rates = [s.expected_seu_rate_per_bit_day for s in states if not s.in_saa]
    assert min(saa_rates) > max(quiet_rates)

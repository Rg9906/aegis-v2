"""
Unit tests for LEO Radiation Environment and SAA Orbit Physics.
"""

import pytest
from aegis.config import DEFAULT_RAD_CONFIG
from aegis.fault_engine.radiation_model import LEORadiationEnvironment


def test_saa_periodic_passes():
    """Verifies that the satellite passes through SAA at expected intervals."""
    env = LEORadiationEnvironment(random_seed=123)

    # At minute 0, should be in SAA (pass begins at minute 0 of pass interval)
    assert env.is_inside_saa(0.0) is True

    # After pass duration (e.g. 15 minutes), should be outside SAA
    assert env.is_inside_saa(15.0) is False

    # Check next pass (1440 / 6 = 240 minutes)
    assert env.is_inside_saa(240.0) is True
    assert env.is_inside_saa(255.0) is False


def test_radiation_step_generation():
    """Verifies Poisson event sampling during orbit propagation."""
    env = LEORadiationEnvironment(random_seed=42)
    # Simulate ResNet-18 parameter size (~11.18M parameters = ~357.8 Mbits)
    model_bits = 11_181_642 * 32

    # Step by 1 full day (1440 minutes)
    state = env.step(delta_minutes=1440.0, total_model_bits=model_bits)

    assert state.elapsed_days == 1.0
    assert state.orbit_index == 15  # 15 orbits per day
    # Verify that upsets were generated
    total_events = state.total_sbus_generated + state.total_mbus_generated
    assert total_events > 0, "Expected non-zero SEUs over 1 day for 357 Mbits"
    # MBU ratio should be roughly ~20%
    assert env.cumulative_sbus >= env.cumulative_mbus


def test_radiation_reset():
    """Verifies reset resets clock and counters."""
    env = LEORadiationEnvironment()
    env.step(delta_minutes=500.0, total_model_bits=1_000_000)
    assert env.elapsed_minutes > 0
    env.reset()
    assert env.elapsed_minutes == 0.0
    assert env.cumulative_sbus == 0
    assert env.cumulative_mbus == 0

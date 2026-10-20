"""
Aegis Fault Engine Subpackage.
"""

from aegis.fault_engine.ieee754 import (
    FLOAT32_EXPONENT_MSB,
    FLOAT32_EXPONENT_RANGE,
    FLOAT32_MANTISSA_RANGE,
    FLOAT32_SIGN_BIT,
    TargetFieldType,
    flip_bits_in_tensor,
    inject_mbu_cluster,
    inject_random_seus,
    inject_single_bit_flip,
    inspect_float32_bits,
)
from aegis.fault_engine.radiation_model import LEORadiationEnvironment, OrbitalState
from aegis.fault_engine.injector import InjectedFaultRecord, ModelFaultInjector

__all__ = [
    "FLOAT32_EXPONENT_MSB",
    "FLOAT32_EXPONENT_RANGE",
    "FLOAT32_MANTISSA_RANGE",
    "FLOAT32_SIGN_BIT",
    "TargetFieldType",
    "flip_bits_in_tensor",
    "inject_single_bit_flip",
    "inject_random_seus",
    "inject_mbu_cluster",
    "inspect_float32_bits",
    "LEORadiationEnvironment",
    "OrbitalState",
    "ModelFaultInjector",
    "InjectedFaultRecord",
]

"""Shared helpers for the console engine and missions: JSON-safe numbers, bit views, fingerprints."""

import math
import struct
import zlib
from typing import Any, Callable, Dict, Optional

from aegis.fault_engine.ieee754 import FLOAT32_EXPONENT_MSB, FLOAT32_SIGN_BIT
from aegis.fault_engine.injector import InjectedFaultRecord
from aegis.models.layer_registry import LayerRegistry

# emit(event_type, payload): the server forwards these to every connected console
Emit = Callable[[str, Dict[str, Any]], None]


def bit_field(bit: int) -> str:
    """IEEE-754 field a bit position belongs to (30 is singled out: it does the damage)."""
    if bit == FLOAT32_SIGN_BIT:
        return "sign"
    if bit == FLOAT32_EXPONENT_MSB:
        return "exponent_msb"
    return "exponent" if bit >= 23 else "mantissa"


def finite(x: float, digits: int = 6) -> Any:
    """Float for JSON; non-finite values become strings (JSON has no NaN/Infinity)."""
    x = float(x)
    if math.isnan(x):
        return "NaN"
    if math.isinf(x):
        return "Infinity" if x > 0 else "-Infinity"
    return float(f"{x:.{digits}g}")


def bits32(x: float) -> str:
    """32-character IEEE-754 single-precision bit string (sign, 8 exponent, 23 mantissa)."""
    return f"{struct.unpack('>I', struct.pack('>f', float(x)))[0]:032b}"


def severity(param: str, bit: int, pre: float, post: float) -> str:
    """
    How much one hit changes the stored value. Mirrors frontend/src/lib/data.ts severity():
    non-finite results and sign flips on BN variances (sqrt of a negative -> NaN) are critical.
    """
    if not math.isfinite(post):
        return "critical"
    if param.endswith("running_var") and post < 0:
        return "critical"
    ratio = abs(post) / max(abs(pre), 1e-12)
    if bit == FLOAT32_EXPONENT_MSB or ratio > 1e3:
        return "critical"
    if ratio > 3 or ratio < 1 / 3:
        return "moderate"
    return "minor"


def hit_dict(rec: InjectedFaultRecord, detail: bool = False) -> Dict[str, Any]:
    """Compact fault record (same keys as the exported campaign JSON), optionally with bit strings."""
    out = {
        "b": rec.block_id,
        "p": rec.param_name,
        "bit": rec.bit_position,
        "from": finite(rec.pre_value),
        "to": finite(rec.post_value),
        "mbu": rec.is_mbu,
        "sev": severity(rec.param_name, rec.bit_position, rec.pre_value, rec.post_value),
    }
    if detail:
        out.update({
            "index": rec.flat_index,
            "field": bit_field(rec.bit_position),
            "from_bits": bits32(rec.pre_value),
            "to_bits": bits32(rec.post_value),
        })
    return out


def block_crcs(registry: LayerRegistry) -> Dict[str, int]:
    """
    CRC-32 fingerprint of every block's stored bits (XOR of per-tensor CRCs, as in Tier 3).
    Used for display and caching only; it does not touch the monitors' scan counters.
    """
    state = registry.model.state_dict()
    out: Dict[str, int] = {}
    for block_id in registry.block_ids:
        crc = 0
        for key in sorted(registry.get_block_metadata(block_id).state_dict_keys):
            crc ^= zlib.crc32(state[key].detach().cpu().contiguous().numpy().tobytes())
        out[block_id] = crc
    return out


def corrupted_blocks(registry: LayerRegistry, golden: Dict[str, int],
                     current: Optional[Dict[str, int]] = None) -> list:
    """Blocks whose stored bits differ from the golden fingerprint, in forward order."""
    current = current if current is not None else block_crcs(registry)
    return [b for b in registry.block_ids if current[b] != golden[b]]

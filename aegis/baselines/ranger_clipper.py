"""
Ranger / FT-ClipAct Baseline: Activation Range Restriction Substructure.

Clamps intermediate layer activations to pre-characterized bounds [L_l, U_l]
during inference without reloading or repairing model weights in memory.
Reference: Chen et al., 'A Low-cost Fault Corrector for Deep Neural Networks through Range Restriction' (DSN 2021).
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn


class RangerActivationClipper:
    """
    In-situ fault mitigation via activation clamping.
    Suppresses extreme activation spikes (e.g. from exponent-MSB flips) into tolerable bounds.
    """

    def __init__(self, bounds: Optional[Dict[str, Tuple[float, float]]] = None):
        self.bounds: Dict[str, Tuple[float, float]] = bounds or {}
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []

    def calibrate(self, model: nn.Module, calib_batch: torch.Tensor, margin: float = 0.1) -> None:
        """
        Profiles per-layer bounds as the observed clean [min, max] (Ranger, DSN'21), widened by
        `margin` x range so that unseen clean chips are not clipped.
        """
        self.bounds.clear()
        stats: Dict[str, List[float]] = {}
        hooks: List[torch.utils.hooks.RemovableHandle] = []

        def make_calib_hook(name: str):
            def hook(module: nn.Module, inp: Tuple[torch.Tensor, ...], out: torch.Tensor):
                with torch.no_grad():
                    lo, hi = float(out.min().item()), float(out.max().item())
                    prev = stats.get(name, [lo, hi])
                    stats[name] = [min(prev[0], lo), max(prev[1], hi)]
            return hook

        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                hooks.append(module.register_forward_hook(make_calib_hook(name)))

        with torch.no_grad():
            for chunk in calib_batch.split(32):
                model(chunk)

        for h in hooks:
            h.remove()

        for name, (lo, hi) in stats.items():
            span = max(hi - lo, 1e-6)
            self.bounds[name] = (lo - margin * span, hi + margin * span)

    def attach(self, model: nn.Module) -> None:
        """Attaches clipping forward hooks to suppress out-of-bounds activations."""
        self.detach()

        def make_clip_hook(name: str):
            def hook(module: nn.Module, inp: Tuple[torch.Tensor, ...], out: torch.Tensor) -> torch.Tensor:
                if name in self.bounds:
                    lower, upper = self.bounds[name]
                    # Clamp NaNs, Infs, and extreme values into safe range
                    clamped = torch.nan_to_num(out, nan=0.0, posinf=upper, neginf=lower)
                    return torch.clamp(clamped, min=lower, max=upper)
                return out
            return hook

        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                self._hooks.append(module.register_forward_hook(make_clip_hook(name)))

    def detach(self) -> None:
        """Removes all clamping hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

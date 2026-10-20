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

    def calibrate(self, model: nn.Module, calib_batch: torch.Tensor, multiplier: float = 6.0) -> None:
        """Profiles upper and lower limits from clean calibration activations."""
        self.bounds.clear()
        stats: Dict[str, List[torch.Tensor]] = {}
        hooks: List[torch.utils.hooks.RemovableHandle] = []

        def make_calib_hook(name: str):
            def hook(module: nn.Module, inp: Tuple[torch.Tensor, ...], out: torch.Tensor):
                with torch.no_grad():
                    stats.setdefault(name, []).append(out.detach().float().view(-1))
            return hook

        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                hooks.append(module.register_forward_hook(make_calib_hook(name)))

        with torch.no_grad():
            model(calib_batch)

        for h in hooks:
            h.remove()

        for name, tensors in stats.items():
            all_acts = torch.cat(tensors)
            mean = all_acts.mean().item()
            std = all_acts.std().item() + 1e-6
            self.bounds[name] = (mean - multiplier * std, mean + multiplier * std)

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

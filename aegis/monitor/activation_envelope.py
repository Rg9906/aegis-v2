"""
Tier 1: Activation Magnitude Envelope Monitor (Ranger-Style Dynamic Clamping & Anomaly Observer).

Monitors intermediate layer activations during inference to detect catastrophic
explosions (NaN, Inf, or extreme values exceeding pre-radiation statistical bounds).
Provides O(1) instantaneous detection of exponent-MSB radiation bit-flips.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn


class ActivationEnvelopeMonitor:
    """
    Hooks into model layers to observe intermediate activation tensors during forward passes.
    Flags any layer whose activations exceed profiled healthy upper/lower limits or produce NaNs/Infs.
    """

    def __init__(self, multiplier: float = 6.0):
        self.multiplier = multiplier
        self._bounds: Dict[str, Tuple[float, float]] = {}  # layer_name -> (min_bound, max_bound)
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._latest_violations: Dict[str, Dict[str, object]] = {}

    @property
    def bounds(self) -> Dict[str, Tuple[float, float]]:
        return self._bounds

    @property
    def latest_violations(self) -> Dict[str, Dict[str, object]]:
        return self._latest_violations

    def profile_clean_bounds(
        self,
        model: nn.Module,
        calibration_loader_or_batch: torch.Tensor,
        tracked_layer_names: Optional[List[str]] = None,
    ) -> None:
        """
        Profiles normal activation distributions using a small set of clean images.
        Establishes upper/lower envelopes: [mu - k*sigma, mu + k*sigma].
        """
        model.eval()
        stats: Dict[str, List[torch.Tensor]] = {}
        hooks: List[torch.utils.hooks.RemovableHandle] = []

        def make_profile_hook(name: str):
            def hook(module: nn.Module, inp: Tuple[torch.Tensor, ...], out: torch.Tensor):
                with torch.no_grad():
                    # Compute mean and std of activations
                    flat_out = out.detach().float().view(-1)
                    stats.setdefault(name, []).append(flat_out)
            return hook

        # Register profiling hooks on Conv2d and Linear layers
        for name, module in model.named_modules():
            if tracked_layer_names and name not in tracked_layer_names:
                continue
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                hooks.append(module.register_forward_hook(make_profile_hook(name)))

        # Run forward pass on calibration imagery
        with torch.no_grad():
            if isinstance(calibration_loader_or_batch, torch.Tensor):
                model(calibration_loader_or_batch)
            else:
                for batch in calibration_loader_or_batch:
                    imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
                    model(imgs)
                    break

        # Remove temporary profiling hooks
        for h in hooks:
            h.remove()

        # Compute empirical bounds
        for name, act_tensors in stats.items():
            all_acts = torch.cat(act_tensors)
            mean = all_acts.mean().item()
            std = all_acts.std().item() + 1e-6
            lower = mean - (self.multiplier * std)
            upper = mean + (self.multiplier * std)
            # Ensure safe bounds
            self._bounds[name] = (min(lower, -10.0), max(upper, 10.0))

    def attach(self, model: nn.Module, tracked_layer_names: Optional[List[str]] = None) -> None:
        """
        Attaches runtime monitoring hooks to the model.
        """
        self.detach()  # Clear existing hooks if any
        self._latest_violations.clear()

        def make_monitor_hook(name: str):
            def hook(module: nn.Module, inp: Tuple[torch.Tensor, ...], out: torch.Tensor):
                with torch.no_grad():
                    # Check for NaNs and Infs
                    has_nan = bool(torch.isnan(out).any().item())
                    has_inf = bool(torch.isinf(out).any().item())

                    max_val = float(out.max().item()) if not (has_nan or has_inf) else float("inf")
                    min_val = float(out.min().item()) if not (has_nan or has_inf) else float("-inf")

                    bound = self._bounds.get(name, (-1e4, 1e4))
                    is_out_of_bounds = (max_val > bound[1]) or (min_val < bound[0])

                    if has_nan or has_inf or is_out_of_bounds:
                        self._latest_violations[name] = {
                            "has_nan": has_nan,
                            "has_inf": has_inf,
                            "max_val": max_val,
                            "min_val": min_val,
                            "expected_bound": bound,
                        }
            return hook

        for name, module in model.named_modules():
            if tracked_layer_names and name not in tracked_layer_names:
                continue
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                self._hooks.append(module.register_forward_hook(make_monitor_hook(name)))

    def detach(self) -> None:
        """Removes all monitoring hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def clear_violations(self) -> None:
        """Resets violation state for next forward pass."""
        self._latest_violations.clear()

    @property
    def has_violations(self) -> bool:
        return len(self._latest_violations) > 0

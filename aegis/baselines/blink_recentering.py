"""
BLINK In-Situ Re-Centering Baseline Substructure.

Detects statistical distribution shifts at Batch Normalization layers and
re-centers activation tensors in-situ without repairing corrupted weights in memory.
Reference: arXiv:2609.06781 (September 2026).
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn


class BlinkRecenteringMitigator:
    """
    BLINK in-situ feature re-centering.
    When a BN layer exhibits statistical divergence, normalizes activations
    using current batch statistics and scales to match reference running statistics.
    """

    def __init__(self, drift_threshold: float = 3.0):
        self.drift_threshold = drift_threshold
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._recentered_layers_log: List[str] = []

    def attach(self, model: nn.Module) -> None:
        """Attaches re-centering forward hooks to BatchNorm2d modules."""
        self.detach()
        self._recentered_layers_log.clear()

        def make_recenter_hook(name: str):
            def hook(module: nn.BatchNorm2d, inp: Tuple[torch.Tensor, ...], out: torch.Tensor) -> torch.Tensor:
                x = inp[0]
                if x.dim() != 4:
                    return out

                # Compute batch statistics per channel
                b_mean = x.mean(dim=(0, 2, 3), keepdim=True)
                b_var = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
                b_std = torch.sqrt(b_var + module.eps)

                r_mean = module.running_mean.view(1, -1, 1, 1)
                r_var = module.running_var.view(1, -1, 1, 1)
                r_std = torch.sqrt(r_var + module.eps)

                # Compute normalized drift score
                drift = (torch.abs(b_mean - r_mean) / r_std).mean().item()

                if drift > self.drift_threshold or torch.isnan(b_mean).any():
                    # Apply in-situ re-centering to match reference statistics
                    normalized = (x - b_mean) / b_std
                    # Replace NaNs/Infs if any
                    normalized = torch.nan_to_num(normalized, nan=0.0, posinf=5.0, neginf=-5.0)
                    recentered = normalized * r_std + r_mean
                    self._recentered_layers_log.append(name)
                    # Pass through affine weights and biases if present
                    if module.affine:
                        w = module.weight.view(1, -1, 1, 1)
                        b = module.bias.view(1, -1, 1, 1)
                        return recentered * w + b
                    return recentered

                return out
            return hook

        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                self._hooks.append(module.register_forward_hook(make_recenter_hook(name)))

    def detach(self) -> None:
        """Removes all hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    @property
    def recentered_layers_count(self) -> int:
        return len(self._recentered_layers_log)

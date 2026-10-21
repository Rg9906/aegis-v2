"""
Tier 2: Batch Normalization Statistical Drift Monitor (BLINK-Style In-Situ Observer).

Measures statistical divergence between runtime batch feature distributions
and clean pre-characterized Batch Normalization running statistics.
Detects subtle mantissa and mid-exponent radiation corruptions that bypass Tier 1.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn


class BNDriftMonitor:
    """
    BLINK-style monitor tracking distribution shifts at BatchNorm2d layers.
    Calculates per-channel Z-score drift against golden running_mean and running_var.
    """

    def __init__(self, drift_threshold: float = 3.5):
        self.drift_threshold = drift_threshold
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._latest_drifts: Dict[str, Dict[str, float]] = {}
        self._calibrated_tolerances: Dict[str, float] = {}

    @property
    def latest_drifts(self) -> Dict[str, Dict[str, float]]:
        return self._latest_drifts

    def calibrate_clean_reference(
        self,
        model: nn.Module,
        calibration_batch: torch.Tensor,
        safety_margin: float = 2.5,
        batch_size: int = 16,
    ) -> None:
        """
        Profiles normal batch variance on clean imagery to set layer-specific drift thresholds.
        Drift is batch-size dependent, so calibration chunks should match the runtime batch size;
        the threshold is set from the worst clean chunk.
        """
        model.eval()
        clean_drifts: Dict[str, List[float]] = {}
        hooks: List[torch.utils.hooks.RemovableHandle] = []

        def make_calib_hook(name: str):
            def hook(module: nn.BatchNorm2d, inp: Tuple[torch.Tensor, ...], out: torch.Tensor):
                with torch.no_grad():
                    drift = self._compute_layer_drift(module, inp[0])
                    clean_drifts.setdefault(name, []).append(drift)
            return hook

        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                hooks.append(module.register_forward_hook(make_calib_hook(name)))

        with torch.no_grad():
            for chunk in calibration_batch.split(batch_size):
                model(chunk)

        for h in hooks:
            h.remove()

        for name, values in clean_drifts.items():
            base_drift = float(max(values))
            # Set dynamic threshold above clean variation
            self._calibrated_tolerances[name] = max(self.drift_threshold, base_drift * safety_margin)

    def attach(self, model: nn.Module) -> None:
        """
        Attaches runtime monitoring hooks to all BatchNorm2d layers.
        """
        self.detach()
        self._latest_drifts.clear()

        def make_monitor_hook(name: str):
            def hook(module: nn.BatchNorm2d, inp: Tuple[torch.Tensor, ...], out: torch.Tensor):
                with torch.no_grad():
                    x = inp[0]  # [B, C, H, W]
                    drift_score = self._compute_layer_drift(module, x)
                    threshold = self._calibrated_tolerances.get(name, self.drift_threshold)
                    is_anomalous = drift_score > threshold

                    self._latest_drifts[name] = {
                        "drift_score": float(drift_score),
                        "threshold": float(threshold),
                        "is_anomalous": bool(is_anomalous),
                    }
            return hook

        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                self._hooks.append(module.register_forward_hook(make_monitor_hook(name)))

    def _compute_layer_drift(self, bn: nn.BatchNorm2d, x: torch.Tensor) -> float:
        """
        Computes the normalized deviation between batch statistics and running statistics.
        """
        if x.dim() != 4:
            return 0.0

        # Compute batch mean and std per channel
        # x is [B, C, H, W] -> reduce over (0, 2, 3)
        b_mean = x.mean(dim=(0, 2, 3))
        b_var = x.var(dim=(0, 2, 3), unbiased=False)
        b_std = torch.sqrt(b_var + bn.eps)

        r_mean = bn.running_mean
        r_var = bn.running_var
        r_std = torch.sqrt(r_var + bn.eps)

        # Handle NaNs or Infinities immediately
        if torch.isnan(b_mean).any() or torch.isinf(b_mean).any():
            return 9999.0

        # Mean drift normalized by running std
        mean_drift = torch.abs(b_mean - r_mean) / r_std
        # Variance drift ratio
        std_drift = torch.abs(b_std - r_std) / r_std

        total_drift = (mean_drift + std_drift).mean().item()
        return float(total_drift)

    def detach(self) -> None:
        """Removes all monitoring hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def clear_drifts(self) -> None:
        """Clears stored drift metrics."""
        self._latest_drifts.clear()

    @property
    def has_anomalies(self) -> bool:
        return any(d["is_anomalous"] for d in self._latest_drifts.values())

    @property
    def anomalous_layers(self) -> List[str]:
        return [name for name, d in self._latest_drifts.items() if d["is_anomalous"]]

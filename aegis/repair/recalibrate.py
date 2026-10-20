"""
Label-Free Batch Normalization Recalibration Substructure.

Performs post-repair feature recalibration on unlabeled streaming satellite imagery
without requiring ground-truth supervision, backpropagation, or ground contact.
"""

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional
import torch
import torch.nn as nn


@dataclass
class RecalibrationReport:
    """Telemetry report after label-free BN recalibration."""
    samples_processed: int
    batches_processed: int
    recalibrated_layers: List[str]


class BNRecalibrator:
    """
    Maintains an onboard circular buffer of recent unlabeled Earth Observation imagery chips.
    Recalibrates downstream Batch Normalization running statistics after parameter repairs.
    """

    def __init__(self, buffer_capacity: int = 128, recalibration_batch_size: int = 32):
        self.buffer_capacity = buffer_capacity
        self.batch_size = recalibration_batch_size
        self._image_buffer: Deque[torch.Tensor] = deque(maxlen=buffer_capacity)

    @property
    def buffer_size(self) -> int:
        return len(self._image_buffer)

    def ingest_unlabeled_imagery(self, images: torch.Tensor) -> None:
        """
        Adds newly acquired unlabeled satellite images to the circular buffer.
        """
        # images is [B, C, H, W]
        for i in range(images.size(0)):
            self._image_buffer.append(images[i].detach().cpu())

    def recalibrate(self, model: nn.Module) -> RecalibrationReport:
        """
        Performs forward passes on buffered imagery in train mode with frozen gradients.
        Updates running_mean and running_var across all BatchNorm2d layers.
        """
        if len(self._image_buffer) == 0:
            return RecalibrationReport(0, 0, [])

        # Collect batches
        buffered_tensors = list(self._image_buffer)
        all_imgs = torch.stack(buffered_tensors)
        total_samples = all_imgs.size(0)

        # Track which BN layers exist
        bn_names = [name for name, mod in model.named_modules() if isinstance(mod, nn.BatchNorm2d)]

        # Temporarily enable train mode to update running statistics
        model.train()
        # Ensure parameters do not compute gradients
        with torch.no_grad():
            num_batches = (total_samples + self.batch_size - 1) // self.batch_size
            for b in range(num_batches):
                start_idx = b * self.batch_size
                end_idx = min(start_idx + self.batch_size, total_samples)
                batch = all_imgs[start_idx:end_idx].to(next(model.parameters()).device)
                model(batch)

        # Restore evaluation mode
        model.eval()

        return RecalibrationReport(
            samples_processed=total_samples,
            batches_processed=num_batches,
            recalibrated_layers=bn_names,
        )

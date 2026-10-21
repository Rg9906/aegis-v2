"""
EuroSAT ResNet-18 Classifier Substructure.

Loads, configures, and serves the 10-class Sentinel-2 land cover model,
interfacing with the LayerRegistry for selective parameter restoration.
"""

import os
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torchvision.models as models
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from aegis.config import (
    DEFAULT_EUROSAT_MODEL_REPO,
    DEFAULT_EUROSAT_WEIGHTS_FILE,
    EUROSAT_CLASSES,
    EUROSAT_NUM_CLASSES,
)
from aegis.models.layer_registry import LayerRegistry


class EuroSATResNet18(nn.Module):
    """
    ResNet-18 classifier customized for 10-class EuroSAT Sentinel-2 satellite imagery.
    Integrates directly with the Aegis LayerRegistry and fault-injection hooks.
    """

    def __init__(self, pretrained: bool = True, weights_path: Optional[str] = None):
        super().__init__()
        # Initialize standard torchvision ResNet-18 architecture with 10 output classes
        self.backbone = models.resnet18(weights=None, num_classes=EUROSAT_NUM_CLASSES)
        self.classes = EUROSAT_CLASSES
        self.registry = LayerRegistry(self.backbone)

        if pretrained:
            self.load_pretrained_weights(weights_path)

        self.eval()

    def load_pretrained_weights(self, weights_path: Optional[str] = None) -> None:
        """
        Loads pre-trained EuroSAT checkpoint. Downloads from Hugging Face Hub if not local.
        """
        if weights_path is None or not os.path.exists(weights_path):
            # Prefer the local Hugging Face cache; only hit the network if it is missing
            try:
                weights_path = hf_hub_download(
                    repo_id=DEFAULT_EUROSAT_MODEL_REPO,
                    filename=DEFAULT_EUROSAT_WEIGHTS_FILE,
                    local_files_only=True,
                )
            except Exception:
                weights_path = hf_hub_download(
                    repo_id=DEFAULT_EUROSAT_MODEL_REPO,
                    filename=DEFAULT_EUROSAT_WEIGHTS_FILE,
                )

        state_dict = load_file(weights_path)
        # Strict load: a silent key mismatch would leave the backbone randomly initialized
        self.backbone.load_state_dict(state_dict, strict=True)
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ResNet-18 backbone."""
        return self.backbone(x)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs inference and returns (predicted_class_indices, softmax_probabilities).
        """
        self.eval()
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        return preds, probs

    @torch.no_grad()
    def evaluate(self, images: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """
        Evaluates batch accuracy and average cross-entropy loss.
        """
        self.eval()
        logits = self.forward(x=images)
        loss = nn.functional.cross_entropy(logits, labels).item()
        preds = torch.argmax(logits, dim=-1)
        correct = (preds == labels).sum().item()
        total = labels.numel()
        accuracy = (correct / total) if total > 0 else 0.0

        return {
            "accuracy": float(accuracy),
            "loss": float(loss),
            "total_samples": total,
            "correct_samples": correct,
        }

    def clone_clean_state(self) -> Dict[str, torch.Tensor]:
        """Creates an in-memory deep copy of current parameters as a baseline golden copy."""
        return {k: v.clone() for k, v in self.backbone.state_dict().items()}

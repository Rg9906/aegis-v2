"""
Architecture Factory for Campaign Backbones.

resnet18:      public EuroSAT checkpoint (cm93/resnet18-eurosat); training split unknown.
mobilenet_v2:  fine-tuned in-project by experiments/train_mobilenet.py on chips disjoint from
               the campaign's calibration/evaluation splits; the checkpoint's JSON sidecar
               lists every held-out chip so leakage can be checked before a campaign runs.
"""

import json
import os
from typing import Iterable, Optional

import torch.nn as nn
import torchvision.models as models
from safetensors.torch import load_file

from aegis.config import EUROSAT_NUM_CLASSES

ARCHITECTURES = ("resnet18", "mobilenet_v2")
DEFAULT_MOBILENET_CHECKPOINT = "checkpoints/mobilenet_v2_eurosat.safetensors"


def build_mobilenet_v2() -> nn.Module:
    """torchvision MobileNetV2 with a 10-class EuroSAT head (no weights loaded)."""
    return models.mobilenet_v2(weights=None, num_classes=EUROSAT_NUM_CLASSES)


def checkpoint_metadata(checkpoint_path: str) -> dict:
    with open(os.path.splitext(checkpoint_path)[0] + ".json", encoding="utf-8") as f:
        return json.load(f)


def assert_held_out(checkpoint_path: str, chip_paths: Iterable[str]) -> None:
    """Raises if any campaign chip was available to the fine-tuned checkpoint during training."""
    held_out = set(checkpoint_metadata(checkpoint_path)["held_out_chips"])
    leaked = [p for p in chip_paths if p not in held_out]
    if leaked:
        raise ValueError(
            f"{len(leaked)} calibration/evaluation chips (e.g. {leaked[0]}) were in the training set of "
            f"{checkpoint_path}. Re-train with this split config in --exclude-splits."
        )


def load_backbone(arch: str = "resnet18", checkpoint_path: Optional[str] = None) -> nn.Module:
    """Pretrained EuroSAT backbone in eval mode."""
    if arch == "resnet18":
        from aegis.models.eurosat_classifier import EuroSATResNet18

        return EuroSATResNet18(pretrained=True, weights_path=checkpoint_path).backbone
    if arch == "mobilenet_v2":
        path = checkpoint_path or DEFAULT_MOBILENET_CHECKPOINT
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found; run experiments/train_mobilenet.py first.")
        model = build_mobilenet_v2()
        model.load_state_dict(load_file(path), strict=True)
        return model.eval()
    raise ValueError(f"Unknown architecture '{arch}'. Choose from {ARCHITECTURES}.")

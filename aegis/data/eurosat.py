"""
EuroSAT Real-Imagery Loader Substructure.

Loads Sentinel-2 RGB chips from the extracted EuroSAT dataset (class-named folders),
draws disjoint, class-stratified calibration and evaluation splits, and applies the
pretrained checkpoint's preprocessing (bicubic upsampling to 224 px + channel normalization).

Caveat: the training split of the pretrained checkpoint is not published, so evaluation
chips may overlap its training data. Absolute clean accuracy is therefore an upper bound;
relative accuracy under faults (the quantity Aegis studies) is unaffected.
"""

import os
import random
from dataclasses import dataclass, field
from typing import List, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from aegis.config import (
    DEFAULT_EUROSAT_DATA_DIR,
    EUROSAT_CLASSES,
    EUROSAT_NORM_MEAN,
    EUROSAT_NORM_STD,
    MODEL_INPUT_SIZE,
)


@dataclass
class EuroSATSplits:
    """Disjoint preprocessed tensors: unlabeled calibration imagery and labeled evaluation imagery."""
    calib_images: torch.Tensor
    eval_images: torch.Tensor
    eval_labels: torch.Tensor
    # Chip paths relative to the data dir ("Class/file.jpg"), for leakage checks
    calib_paths: List[str] = field(default_factory=list)
    eval_paths: List[str] = field(default_factory=list)


def preprocess_chips(chips_uint8: torch.Tensor, input_size: int = MODEL_INPUT_SIZE) -> torch.Tensor:
    """Converts [N, 3, 64, 64] uint8 chips to normalized [N, 3, input_size, input_size] float32."""
    x = chips_uint8.float() / 255.0
    mean = torch.tensor(EUROSAT_NORM_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(EUROSAT_NORM_STD).view(1, 3, 1, 1)
    x = (x - mean) / std
    if x.shape[-1] != input_size:
        x = F.interpolate(x, size=input_size, mode="bicubic", align_corners=False)
    return x.contiguous()


def _load_chips(paths: List[str]) -> torch.Tensor:
    arrays = [np.array(Image.open(p).convert("RGB"), dtype=np.uint8) for p in paths]
    return torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).contiguous()


def _class_files(data_dir: str, class_name: str) -> List[str]:
    class_dir = os.path.join(data_dir, class_name)
    return sorted(f for f in os.listdir(class_dir) if f.lower().endswith((".jpg", ".png", ".tif")))


def split_eurosat_paths(
    data_dir: str = DEFAULT_EUROSAT_DATA_DIR,
    calib_per_class: int = 4,
    eval_per_class: int = 20,
    seed: int = 0,
) -> Tuple[List[str], List[Tuple[str, int]]]:
    """
    Class-stratified, disjoint calibration paths and (eval path, label) items, shuffled so any
    batch slice is class-mixed. Paths are relative to `data_dir` ("Class/file.jpg").
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"EuroSAT data not found at '{data_dir}'. Download "
            "https://huggingface.co/datasets/torchgeo/eurosat/resolve/main/EuroSAT.zip "
            "and extract it into data/eurosat/."
        )

    rng = random.Random(seed)
    calib_paths: List[str] = []
    eval_items: List[Tuple[str, int]] = []

    for label, class_name in enumerate(EUROSAT_CLASSES):
        files = _class_files(data_dir, class_name)
        needed = calib_per_class + eval_per_class
        if len(files) < needed:
            raise ValueError(f"Class '{class_name}' has {len(files)} chips, need {needed}.")
        chosen = rng.sample(files, needed)
        calib_paths += [f"{class_name}/{f}" for f in chosen[:calib_per_class]]
        eval_items += [(f"{class_name}/{f}", label) for f in chosen[calib_per_class:]]

    # Shuffle so that any batch slice is class-mixed (monitor batches see all classes)
    rng.shuffle(calib_paths)
    rng.shuffle(eval_items)
    return calib_paths, eval_items


def load_eurosat_splits(
    data_dir: str = DEFAULT_EUROSAT_DATA_DIR,
    calib_per_class: int = 4,
    eval_per_class: int = 20,
    seed: int = 0,
    input_size: int = MODEL_INPUT_SIZE,
) -> EuroSATSplits:
    """
    Draws class-stratified, disjoint calibration and evaluation splits.
    Labels are indices into EUROSAT_CLASSES (checkpoint logit order).
    """
    calib_paths, eval_items = split_eurosat_paths(data_dir, calib_per_class, eval_per_class, seed)
    eval_paths = [p for p, _ in eval_items]
    calib_images = preprocess_chips(_load_chips([os.path.join(data_dir, p) for p in calib_paths]), input_size)
    eval_images = preprocess_chips(_load_chips([os.path.join(data_dir, p) for p in eval_paths]), input_size)
    eval_labels = torch.tensor([label for _, label in eval_items], dtype=torch.long)

    return EuroSATSplits(calib_images=calib_images, eval_images=eval_images, eval_labels=eval_labels,
                         calib_paths=calib_paths, eval_paths=eval_paths)


def held_out_paths(
    split_configs: Sequence[Tuple[int, int]],
    data_dir: str = DEFAULT_EUROSAT_DATA_DIR,
    seed: int = 0,
) -> Set[str]:
    """Union of calibration + evaluation paths over (calib_per_class, eval_per_class) configs."""
    reserved: Set[str] = set()
    for calib_per_class, eval_per_class in split_configs:
        calib_paths, eval_items = split_eurosat_paths(data_dir, calib_per_class, eval_per_class, seed)
        reserved.update(calib_paths)
        reserved.update(p for p, _ in eval_items)
    return reserved


def load_training_chips(
    exclude: Set[str],
    data_dir: str = DEFAULT_EUROSAT_DATA_DIR,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """All chips not in `exclude`, as raw [N, 3, 64, 64] uint8 plus labels (EUROSAT_CLASSES order)."""
    paths, labels = [], []
    for label, class_name in enumerate(EUROSAT_CLASSES):
        for f in _class_files(data_dir, class_name):
            rel = f"{class_name}/{f}"
            if rel not in exclude:
                paths.append(os.path.join(data_dir, rel))
                labels.append(label)
    return _load_chips(paths), torch.tensor(labels, dtype=torch.long)

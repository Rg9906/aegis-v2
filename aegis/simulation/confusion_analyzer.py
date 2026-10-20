"""
Earth Observation Semantic Confusion Matrix Analyzer Substructure.

Analyzes class-confusion structure in Sentinel-2 / EuroSAT land-cover predictions
under radiation-induced weight corruption and evaluates semantic mission risk.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from aegis.config import EUROSAT_CLASSES, EUROSAT_NUM_CLASSES


@dataclass
class ConfusionDeltaReport:
    """Detailed report on semantic confusion induced by radiation faults."""
    clean_accuracy: float
    corrupted_accuracy: float
    accuracy_drop: float
    top_confusions: List[Dict[str, object]]
    confusion_matrix_clean: np.ndarray
    confusion_matrix_corrupted: np.ndarray


class EOConfusionAnalyzer:
    """
    Computes confusion matrices and quantifies semantic misclassifications
    (e.g. River <-> Highway, SeaLake <-> River) on EuroSAT satellite imagery.
    """

    def __init__(self, classes: Tuple[str, ...] = EUROSAT_CLASSES):
        self.classes = classes
        self.num_classes = len(classes)

    def compute_confusion_matrix(
        self,
        model: nn.Module,
        val_images: torch.Tensor,
        val_labels: torch.Tensor,
    ) -> Tuple[np.ndarray, float]:
        """
        Runs inference and computes normalized confusion matrix and top-1 accuracy.
        """
        model.eval()
        device = next(model.parameters()).device
        val_images = val_images.to(device)
        val_labels = val_labels.to(device)

        with torch.no_grad():
            logits = model(val_images)
            preds = torch.argmax(logits, dim=-1)

        total = val_labels.numel()
        correct = (preds == val_labels).sum().item()
        accuracy = correct / total if total > 0 else 0.0

        cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        for true_l, pred_l in zip(val_labels.cpu().numpy(), preds.cpu().numpy()):
            cm[true_l, pred_l] += 1

        return cm, float(accuracy)

    def analyze_confusion_delta(
        self,
        clean_cm: np.ndarray,
        clean_acc: float,
        corrupted_cm: np.ndarray,
        corrupted_acc: float,
        top_k: int = 5,
    ) -> ConfusionDeltaReport:
        """
        Calculates delta matrix (corrupted_cm - clean_cm) and identifies top semantic confusions.
        """
        delta_cm = corrupted_cm.astype(np.float64) - clean_cm.astype(np.float64)
        confusions: List[Dict[str, object]] = []

        for i in range(self.num_classes):
            for j in range(self.num_classes):
                if i != j:  # Misclassification off-diagonal
                    increase = delta_cm[i, j]
                    if increase > 0:
                        confusions.append({
                            "true_class": self.classes[i],
                            "predicted_as": self.classes[j],
                            "misclassification_increase": int(increase),
                            "corrupted_count": int(corrupted_cm[i, j]),
                            "clean_count": int(clean_cm[i, j]),
                        })

        # Sort by largest increase in misclassification
        confusions.sort(key=lambda x: x["misclassification_increase"], reverse=True)

        return ConfusionDeltaReport(
            clean_accuracy=clean_acc,
            corrupted_accuracy=corrupted_acc,
            accuracy_drop=round(clean_acc - corrupted_acc, 4),
            top_confusions=confusions[:top_k],
            confusion_matrix_clean=clean_cm,
            confusion_matrix_corrupted=corrupted_cm,
        )

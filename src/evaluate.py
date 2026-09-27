"""
Comprehensive Clinical Evaluation & Metrics Suite
Calculates Accuracy, Precision, Recall/Sensitivity, Specificity, F1-Score,
Matthews Correlation Coefficient (MCC), and Multi-Class ROC-AUC with visualization tools.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    auc,
)
from typing import Dict, Any, List, Optional


class ClinicalEvaluator:
    """
    Evaluates predictions against ground truth labels and exports publication-ready figures.
    """

    def __init__(self, class_names: List[str], output_dir: str = "experiments/results"):
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def calculate_specificity(self, cm: np.ndarray) -> np.ndarray:
        """
        Calculates per-class Specificity from confusion matrix:
        Specificity = TN / (TN + FP)
        """
        specificities = []
        for i in range(self.num_classes):
            tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
            fp = np.sum(cm[:, i]) - cm[i, i]
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            specificities.append(spec)
        return np.array(specificities)

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_probs: Optional[np.ndarray] = None,
        prefix: str = "test",
    ) -> Dict[str, Any]:
        """
        Computes all standard IEEE benchmark metrics.
        """
        acc = accuracy_score(y_true, y_pred)
        prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
        prec_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
        rec_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)

        cm = confusion_matrix(y_true, y_pred, labels=list(range(self.num_classes)))
        specificities = self.calculate_specificity(cm)
        spec_macro = float(np.mean(specificities))

        # Per-class metrics
        prec_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        rec_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

        # Multi-class ROC-AUC
        roc_auc_val = 0.0
        if y_probs is not None:
            try:
                if self.num_classes == 2:
                    roc_auc_val = float(roc_auc_score(y_true, y_probs[:, 1]))
                else:
                    roc_auc_val = float(roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro"))
            except Exception:
                roc_auc_val = 0.0

        metrics = {
            "accuracy": float(acc),
            "precision_macro": float(prec_macro),
            "precision_weighted": float(prec_weighted),
            "recall_macro": float(rec_macro),
            "recall_weighted": float(rec_weighted),
            "specificity_macro": float(spec_macro),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
            "mcc": float(mcc),
            "roc_auc": float(roc_auc_val),
            "confusion_matrix": cm.tolist(),
            "per_class": {
                self.class_names[i]: {
                    "precision": float(prec_per_class[i]),
                    "recall": float(rec_per_class[i]),
                    "specificity": float(specificities[i]),
                    "f1_score": float(f1_per_class[i]),
                }
                for i in range(self.num_classes)
            },
        }

        # Generate publication plots
        self.plot_confusion_matrix(cm, prefix=prefix)
        if y_probs is not None and roc_auc_val > 0:
            self.plot_roc_curves(y_true, y_probs, prefix=prefix)

        return metrics

    def plot_confusion_matrix(self, cm: np.ndarray, prefix: str = "test"):
        plt.figure(figsize=(7, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar=True,
        )
        plt.title(f"Confusion Matrix ({prefix.capitalize()})", fontsize=13, fontweight="bold")
        plt.xlabel("Predicted Disease Class", fontsize=11)
        plt.ylabel("True Clinical Label", fontsize=11)
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f"{prefix}_confusion_matrix.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_roc_curves(self, y_true: np.ndarray, y_probs: np.ndarray, prefix: str = "test"):
        plt.figure(figsize=(8, 6))
        for i in range(self.num_classes):
            # One-vs-Rest binarization
            y_bin = (y_true == i).astype(int)
            fpr, tpr, _ = roc_curve(y_bin, y_probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(
                fpr,
                tpr,
                lw=2,
                label=f"{self.class_names[i]} (AUC = {roc_auc:.3f})",
            )

        plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random Classifier")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
        plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11)
        plt.title(f"Multi-Class ROC-AUC Curves ({prefix.capitalize()})", fontsize=13, fontweight="bold")
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f"{prefix}_roc_curves.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

"""
Loss Functions Suite for Medical Image Classification
Implements Multi-Class Focal Loss, Label Smoothing, and Weighted Cross-Entropy
to effectively tackle clinical class imbalance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MultiClassFocalLoss(nn.Module):
    """
    Multi-Class Focal Loss:
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    Down-weights easy examples and focuses training on hard negative and minority disease classes.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super(MultiClassFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        inputs: Logits of shape (BatchSize, NumClasses)
        targets: Ground truth class indices of shape (BatchSize)
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss) # Probability of true class

        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy loss with label smoothing to prevent overconfidence.
    """

    def __init__(self, smoothing: float = 0.1, reduction: str = "mean"):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = inputs.size(-1)
        log_preds = F.log_softmax(inputs, dim=-1)

        loss = -log_preds.sum(dim=-1)
        nll = F.nll_loss(log_preds, targets, reduction="none")

        smoothed_loss = (1.0 - self.smoothing) * nll + (self.smoothing / num_classes) * loss

        if self.reduction == "mean":
            return smoothed_loss.mean()
        elif self.reduction == "sum":
            return smoothed_loss.sum()
        else:
            return smoothed_loss

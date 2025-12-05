"""Loss functions for MOH-DETECTOR."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice loss for segmentation."""

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> float:  # type: ignore
        probs = torch.sigmoid(logits)
        num = 2 * (probs * targets).sum() + self.smooth
        den = probs.sum() + targets.sum() + self.smooth
        return 1 - num / den


class BCEWithLogits(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.loss(logits, targets)


def focal_loss(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0, alpha: float = 0.25) -> torch.Tensor:
    """Compute focal loss for binary segmentation."""
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    prob = torch.sigmoid(logits)
    p_t = prob * targets + (1 - prob) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * (1 - p_t) ** gamma * bce
    return loss.mean()


__all__ = ["DiceLoss", "BCEWithLogits", "focal_loss"]

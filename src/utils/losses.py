import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):

    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = float(gamma)
        self.reduction = reduction
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # CE per-sample
        ce = F.cross_entropy(
            logits,
            target,
            reduction="none",
            weight=self.alpha,
            label_smoothing=self.label_smoothing,
        )

        # pt = P(correct class)
        pt = torch.exp(-ce)  # because ce = -log(pt)
        loss = ((1.0 - pt) ** self.gamma) * ce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

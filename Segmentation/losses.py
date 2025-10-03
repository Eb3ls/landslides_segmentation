import torch
import torch.nn as nn

def _flatten_per_sample(x: torch.Tensor) -> torch.Tensor:
    return x.view(x.size(0), -1)

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0, eps: float = 1e-7):
        super().__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        p = _flatten_per_sample(probs)
        t = _flatten_per_sample(targets)
        intersection = (p * t).sum(dim=1)
        denom = p.sum(dim=1) + t.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth + self.eps)
        return 1.0 - dice.mean()

class SquaredDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0, eps: float = 1e-7):
        super().__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        p = _flatten_per_sample(probs)
        t = _flatten_per_sample(targets)
        intersection = (p * t).sum(dim=1)
        denom = p.pow(2).sum(dim=1) + t.pow(2).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth + self.eps)
        return 1.0 - dice.mean()

class BCEDiceLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 0.5,
        smooth: float = 1.0,
        eps: float = 1e-7,
        pos_weight: torch.Tensor | None = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=reduction)
        self.bce_weight = bce_weight
        self.dice = DiceLoss(smooth=smooth, eps=eps)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, targets.float())
        d = self.dice(logits, targets)
        return self.bce_weight * bce + (1.0 - self.bce_weight) * d
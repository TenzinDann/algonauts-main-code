"""Loss functions for brain encoding models.

Combines MSE (for output scale anchoring) with Pearson correlation loss
(for direct optimization of the evaluation metric), following best practices
from top Algonauts 2025 teams [1, 2].
"""

import torch
import torch.nn as nn


class PearsonLoss(nn.Module):
    """Pearson correlation loss: 1 - mean(r) across output parcels."""

    def forward(self, pred, target):
        pred_c = pred - pred.mean(dim=0, keepdim=True)
        target_c = target - target.mean(dim=0, keepdim=True)
        pred_n = torch.sqrt((pred_c ** 2).sum(dim=0) + 1e-8)
        target_n = torch.sqrt((target_c ** 2).sum(dim=0) + 1e-8)
        corr = (pred_c * target_c).sum(dim=0) / (pred_n * target_n + 1e-8)
        return 1.0 - corr.mean()


class CombinedLoss(nn.Module):
    """Weighted combination of MSE and Pearson correlation loss.

    Parameters
    ----------
    lambda_mse : float
        Weight for MSE term (default: 0.03).
    lambda_pearson : float
        Weight for Pearson loss term (default: 1.0).
    """

    def __init__(self, lambda_mse=0.03, lambda_pearson=1.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.pearson = PearsonLoss()
        self.lambda_mse = lambda_mse
        self.lambda_pearson = lambda_pearson

    def forward(self, pred, target):
        return (self.lambda_mse * self.mse(pred, target) +
                self.lambda_pearson * self.pearson(pred, target))

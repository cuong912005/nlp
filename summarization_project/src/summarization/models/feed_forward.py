"""Position-wise feed-forward network."""

from __future__ import annotations

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """Two-layer MLP applied independently to each token position."""

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

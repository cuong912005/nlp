"""Position-wise feed-forward network."""

from __future__ import annotations

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """Two-layer MLP applied independently to each token position."""

    def __init__(self, d_model: int, d_ff: int, dropout: float, activation: str = "gelu"):
        super().__init__()
        if activation == "relu":
            activation_layer = nn.ReLU()
        elif activation == "gelu":
            activation_layer = nn.GELU()
        else:
            raise ValueError("activation must be 'gelu' or 'relu'")

        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            activation_layer,
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

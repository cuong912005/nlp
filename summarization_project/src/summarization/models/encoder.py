"""Transformer encoder blocks."""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .feed_forward import FeedForward


class EncoderLayer(nn.Module):
    """Pre-LN encoder layer for stable training."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, source_mask: torch.Tensor | None) -> torch.Tensor:
        # Pre-LN: normalize trước attention để gradient ổn định hơn khi stack nhiều layer.
        norm_x = self.norm1(x)
        x = x + self.dropout(self.self_attn(norm_x, norm_x, norm_x, source_mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    """Stack of encoder layers."""

    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, source_mask: torch.Tensor | None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, source_mask)
        return self.final_norm(x)

"""Transformer decoder blocks."""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .feed_forward import FeedForward


class DecoderLayer(nn.Module):
    """Decoder layer with selectable Pre-LN or Post-LN layout."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float,
        norm_type: str = "pre",
        activation: str = "gelu",
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout, activation=activation)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm_type = norm_type

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        source_mask: torch.Tensor | None,
        target_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.norm_type == "pre":
            # Self-attention dùng causal mask để decoder không nhìn trước đáp án.
            norm_x = self.norm1(x)
            x = x + self.dropout(self.self_attn(norm_x, norm_x, norm_x, target_mask))

            # Cross-attention cho phép summary nhìn vào toàn bộ article đã encode.
            norm_x = self.norm2(x)
            x = x + self.dropout(self.cross_attn(norm_x, memory, memory, source_mask))

            x = x + self.dropout(self.ffn(self.norm3(x)))
            return x

        # Post-LN: Add & Norm sau từng sublayer như sơ đồ Transformer gốc.
        self_attn_out = self.self_attn(x, x, x, target_mask)
        x = self.norm1(x + self.dropout(self_attn_out))

        cross_attn_out = self.cross_attn(x, memory, memory, source_mask)
        x = self.norm2(x + self.dropout(cross_attn_out))

        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x


class TransformerDecoder(nn.Module):
    """Stack of decoder layers."""

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float,
        norm_type: str = "pre",
        activation: str = "gelu",
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_model,
                    num_heads,
                    d_ff,
                    dropout,
                    norm_type=norm_type,
                    activation=activation,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model) if norm_type == "pre" else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        source_mask: torch.Tensor | None,
        target_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)
        return self.final_norm(x)

"""From-scratch encoder-decoder Transformer for summarization."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .config import TransformerConfig
from .decoder import TransformerDecoder
from .encoder import TransformerEncoder
from .positional_encoding import PositionalEncoding


class SummarizationTransformer(nn.Module):
    """Transformer seq2seq model without using nn.Transformer."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.source_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_id)
        self.target_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_id)
        self.position = PositionalEncoding(config.d_model, config.max_position, config.dropout)

        self.encoder = TransformerEncoder(
            config.num_encoder_layers,
            config.d_model,
            config.num_heads,
            config.d_ff,
            config.dropout,
        )
        self.decoder = TransformerDecoder(
            config.num_decoder_layers,
            config.d_model,
            config.num_heads,
            config.d_ff,
            config.dropout,
        )

        self.output_projection = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.share_embeddings:
            # Source và summary cùng tiếng Việt nên dùng chung embedding giúp giảm tham số.
            self.target_embedding.weight = self.source_embedding.weight

        if config.weight_tying:
            # Output projection dùng lại target embedding để giảm tham số và thường tổng quát tốt hơn.
            self.output_projection.weight = self.target_embedding.weight

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize trainable weights."""
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def encode(self, source: torch.Tensor, source_mask: torch.Tensor | None) -> torch.Tensor:
        """Encode source article tokens."""
        x = self.source_embedding(source) * math.sqrt(self.config.d_model)
        x = self.position(x)
        return self.encoder(x, source_mask)

    def decode(
        self,
        target_input: torch.Tensor,
        memory: torch.Tensor,
        source_mask: torch.Tensor | None,
        target_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Decode summary tokens with encoder memory."""
        x = self.target_embedding(target_input) * math.sqrt(self.config.d_model)
        x = self.position(x)
        x = self.decoder(x, memory, source_mask, target_mask)
        return self.output_projection(x)

    def forward(
        self,
        source: torch.Tensor,
        target_input: torch.Tensor,
        source_mask: torch.Tensor | None,
        target_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return logits for next-token prediction."""
        memory = self.encode(source, source_mask)
        return self.decode(target_input, memory, source_mask, target_mask)

    def count_parameters(self) -> int:
        """Count trainable parameters for reporting."""
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

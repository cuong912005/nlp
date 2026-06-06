"""Configuration for the from-scratch Transformer baseline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TransformerConfig:
    """Small defaults are chosen so local smoke tests are cheap."""

    vocab_size: int
    pad_id: int = 0
    bos_id: int = 2
    eos_id: int = 3

    d_model: int = 256
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    num_heads: int = 8
    d_ff: int = 1024
    dropout: float = 0.1
    max_position: int = 1024

    label_smoothing: float = 0.1
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    grad_clip: float = 1.0

    weight_tying: bool = True
    share_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

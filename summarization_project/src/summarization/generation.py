"""Greedy and beam-search decoding for summarization."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .dataset import make_causal_mask
from .models import SummarizationTransformer


@dataclass
class GenerationConfig:
    """Decoding settings used during validation/test inference."""

    max_len: int = 128
    min_len: int = 20
    beam_size: int = 4
    length_penalty: float = 0.8
    no_repeat_ngram_size: int = 3


def source_padding_mask(source: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Build encoder padding mask from source IDs."""
    return (source != pad_id).unsqueeze(1).unsqueeze(2)


def target_causal_mask(target: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Build decoder mask during autoregressive decoding."""
    padding_mask = (target != pad_id).unsqueeze(1).unsqueeze(2)
    causal_mask = make_causal_mask(target.size(1), target.device).unsqueeze(0).unsqueeze(0)
    return padding_mask & causal_mask


def blocked_tokens(tokens: list[int], ngram_size: int) -> set[int]:
    """Return tokens that would create a repeated n-gram."""
    if ngram_size <= 0 or len(tokens) < ngram_size - 1:
        return set()

    prefix = tuple(tokens[-(ngram_size - 1) :])
    blocked: set[int] = set()

    # Nếu cùng prefix đã xuất hiện, chặn token tiếp theo để không lặp n-gram.
    for idx in range(len(tokens) - ngram_size + 1):
        ngram = tuple(tokens[idx : idx + ngram_size])
        if ngram[:-1] == prefix:
            blocked.add(ngram[-1])

    return blocked


@torch.no_grad()
def greedy_decode(
    model: SummarizationTransformer,
    source: torch.Tensor,
    config: GenerationConfig,
) -> list[int]:
    """Fast greedy decoding for debugging."""
    model.eval()
    device = next(model.parameters()).device
    source = source.to(device)
    src_mask = source_padding_mask(source, model.config.pad_id)
    memory = model.encode(source, src_mask)

    generated = torch.tensor([[model.config.bos_id]], dtype=torch.long, device=device)

    for step in range(config.max_len - 1):
        tgt_mask = target_causal_mask(generated, model.config.pad_id)
        logits = model.decode(generated, memory, src_mask, tgt_mask)[:, -1, :]

        if generated.size(1) < config.min_len:
            # Ép chưa kết thúc quá sớm để summary không bị cụt.
            logits[:, model.config.eos_id] = -1e9

        for token_id in blocked_tokens(generated[0].tolist(), config.no_repeat_ngram_size):
            logits[:, token_id] = -1e9

        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)

        if next_token.item() == model.config.eos_id and step + 1 >= config.min_len:
            break

    return generated[0].tolist()


def normalized_score(log_prob: float, length: int, length_penalty: float) -> float:
    """Apply length penalty so beam search does not prefer very short summaries."""
    return log_prob / ((max(1, length) ** length_penalty))


@torch.no_grad()
def beam_search_decode(
    model: SummarizationTransformer,
    source: torch.Tensor,
    config: GenerationConfig,
) -> list[int]:
    """Beam search with length penalty and no-repeat n-gram blocking."""
    model.eval()
    device = next(model.parameters()).device
    source = source.to(device)
    src_mask = source_padding_mask(source, model.config.pad_id)
    memory = model.encode(source, src_mask)

    beams: list[tuple[list[int], float, bool]] = [([model.config.bos_id], 0.0, False)]

    for _ in range(config.max_len - 1):
        candidates: list[tuple[list[int], float, bool]] = []

        for tokens, score, finished in beams:
            if finished:
                candidates.append((tokens, score, True))
                continue

            target = torch.tensor([tokens], dtype=torch.long, device=device)
            tgt_mask = target_causal_mask(target, model.config.pad_id)
            logits = model.decode(target, memory, src_mask, tgt_mask)[:, -1, :]

            if len(tokens) < config.min_len:
                logits[:, model.config.eos_id] = -1e9

            for token_id in blocked_tokens(tokens, config.no_repeat_ngram_size):
                logits[:, token_id] = -1e9

            log_probs = F.log_softmax(logits, dim=-1).squeeze(0)
            top_scores, top_ids = torch.topk(log_probs, k=config.beam_size)

            for next_score, next_id in zip(top_scores.tolist(), top_ids.tolist()):
                new_tokens = tokens + [int(next_id)]
                done = int(next_id) == model.config.eos_id and len(new_tokens) >= config.min_len
                candidates.append((new_tokens, score + float(next_score), done))

        beams = sorted(
            candidates,
            key=lambda item: normalized_score(item[1], len(item[0]), config.length_penalty),
            reverse=True,
        )[: config.beam_size]

        if all(done for _, _, done in beams):
            break

    best_tokens, _, _ = max(
        beams,
        key=lambda item: normalized_score(item[1], len(item[0]), config.length_penalty),
    )
    return best_tokens

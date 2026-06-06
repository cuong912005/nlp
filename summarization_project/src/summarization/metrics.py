"""Evaluation metrics for generated summaries."""

from __future__ import annotations

from collections import Counter


def repetition_rate(texts: list[str], n: int = 3) -> float:
    """Return fraction of repeated n-grams across generated texts."""
    repeated = 0
    total = 0

    for text in texts:
        tokens = text.split()
        if len(tokens) < n:
            continue

        ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        counts = Counter(ngrams)
        total += len(ngrams)
        repeated += sum(count - 1 for count in counts.values() if count > 1)

    return repeated / max(1, total)


def compression_ratio(sources: list[str], predictions: list[str]) -> float:
    """Average prediction/source word-length ratio."""
    ratios = []
    for source, prediction in zip(sources, predictions):
        source_len = max(1, len(source.split()))
        ratios.append(len(prediction.split()) / source_len)
    return sum(ratios) / max(1, len(ratios))

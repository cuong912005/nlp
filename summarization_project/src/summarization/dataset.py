"""Dataset and batching utilities for cached summarization token IDs."""

from __future__ import annotations

import pickle
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset


class SummaryDataset(Dataset):
    """Load tokenized source-summary pairs from a pickle cache."""

    def __init__(self, cache_path: str | Path):
        with Path(cache_path).open("rb") as f:
            payload = pickle.load(f)

        self.records = payload["records"]
        self.pad_id = payload["pad_id"]
        self.vocab_size = payload["vocab_size"]
        self.max_source_len = payload["max_source_len"]
        self.max_target_len = payload["max_target_len"]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        item = self.records[idx]

        return {
            "id": item["id"],
            "source_ids": torch.tensor(item["source_ids"], dtype=torch.long),
            "target_ids": torch.tensor(item["target_ids"], dtype=torch.long),
        }


def make_causal_mask(size: int, device: torch.device | None = None) -> torch.Tensor:
    """Create lower-triangular mask so decoder cannot see future tokens."""
    return torch.tril(torch.ones((size, size), dtype=torch.bool, device=device))


def collate_summary_batch(batch: list[dict], pad_id: int) -> dict:
    """Pad variable-length examples and build Transformer masks."""
    source = [item["source_ids"] for item in batch]
    target = [item["target_ids"] for item in batch]
    ids = torch.tensor([item["id"] for item in batch], dtype=torch.long)

    # Dynamic padding giảm số token PAD so với padding cố định toàn dataset.
    source_padded = pad_sequence(source, batch_first=True, padding_value=pad_id)
    target_padded = pad_sequence(target, batch_first=True, padding_value=pad_id)

    # Encoder chỉ được attention vào token thật, không nhìn PAD.
    source_mask = (source_padded != pad_id).unsqueeze(1).unsqueeze(2)

    # Decoder vừa bỏ PAD vừa chặn nhìn token tương lai.
    target_padding_mask = (target_padded != pad_id).unsqueeze(1).unsqueeze(2)
    causal_mask = make_causal_mask(target_padded.size(1)).unsqueeze(0).unsqueeze(0)
    target_mask = target_padding_mask & causal_mask

    return {
        "ids": ids,
        "source": source_padded,
        "target": target_padded,
        "source_mask": source_mask,
        "target_mask": target_mask,
    }


def create_dataloader(
    dataset: SummaryDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    limit: int | None = None,
) -> DataLoader:
    """Create a DataLoader, optionally limiting examples for quick overfit tests."""
    data = dataset
    if limit is not None:
        # Subset nhỏ giúp kiểm tra mô hình có overfit được trước khi train tốn tiền.
        data = Subset(dataset, range(min(limit, len(dataset))))

    return DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_summary_batch(batch, dataset.pad_id),
    )

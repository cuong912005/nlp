"""Tokenize cleaned JSONL files and cache token ids for model training."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Ép stdout UTF-8 để log tiếng Việt không lỗi trên Windows.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from summarization.jsonl_io import read_jsonl
from summarization.sp_tokenizer import PAD_ID, SummaryTokenizer


def tokenize_split(
    input_path: Path,
    output_path: Path,
    tokenizer: SummaryTokenizer,
    max_source_len: int,
    max_target_len: int,
) -> dict:
    """Tokenize one split and save a pickle cache."""
    records = []
    source_lengths = []
    target_lengths = []
    source_at_max = 0
    target_at_max = 0

    for item in tqdm(list(read_jsonl(input_path)), desc=f"Tokenizing {input_path.name}"):
        # Source cũng có BOS/EOS để encoder thấy ranh giới văn bản.
        source_ids = tokenizer.encode(item["source"], add_bos=True, add_eos=True, max_len=max_source_len)

        # Target có BOS/EOS để train decoder bằng teacher forcing.
        target_ids = tokenizer.encode(item["target"], add_bos=True, add_eos=True, max_len=max_target_len)

        records.append(
            {
                "id": item["id"],
                "source_ids": source_ids,
                "target_ids": target_ids,
            }
        )
        source_lengths.append(len(source_ids))
        target_lengths.append(len(target_ids))
        source_at_max += int(len(source_ids) == max_source_len)
        target_at_max += int(len(target_ids) == max_target_len)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(
            {
                "records": records,
                "pad_id": PAD_ID,
                "vocab_size": tokenizer.vocab_size,
                "max_source_len": max_source_len,
                "max_target_len": max_target_len,
            },
            f,
        )

    return {
        "input": str(input_path),
        "output": str(output_path),
        "examples": len(records),
        "avg_source_tokens": sum(source_lengths) / max(1, len(source_lengths)),
        "avg_target_tokens": sum(target_lengths) / max(1, len(target_lengths)),
        "max_source_tokens": max(source_lengths) if source_lengths else 0,
        "max_target_tokens": max(target_lengths) if target_lengths else 0,
        "source_at_max_len": source_at_max,
        "target_at_max_len": target_at_max,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path, default=PROJECT_ROOT / "tokenizer" / "tokenizer_models" / "summary_bpe.model")
    parser.add_argument("--processed-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "data" / "cached")
    parser.add_argument("--max-source-len", type=int, default=512)
    parser.add_argument("--max-target-len", type=int, default=128)
    args = parser.parse_args()

    tokenizer = SummaryTokenizer(args.tokenizer)

    train_stats = tokenize_split(
        args.processed_dir / "train.jsonl",
        args.cache_dir / "train_tokenized.pkl",
        tokenizer,
        args.max_source_len,
        args.max_target_len,
    )
    valid_stats = tokenize_split(
        args.processed_dir / "valid.jsonl",
        args.cache_dir / "valid_tokenized.pkl",
        tokenizer,
        args.max_source_len,
        args.max_target_len,
    )

    stats_path = args.cache_dir / "tokenization_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        # Lưu stats để báo cáo và chọn max length khi train trên server.
        json.dump({"train": train_stats, "valid": valid_stats}, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print(train_stats)
    print(valid_stats)
    print(f"Stats saved to: {stats_path}")


if __name__ == "__main__":
    main()

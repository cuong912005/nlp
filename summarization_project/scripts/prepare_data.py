"""Prepare clean JSONL files from raw parquet data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Ép stdout UTF-8 để log tiếng Việt không lỗi trên Windows.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from summarization.data_processing import (
    FilterConfig,
    dataframe_to_examples,
    detect_columns,
    filter_examples,
    read_parquet,
    write_jsonl,
    write_stats,
)


def prepare_split(
    parquet_path: Path,
    output_path: Path,
    source_col: str | None,
    target_col: str | None,
    filter_config: FilterConfig,
) -> dict:
    """Read, normalize, filter, and save one data split."""
    df = read_parquet(parquet_path)

    guessed_source, guessed_target = detect_columns(df.columns)
    source_col = source_col or guessed_source
    target_col = target_col or guessed_target

    if source_col is None or target_col is None:
        raise SystemExit(
            "Cannot detect source/target columns. "
            f"Columns are: {list(df.columns)}. "
            "Please pass --source-col and --target-col."
        )

    if source_col not in df.columns or target_col not in df.columns:
        raise SystemExit(
            f"Invalid columns: source={source_col}, target={target_col}. "
            f"Available columns: {list(df.columns)}"
        )

    print(f"\nPreparing {parquet_path}")
    print(f"Using source column: {source_col}")
    print(f"Using target column: {target_col}")

    examples = dataframe_to_examples(df, source_col, target_col)
    kept, reasons = filter_examples(examples, filter_config)
    write_jsonl(output_path, kept)

    return {
        "input_file": str(parquet_path),
        "output_file": str(output_path),
        "source_col": source_col,
        "target_col": target_col,
        "raw_examples": len(examples),
        "kept_examples": len(kept),
        "filter_reasons": dict(reasons),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--valid", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--source-col", type=str, default=None)
    parser.add_argument("--target-col", type=str, default=None)
    parser.add_argument("--min-source-words", type=int, default=30)
    parser.add_argument("--min-target-words", type=int, default=3)
    parser.add_argument("--max-source-words", type=int, default=900)
    parser.add_argument("--max-target-words", type=int, default=180)
    parser.add_argument("--max-target-source-ratio", type=float, default=0.9)
    args = parser.parse_args()

    filter_config = FilterConfig(
        min_source_words=args.min_source_words,
        min_target_words=args.min_target_words,
        max_source_words=args.max_source_words,
        max_target_words=args.max_target_words,
        max_target_source_ratio=args.max_target_source_ratio,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_stats = prepare_split(
        args.train,
        args.output_dir / "train.jsonl",
        args.source_col,
        args.target_col,
        filter_config,
    )
    valid_stats = prepare_split(
        args.valid,
        args.output_dir / "valid.jsonl",
        args.source_col,
        args.target_col,
        filter_config,
    )

    stats = {"train": train_stats, "valid": valid_stats}
    write_stats(args.output_dir / "preprocess_stats.json", stats)

    print("\nDone.")
    print(f"Train kept: {train_stats['kept_examples']:,}/{train_stats['raw_examples']:,}")
    print(f"Valid kept: {valid_stats['kept_examples']:,}/{valid_stats['raw_examples']:,}")
    print(f"Stats saved to: {args.output_dir / 'preprocess_stats.json'}")


if __name__ == "__main__":
    main()

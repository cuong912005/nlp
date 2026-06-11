"""Prepare test JSONL without using it for training or model selection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Ép stdout UTF-8 để log tiếng Việt không lỗi trên Windows/Kaggle.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from summarization.data_processing import detect_columns, read_parquet, write_jsonl, write_stats
from summarization.text_cleaning import normalize_text, word_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--source-col", type=str, default=None)
    parser.add_argument("--target-col", type=str, default=None)
    args = parser.parse_args()

    df = read_parquet(args.test)
    guessed_source, guessed_target = detect_columns(df.columns)
    source_col = args.source_col or guessed_source
    target_col = args.target_col or guessed_target

    if source_col is None:
        raise SystemExit(f"Cannot detect source column. Available columns: {list(df.columns)}")
    if source_col not in df.columns:
        raise SystemExit(f"Invalid source column: {source_col}. Available columns: {list(df.columns)}")

    has_target = target_col is not None and target_col in df.columns

    print(f"Preparing test file: {args.test}")
    print(f"Using source column: {source_col}")
    print(f"Using target column: {target_col if has_target else '(none)'}")

    examples = []
    for idx, row in df.iterrows():
        source = normalize_text(row[source_col])
        target = normalize_text(row[target_col]) if has_target else ""

        # Không lọc test để metric tính trên toàn bộ public test.
        examples.append(
            {
                "source": source,
                "target": target,
                "source_words": word_count(source),
                "target_words": word_count(target),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "test.jsonl", examples)
    write_stats(
        args.output_dir / "test_stats.json",
        {
            "input_file": str(args.test),
            "output_file": str(args.output_dir / "test.jsonl"),
            "source_col": source_col,
            "target_col": target_col if has_target else None,
            "examples": len(examples),
            "has_reference_summary": has_target,
        },
    )

    print(f"Done. Test examples: {len(examples):,}")
    print(f"Saved to: {args.output_dir / 'test.jsonl'}")


if __name__ == "__main__":
    main()

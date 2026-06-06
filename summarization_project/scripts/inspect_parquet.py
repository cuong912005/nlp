"""Inspect parquet schemas before preprocessing."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Ép stdout UTF-8 để in tiếng Việt trên Windows không bị lỗi cp1252.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from summarization.data_processing import detect_columns, read_parquet


def preview(path: Path, rows: int) -> None:
    """Print schema and a small preview for one parquet file."""
    df = read_parquet(path)
    source_col, target_col = detect_columns(df.columns)

    print(f"\nFile: {path}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    print(f"Guessed source column: {source_col}")
    print(f"Guessed target column: {target_col}")

    # In vài dòng đầu để kiểm tra cột đoán có đúng nghĩa không.
    print("\nPreview:")
    print(df.head(rows).to_string())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--valid", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=3)
    args = parser.parse_args()

    preview(args.train, args.rows)
    preview(args.valid, args.rows)


if __name__ == "__main__":
    main()

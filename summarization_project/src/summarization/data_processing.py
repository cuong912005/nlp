"""Data loading and preprocessing for abstractive summarization."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

from .text_cleaning import keep_pair, normalize_text, word_count


SOURCE_CANDIDATES = (
    "source",
    "document",
    "article",
    "text",
    "content",
    "body",
    "input",
    "news",
)

TARGET_CANDIDATES = (
    "target",
    "summary",
    "highlights",
    "abstract",
    "title",
    "output",
)


@dataclass
class FilterConfig:
    """Filtering values are word-based so they work before tokenizer training."""

    min_source_words: int = 30
    min_target_words: int = 3
    max_source_words: int = 900
    max_target_words: int = 180
    max_target_source_ratio: float = 0.9


def require_pandas():
    """Import pandas lazily so the error message is easy to understand."""
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: pandas/pyarrow. Install with `pip install -r requirements.txt`."
        ) from exc
    return pd


def read_parquet(path: str | Path):
    """Read a parquet file with pandas."""
    pd = require_pandas()
    return pd.read_parquet(path)


def detect_columns(columns: Iterable[str]) -> tuple[str | None, str | None]:
    """Guess source and target columns from common summarization names."""
    normalized = {col.lower(): col for col in columns}

    source_col = next((normalized[name] for name in SOURCE_CANDIDATES if name in normalized), None)
    target_col = next((normalized[name] for name in TARGET_CANDIDATES if name in normalized), None)

    return source_col, target_col


def dataframe_to_examples(df, source_col: str, target_col: str) -> list[dict]:
    """Convert a dataframe to normalized source-target examples."""
    examples: list[dict] = []

    for source_raw, target_raw in tqdm(
        zip(df[source_col].tolist(), df[target_col].tolist()),
        total=len(df),
        desc="Normalizing",
    ):
        # Chuẩn hóa ngay khi đọc để các bước sau dùng cùng một biểu diễn text.
        source = normalize_text(source_raw)
        target = normalize_text(target_raw)

        examples.append({"source": source, "target": target})

    return examples


def filter_examples(examples: list[dict], config: FilterConfig) -> tuple[list[dict], Counter]:
    """Filter noisy examples and return kept data plus reason counts."""
    kept: list[dict] = []
    reasons: Counter = Counter()

    for item in tqdm(examples, desc="Filtering"):
        ok, reason = keep_pair(
            item["source"],
            item["target"],
            min_source_words=config.min_source_words,
            min_target_words=config.min_target_words,
            max_source_words=config.max_source_words,
            max_target_words=config.max_target_words,
            max_target_source_ratio=config.max_target_source_ratio,
        )
        reasons[reason] += 1

        if ok:
            # Lưu thêm độ dài word để phân tích dữ liệu và bucket batching sau này.
            kept.append(
                {
                    "source": item["source"],
                    "target": item["target"],
                    "source_words": word_count(item["source"]),
                    "target_words": word_count(item["target"]),
                }
            )

    return kept, reasons


def write_jsonl(path: str | Path, examples: list[dict]) -> None:
    """Write examples as JSONL so train/evaluate scripts can stream them later."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for idx, item in enumerate(examples):
            # id cố định giúp trace prediction về lại mẫu gốc khi phân tích lỗi.
            record = {"id": idx, **item}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_stats(path: str | Path, stats: dict) -> None:
    """Write preprocessing statistics for reports and debugging."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

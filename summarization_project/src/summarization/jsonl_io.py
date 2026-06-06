"""Small JSONL helpers used by data and tokenizer scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator


def read_jsonl(path: str | Path) -> Iterator[dict]:
    """Yield one JSON object per line."""
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, records: list[dict]) -> None:
    """Write records as UTF-8 JSONL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

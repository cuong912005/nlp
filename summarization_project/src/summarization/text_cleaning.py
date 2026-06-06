"""Text cleaning helpers for summarization data."""

from __future__ import annotations

import re
import unicodedata


def normalize_text(text: str) -> str:
    """Normalize text before filtering and tokenization."""
    if text is None:
        return ""

    # Chuẩn hóa Unicode để dấu tiếng Việt có cùng một kiểu biểu diễn.
    text = unicodedata.normalize("NFKC", str(text))

    # Gom nhiều khoảng trắng/xuống dòng thành một khoảng trắng.
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def word_count(text: str) -> int:
    """Count whitespace-separated words after basic normalization."""
    text = normalize_text(text)
    if not text:
        return 0
    return len(text.split())


def keep_pair(
    source: str,
    target: str,
    min_source_words: int,
    min_target_words: int,
    max_source_words: int,
    max_target_words: int,
    max_target_source_ratio: float,
) -> tuple[bool, str]:
    """Return whether a source-summary pair is usable and why."""
    source_words = word_count(source)
    target_words = word_count(target)

    # Bỏ mẫu rỗng vì không học được quan hệ source -> summary.
    if source_words == 0 or target_words == 0:
        return False, "empty"

    # Source quá ngắn thường không đúng bản chất bài toán tóm tắt.
    if source_words < min_source_words:
        return False, "source_too_short"

    # Summary quá ngắn dễ là nhãn lỗi hoặc tiêu đề cụt.
    if target_words < min_target_words:
        return False, "target_too_short"

    # Cắt mẫu quá dài để tránh vỡ bộ nhớ ở giai đoạn train baseline.
    if source_words > max_source_words:
        return False, "source_too_long"
    if target_words > max_target_words:
        return False, "target_too_long"

    # Summary gần dài bằng source thì không còn là tóm tắt tốt.
    if target_words >= source_words * max_target_source_ratio:
        return False, "target_not_shorter"

    return True, "kept"

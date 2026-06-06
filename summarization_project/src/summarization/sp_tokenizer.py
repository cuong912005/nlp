"""SentencePiece tokenizer wrapper for source and summary text."""

from __future__ import annotations

from pathlib import Path


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3


class SummaryTokenizer:
    """Thin wrapper so training code uses stable encode/decode methods."""

    def __init__(self, model_path: str | Path):
        try:
            import sentencepiece as spm
        except ModuleNotFoundError as exc:
            raise SystemExit("Missing dependency: sentencepiece. Install with `pip install -r requirements.txt`.") from exc

        self.processor = spm.SentencePieceProcessor()
        # Tránh lỗi path Unicode trên Windows bằng cách load model từ bytes.
        model_bytes = Path(model_path).read_bytes()
        self.processor.LoadFromSerializedProto(model_bytes)

    @property
    def vocab_size(self) -> int:
        """Return tokenizer vocabulary size."""
        return int(self.processor.get_piece_size())

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True, max_len: int | None = None) -> list[int]:
        """Encode text and optionally add BOS/EOS tokens."""
        ids = list(self.processor.encode(text, out_type=int))

        # BOS/EOS giúp decoder biết điểm bắt đầu và kết thúc summary.
        if add_bos:
            ids = [BOS_ID] + ids
        if add_eos:
            ids = ids + [EOS_ID]

        # Nếu cắt chuỗi, giữ EOS ở cuối để mô hình vẫn học điểm dừng.
        if max_len is not None and len(ids) > max_len:
            ids = ids[:max_len]
            if add_eos:
                ids[-1] = EOS_ID

        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode token ids, skipping special tokens."""
        cleaned = [idx for idx in ids if idx not in {PAD_ID, BOS_ID, EOS_ID}]
        return self.processor.decode(cleaned)

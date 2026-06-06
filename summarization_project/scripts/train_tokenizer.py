"""Train a SentencePiece tokenizer from cleaned JSONL data."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Ép stdout UTF-8 để log tiếng Việt không lỗi trên Windows.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from summarization.jsonl_io import read_jsonl
from summarization.sp_tokenizer import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN


def write_training_text(jsonl_path: Path, output_txt: Path) -> int:
    """Write source and target text into a plain corpus for tokenizer training."""
    count = 0
    with output_txt.open("w", encoding="utf-8") as f:
        for item in read_jsonl(jsonl_path):
            # Train tokenizer trên cả source và summary vì hai phía cùng ngôn ngữ.
            f.write(item["source"] + "\n")
            f.write(item["target"] + "\n")
            count += 2
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data" / "processed" / "train.jsonl")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "tokenizer" / "tokenizer_models")
    parser.add_argument("--vocab-size", type=int, default=16000)
    parser.add_argument("--model-prefix", type=str, default="summary_bpe")
    args = parser.parse_args()

    try:
        import sentencepiece as spm
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: sentencepiece. Install with `pip install -r requirements.txt`.") from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        corpus_path = tmp_path / "tokenizer_corpus.txt"
        temp_model_prefix = tmp_path / args.model_prefix
        line_count = write_training_text(args.input, corpus_path)

        print(f"Tokenizer corpus lines: {line_count:,}")
        print(f"Training tokenizer in temp dir: {temp_model_prefix}")

        spm.SentencePieceTrainer.train(
            input=str(corpus_path),
            # SentencePiece trên Windows dễ lỗi với đường dẫn có dấu, nên train ở temp ASCII trước.
            model_prefix=str(temp_model_prefix),
            vocab_size=args.vocab_size,
            model_type="bpe",
            max_sentence_length=10000,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            pad_piece=PAD_TOKEN,
            unk_piece=UNK_TOKEN,
            bos_piece=BOS_TOKEN,
            eos_piece=EOS_TOKEN,
            character_coverage=1.0,
        )

        final_model = args.output_dir / f"{args.model_prefix}.model"
        final_vocab = args.output_dir / f"{args.model_prefix}.vocab"

        # Copy model về project để dùng lại trên local/server.
        shutil.copy2(f"{temp_model_prefix}.model", final_model)
        shutil.copy2(f"{temp_model_prefix}.vocab", final_vocab)

    print("Done.")
    print(f"Model: {args.output_dir / (args.model_prefix + '.model')}")
    print(f"Vocab: {args.output_dir / (args.model_prefix + '.vocab')}")


if __name__ == "__main__":
    main()

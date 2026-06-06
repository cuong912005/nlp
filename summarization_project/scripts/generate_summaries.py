"""Generate summaries from a trained checkpoint and write JSONL output."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Ép stdout UTF-8 để log tiếng Việt không lỗi trên Windows.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from summarization.generation import GenerationConfig, beam_search_decode, greedy_decode
from summarization.jsonl_io import read_jsonl
from summarization.models import SummarizationTransformer, TransformerConfig
from summarization.sp_tokenizer import SummaryTokenizer


def load_records(cache_path: Path) -> list[dict]:
    """Load tokenized records from cache."""
    with cache_path.open("rb") as f:
        payload = pickle.load(f)
    return payload["records"]


def load_model(checkpoint_path: Path, device: torch.device) -> SummarizationTransformer:
    """Restore model and config from a train checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = TransformerConfig(**checkpoint["config"])
    model = SummarizationTransformer(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints" / "baseline" / "best.pt")
    parser.add_argument("--cache", type=Path, default=PROJECT_ROOT / "data" / "cached" / "valid_tokenized.pkl")
    parser.add_argument("--processed-jsonl", type=Path, default=PROJECT_ROOT / "data" / "processed" / "valid.jsonl")
    parser.add_argument("--tokenizer", type=Path, default=PROJECT_ROOT / "tokenizer" / "tokenizer_models" / "summary_bpe.model")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "valid_predictions.jsonl")
    parser.add_argument("--method", choices=["greedy", "beam"], default="beam")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--min-len", type=int, default=20)
    parser.add_argument("--beam-size", type=int, default=4)
    parser.add_argument("--length-penalty", type=float, default=0.8)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    tokenizer = SummaryTokenizer(args.tokenizer)
    records = load_records(args.cache)
    raw_items = list(read_jsonl(args.processed_jsonl))

    if args.limit is not None:
        records = records[: args.limit]
        raw_items = raw_items[: args.limit]

    gen_config = GenerationConfig(
        max_len=args.max_len,
        min_len=args.min_len,
        beam_size=args.beam_size,
        length_penalty=args.length_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for record, raw_item in tqdm(list(zip(records, raw_items)), desc="Generating"):
            source = torch.tensor([record["source_ids"]], dtype=torch.long)

            if args.method == "greedy":
                pred_ids = greedy_decode(model, source, gen_config)
            else:
                pred_ids = beam_search_decode(model, source, gen_config)

            prediction = tokenizer.decode(pred_ids)

            # JSONL giúp evaluate và phân tích lỗi dễ trace theo từng mẫu.
            output_item = {
                "id": raw_item["id"],
                "source": raw_item["source"],
                "reference": raw_item["target"],
                "prediction": prediction,
                "method": args.method,
            }
            f.write(json.dumps(output_item, ensure_ascii=False) + "\n")

    print(f"Saved predictions to: {args.output}")


if __name__ == "__main__":
    main()

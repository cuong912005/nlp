"""Evaluate generated summaries with ROUGE and simple repetition stats."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Ép stdout UTF-8 để log tiếng Việt không lỗi trên Windows.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from summarization.jsonl_io import read_jsonl
from summarization.metrics import compression_ratio, repetition_rate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=PROJECT_ROOT / "outputs" / "valid_predictions.jsonl")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "valid_rouge.json")
    args = parser.parse_args()

    try:
        from rouge_score import rouge_scorer
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: rouge-score. Install with `pip install -r requirements.txt`.") from exc

    items = list(read_jsonl(args.predictions))
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)

    sums = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    references = []
    predictions = []
    sources = []

    for item in items:
        reference = item["reference"]
        prediction = item["prediction"]
        scores = scorer.score(reference, prediction)

        # Dùng F1 vì cân bằng giữa precision và recall của summary.
        for key in sums:
            sums[key] += scores[key].fmeasure

        references.append(reference)
        predictions.append(prediction)
        sources.append(item["source"])

    count = max(1, len(items))
    results = {
        "examples": len(items),
        "rouge-1": sums["rouge1"] / count,
        "rouge-2": sums["rouge2"] / count,
        "rouge-l": sums["rougeL"] / count,
        "repetition-3gram": repetition_rate(predictions, n=3),
        "compression-ratio": compression_ratio(sources, predictions),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"Saved metrics to: {args.output}")


if __name__ == "__main__":
    main()

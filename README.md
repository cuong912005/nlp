# Vietnamese Abstractive Summarization - Task 1

Task 1 implements an encoder-decoder Transformer **from scratch** for Vietnamese abstractive text summarization. The implementation uses basic PyTorch modules and does not rely on high-level modules such as `nn.Transformer`.

The goal is to generate a concise summary from a long Vietnamese document while preserving the main information.

## Main Features

- Transformer encoder-decoder implemented from scratch.
- SentencePiece BPE tokenizer trained on the training split.
- Sinusoidal positional encoding.
- Multi-head self-attention and encoder-decoder cross-attention.
- Padding mask for source/cross-attention and causal mask for decoder self-attention.
- Pre-LN and Post-LN support for ablation.
- GELU and ReLU feed-forward activation support.
- Label smoothing.
- Shared source-target embeddings and output weight tying.
- AdamW with warmup and inverse-square-root learning-rate decay.
- Gradient clipping.
- Greedy decoding and beam search.
- Length penalty and no-repeat n-gram blocking.
- ROUGE-1, ROUGE-2, ROUGE-L, repetition rate, and compression-ratio evaluation.

## Project Structure

```text
summarization_project/
├── configs/
│   └── base.yaml
├── notebooks/
│   ├── kaggle_train_eval_report.ipynb
│   └── kaggle_transformer_ablation_report.ipynb
├── scripts/
│   ├── inspect_parquet.py
│   ├── prepare_data.py
│   ├── prepare_test_data.py
│   ├── train_tokenizer.py
│   ├── tokenize_cache.py
│   ├── train_baseline.py
│   ├── generate_summaries.py
│   └── evaluate_rouge.py
└── src/
    └── summarization/
        ├── dataset.py
        ├── generation.py
        ├── metrics.py
        ├── sp_tokenizer.py
        └── models/
            ├── attention.py
            ├── config.py
            ├── decoder.py
            ├── encoder.py
            ├── feed_forward.py
            ├── positional_encoding.py
            └── transformer.py
```

## Environment Setup

From the repository root:

```bash
pip install -r requirements.txt
```

Then enter the Task 1 project folder:

```bash
cd summarization_project
```

Main dependencies:

- `torch`
- `pandas`
- `pyarrow`
- `sentencepiece`
- `rouge-score`
- `tqdm`
- `pyyaml`

## Data Format

The expected raw data files are parquet files containing at least:

- input document column, usually `article`
- target summary column, usually `summary`

Example files:

```text
train-00000-of-00001.parquet
valid-00000-of-00001.parquet
test-00000-of-00001.parquet
```

## 1. Inspect Dataset Schema

```bash
python scripts/inspect_parquet.py \
  --train ../train-00000-of-00001.parquet \
  --valid ../valid-00000-of-00001.parquet
```

This step previews the parquet files and helps confirm the correct source/target column names.

## 2. Prepare Train and Validation Data

If the script can automatically detect the columns:

```bash
python scripts/prepare_data.py \
  --train ../train-00000-of-00001.parquet \
  --valid ../valid-00000-of-00001.parquet
```

If the columns must be specified explicitly:

```bash
python scripts/prepare_data.py \
  --train ../train-00000-of-00001.parquet \
  --valid ../valid-00000-of-00001.parquet \
  --source-col article \
  --target-col summary
```

This creates:

```text
data/processed/train.jsonl
data/processed/valid.jsonl
data/processed/preprocess_stats.json
```

## 3. Prepare Test Data

```bash
python scripts/prepare_test_data.py \
  --test ../test-00000-of-00001.parquet \
  --source-col article \
  --target-col summary
```

This creates:

```text
data/processed/test.jsonl
```

## 4. Train SentencePiece BPE Tokenizer

```bash
python scripts/train_tokenizer.py \
  --input data/processed/train.jsonl \
  --vocab-size 16000
```

This creates:

```text
tokenizer/tokenizer_models/summary_bpe.model
tokenizer/tokenizer_models/summary_bpe.vocab
```

Special tokens:

```text
<PAD>, <UNK>, <BOS>, <EOS>
```

## 5. Tokenize and Cache Data

For train and validation:

```bash
python scripts/tokenize_cache.py \
  --tokenizer tokenizer/tokenizer_models/summary_bpe.model \
  --splits train valid \
  --max-source-len 512 \
  --max-target-len 128
```

For train, validation, and test:

```bash
python scripts/tokenize_cache.py \
  --tokenizer tokenizer/tokenizer_models/summary_bpe.model \
  --splits train valid test \
  --max-source-len 512 \
  --max-target-len 128
```

This creates:

```text
data/cached/train_tokenized.pkl
data/cached/valid_tokenized.pkl
data/cached/test_tokenized.pkl
data/cached/tokenization_stats.json
```

## 6. Train the Transformer

Small CPU smoke test:

```bash
python scripts/train_baseline.py \
  --epochs 1 \
  --batch-size 2 \
  --limit-train 8 \
  --limit-valid 4 \
  --d-model 64 \
  --layers 1 \
  --heads 4 \
  --d-ff 128 \
  --warmup-steps 10 \
  --device cpu
```

Full training configuration used for the report:

```bash
python scripts/train_baseline.py \
  --epochs 10 \
  --batch-size 4 \
  --d-model 256 \
  --layers 4 \
  --heads 8 \
  --d-ff 1024 \
  --device cuda \
  --save-dir checkpoints/full_improvements
```

Important default options:

```text
norm_type = pre
activation = gelu
label_smoothing = 0.1
share_embeddings = true
weight_tying = true
use_scheduler = true
grad_clip = 1.0
```

Training outputs:

```text
checkpoints/<run_name>/best.pt
checkpoints/<run_name>/latest.pt
checkpoints/<run_name>/history.json
```

## 7. Run Ablation Experiments

The notebook `notebooks/kaggle_transformer_ablation_report.ipynb` runs the ablation suite used in the report.

Paper-style Transformer, closer to the original Transformer layer layout:

```bash
python scripts/train_baseline.py \
  --epochs 10 \
  --batch-size 4 \
  --d-model 256 \
  --layers 4 \
  --heads 8 \
  --d-ff 1024 \
  --norm-type post \
  --activation relu \
  --device cuda \
  --save-dir checkpoints/paper_style_transformer
```

Full-improvement Transformer:

```bash
python scripts/train_baseline.py \
  --epochs 10 \
  --batch-size 4 \
  --d-model 256 \
  --layers 4 \
  --heads 8 \
  --d-ff 1024 \
  --device cuda \
  --save-dir checkpoints/full_improvements
```

No label smoothing:

```bash
python scripts/train_baseline.py \
  --epochs 10 \
  --batch-size 4 \
  --d-model 256 \
  --layers 4 \
  --heads 8 \
  --d-ff 1024 \
  --label-smoothing 0.0 \
  --device cuda \
  --save-dir checkpoints/no_label_smoothing
```

No shared embeddings and no weight tying:

```bash
python scripts/train_baseline.py \
  --epochs 10 \
  --batch-size 4 \
  --d-model 256 \
  --layers 4 \
  --heads 8 \
  --d-ff 1024 \
  --no-share-embeddings \
  --no-weight-tying \
  --device cuda \
  --save-dir checkpoints/no_shared_no_tying
```

## 8. Generate Summaries

Beam search generation on validation:

```bash
python scripts/generate_summaries.py \
  --checkpoint checkpoints/full_improvements/best.pt \
  --cache data/cached/valid_tokenized.pkl \
  --processed-jsonl data/processed/valid.jsonl \
  --tokenizer tokenizer/tokenizer_models/summary_bpe.model \
  --method beam \
  --beam-size 4 \
  --length-penalty 0.8 \
  --no-repeat-ngram-size 3 \
  --device cuda \
  --output outputs/valid_predictions_beam.jsonl
```

Greedy generation on validation:

```bash
python scripts/generate_summaries.py \
  --checkpoint checkpoints/full_improvements/best.pt \
  --cache data/cached/valid_tokenized.pkl \
  --processed-jsonl data/processed/valid.jsonl \
  --tokenizer tokenizer/tokenizer_models/summary_bpe.model \
  --method greedy \
  --device cuda \
  --output outputs/valid_predictions_greedy.jsonl
```

Beam search generation on test:

```bash
python scripts/generate_summaries.py \
  --checkpoint checkpoints/full_improvements/best.pt \
  --cache data/cached/test_tokenized.pkl \
  --processed-jsonl data/processed/test.jsonl \
  --tokenizer tokenizer/tokenizer_models/summary_bpe.model \
  --method beam \
  --beam-size 4 \
  --length-penalty 0.8 \
  --no-repeat-ngram-size 3 \
  --device cuda \
  --output outputs/test_predictions_beam.jsonl
```

## 9. Evaluate ROUGE

Validation:

```bash
python scripts/evaluate_rouge.py \
  --predictions outputs/valid_predictions_beam.jsonl \
  --output outputs/valid_rouge_beam.json
```

Test:

```bash
python scripts/evaluate_rouge.py \
  --predictions outputs/test_predictions_beam.jsonl \
  --output outputs/test_rouge_beam.json
```

The evaluator reports:

```text
examples
rouge-1
rouge-2
rouge-l
repetition-3gram
compression-ratio
```

## Reported Results

Task 1 ablation results on the full validation set:

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|
| Paper-style Transformer | 0.5673 | 0.1937 | 0.3005 |
| Full improvements | 0.5815 | 0.2119 | 0.3076 |
| No label smoothing | 0.5787 | 0.2095 | 0.3069 |
| No shared embedding / no weight tying | **0.5899** | **0.2143** | **0.3110** |

Validation decoding comparison with the same trained checkpoint:

| Decoding method | ROUGE-1 | ROUGE-2 | ROUGE-L | Compression |
|---|---:|---:|---:|---:|
| Greedy decoding | 0.5724 | 0.2003 | 0.3057 | 0.1718 |
| Beam + length penalty + no-repeat 3-gram | 0.5768 | 0.2087 | 0.3053 | 0.1617 |

Full test result:

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|
| Scratch Transformer, beam decoding | 0.5747 | 0.2038 | 0.3045 |

## Notes on Interpreting the Ablations

- `paper_style_transformer` mainly uses Post-LN and ReLU to be closer to the original Transformer design.
- `full_improvements` uses Pre-LN and GELU by default, together with practical training and decoding improvements.
- `no_label_smoothing` has lower numerical cross-entropy loss but slightly lower ROUGE than the full-improvement model.
- `no_shared_no_tying` obtains the best validation ROUGE in this run, likely because separate source embedding, target embedding, and output projection increase model capacity. The trade-off is a larger parameter count.
- Beam search improves ROUGE-1 and ROUGE-2 over greedy decoding, while ROUGE-L is similar.

## Code Map

```text
src/summarization/models/config.py
  Transformer hyperparameters and token IDs.

src/summarization/models/attention.py
  Multi-head scaled dot-product attention.

src/summarization/models/positional_encoding.py
  Sinusoidal positional encoding.

src/summarization/models/feed_forward.py
  Position-wise FFN with GELU/ReLU.

src/summarization/models/encoder.py
  Encoder layer and encoder stack.

src/summarization/models/decoder.py
  Decoder layer and decoder stack.

src/summarization/models/transformer.py
  Full summarization Transformer with encode/decode/forward.

src/summarization/dataset.py
  Dataset loading, padding, source mask, target causal mask.

src/summarization/generation.py
  Greedy decoding, beam search, length penalty, no-repeat n-gram.

scripts/train_baseline.py
  Training loop, optimizer, scheduler, checkpoint saving.

scripts/generate_summaries.py
  Inference script for validation/test predictions.

scripts/evaluate_rouge.py
  ROUGE and auxiliary metric evaluation.
```

## Reproducibility Notes

- Use validation results for model selection.
- Do not tune hyperparameters on the test split.
- Smoke-test checkpoints are only for checking that the pipeline runs; they should not be reported as final model quality.
- The main report uses full validation and full test results from the exported Kaggle outputs.


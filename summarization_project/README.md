# Abstractive Summarization From Scratch

Giai đoạn 1 tập trung vào dữ liệu:

1. Inspect file parquet để biết schema.
2. Chuẩn hóa dữ liệu về cặp `source` và `target`.
3. Làm sạch, lọc mẫu lỗi, lưu JSONL.
4. Huấn luyện tokenizer subword từ dữ liệu train.
5. Tokenize và cache dữ liệu để train Transformer.

## Cài môi trường

```bash
pip install -r ../requirements.txt
```

## Inspect schema

```bash
python scripts/inspect_parquet.py --train ../train-00000-of-00001.parquet --valid ../valid-00000-of-00001.parquet
```

## Chuẩn bị dữ liệu

Nếu script tự nhận đúng cột:

```bash
python scripts/prepare_data.py --train ../train-00000-of-00001.parquet --valid ../valid-00000-of-00001.parquet
```

Nếu cần chỉ định cột:

```bash
python scripts/prepare_data.py --train ../train-00000-of-00001.parquet --valid ../valid-00000-of-00001.parquet --source-col article --target-col summary
```

## Train tokenizer

```bash
python scripts/train_tokenizer.py --input data/processed/train.jsonl
```

## Tokenize và cache

```bash
python scripts/tokenize_cache.py --tokenizer tokenizer/tokenizer_models/summary_bpe.model
```

Sau giai đoạn này cần có:

- `data/processed/train.jsonl`
- `data/processed/valid.jsonl`
- `data/processed/preprocess_stats.json`
- `tokenizer/tokenizer_models/summary_bpe.model`
- `data/cached/train_tokenized.pkl`
- `data/cached/valid_tokenized.pkl`

## Train baseline Transformer

Smoke test local với model nhỏ:

```bash
python scripts/train_baseline.py --epochs 1 --batch-size 2 --limit-train 8 --limit-valid 4 --d-model 64 --layers 1 --heads 4 --d-ff 128 --warmup-steps 10 --device cpu
```

Train baseline thật trên server:

```bash
python scripts/train_baseline.py --epochs 10 --batch-size 8 --d-model 256 --layers 4 --heads 8 --d-ff 1024 --device cuda
```

## Giai đoạn 3: cải tiến và ablation

Các cải tiến đã có trong code:

- Pre-LN encoder/decoder để train ổn định hơn.
- Label smoothing để giảm overconfidence.
- Shared embeddings vì source và summary cùng tiếng Việt.
- Weight tying để giảm tham số output projection.
- Warmup + inverse-sqrt scheduler.
- Gradient clipping.
- Beam search + length penalty.
- No-repeat n-gram blocking.

Ví dụ chạy ablation trên server:

```bash
# Transformer gốc kiểu Attention Is All You Need hơn: Post-LN + ReLU,
# không label smoothing, không shared embedding, không weight tying.
python scripts/train_baseline.py --epochs 10 --batch-size 8 --norm-type post --activation relu --label-smoothing 0.0 --no-share-embeddings --no-weight-tying --device cuda

# Bản đầy đủ
python scripts/train_baseline.py --epochs 10 --batch-size 8 --device cuda

# Tắt label smoothing
python scripts/train_baseline.py --epochs 10 --batch-size 8 --label-smoothing 0.0 --device cuda

# Tắt shared embedding và weight tying
python scripts/train_baseline.py --epochs 10 --batch-size 8 --no-share-embeddings --no-weight-tying --device cuda
```

Sinh summary trên validation:

```bash
python scripts/generate_summaries.py --checkpoint checkpoints/baseline/best.pt --method beam --beam-size 4 --length-penalty 0.8 --no-repeat-ngram-size 3 --device cuda
```

Đánh giá ROUGE:

```bash
python scripts/evaluate_rouge.py --predictions outputs/valid_predictions.jsonl
```

Lưu ý: checkpoint tạo từ smoke test local chỉ dùng để kiểm tra pipeline chạy được, không dùng số ROUGE đó để kết luận chất lượng mô hình.

Bảng nên ghi vào báo cáo:

```text
Model                         ROUGE-1  ROUGE-2  ROUGE-L  repetition-3gram  Params
Full improvements             ...
No label smoothing            ...
No shared emb / weight tying   ...
Greedy decoding               ...
Beam + length penalty          ...
Beam + no-repeat 3gram         ...
```

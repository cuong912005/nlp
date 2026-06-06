# Ghi chú viết báo cáo

## 1. Bài toán

Xây dựng hệ thống tóm tắt văn bản tự động dạng abstractive summarization. Đầu vào là một bài viết dài (`article`), đầu ra là bản tóm tắt ngắn hơn (`summary`) nhưng vẫn giữ ý chính.

Yêu cầu kỹ thuật chính: tự cài Transformer encoder-decoder từ đầu bằng PyTorch, không dùng `nn.Transformer` hoặc mô hình pretrained.

## 2. Dữ liệu

Dữ liệu hiện có:

- Train: `10,775` mẫu.
- Validation: `1,349` mẫu.
- Cột input: `article`.
- Cột output: `summary`.

Pipeline tiền xử lý:

1. Đọc file parquet.
2. Chuẩn hóa Unicode.
3. Xóa khoảng trắng thừa.
4. Lọc mẫu rỗng, quá ngắn, quá dài hoặc summary không ngắn hơn article.
5. Lưu dữ liệu sạch thành JSONL.
6. Train tokenizer SentencePiece/BPE từ dữ liệu train.
7. Tokenize và cache token IDs để train nhanh hơn.

Kết quả lọc hiện tại:

```text
Train kept: 10,775 / 10,775
Valid kept: 1,349 / 1,349
```

Tokenizer:

```text
SentencePiece BPE
Vocab size: 16,000
Special tokens: <PAD>, <UNK>, <BOS>, <EOS>
```

## 3. Mô hình

Mô hình chính là Transformer encoder-decoder tự cài:

- Token embedding.
- Sinusoidal positional encoding.
- Multi-head self-attention.
- Encoder stack.
- Decoder stack.
- Cross-attention trong decoder.
- Feed-forward network.
- Residual connection.
- LayerNorm.
- Padding mask.
- Causal mask.
- Output projection sang vocabulary.

Không sử dụng:

- `nn.Transformer`
- `nn.TransformerEncoder`
- `nn.TransformerDecoder`
- BART/T5/Pegasus hoặc mô hình pretrained.

Cấu hình baseline đề xuất:

```text
d_model = 256
encoder_layers = 4
decoder_layers = 4
heads = 8
d_ff = 1024
vocab_size = 16000
max_source_len = 512
max_target_len = 128
```

Số tham số:

```text
baseline_default: 11,469,824 parameters
```

## 4. Huấn luyện

Training dùng teacher forcing:

```text
decoder_input = target[:, :-1]
label = target[:, 1:]
```

Loss:

```text
CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)
```

Optimizer:

```text
AdamW
Warmup + inverse-sqrt scheduler
Gradient clipping
```

Checkpoint:

- `latest.pt`
- `best.pt`
- `history.json`

## 5. Cải tiến đã cài

| Cải tiến | Lý do tốt hơn |
|---|---|
| Pre-LayerNorm | Giúp train Transformer ổn định hơn, nhất là khi stack nhiều layer. |
| Label smoothing | Tránh mô hình quá tự tin vào một cách diễn đạt duy nhất, phù hợp với tóm tắt vì có nhiều summary đúng. |
| Shared embeddings | Source và target cùng tiếng Việt nên chia sẻ embedding giúp giảm tham số và học biểu diễn từ nhất quán. |
| Weight tying | Dùng lại target embedding cho output projection, giảm tham số và thường tổng quát tốt hơn. |
| Warmup scheduler | Transformer nhạy với learning rate ban đầu; warmup giúp tránh cập nhật quá mạnh lúc đầu. |
| Gradient clipping | Giảm nguy cơ gradient spike khi văn bản dài. |
| Beam search | Tốt hơn greedy vì xét nhiều ứng viên tóm tắt. |
| Length penalty | Tránh beam search thiên về summary quá ngắn. |
| No-repeat n-gram | Giảm lỗi lặp cụm từ khi sinh summary. |

## 6. Chứng minh thực nghiệm cần làm

Để chứng minh cải tiến tốt hơn, cần chạy ablation trên validation:

```text
Model                         ROUGE-1  ROUGE-2  ROUGE-L  repetition-3gram
Full improvements             ...
No label smoothing            ...
No shared emb / weight tying   ...
Greedy decoding               ...
Beam + length penalty          ...
Beam + no-repeat 3gram         ...
```

Hiện tại code đã hỗ trợ ablation, nhưng chưa có kết quả train full trên GPU.

## 7. Đánh giá

Metric bắt buộc:

- ROUGE-1
- ROUGE-2
- ROUGE-L

Metric bổ sung đã cài:

- Repetition 3-gram.
- Compression ratio.

Lệnh đánh giá:

```bash
python scripts/generate_summaries.py --checkpoint checkpoints/baseline/best.pt --method beam --beam-size 4 --length-penalty 0.8 --no-repeat-ngram-size 3 --device cuda
python scripts/evaluate_rouge.py --predictions outputs/valid_predictions.jsonl
```

## 8. Trạng thái hiện tại

Đã hoàn thành:

- Giai đoạn 1: tiền xử lý dữ liệu, tokenizer, cache.
- Giai đoạn 2: Transformer baseline tự cài, train script, smoke test.
- Giai đoạn 3: cải tiến decoding, ablation flags, ROUGE evaluation.

Chưa hoàn thành:

- Train full trên GPU.
- Kết quả ROUGE chính thức.
- Ablation study thực nghiệm.
- Đánh giá trên public test vì test chưa công bố.

Lưu ý: kết quả smoke test hiện tại chỉ dùng để kiểm tra pipeline chạy được, không dùng làm kết quả báo cáo.

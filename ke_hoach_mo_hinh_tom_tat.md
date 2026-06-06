# Kế hoạch xây dựng mô hình tóm tắt văn bản tự động

## 1. Mục tiêu và phạm vi

Bài toán cần xây dựng hệ thống tóm tắt văn bản tự động dạng abstractive summarization: đầu vào là một văn bản dài, đầu ra là một bản tóm tắt ngắn hơn nhưng vẫn giữ các ý chính. Mô hình chính phải là Transformer encoder-decoder tự lập trình bằng PyTorch, không dùng `nn.Transformer`, không dùng các mô hình pretrained như BART/T5/Pegasus để thay thế phần mô hình chính.

Repo `NLP/Problem 1` có thể tham khảo về cách tổ chức code, tokenizer, attention, encoder, decoder, train/evaluate, checkpoint và beam search. Tuy nhiên bài toán dịch máy trong repo đó khác tóm tắt văn bản, nên cần thay đổi dữ liệu, mục tiêu huấn luyện, decoding, metric và các cải tiến phù hợp với văn bản dài.

## 2. Pipeline tổng thể

1. Chuẩn bị dữ liệu từ các file `train-00000-of-00001.parquet` và `valid-00000-of-00001.parquet`.
2. Xác định các cột văn bản nguồn và tóm tắt đích, ví dụ `document`/`summary`, `article`/`highlights`, hoặc tên tương tự.
3. Làm sạch văn bản: chuẩn hóa Unicode, xóa khoảng trắng dư, loại mẫu rỗng, loại mẫu có nguồn hoặc tóm tắt quá ngắn.
4. Chia dữ liệu: dùng `train` để huấn luyện, dùng `valid` để chọn mô hình và làm ablation. Vì test public sẽ được công bố sau, chưa dùng test trong giai đoạn phát triển.
5. Huấn luyện tokenizer subword BPE/SentencePiece từ đầu trên cả văn bản nguồn và tóm tắt.
6. Tokenize và cache dữ liệu sang `.pkl` hoặc `.pt` để tăng tốc huấn luyện.
7. Huấn luyện Transformer baseline.
8. Thêm từng cải tiến có kiểm soát và đo bằng ablation study.
9. Sinh tóm tắt trên validation bằng greedy và beam search trong lúc phát triển; khi có public test thì chạy inference lại trên test.
10. Đánh giá bằng ROUGE-1, ROUGE-2, ROUGE-L; thêm BLEU, METEOR hoặc BERTScore nếu kịp.

## 2.1. Checklist yêu cầu cơ bản

| Yêu cầu | Trạng thái trong kế hoạch | Cần làm khi code |
|---|---|---|
| Bài toán abstractive summarization | Đã có | Dữ liệu phải map đúng `source` -> `summary`, không làm extractive đơn thuần. |
| Transformer tự xây dựng từ đầu | Đã có | Không gọi `nn.Transformer`, `nn.TransformerEncoder`, `nn.TransformerDecoder` hoặc mô hình pretrained. |
| Có Encoder, Decoder, Multi-Head Attention, Positional Encoding, FFN | Đã có | Tự cài từng module trong `models/`. |
| Huấn luyện mô hình | Đã có | Có script train, checkpoint, resume, log loss. |
| Đánh giá ROUGE-1, ROUGE-2, ROUGE-L | Đã có | Cài `trainer/evaluate.py`, chạy trên validation trước, test public sau. |
| Có cải tiến phương pháp | Đã có đề xuất | Cần chạy thực nghiệm ablation để chứng minh. |

Kết luận: kế hoạch đã bao phủ yêu cầu cơ bản của đề. Tuy nhiên ở thời điểm kế hoạch, phần "đạt yêu cầu" mới là về thiết kế. Khi triển khai cần có code chạy được, log huấn luyện, checkpoint và bảng ROUGE thì mới coi là hoàn thành thực nghiệm.

## 3. Cấu trúc thư mục đề xuất

```text
summarization_project/
├── data/
│   ├── raw/
│   ├── processed/
│   └── cached/
├── tokenizer/
│   ├── train_tokenizer.py
│   └── tokenizer_models/
├── models/
│   ├── config.py
│   ├── transformer.py
│   ├── encoder.py
│   ├── decoder.py
│   ├── attention.py
│   ├── feed_forward.py
│   ├── embeddings.py
│   ├── positional_encoding.py
│   ├── layer_norm.py
│   ├── beam_search.py
│   └── label_smoothing.py
├── utils/
│   ├── data_processing.py
│   ├── batching.py
│   └── text_cleaning.py
├── trainer/
│   ├── train.py
│   ├── inference.py
│   └── evaluate.py
├── checkpoints/
├── outputs/
└── README.md
```

Cấu trúc này lấy cảm hứng từ `NLP/Problem 1`, nhưng đổi tên và trách nhiệm cho bài toán tóm tắt. Các module `attention.py`, `encoder.py`, `decoder.py`, `beam_search.py`, `label_smoothing.py` có thể tham khảo logic, còn `data_processing.py`, `evaluate.py`, `inference.py` cần viết lại theo dữ liệu tóm tắt và ROUGE.

## 4. Tiền xử lý dữ liệu

Bước đầu tiên là kiểm tra schema của file parquet để biết chính xác tên cột. Sau đó chuẩn hóa mỗi mẫu thành:

```python
{
    "source": "văn bản dài",
    "target": "bản tóm tắt"
}
```

Các quy tắc lọc nên dùng:

- Bỏ mẫu thiếu `source` hoặc `target`.
- Bỏ mẫu có `source` quá ngắn vì không đúng bản chất tóm tắt.
- Bỏ mẫu có `target` dài hơn hoặc gần bằng `source`.
- Giới hạn `source_max_len`, ví dụ 512 hoặc 768 token ở giai đoạn đầu; giới hạn `target_max_len`, ví dụ 128 hoặc 160 token.
- Lưu thống kê độ dài nguồn/đích trước và sau khi lọc để giải thích quyết định trong báo cáo.

Vì tóm tắt thường có nguồn dài hơn dịch máy, nên cần dùng dynamic padding theo batch và có thể bucket theo độ dài. Cách này giảm số token padding, tiết kiệm GPU và giúp batch thực sự chứa nhiều thông tin hơn.

## 5. Tokenizer

Sử dụng tokenizer subword tự huấn luyện, có thể tham khảo thư mục `NLP/Problem 1/SentencePiece-from-scratch`.

Cấu hình đề xuất:

- Vocab size: 16k hoặc 32k.
- Special tokens: `<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`.
- Huấn luyện tokenizer trên cả source và target để dùng chung vocabulary.

Lý do dùng tokenizer chung: bài toán tóm tắt có đầu vào và đầu ra cùng ngôn ngữ, nên source và target chia sẻ nhiều từ/cụm từ. Dùng chung vocab giúp giảm số tham số và hỗ trợ copy các thuật ngữ, tên riêng từ văn bản gốc sang bản tóm tắt.

## 6. Mô hình baseline

Baseline là Transformer encoder-decoder tự cài đặt:

- Token embedding.
- Positional encoding.
- Multi-head self-attention.
- Cross-attention trong decoder.
- Feed-forward network.
- Residual connection.
- Layer normalization.
- Padding mask cho source.
- Causal mask cho target.
- Output projection sang vocabulary.

Cấu hình baseline nên vừa đủ để chạy ổn:

```text
d_model = 256 hoặc 512
encoder_layers = 4 hoặc 6
decoder_layers = 4 hoặc 6
n_heads = 8
d_ff = 4 * d_model
dropout = 0.1
label_smoothing = 0.1
max_source_len = 512
max_target_len = 128
```

Huấn luyện bằng teacher forcing: decoder nhận `target[:-1]`, mô hình dự đoán `target[1:]`. Loss chính là cross-entropy, bỏ qua `<PAD>`.

## 7. Các cải tiến đề xuất và lý do

| Cải tiến | Vì sao tốt hơn baseline | Cách chứng minh |
|---|---|---|
| Pre-Layer Normalization | Với Transformer sâu, đặt LayerNorm trước attention/FFN giúp gradient ổn định hơn, ít bị lỗi loss dao động hoặc không học. Repo `NLP/Problem 1` đã dùng hướng này và có thể tham khảo. | So sánh đường loss và ROUGE giữa Post-LN baseline và Pre-LN. |
| Weight tying | Dùng chung trọng số embedding đầu ra và output projection giúp giảm tham số. Với tóm tắt cùng ngôn ngữ, phân phối từ ở input/output gần nhau hơn dịch máy, nên cách này hợp lý. | Báo cáo số tham số giảm và ROUGE không giảm hoặc tăng nhẹ. |
| Shared source-target embeddings | Vì source và summary cùng ngôn ngữ, encoder và decoder có thể dùng chung embedding. Điều này giúp mô hình học biểu diễn từ nhất quán giữa văn bản gốc và bản tóm tắt. | So sánh số tham số, tốc độ hội tụ và ROUGE. |
| Label smoothing | Tóm tắt có nhiều cách diễn đạt đúng. Label smoothing tránh mô hình quá tự tin vào đúng một token tham chiếu, giúp tổng quát hóa tốt hơn. | So sánh validation loss, ROUGE và ví dụ sinh tóm tắt. |
| Beam search + length penalty | Greedy dễ chọn token cục bộ và sinh tóm tắt quá ngắn. Beam search giữ nhiều ứng viên, length penalty giúp cân bằng giữa câu quá ngắn và quá dài. | So sánh greedy, beam size 3/5 và các length penalty 0.6/0.8/1.0 bằng ROUGE-L. |
| Coverage penalty trong decoding | Tóm tắt dễ bỏ sót ý chính hoặc lặp ý. Coverage penalty khuyến khích decoder chú ý rộng hơn đến các phần quan trọng của source. | Đo ROUGE và đếm tỷ lệ lặp n-gram trong output. |
| No-repeat n-gram blocking | Mô hình seq2seq nhỏ dễ lặp cụm từ. Chặn lặp 3-gram hoặc 4-gram khi decoding giúp output tự nhiên hơn. | So sánh số n-gram bị lặp và đọc thủ công một số ví dụ. |
| Length-aware batching | Gom mẫu có độ dài gần nhau vào cùng batch làm giảm padding. Tóm tắt có source dài nên lợi ích rõ hơn bài toán câu ngắn. | Báo cáo tokens/second, thời gian/epoch, GPU memory. |
| Warmup + cosine/inverse-sqrt scheduler | Transformer nhạy với learning rate ban đầu. Warmup giúp tránh cập nhật quá mạnh lúc embedding và attention chưa ổn định. | So sánh loss vài epoch đầu với scheduler cố định. |
| Gradient clipping | Khi source dài, attention và decoder có thể gây gradient spike. Clipping giúp huấn luyện ổn định hơn. | Theo dõi gradient norm và số lần loss bị NaN/dao động. |
| Mixed precision | Không cải thiện chất lượng trực tiếp nhưng tăng tốc và giảm bộ nhớ, cho phép batch lớn hơn hoặc max length dài hơn. | Báo cáo tốc độ train và GPU memory. |

Các cải tiến nên thêm theo thứ tự: Pre-LN, label smoothing, weight tying/shared embedding, scheduler, beam search, no-repeat n-gram, coverage penalty. Không nên thêm tất cả cùng lúc ngay từ đầu vì sẽ khó giải thích cải tiến nào tạo ra hiệu quả.

## 7.1. Chứng minh thực nghiệm cho cải tiến

Hiện tại các cải tiến mới có cơ sở lý thuyết và lý do kỹ thuật, chưa có chứng minh thực nghiệm vì mô hình chưa được huấn luyện. Để phần cải tiến được coi là thuyết phục, cần chạy ablation study trên validation set.

Bảng thực nghiệm tối thiểu:

```text
Thí nghiệm                      ROUGE-1  ROUGE-2  ROUGE-L  Val loss  Ghi chú
Baseline Transformer            ...
+ Pre-LN                        ...
+ Label smoothing               ...
+ Weight tying/shared emb       ...
+ Beam search + length penalty  ...
+ No-repeat n-gram              ...
```

Quy tắc chứng minh:

- Mỗi lần chỉ thêm một nhóm cải tiến chính để biết cải tiến nào có tác dụng.
- Giữ cùng dữ liệu, tokenizer, số epoch hoặc số update để so sánh công bằng.
- Báo cáo cả metric và ví dụ sinh tóm tắt, vì ROUGE không bắt hết lỗi lặp, sai ý hoặc hallucination.
- Nếu cải tiến không tăng ROUGE nhưng giảm tham số, tăng tốc hoặc giảm lặp, vẫn có thể giữ nếu giải thích được lợi ích.

## 8. Huấn luyện

Chiến lược huấn luyện:

1. Chạy mô hình nhỏ để kiểm tra pipeline, ví dụ `d_model=256`, 2-4 layer, 1-2 epoch.
2. Kiểm tra overfit trên 100-500 mẫu. Nếu mô hình không thể overfit tập nhỏ thì có lỗi mask, shift target, tokenizer hoặc loss.
3. Chạy baseline đầy đủ.
4. Lưu checkpoint tốt nhất theo validation ROUGE-L hoặc validation loss.
5. Sau mỗi epoch, sinh thử 20-50 mẫu validation để kiểm tra lỗi lặp, quá ngắn, copy nguyên văn hoặc hallucination.
6. Chạy ablation cho từng cải tiến.

Thông số khởi đầu:

```text
optimizer = AdamW
learning_rate = 1e-4 đến 3e-4
betas = (0.9, 0.98)
eps = 1e-9
weight_decay = 1e-4
warmup_steps = 4000 hoặc 8000
grad_clip = 1.0
batch_size = tùy GPU, ưu tiên tính theo số token nếu có thể
epochs = 10 đến 20
```

## 9. Inference

Cần có hai chế độ:

- Greedy decoding: nhanh, dùng để debug.
- Beam search: dùng cho kết quả chính.

Ràng buộc decoding đề xuất:

- `min_decode_len`: tránh tóm tắt quá ngắn.
- `max_decode_len`: tránh sinh quá dài.
- `length_penalty`: cân bằng độ dài.
- `no_repeat_ngram_size`: giảm lặp.
- `coverage_penalty`: giảm bỏ sót ý.

Đầu ra nên lưu dạng JSONL:

```json
{"id": 0, "source": "...", "reference": "...", "prediction": "..."}
```

Dạng này tiện cho đánh giá, phân tích lỗi và đưa ví dụ vào báo cáo.

## 10. Đánh giá

Metric bắt buộc:

- ROUGE-1: đo trùng unigram, phản ánh mức giữ nội dung.
- ROUGE-2: đo trùng bigram, phản ánh cụm từ và độ mạch lạc cục bộ.
- ROUGE-L: dựa trên longest common subsequence, phản ánh cấu trúc câu/tóm tắt.

Metric cộng thêm:

- BLEU: tham khảo thêm nhưng không phải metric chính cho summarization.
- METEOR: tốt hơn BLEU ở mức đồng nghĩa/biến thể từ nếu thư viện hỗ trợ.
- BERTScore: đo tương đồng ngữ nghĩa tốt hơn ROUGE, dùng cho phân tích bổ sung.
- Compression ratio: độ dài summary/source.
- Repetition rate: tỷ lệ lặp n-gram trong bản tóm tắt.

Báo cáo nên có bảng:

```text
Model                          ROUGE-1  ROUGE-2  ROUGE-L  Params  Notes
Baseline Post-LN               ...
+ Pre-LN                       ...
+ Label smoothing              ...
+ Weight tying/shared emb      ...
+ Beam + length penalty        ...
+ No-repeat + coverage penalty ...
```

Vì test public sẽ được công bố sau, trong giai đoạn phát triển chỉ dùng validation để:

- chọn checkpoint tốt nhất;
- so sánh các cải tiến;
- kiểm tra lỗi sinh tóm tắt;
- khóa cấu hình cuối cùng.

Khi có public test, không chỉnh mô hình theo test. Chỉ chạy inference bằng checkpoint/cấu hình đã chọn, sau đó báo cáo ROUGE test.

## 10.1. Quy trình code local, train server

Vì dự định code trên máy local rồi train trên server thuê, cần thiết kế project sao cho dễ chuyển môi trường:

- Toàn bộ đường dẫn cấu hình qua `config.py` hoặc tham số dòng lệnh, không hard-code ổ đĩa local.
- Có `requirements.txt` ghi rõ phiên bản thư viện chính: `torch`, `numpy`, `tqdm`, thư viện đọc parquet, thư viện ROUGE.
- Có script chuẩn bị dữ liệu riêng: `prepare_data.py`.
- Có script train chạy được bằng command line, ví dụ `python trainer/train.py --config configs/base.yaml`.
- Có script resume từ checkpoint để tránh mất tiến trình khi server ngắt.
- Log kết quả vào file `.json` hoặc `.csv`, không chỉ in ra terminal.
- Lưu tokenizer, config và checkpoint cùng nhau để inference trên server/test public không lệch cấu hình.

Luồng làm việc đề xuất:

```text
Local:
1. Code module.
2. Chạy unit test nhỏ.
3. Overfit 100-500 mẫu để kiểm tra mô hình học được.
4. Commit hoặc nén project.

Server:
1. Tạo môi trường Python.
2. Cài requirements.
3. Copy dữ liệu, tokenizer hoặc train tokenizer lại.
4. Chạy train baseline.
5. Chạy ablation.
6. Lưu checkpoint, log, output validation.
7. Khi có public test, chạy inference và evaluate.
```

## 10.2. Quy ước comment khi code

Khi triển khai code, cần comment ngắn gọn và liên tục ở các điểm quan trọng để người đọc dễ theo dõi. Comment không cần dài, nhưng phải giải thích "vì sao làm vậy" ở những chỗ dễ nhầm.

Quy tắc comment:

- Comment trước mỗi bước chính trong pipeline: đọc dữ liệu, chọn cột, làm sạch, lọc, split, tokenize, cache.
- Comment tại các logic dễ sai: shift target, padding mask, causal mask, special tokens, giới hạn độ dài.
- Comment tại các cải tiến: Pre-LN, label smoothing, weight tying, beam search, length penalty, no-repeat n-gram.
- Không comment kiểu lặp lại tên biến, ví dụ không viết `# set source to source`.
- Mỗi comment nên ngắn, dễ hiểu, ưu tiên tiếng Việt.

Ví dụ comment tốt:

```python
# Chuẩn hóa Unicode để các dấu tiếng Việt có cùng biểu diễn.
text = unicodedata.normalize("NFKC", text)

# Bỏ mẫu có summary quá dài vì không còn đúng bản chất tóm tắt.
if target_len >= source_len * 0.9:
    return False
```

## 11. Phân tích lỗi cần làm

Sau khi có kết quả, chọn khoảng 10-20 mẫu để phân tích:

- Tóm tắt đúng ý chính.
- Tóm tắt bỏ sót ý.
- Tóm tắt lặp.
- Tóm tắt quá ngắn.
- Tóm tắt copy quá nhiều từ source.
- Tóm tắt sinh thông tin không có trong source.

Phần này quan trọng vì ROUGE không phản ánh hết chất lượng abstractive summarization. Nếu một cải tiến tăng ROUGE nhưng làm summary kém tự nhiên, cần nêu rõ.

## 12. Lộ trình thực hiện

### Giai đoạn 1: dựng pipeline

- Đọc parquet và xác định schema.
- Chuẩn hóa dữ liệu về `source`/`target`.
- Làm sạch, lọc, split dữ liệu.
- Huấn luyện tokenizer.
- Tokenize và cache.

### Giai đoạn 2: baseline Transformer

- Cài embedding, positional encoding, attention, encoder, decoder.
- Cài mask đúng cho padding và causal decoding.
- Cài loss với target shifting.
- Chạy overfit tập nhỏ.
- Chạy baseline trên toàn bộ train.

### Giai đoạn 3: cải tiến

- Thêm Pre-LN.
- Thêm label smoothing.
- Thêm weight tying/shared embedding.
- Thêm scheduler warmup.
- Thêm beam search + length penalty.
- Thêm no-repeat n-gram và coverage penalty.

### Giai đoạn 4: đánh giá và báo cáo

- Sinh output trên test.
- Tính ROUGE-1/2/L và metric bổ sung.
- Làm ablation study.
- Phân tích lỗi định tính.
- Viết phần giải thích vì sao từng cải tiến tốt hơn baseline.

## 13. Rủi ro và cách xử lý

- Dữ liệu quá dài làm hết GPU memory: giảm `max_source_len`, dùng bucket batching, gradient accumulation hoặc mixed precision.
- Mô hình sinh lặp: thêm no-repeat n-gram, coverage penalty, tăng dropout hoặc kiểm tra decoding.
- Mô hình tóm tắt quá ngắn: dùng `min_decode_len`, chỉnh length penalty, kiểm tra phân phối độ dài target.
- ROUGE thấp dù loss giảm: kiểm tra tokenizer decode, target shift, special tokens, beam search và độ dài output.
- Copy nguyên văn quá nhiều: giảm max output, dùng length penalty phù hợp, đánh giá compression ratio.

## 14. Kết luận hướng mô hình

Hướng nên chọn là Transformer encoder-decoder tự cài từ đầu, sử dụng tokenizer subword tự huấn luyện và pipeline train/evaluate tách rõ như repo `NLP/Problem 1`. Baseline cần đơn giản, đúng và đo được. Các cải tiến nên tập trung vào ba vấn đề thật của summarization: ổn định huấn luyện với văn bản dài, sinh tóm tắt đủ ý nhưng không lặp, và giảm tham số nhờ đặc thù source/target cùng ngôn ngữ. Mỗi cải tiến phải được chứng minh bằng ablation và giải thích bằng cả metric lẫn ví dụ đầu ra.

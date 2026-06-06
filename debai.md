1. Đề bài: Xây dựng hệ thống tóm tắt văn bản tự động. 
2. Mục tiêu: Các nhóm cần xây dựng một hệ thống tóm tắt văn bản tự động, trong đó hệ 
thống nhận đầu vào là một văn bản dài và sinh ra bản tóm tắt ngắn hơn nhưng vẫn giữ được các ý chính (Abstractive Summary). 
1. Yêu cầu cốt lõi:
2. Xây dựng và huấn luyện mô hình Transformer hoàn toàn từ đầu. 
4. Yêu cầu kỹ thuật: 
4.1. Xây dựng kiến trúc transformer từ đầu: 
• Các nhóm phải tự lập trình kiến trúc Transformer (bao gồm Encoder, Decoder, 
Multi-Head Attention, Positional Encoding, Feed-Forward Network, v.v.) bằng các 
framework cơ bản (ví dụ: Pytorch). Không được phép gọi các lớp Transformer có 
sẵn trong thư viện ngoài (ví dụ: nn.Transformer của PyTorch). 

4.3. Các cải tiến phương pháp: 
• Các yêu cầu tối thiểu là điều kiện đủ để các nhóm đạt điểm cơ bản. Ngoài ra các 
nhóm cấn áp dụng các phương pháp cải tiến để đạt điểm cao 
hơn. VD: các kiến trúc cải tiến của Transformer

5. Dữ liệu và Đánh giá:
   Đánh giá: Sử dụng các độ đo tiêu chuẩn cho bài toán tóm tắt văn bản. Tối thiểu 
phải có: ROUGE (ROUGE-1, ROUGE-2, ROUGE-L). Sử dụng thêm nhiều độ đo 
khác là điểm cộng.
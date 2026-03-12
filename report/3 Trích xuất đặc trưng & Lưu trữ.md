# Báo cáo Bước 3 – Trích xuất Đặc trưng & Lưu trữ
## Bài toán: Phân loại văn bản đa nhãn bài báo khoa học

---

## 1. Tổng quan

Bước này biến đổi văn bản đã làm sạch thành vector số – đầu vào cho mô hình phân loại. Hai phương pháp được sử dụng song song:

| Phương pháp | Loại | Thư viện |
|-------------|------|----------|
| **TF-IDF** | Truyền thống (Bag-of-Words) | scikit-learn `TfidfVectorizer` |
| **BERT Embeddings** | Hiện đại (Deep Learning) | HuggingFace `sentence-transformers` |

Toàn bộ logic nằm trong module `src/modules/feature_extraction.py`.

---

## 2. Phương pháp 1: TF-IDF

### 2.1. Cấu hình

| Tham số | Giá trị | Mô tả |
|---------|---------|-------|
| `max_features` | 50,000 | Số lượng từ/bigram tối đa |
| `ngram_range` | (1, 2) | Sử dụng unigram + bigram |
| `sublinear_tf` | True | Dùng $1 + \log(tf)$ thay vì $tf$ thô |
| `min_df` | 2 | Từ phải xuất hiện trong ≥ 2 tài liệu |
| `strip_accents` | unicode | Chuẩn hóa dấu |

### 2.2. Kết quả

| Chỉ số | Giá trị |
|--------|---------|
| Vocabulary size | **50,000** từ/bigram |
| X_train shape | **(16,777 × 50,000)** |
| X_val shape | **(4,195 × 50,000)** |
| Kiểu dữ liệu | Sparse matrix (CSR) |

### 2.3. Nhận xét

- Ma trận TF-IDF rất thưa (sparse) → sử dụng `scipy.sparse` để lưu trữ hiệu quả.
- Bigram giúp bắt được cụm từ quan trọng (ví dụ: "machine learning", "neural network").
- `sublinear_tf=True` giúp giảm ảnh hưởng của các từ xuất hiện quá nhiều lần.

---

## 3. Phương pháp 2: BERT Embeddings

### 3.1. Cấu hình

| Tham số | Giá trị | Mô tả |
|---------|---------|-------|
| Model | `all-MiniLM-L6-v2` | SentenceTransformer nhẹ, hiệu quả |
| Embedding dim | 384 | Số chiều vector đầu ra |
| Batch size | 64 | Số mẫu encode mỗi batch |

### 3.2. Kết quả

| Chỉ số | Giá trị |
|--------|---------|
| X_train shape | **(16,777 × 384)** |
| X_val shape | **(4,195 × 384)** |
| Kiểu dữ liệu | Dense matrix (numpy float32) |

### 3.3. Nhận xét

- BERT tạo ra vector đặc trưng dense 384 chiều, nhỏ gọn hơn TF-IDF (50,000 chiều) rất nhiều.
- Model `all-MiniLM-L6-v2` nắm bắt được ngữ nghĩa (semantic) của văn bản – không chỉ dựa trên tần suất từ.
- L2-norm của các vector ≈ 1.0 (đã chuẩn hóa sẵn).
- **Lưu ý quan trọng:** BERT chỉ dùng để **trích xuất đặc trưng** (embedding), KHÔNG dùng để dự đoán trực tiếp. Đầu ra được đưa vào các mô hình sklearn phân loại → đúng yêu cầu pipeline tách biệt.

---

## 4. So sánh hai phương pháp

| Tiêu chí | TF-IDF | BERT |
|----------|--------|------|
| Số chiều | 50,000 | 384 |
| Kiểu ma trận | Sparse | Dense |
| Bắt ngữ nghĩa | Không (BoW) | Có (contextual) |
| Tốc độ tạo | Rất nhanh (~vài giây) | Chậm (~5-15 phút) |
| Kích thước file | Nhỏ (sparse → .npz) | Trung bình (.npy) |
| Phù hợp | Baseline, từ khóa rõ | Văn bản phức tạp, ngữ cảnh |

---

## 5. File đầu ra

| File | Định dạng | Mô tả |
|------|-----------|-------|
| `src/features/tfidf_train.npz` | scipy sparse | Ma trận TF-IDF train |
| `src/features/tfidf_val.npz` | scipy sparse | Ma trận TF-IDF val |
| `src/features/bert_train.npy` | numpy array | BERT embeddings train |
| `src/features/bert_val.npy` | numpy array | BERT embeddings val |

Các file này có thể tải lại bằng `load_tfidf()` và `load_bert()` mà không cần chạy lại quá trình trích xuất.
# Báo cáo Bước 5 – Đánh giá Mô hình
## Bài toán: Phân loại văn bản đa nhãn bài báo khoa học

---

## 1. Tổng quan

Đánh giá 6 tổ hợp mô hình trên tập Validation (4,195 mẫu) bằng hai chỉ số chính:
- **Micro-F1**: Tính F1 trên toàn bộ nhãn gộp lại → ưu tiên nhãn có nhiều mẫu.
- **Macro-F1**: Trung bình F1 của từng nhãn → phản ánh hiệu quả trên nhãn thiểu số.

Module sử dụng: `src/modules/evaluation.py`.

---

## 2. Bảng xếp hạng tổng hợp

| # | Feature | Model | Micro-F1 | Macro-F1 |
|---|---------|-------|----------|----------|
| **1** | **TF-IDF** | **LinearSVC** | **0.8254** | **0.7071** |
| 2 | TF-IDF | NaiveBayes | 0.8193 | 0.7364 |
| 3 | BERT | LinearSVC | 0.8189 | 0.7131 |
| 4 | TF-IDF | LogisticRegression | 0.8168 | 0.5671 |
| 5 | BERT | LogisticRegression | 0.8149 | 0.6801 |
| 6 | BERT | NaiveBayes | 0.7831 | 0.6897 |

**Mô hình tốt nhất (Micro-F1): TF-IDF + LinearSVC = 0.8254**

---

## 3. Phân tích chi tiết – Mô hình tốt nhất (TF-IDF + LinearSVC)

| Nhãn | Precision | Recall | F1-score | Support |
|------|-----------|--------|----------|---------|
| Computer Science | 0.830 | 0.851 | 0.840 | 1,732 |
| Physics | 0.946 | 0.865 | 0.903 | 1,204 |
| Mathematics | 0.853 | 0.791 | 0.821 | 1,092 |
| Statistics | 0.786 | 0.740 | 0.762 | 1,040 |
| Quantitative Biology | 0.700 | 0.248 | 0.366 | 113 |
| Quantitative Finance | 0.962 | 0.385 | 0.550 | 65 |

**Nhận xét:**
- **Physics** đạt F1 cao nhất (0.903) – từ vựng chuyên ngành rõ ràng, mô hình dễ nhận diện.
- **Computer Science** (0.840) và **Mathematics** (0.821) có F1 tốt – đây là những nhãn có nhiều mẫu.
- **Quantitative Biology** (0.366) và **Quantitative Finance** (0.550) có F1 rất thấp, chủ yếu do Recall thấp (0.248 và 0.385) – mô hình bỏ sót nhiều mẫu thuộc nhãn thiểu số.
- Precision của QuantFin rất cao (0.962) → khi mô hình dự đoán nhãn này, gần như luôn đúng; nhưng nó rất "dè dặt" – ít khi dự đoán.

---

## 4. So sánh TF-IDF vs BERT

### 4.1. Micro-F1

| Thuật toán | TF-IDF | BERT | Chênh lệch |
|------------|--------|------|-------------|
| LogisticRegression | 0.8168 | 0.8149 | TF-IDF +0.0019 |
| NaiveBayes | 0.8193 | 0.7831 | TF-IDF +0.0362 |
| LinearSVC | **0.8254** | 0.8189 | TF-IDF +0.0065 |

### 4.2. Macro-F1

| Thuật toán | TF-IDF | BERT | Chênh lệch |
|------------|--------|------|-------------|
| LogisticRegression | 0.5671 | **0.6801** | BERT +0.1130 |
| NaiveBayes | **0.7364** | 0.6897 | TF-IDF +0.0467 |
| LinearSVC | 0.7071 | **0.7131** | BERT +0.0060 |

### 4.3. Nhận xét so sánh

- **Micro-F1:** TF-IDF tốt hơn BERT ở tất cả 3 thuật toán. Chênh lệch lớn nhất ở NaiveBayes (+0.036). Lý do: TF-IDF với 50,000 features bigram bắt được từ khóa chuyên ngành rất tốt, còn BERT (384 dims) nén thông tin nhiều hơn.

- **Macro-F1:** Kết quả pha trộn:
  - BERT tốt hơn ở LR (+0.113) và SVC (+0.006) → embedding ngữ nghĩa giúp nhận diện nhãn thiểu số tốt hơn.
  - TF-IDF+NB tốt hơn BERT+NB (+0.047) → ComplementNB được thiết kế đặc biệt cho dữ liệu mất cân bằng.

- **Kết luận:** Trong bài toán phân loại bài báo khoa học, **TF-IDF truyền thống vẫn rất cạnh tranh** với BERT embeddings. Điều này hợp lý vì:
  - Bài báo khoa học có từ vựng chuyên ngành rõ ràng → BoW/TF-IDF bắt được tốt.
  - BERT `all-MiniLM-L6-v2` là model nhẹ (6 layers, 384 dims) – các model lớn hơn có thể cho kết quả tốt hơn.
  - BERT nén ngữ nghĩa tốt hơn → lợi thế ở Macro-F1 (nhãn thiểu số).

---

## 5. Phân tích theo nhãn – So sánh TF-IDF|SVC vs BERT|SVC

| Nhãn | TF-IDF F1 | BERT F1 | Tốt hơn |
|------|-----------|---------|---------|
| Computer Science | 0.840 | 0.842 | BERT +0.002 |
| Physics | 0.903 | 0.894 | TF-IDF +0.009 |
| Mathematics | 0.821 | 0.801 | TF-IDF +0.020 |
| Statistics | 0.762 | 0.762 | Bằng nhau |
| Quantitative Biology | 0.366 | 0.366 | Bằng nhau |
| Quantitative Finance | 0.550 | 0.614 | BERT +0.064 |

**Nhận xét:**
- BERT nổi trội ở **Quantitative Finance** (+0.064) – nhãn thiểu số nhất (chỉ 65 mẫu val). Embedding ngữ nghĩa giúp nhận diện văn bản tài chính tốt hơn khi mẫu ít.
- TF-IDF nổi trội ở **Mathematics** (+0.020) – từ vựng toán học (equation, theorem, proof) rất đặc trưng, BoW bắt tốt.

---

## 6. Vấn đề và Hạn chế

### 6.1. Mất cân bằng nhãn (Label Imbalance)

Đây là vấn đề lớn nhất ảnh hưởng hiệu quả mô hình:
- **CS**: 1,732 mẫu val → F1 = 0.840
- **QuantFin**: 65 mẫu val → F1 = 0.550

Chênh lệch số mẫu 27 lần → chênh lệch F1 gần 0.3.

### 6.2. Gợi ý cải thiện

1. **Threshold tuning**: Điều chỉnh ngưỡng quyết định cho từng nhãn thay vì dùng 0.5 cố định.
2. **Class weighting**: Sử dụng `class_weight='balanced'` trong LR và SVC.
3. **Larger BERT model**: Thử `all-mpnet-base-v2` (768 dims) để tăng chất lượng embedding.
4. **Ensemble**: Kết hợp TF-IDF và BERT features.

---

## 7. Tổng kết

| Tiêu chí | Kết quả |
|----------|---------|
| Mô hình tốt nhất (Micro-F1) | TF-IDF + LinearSVC (**0.8254**) |
| Mô hình tốt nhất (Macro-F1) | TF-IDF + NaiveBayes (**0.7364**) |
| TF-IDF vs BERT | TF-IDF tốt hơn Micro-F1; BERT tốt hơn ở nhãn thiểu số |
| Nhãn khó nhất | Quantitative Biology (F1 ≈ 0.37) |
| Nhãn dễ nhất | Physics (F1 ≈ 0.90) |

Pipeline phân loại đa nhãn hoạt động hiệu quả với Micro-F1 > 0.82, đặc biệt tốt trên các nhãn có đủ mẫu. Vấn đề chính cần giải quyết là hiệu quả trên nhãn thiểu số.
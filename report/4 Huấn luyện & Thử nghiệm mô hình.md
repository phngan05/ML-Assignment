# Báo cáo Bước 4 – Huấn luyện & Thử nghiệm Mô hình
## Bài toán: Phân loại văn bản đa nhãn bài báo khoa học

---

## 1. Tổng quan

Bước này huấn luyện **6 tổ hợp mô hình** (3 thuật toán × 2 loại đặc trưng) sử dụng chiến lược `OneVsRestClassifier` để xử lý bài toán đa nhãn (multi-label). Toàn bộ logic nằm trong module `src/modules/models.py`.

**Pipeline:**
$$\text{Đặc trưng (TF-IDF / BERT)} \longrightarrow \text{OneVsRestClassifier} \longrightarrow \text{Dự đoán nhãn}$$

---

## 2. Chiến lược OneVsRestClassifier

- Mỗi nhãn (6 nhãn) được huấn luyện một classifier nhị phân riêng biệt.
- Phù hợp cho multi-label vì các nhãn có thể chồng chéo (1 bài thuộc nhiều nhãn).
- Sử dụng thư viện `sklearn.multiclass.OneVsRestClassifier`.

---

## 3. Các thuật toán phân loại

### 3.1. Logistic Regression

| Tham số | Giá trị |
|---------|---------|
| `C` | 1.0 |
| `max_iter` | 1,000 |
| `solver` | lbfgs |

- Mô hình tuyến tính đơn giản, hiệu quả cao cho dữ liệu văn bản.
- Hỗ trợ tốt cả sparse (TF-IDF) và dense (BERT).

### 3.2. Naive Bayes

**Cho TF-IDF:** ComplementNB (`alpha=0.1`)
- Biến thể cải tiến của MultinomialNB, hiệu quả hơn cho dữ liệu mất cân bằng nhãn.
- Yêu cầu giá trị không âm → phù hợp TF-IDF.

**Cho BERT:** GaussianNB
- Hỗ trợ giá trị âm → phù hợp BERT embeddings.
- Giả định phân phối Gaussian cho đặc trưng.

### 3.3. LinearSVC (Support Vector Machine)

| Tham số | Giá trị |
|---------|---------|
| `C` | 1.0 |
| `max_iter` | 2,000 |

- SVM tuyến tính, thường cho kết quả tốt nhất trên dữ liệu văn bản chiều cao.
- Tối ưu margin giữa các lớp.

---

## 4. Ma trận thí nghiệm

| # | Feature | Model | Ghi chú |
|---|---------|-------|---------|
| 1 | TF-IDF (50K dims) | LogisticRegression | Baseline |
| 2 | TF-IDF (50K dims) | ComplementNB | Tốt cho imbalanced |
| 3 | TF-IDF (50K dims) | LinearSVC | Best traditional |
| 4 | BERT (384 dims) | LogisticRegression | Semantic baseline |
| 5 | BERT (384 dims) | GaussianNB | Hỗ trợ giá trị âm |
| 6 | BERT (384 dims) | LinearSVC | Semantic + SVM |

---

## 5. Dynamic Config (Cấu hình linh hoạt)

Tham số mô hình có thể thay đổi tuỳ thí nghiệm:

```python
MODEL_CONFIG_TFIDF = dict(
    feature_type = "tfidf",   # → ComplementNB
    lr_C         = 1.0,       # Điều chỉnh regularization
    lr_max_iter  = 1000,
    nb_alpha     = 0.1,       # Làm trơn NB
    svc_C        = 1.0,
    svc_max_iter = 2000,
)
models = build_models(**MODEL_CONFIG_TFIDF)
```

---

## 6. Lưu trữ mô hình

Tất cả mô hình đã huấn luyện được lưu dưới dạng `.joblib`:

| File | Mô tả |
|------|-------|
| `src/models/tfidf_LogisticRegression.joblib` | LR trên TF-IDF |
| `src/models/tfidf_NaiveBayes.joblib` | ComplementNB trên TF-IDF |
| `src/models/tfidf_LinearSVC.joblib` | SVC trên TF-IDF |
| `src/models/bert_LogisticRegression.joblib` | LR trên BERT |
| `src/models/bert_NaiveBayes.joblib` | GaussianNB trên BERT |
| `src/models/bert_LinearSVC.joblib` | SVC trên BERT |

Có thể tải lại bằng `load_model("tfidf_LinearSVC.joblib")` mà không cần huấn luyện lại.
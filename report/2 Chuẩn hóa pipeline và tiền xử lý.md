# Báo cáo Bước 2 – Chuẩn hóa Pipeline và Tiền xử lý
## Bài toán: Phân loại văn bản đa nhãn bài báo khoa học

---

## 1. Tổng quan

Bước tiền xử lý chuyển đổi dữ liệu thô (TITLE + ABSTRACT) thành văn bản sạch sẵn sàng cho trích xuất đặc trưng. Toàn bộ logic được đóng gói trong module `src/modules/preprocessing.py` với **dynamic pipeline** – các tham số cấu hình linh hoạt có thể thay đổi tuỳ thí nghiệm.

---

## 2. Pipeline tiền xử lý

### 2.1. Ghép văn bản

- Ghép cột `TITLE` và `ABSTRACT` thành cột `text` duy nhất (nối bằng dấu cách).
- Mục đích: tận dụng thông tin từ cả tiêu đề và tóm tắt để tạo đặc trưng phong phú hơn.

### 2.2. Làm sạch văn bản (`clean_text`)

Cấu hình pipeline được sử dụng:

| Tham số | Giá trị | Mô tả |
|---------|---------|-------|
| `remove_stopwords` | `True` | Loại bỏ ~80 stopwords tiếng Anh |
| `remove_numbers` | `True` | Xóa tất cả chữ số |
| `remove_latex` | `True` | Xóa ký hiệu LaTeX ($...$, \command{...}) |
| `min_word_len` | `3` | Chỉ giữ từ có ≥ 3 ký tự |

**Trình tự xử lý:**
1. Chuyển về chữ thường
2. Xóa ký hiệu LaTeX (inline `$...$`, block `$$...$$`, lệnh `\command{...}`)
3. Xóa URL (http/https/www)
4. Xóa chữ số
5. Xóa ký tự đặc biệt (chỉ giữ a-z và khoảng trắng)
6. Tách từ (tokenization bằng whitespace)
7. Lọc: loại stopwords + loại từ ngắn (< 3 ký tự)

**Lưu ý:** Stopwords được định nghĩa thủ công (~80 từ) ngay trong module, không phụ thuộc thư viện NLTK bên ngoài.

### 2.3. Thống kê sau làm sạch

| Chỉ số | Giá trị |
|--------|---------|
| Số từ tối thiểu (sau cleaning) | **4** |
| Số từ tối đa | **256** |
| Số từ trung bình | **96.5** |
| Số từ trung vị | **95.0** |
| Số bài có < 10 từ sau cleaning | **5** bài |

**Nhận xét:** Quá trình làm sạch giữ lại trung bình ~96 từ/bài (từ ~148 từ ban đầu trong ABSTRACT), đảm bảo đủ thông tin cho mô hình. Chỉ có 5 bài rất ngắn (< 10 từ) – số lượng không đáng kể.

---

## 3. Chia dữ liệu Train / Validation

| Tập | Số mẫu | Tỉ lệ |
|-----|--------|-------|
| **Train** | 16,777 | 80.0% |
| **Validation** | 4,195 | 20.0% |
| **Tổng** | 20,972 | 100% |

**Phương pháp:** `train_test_split` với `random_state=42`, stratify theo số nhãn/bài (nhóm 0, 1, 2+ nhãn) để đảm bảo phân phối nhãn đồng đều giữa train và val.

### Phân phối nhãn sau chia:

| Nhãn | Train (%) | Val (%) | Chênh lệch |
|------|-----------|---------|-------------|
| Computer Science | ~41.0% | ~41.0% | < 0.5% |
| Physics | ~28.7% | ~28.7% | < 0.5% |
| Mathematics | ~26.8% | ~26.8% | < 0.5% |
| Statistics | ~24.8% | ~24.8% | < 0.5% |
| Quantitative Biology | ~2.8% | ~2.8% | < 0.5% |
| Quantitative Finance | ~1.2% | ~1.2% | < 0.5% |

**Nhận xét:** Phân phối nhãn giữa train và val rất đồng đều (chênh lệch < 0.5% cho mọi nhãn), chứng tỏ chiến lược stratify hoạt động hiệu quả.

---

## 4. File đầu ra

| File | Mô tả |
|------|-------|
| `src/data/train_split.csv` | Tập train (16,777 dòng) |
| `src/data/val_split.csv` | Tập validation (4,195 dòng) |

---

## 5. Tính linh hoạt (Dynamic Pipeline)

Module `preprocessing.py` cho phép thay đổi tham số pipeline mà không cần sửa code:

```python
# Ví dụ: Thử nghiệm giữ stopwords, bỏ filter số
PIPELINE_CONFIG = dict(
    remove_stopwords = False,  # Giữ stopwords
    remove_numbers   = False,  # Giữ số
    remove_latex     = True,
    min_word_len     = 2,      # Giảm ngưỡng từ tối thiểu
)
df_clean = combine_and_clean(df, **PIPELINE_CONFIG)
```

Đây là yêu cầu bắt buộc của đề bài: code phải viết theo kiểu có tham số cấu hình linh hoạt.
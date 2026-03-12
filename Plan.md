# 📋 Kế hoạch hoàn thiện Bài tập lớn Học Máy

## 📝 Yêu cầu đề bài (Tóm tắt)
- **Đề bài:** Phân loại văn bản đa nhãn bài báo khoa học.
- **Yêu cầu bắt buộc:**
    - Hiện thực ít nhất 2 phương pháp trích xuất đặc trưng: TF-IDF (Truyền thống) và BERT/Word2Vec (Hiện đại).
    - Lưu kết quả trích xuất embedding thành file `.npy`.
    - Pipeline linh hoạt, so sánh hiệu quả giữa truyền thống và hiện đại.
- **Sản phẩm nộp:** File Colab, báo cáo PDF, các file đặc trưng `.npy`, và cấu trúc thư mục chuẩn (`modules/`, `notebooks/`, ...).

---

## 🗺️ Lộ trình về đích (Updated)

Dựa trên tiến độ hiện tại (Đã xong phần thống kê cơ bản và độ dài văn bản), dưới đây là các bước tiếp theo cần thực hiện:

### Bước 1: Hoàn thiện EDA (Ưu tiên ngay)
Nhóm cần bổ sung các cell code sau vào `main.ipynb` để xong hoàn toàn phần EDA (40% số điểm):
- [ ] **Phân tích nhãn đa lớp:** Đếm số lượng bài báo thuộc 1 nhãn, 2 nhãn, 3 nhãn... (Check sự chồng lấn giữa các lĩnh vực).
- [ ] **Ma trận tương quan nhãn (Heatmap):** Sử dụng `sns.heatmap` trên ma trận tương quan của 6 cột nhãn.
- [ ] **So sánh đặc thù chuyên ngành:** Tính độ dài `ABSTRACT` trung bình theo từng nhãn (VD: Nhãn Physics có tóm tắt dài hơn Math không?).
- [ ] **WordCloud/Từ vựng:** Trực quan hóa các từ xuất hiện nhiều nhất trong dữ liệu (sử dụng thư viện `wordcloud`).

### Bước 2: Chuẩn hóa Pipeline Tiền xử lý
Sử dụng module `src/modules/preprocessing.py`:
- [ ] Ghép `TITLE` và `ABSTRACT` thành một cột văn bản tổng hợp duy nhất.
- [ ] Thực hiện làm sạch: chữ thường, xóa stopwords, xóa ký tự đặc biệt.
- [ ] **Chia dữ liệu:** Chia `train.csv` thành tập `train` và `val` theo tỷ lệ 80/20 (Sử dụng `train_test_split`).

### Bước 3: Trích xuất đặc trưng & Lưu trữ (Mấu chốt)
Sử dụng module `src/modules/feature_extraction.py`:
- [ ] **TF-IDF:** Tạo ma trận đặc trưng truyền thống.
- [ ] **BERT Embeddings:** Sử dụng `SentenceTransformer('all-MiniLM-L6-v2')` để lấy vector đặc trưng (Modern approach).
- [ ] **Lưu file:** Xuất các ma trận đặc trưng này ra thư mục `src/features/` dưới định dạng `.npy` (Ví dụ: `bert_train.npy`).

### Bước 4: Huấn luyện & Thử nghiệm mô hình
Sử dụng module `src/modules/models.py`:
- [ ] Xây dựng mô hình với chiến lược `OneVsRestClassifier`.
- [ ] Thử nghiệm đồng thời 3 thuật toán: `Logistic Regression`, `Naive Bayes`, và `LinearSVC`.
- [ ] Chạy so sánh trên cả 2 loại đầu vào: (TF-IDF vs BERT).

### Bước 5: Đánh giá & Báo cáo PDF
Sử dụng module `src/modules/evaluation.py`:
- [ ] Xuất `classification_report` chi tiết cho từng nhãn.
- [ ] Tính toán `Micro-F1` và `Macro-F1`.
- [ ] **Viết báo cáo:** Tổng hợp kết quả, nhận xét về việc BERT có thực sự tốt hơn TF-IDF trong bài toán này không.

### Bước 6: Đóng gói & Nộp bài
- [ ] Kiểm tra file `main.ipynb` đảm bảo chọn "Run all" chạy thành công không lỗi.
- [ ] Cập nhật file `README.md` trong `src/` với đầy đủ thông tin nhóm và hướng dẫn chạy.
- [ ] Nén thư mục `src/` thành `.zip` để nộp.

> [!tip] Mẹo đạt điểm cộng
> - Hãy tập trung vào việc **giải thích tại sao** mô hình này tốt hơn mô hình kia trong báo cáo.
> - Đảm bảo GitHub của nhóm có lịch sử commit đều đặn từ các thành viên.

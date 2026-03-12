# Báo cáo Phân tích Khám phá Dữ liệu (EDA)
## Bài toán: Phân loại văn bản đa nhãn bài báo khoa học

---

## 1. Tổng quan tập dữ liệu

| Thông tin | Giá trị |
|-----------|---------|
| Số mẫu (bài báo) | **20,972** |
| Số cột | **9** (ID, TITLE, ABSTRACT + 6 cột nhãn) |
| Dữ liệu thiếu (null) | **0** – tập dữ liệu sạch hoàn toàn |
| Dữ liệu trùng lặp | **0** – không có dòng nào trùng TITLE+ABSTRACT |

**Cấu trúc cột:**
- `ID`: định danh số nguyên
- `TITLE`: tiêu đề bài báo (chuỗi văn bản)
- `ABSTRACT`: tóm tắt bài báo (chuỗi văn bản)
- 6 cột nhãn nhị phân: `Computer Science`, `Physics`, `Mathematics`, `Statistics`, `Quantitative Biology`, `Quantitative Finance`

**Nhận xét:** Tập dữ liệu có chất lượng tốt – không thiếu dữ liệu, không trùng lặp. Không cần bước làm sạch sơ bộ.

---

## 2. Phân phối nhãn

### 2.1 Số bài báo theo từng nhãn

| Nhãn | Số bài | Tỉ lệ |
|------|--------|-------|
| Computer Science | 8,594 | 41.0% |
| Physics | 6,013 | 28.7% |
| Mathematics | 5,618 | 26.8% |
| Statistics | 5,206 | 24.8% |
| Quantitative Biology | 587 | 2.8% |
| Quantitative Finance | 249 | 1.2% |

**Nhận xét:**
- Tập dữ liệu **mất cân bằng nhãn** rõ rệt: Computer Science chiếm 41%, trong khi Quantitative Finance chỉ chiếm 1.2% (chênh lệch ~34 lần).
- Nhóm `Quantitative Biology` và `Quantitative Finance` là thiểu số – cần lưu ý khi huấn luyện mô hình (cân nhắc class weighting hoặc oversampling).

### 2.2 Phân tích đa nhãn (Multi-label)

| Số nhãn/bài | Số bài | Tỉ lệ |
|-------------|--------|-------|
| 1 nhãn | 15,928 | **75.9%** |
| 2 nhãn | 4,793 | **22.9%** |
| 3 nhãn | 251 | **1.2%** |

- **Tổng nhãn được gán:** 26,267 (trung bình 1.25 nhãn/bài)
- **Không có bài nào thuộc 0 nhãn** – tập dữ liệu được gán nhãn đầy đủ
- Phần lớn bài báo chỉ thuộc 1 lĩnh vực, nhưng ~23% có sự chồng lấn

---

## 3. Ma trận tương quan giữa các nhãn

| Cặp nhãn | Tương quan | Chiều |
|----------|------------|-------|
| Computer Science – Physics | **-0.423** | Ngược chiều (mạnh nhất) |
| Physics – Statistics | **-0.329** | Ngược chiều |
| Computer Science – Mathematics | **-0.311** | Ngược chiều |
| Physics – Mathematics | **-0.307** | Ngược chiều |
| Computer Science – Statistics | **+0.084** | Cùng chiều (duy nhất dương) |
| Các cặp còn lại | -0.008 đến -0.121 | Ngược chiều yếu |

**Nhận xét:**
- **Computer Science và Physics** ít khi xuất hiện cùng nhau nhất (r = -0.42) – hai lĩnh vực này có từ vựng và hướng nghiên cứu khá tách biệt.
- **Computer Science và Statistics** là cặp duy nhất có xu hướng đồng xuất hiện (r = +0.08) – phù hợp thực tế: nhiều bài học máy/thống kê có thể được gắn cả hai nhãn.
- Tương quan tổng thể thấp → các nhãn tương đối độc lập nhau, phù hợp với chiến lược phân loại **One-vs-Rest (OvR)**.

---

## 4. Phân tích độ dài văn bản

### 4.1 Thống kê mô tả

| Cột | Min | Max | Mean | Median | Std |
|-----|-----|-----|------|--------|-----|
| TITLE – số từ | 1 | 40 | 9.5 | 9.0 | 3.6 |
| TITLE – số ký tự | 7 | 239 | 72.9 | 71.0 | 26.1 |
| ABSTRACT – số từ | 1 | 449 | 148.4 | 145.0 | 60.8 |
| ABSTRACT – số ký tự | 7 | 2,761 | 1,009.1 | 989.0 | 408.6 |

**Các ngoại lai đáng chú ý:**
- Tiêu đề ngắn nhất: 1 từ ("Gamorithm" – ID 704)
- Tóm tắt ngắn nhất: 1 từ ("Yes." – ID 16395) → **ngoại lai bất thường**
- Tóm tắt dài nhất: 449 từ (ID 19446) → vẫn trong phạm vi hợp lý

### 4.2 Phát hiện ngoại lai (IQR × 1.5)

Histogram + KDE cho thấy phân phối độ dài **lệch phải (right-skewed)**:
- Phần lớn bài báo có tiêu đề 6–13 từ và tóm tắt 80–220 từ
- Tồn tại một số bài có abstract rất ngắn (<20 từ) hoặc rất dài (>350 từ)
- Boxplot xác nhận: ABSTRACT có nhiều giá trị ngoại lai hơn TITLE

**Đề xuất tiền xử lý:** Cân nhắc lọc hoặc đánh dấu các bài có abstract < 20 từ trước khi trích xuất đặc trưng.

---

## 5. So sánh độ dài trung bình theo chuyên ngành

Bài báo thuộc các nhãn khác nhau có sự khác biệt về độ dài tóm tắt:

- **Quantitative Finance** và **Quantitative Biology** có abstract trung bình dài hơn các nhãn chính (CS, Physics, Math) – có thể do các bài thuộc nhóm này thường cần giải thích bối cảnh ứng dụng nhiều hơn.
- **Computer Science** có xu hướng tiêu đề dài hơn (mô tả kỹ thuật cụ thể) so với **Mathematics** (tiêu đề ngắn gọn, trừu tượng hơn).
- Sự khác biệt về độ dài văn bản giữa các chuyên ngành là một **đặc trưng tiềm năng** có thể hỗ trợ mô hình phân loại.

---

## 6. Phân tích từ vựng (WordCloud)

### 6.1 Từ khóa nổi bật toàn tập
Các từ xuất hiện nhiều nhất trong toàn bộ ABSTRACT: **system, algorithm, time, network, learning, function, models, analysis, space, structure**.

### 6.2 Từ khóa đặc trưng từng chuyên ngành

| Chuyên ngành | Từ khóa đặc trưng |
|---|---|
| **Computer Science** | algorithm, network, learning, system, task, performance, graph, neural |
| **Physics** | system, field, energy, quantum, state, electron, temperature, magnetic |
| **Mathematics** | equation, space, operator, graph, proof, linear, algebra, distribution |
| **Statistics** | learning, distribution, models, algorithm, estimation, regression, Bayesian |
| **Quantitative Biology** | network, cell, gene, population, protein, brain, species, biological |
| **Quantitative Finance** | price, market, risk, stock, financial, asset, volatility, option |

**Nhận xét:**
- Từ vựng giữa **Computer Science** và **Statistics** có sự chồng lấn đáng kể (`learning`, `algorithm`, `network`) → nhất quán với tương quan dương +0.084 đã tìm thấy.
- **Physics** và **Mathematics** chia sẻ các từ toán học (`space`, `equation`, `function`) nhưng Physics có thêm từ ngữ thực nghiệm (`electron`, `magnetic`, `temperature`).
- **Quantitative Finance** có từ vựng rất riêng biệt (`price`, `market`, `risk`, `stock`) → có thể dễ phân loại nhất dù ít mẫu nhất.
- **Quantitative Biology** cũng có từ đặc trưng rõ (`cell`, `gene`, `protein`, `brain`) nhưng chia sẻ `network`, `learning` với CS/Stats.

---

## 7. Tổng kết & Hàm ý cho các bước tiếp theo

### Những gì đã biết từ EDA:
1. **Dữ liệu sạch** – không cần xử lý null/duplicate
2. **Mất cân bằng nhãn nghiêm trọng** – cần chiến lược xử lý trong huấn luyện
3. **Nhãn tương đối độc lập** – OvR là chiến lược phù hợp
4. **Từ vựng đặc trưng rõ theo nhãn** – TF-IDF sẽ hoạt động tốt; BERT có thể bổ sung ngữ cảnh sâu hơn cho CS/Stats (nhãn hay bị nhầm lẫn)
5. **Tồn tại ngoại lai độ dài** – cần lọc/xử lý trước khi trích xuất đặc trưng

### Đề xuất cho Bước 2 (Tiền xử lý):
- Ghép `TITLE` + `ABSTRACT` thành một trường văn bản duy nhất
- Làm sạch: lowercase, loại stopwords, loại ký tự đặc biệt/LaTeX
- Cân nhắc loại bỏ các bài có abstract < 20 từ trước khi split train/val
- Chia 80/20 stratified theo phân phối nhãn để giữ cân bằng tương đối

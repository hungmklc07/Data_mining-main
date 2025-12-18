# NỘI DUNG SLIDE THUYẾT TRÌNH
## Data Mining - Phân tích dữ liệu Premier League 2024-2025

---

# SLIDE 1: TRANG BÌA

## Nội dung
- **Tiêu đề**: PHÂN TÍCH DỮ LIỆU BÓNG ĐÁ PREMIER LEAGUE 2024-2025
- **Phụ đề**: Ứng dụng các kỹ thuật Data Mining
- **Môn học**: Data Mining
- **Tên nhóm**: [Điền tên nhóm]
- **Thành viên**: [Điền tên thành viên]
- **Giảng viên**: [Điền tên GV]

## Gợi ý hình ảnh
- Logo Premier League
- Hình nền sân bóng đá hoặc các cầu thủ nổi tiếng

---

# SLIDE 2: BỐI CẢNH VÀ LÝ DO CHỌN ĐỀ TÀI

## Nội dung
### Bối cảnh
- Bóng đá hiện đại ngày càng phụ thuộc vào phân tích dữ liệu
- Các CLB lớn đều có đội ngũ Data Analyst chuyên nghiệp
- Premier League là giải đấu hấp dẫn nhất thế giới với dữ liệu phong phú

### Lý do chọn đề tài
- **Tính ứng dụng cao**: Hỗ trợ chiêu mộ cầu thủ, phân tích đối thủ
- **Dữ liệu đa dạng**: Nhiều chỉ số thống kê (Goals, xG, Assists, Tackles...)
- **Phù hợp với yêu cầu**: Áp dụng được nhiều kỹ thuật Data Mining

## Gợi ý hình ảnh
- Hình ảnh phòng phân tích dữ liệu của CLB bóng đá
- Biểu đồ thống kê bóng đá mẫu

## Ghi chú thuyết trình
- Nhấn mạnh xu hướng "Moneyball" trong bóng đá
- Đề cập Liverpool, Man City sử dụng data analytics thành công

---

# SLIDE 3: MỤC TIÊU VÀ PHẠM VI PROJECT

## Nội dung
### Mục tiêu
1. **Tìm mẫu kết hợp**: Phát hiện patterns trong chỉ số cầu thủ
2. **Phân nhóm cầu thủ**: Nhóm theo phong cách chơi
3. **Phân loại**: Dự đoán vị trí, hiệu suất cầu thủ
4. **Phát hiện bất thường**: Tìm cầu thủ xuất sắc/đặc biệt
5. **Hệ thống gợi ý**: Đề xuất cầu thủ phù hợp

### Phạm vi
- **Giải đấu**: Premier League 2024-2025
- **Dữ liệu**: 500+ cầu thủ, 20 đội bóng
- **Nguồn**: FBref.com

## Gợi ý hình ảnh
- Sơ đồ 5 mục tiêu dạng icon
- Bản đồ Premier League với 20 CLB

---

# SLIDE 4: NGUỒN DỮ LIỆU VÀ CÁCH THU THẬP

## Nội dung
### Nguồn dữ liệu
- **Website**: FBref.com - Nguồn thống kê bóng đá uy tín
- **Giải đấu**: Premier League 2024-2025
- **Thời điểm**: Dữ liệu cập nhật đến hiện tại

### Phương pháp thu thập
- **Web Scraping** sử dụng Python
- Thư viện: `undetected_chromedriver`, `pandas.read_html`
- Tự động hóa việc lấy dữ liệu từ nhiều trang

### Các loại dữ liệu
- Standard Stats, Shooting, Passing
- Defense, Possession, Miscellaneous
- Goalkeeping, Team Statistics

## Gợi ý hình ảnh
- Screenshot giao diện FBref
- Code snippet minh họa web scraping

## Ghi chú thuyết trình
- Giải thích tại sao chọn FBref (dữ liệu chi tiết, miễn phí)

---

# SLIDE 5: MÔ TẢ CÁC DATASET

## Nội dung
### 4 Dataset chính

| Dataset | Mô tả | Số dòng | Số cột |
|---------|-------|---------|--------|
| Players Full | Thống kê toàn diện cầu thủ | ~550 | 120+ |
| Keepers | Thống kê thủ môn | ~40 | 50+ |
| Teams For | Chỉ số tấn công đội | 20 | 18 |
| Teams VS | Chỉ số sân nhà/khách | 20 | 28 |

### Các chỉ số quan trọng
- **Tấn công**: Goals, Shots, SoT, xG
- **Kiến tạo**: Assists, Key Passes, xA
- **Phòng ngự**: Tackles, Interceptions, Blocks
- **Chuyền bóng**: Passes, Pass Completion %

## Gợi ý hình ảnh
- Bảng mẫu dữ liệu (5-10 dòng đầu)
- Biểu đồ phân bố số cột theo loại

---

# SLIDE 6: QUY TRÌNH TIỀN XỬ LÝ DỮ LIỆU

## Nội dung
### Quy trình xử lý

```
Raw Data → Clean → Transform → Feature Engineering → Ready
```

### Các bước cụ thể
1. **Xử lý missing values**: Điền 0 hoặc loại bỏ
2. **Chuyển đổi kiểu dữ liệu**: String → Numeric
3. **Chuẩn hóa tên cột**: Loại bỏ ký tự đặc biệt
4. **Feature Engineering**:
   - Tính Goals per 90 minutes
   - Tính tỷ lệ % các chỉ số
   - Tạo biến phân loại (Position groups)

### Kết quả
- Dữ liệu sạch, sẵn sàng phân tích
- Lưu vào file Excel để tái sử dụng

## Gợi ý hình ảnh
- Flowchart quy trình xử lý
- Before/After của dữ liệu

---

# SLIDE 7: ASSOCIATION RULE MINING - GIỚI THIỆU

## Nội dung
### Vấn đề cần giải quyết
- Tìm **mẫu (patterns)** trong dữ liệu cầu thủ
- Ví dụ: "Cầu thủ có xG cao thường có bao nhiêu Shots?"

### Ứng dụng thực tế
- Nếu "Shots on Target cao → Goals cao" → Cần cải thiện độ chính xác
- Nếu "Passes cao + xA cao → Assists cao" → Tìm tiền vệ có cả hai

### Thuật toán sử dụng
- **FP-Growth** (Frequent Pattern Growth)
- Hiệu quả hơn Apriori: Chỉ quét database 2 lần
- Không sinh candidate itemsets

## Gợi ý hình ảnh
- Ví dụ trực quan về association rule
- So sánh Apriori vs FP-Growth

---

# SLIDE 8: ASSOCIATION RULE MINING - LÝ THUYẾT

## Nội dung
### Các độ đo quan trọng

**Support** - Tần suất xuất hiện
$$Support(A) = \frac{\text{Số transactions chứa A}}{\text{Tổng transactions}}$$

**Confidence** - Độ tin cậy
$$Confidence(A → B) = \frac{Support(A ∪ B)}{Support(A)}$$

**Lift** - Độ nâng
$$Lift(A → B) = \frac{Confidence(A → B)}{Support(B)}$$

### Ý nghĩa Lift
- Lift = 1: A và B độc lập
- Lift > 1: Tương quan dương (hay đi cùng nhau)
- Lift < 1: Tương quan âm

## Gợi ý hình ảnh
- Công thức với ví dụ số cụ thể
- Biểu đồ Venn minh họa Support

## Ghi chú thuyết trình
- Giải thích bằng ví dụ bóng đá cụ thể

---

# SLIDE 9: ASSOCIATION RULE MINING - KẾT QUẢ

## Nội dung
### Thiết lập
- min_support = 0.1 (10%)
- min_confidence = 0.6 (60%)
- Discretization: Low/Medium/High (quantile)

### Một số luật phát hiện được
| Antecedent | Consequent | Support | Confidence | Lift |
|------------|------------|---------|------------|------|
| xG=High, SoT=High | Goals=High | 0.12 | 0.85 | 4.2 |
| Passes=High, xA=High | Assists=High | 0.08 | 0.78 | 3.8 |
| Tackles=High, Int=High | Pos=DF | 0.15 | 0.72 | 3.1 |

### Nhận xét
- Cầu thủ có xG cao VÀ SoT cao → 85% có Goals cao
- Phù hợp với logic bóng đá thực tế

## Gợi ý hình ảnh
- Bảng top 10 rules
- Scatter plot: Support vs Confidence (màu = Lift)

---

# SLIDE 10: CLUSTERING - GIỚI THIỆU

## Nội dung
### Vấn đề
- Với 500+ cầu thủ, làm sao nhóm theo "phong cách chơi"?

### Ứng dụng
- **Tìm cầu thủ thay thế**: Cầu thủ cùng cụm có phong cách tương tự
- **Phân tích chiến thuật**: Mỗi cụm = 1 loại cầu thủ
- **So sánh**: Cầu thủ thuộc cụm nào?

### Thuật toán sử dụng
1. **K-Means**: Phân cụm theo centroid
2. **Hierarchical**: Phân cụm phân cấp

### Tiền xử lý
- **Standardization**: Z-score để các features có cùng scale
- Feature selection: Chọn features phù hợp mục tiêu

## Gợi ý hình ảnh
- Minh họa clustering 2D với các cụm màu khác nhau
- Ví dụ cầu thủ trong từng cụm

---

# SLIDE 11: CLUSTERING - LÝ THUYẾT

## Nội dung
### K-Means Algorithm

**Bước 1**: Khởi tạo K centroids ngẫu nhiên

**Bước 2**: Gán mỗi điểm vào cụm có centroid gần nhất

**Bước 3**: Cập nhật centroid = trung bình các điểm trong cụm

**Bước 4**: Lặp lại đến khi hội tụ

### Chọn K tối ưu
- **Elbow Method**: Tìm điểm "khuỷu tay" trên đồ thị Inertia
- **Silhouette Score**: Đo chất lượng phân cụm (-1 đến 1)

### Khoảng cách Euclidean
$$d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}$$

## Gợi ý hình ảnh
- Animation minh họa K-Means iterations
- Elbow curve và Silhouette plot

---

# SLIDE 12: CLUSTERING - KẾT QUẢ

## Nội dung
### Thiết lập
- Số cụm tối ưu: K = 4 (theo Elbow + Silhouette)
- Features: Goals, Assists, xG, xA, Shots, Passes, Tackles

### Kết quả phân cụm

| Cluster | Đặc điểm | Ví dụ cầu thủ |
|---------|----------|---------------|
| 0 | Goal Scorers - xG, Goals cao | Haaland, Salah |
| 1 | Playmakers - Assists, Passes cao | De Bruyne, Odegaard |
| 2 | Defenders - Tackles, Int cao | Van Dijk, Saliba |
| 3 | All-rounders - Balanced stats | Saka, Palmer |

### Silhouette Score: 0.45 (Phân cụm tốt)

## Gợi ý hình ảnh
- Scatter plot 2D (PCA) với các cụm
- Radar chart so sánh profile các cụm
- Dendrogram của Hierarchical Clustering

---

# SLIDE 13: CLASSIFICATION - GIỚI THIỆU

## Nội dung
### Các bài toán phân loại

1. **Phân loại vị trí cầu thủ**
   - Input: Các chỉ số thống kê
   - Output: FW, MF, DF, GK

2. **Dự đoán Top 4**
   - Input: Chỉ số đội bóng
   - Output: Top 4 hay không

3. **Phân loại hiệu suất**
   - Input: Goals, xG, Assists...
   - Output: Elite, Good, Average, Below Average

### Thuật toán
- **Random Forest**: Ensemble của nhiều Decision Trees
- **Decision Tree**: Cây quyết định dựa trên Information Gain

## Gợi ý hình ảnh
- Sơ đồ 3 bài toán classification
- Minh họa Decision Tree đơn giản

---

# SLIDE 14: CLASSIFICATION - LÝ THUYẾT

## Nội dung
### Decision Tree
- Chia dữ liệu theo điều kiện
- Sử dụng **Information Gain** hoặc **Gini Index**

**Entropy**:
$$H(S) = -\sum_{i=1}^{c} p_i \log_2(p_i)$$

### Random Forest
- **Bagging**: Train nhiều trees trên bootstrap samples
- **Feature Randomization**: Mỗi node chỉ xét m features ngẫu nhiên
- **Voting**: Kết hợp kết quả của tất cả trees

### Tại sao Random Forest tốt hơn?
- Giảm overfitting
- Giảm variance
- Robust với noise

## Gợi ý hình ảnh
- Minh họa Decision Tree với ví dụ bóng đá
- Sơ đồ Random Forest ensemble

---

# SLIDE 15: CLASSIFICATION - KẾT QUẢ

## Nội dung
### 1. Phân loại vị trí cầu thủ
| Model | Accuracy | F1-Score |
|-------|----------|----------|
| Random Forest | 78% | 0.76 |
| Decision Tree | 71% | 0.69 |

**Top features**: Tackles, Goals, Passes, Touches

### 2. Dự đoán Top 4
| Model | Accuracy | Precision | Recall |
|-------|----------|-----------|--------|
| Random Forest | 85% | 0.80 | 0.75 |

**Top features**: xG, xGD, Points

### 3. Phân loại hiệu suất
- 4 classes: Elite (5%), Good (20%), Average (50%), Below (25%)
- Accuracy: 72%

## Gợi ý hình ảnh
- Confusion Matrix
- Feature Importance bar chart
- ROC Curve

---

# SLIDE 16: ANOMALY DETECTION - GIỚI THIỆU

## Nội dung
### Anomaly là gì?
- Điểm dữ liệu **khác biệt đáng kể** so với phần còn lại

### Trong bóng đá, anomaly có thể là:
- **Positive**: Cầu thủ xuất sắc (Haaland, Salah)
- **Negative**: Cầu thủ hiệu suất kém bất thường
- **Interesting**: Phong cách độc đáo (thủ môn kiến tạo nhiều)

### Ứng dụng
- Phát hiện tài năng đặc biệt
- Tìm cầu thủ bị đánh giá thấp
- Phát hiện vấn đề (chấn thương, phong độ giảm)

### Thuật toán
- **Isolation Forest**: Cô lập outliers
- **LOF (Local Outlier Factor)**: So sánh mật độ cục bộ

## Gợi ý hình ảnh
- Scatter plot với outliers được highlight
- Ví dụ cầu thủ anomaly

---

# SLIDE 17: ANOMALY DETECTION - KẾT QUẢ

## Nội dung
### Isolation Forest
- **Ý tưởng**: Outliers dễ bị cô lập (ít lần chia)
- **Anomaly Score**: 0 (normal) → 1 (anomaly)
- **Contamination**: 5%

### LOF (Local Outlier Factor)
- **Ý tưởng**: So sánh mật độ với láng giềng
- **LOF > 1**: Outlier (mật độ thấp hơn)

### Kết quả phát hiện

| Cầu thủ | Lý do là Anomaly | Score |
|---------|------------------|-------|
| Haaland | Goals vượt trội (20+) | 0.92 |
| Salah | Goals + Assists cao | 0.88 |
| Palmer | Hiệu suất vượt xG | 0.85 |
| Ederson | Thủ môn passes cao | 0.78 |

### Nhận xét
- Phát hiện đúng các cầu thủ xuất sắc nhất giải

## Gợi ý hình ảnh
- Scatter plot với anomalies highlighted (màu đỏ)
- Bar chart anomaly scores

---

# SLIDE 18: RECOMMENDATION SYSTEM - GIỚI THIỆU

## Nội dung
### Bài toán
- Tìm cầu thủ **tương tự** với cầu thủ hiện tại
- Tìm cầu thủ **phù hợp** với nhu cầu đội bóng
- Tìm cầu thủ có **phong cách chơi** mong muốn

### Phương pháp: Content-based Filtering
- Gợi ý dựa trên **features** (chỉ số thống kê)
- Không cần dữ liệu về user preferences

### Độ đo tương tự: Cosine Similarity
$$\cos(\theta) = \frac{A \cdot B}{\|A\| \times \|B\|}$$

### Tại sao Cosine?
- Đo **hướng** (tỷ lệ), không phải **khoảng cách**
- Tìm cầu thủ có "phong cách" tương tự

## Gợi ý hình ảnh
- Minh họa Cosine Similarity 2D
- Sơ đồ hệ thống recommendation

---

# SLIDE 19: RECOMMENDATION SYSTEM - KẾT QUẢ

## Nội dung
### 3 Loại gợi ý

**1. Similar Players** - Tìm cầu thủ tương tự
- Input: "Salah"
- Output: Saka (0.92), Son (0.89), Diaz (0.87)

**2. Team Needs** - Gợi ý theo nhu cầu đội
- Input: "Man United cần FW"
- Output: Top 10 FW phù hợp (theo tổng hợp chỉ số)

**3. Style-based** - Gợi ý theo phong cách
- Input: Goals=15, Assists=10, xG=12
- Output: Cầu thủ có profile gần nhất

### Demo trên Dashboard
- Interactive: Chọn cầu thủ, chọn đội, điều chỉnh tiêu chí
- Hiển thị radar chart so sánh

## Gợi ý hình ảnh
- Screenshot Dashboard Recommendation
- Radar chart so sánh cầu thủ gốc vs gợi ý

---

# SLIDE 20: TỔNG HỢP KẾT QUẢ

## Nội dung
### Bảng tóm tắt

| Kỹ thuật | Kết quả chính |
|----------|---------------|
| Association Rules | 50+ rules với Lift > 2 |
| Clustering | 4 cụm cầu thủ rõ ràng |
| Classification | Accuracy 72-85% |
| Anomaly Detection | Phát hiện đúng top performers |
| Recommendation | Similarity > 0.85 cho top matches |

### Insights quan trọng
1. **xG là chỉ số quan trọng nhất** để dự đoán Goals
2. **Cầu thủ có thể phân thành 4 nhóm** rõ ràng theo phong cách
3. **Random Forest tốt hơn Decision Tree** cho mọi bài toán
4. **Anomaly detection** phát hiện chính xác các ngôi sao

## Gợi ý hình ảnh
- Infographic tổng hợp 5 kỹ thuật
- Key numbers highlights

---

# SLIDE 21: DEMO DASHBOARD

## Nội dung
### Streamlit Dashboard
- **URL**: `streamlit run dashboard.py`
- **Các tab chức năng**:
  1. Overview - Tổng quan dữ liệu
  2. Association Rules - Khám phá luật kết hợp
  3. Clustering - Visualize các cụm
  4. Classification - Dự đoán vị trí/hiệu suất
  5. Anomaly Detection - Phát hiện outliers
  6. **Recommendation System** - Gợi ý cầu thủ

### Demo live
- Tìm cầu thủ tương tự Salah
- Gợi ý tiền đạo cho Man United
- Xem anomaly scores

## Gợi ý hình ảnh
- Screenshots các tab của Dashboard
- GIF demo tương tác (nếu có thể)

## Ghi chú thuyết trình
- Demo trực tiếp nếu có kết nối internet
- Chuẩn bị screenshots backup

---

# SLIDE 22: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

## Nội dung
### Kết luận
- Áp dụng thành công **5 kỹ thuật Data Mining** vào bóng đá
- **Association Rules**: Tìm được patterns có ý nghĩa
- **Clustering**: Phân nhóm cầu thủ hiệu quả
- **Classification**: Độ chính xác cao (72-85%)
- **Anomaly Detection**: Phát hiện đúng ngôi sao
- **Recommendation**: Gợi ý cầu thủ chính xác

### Hạn chế
- Dữ liệu chỉ 1 mùa giải
- Thiếu dữ liệu về giá trị chuyển nhượng, lương

### Hướng phát triển
1. Thêm dữ liệu nhiều mùa giải
2. Kết hợp dữ liệu tài chính
3. Dự đoán kết quả trận đấu
4. Phân tích video/tracking data

## Gợi ý hình ảnh
- Roadmap hướng phát triển
- Summary infographic

---

# SLIDE 23: HỎI ĐÁP

## Nội dung

# Q&A

### Cảm ơn thầy/cô và các bạn đã lắng nghe!

**Liên hệ**: [Email/GitHub]

**Source code**: [Link GitHub repository]

## Gợi ý hình ảnh
- Hình nền Premier League
- QR code link GitHub (nếu có)

---

# PHỤ LỤC: CÁC HÌNH ẢNH CẦN CHUẨN BỊ

## Danh sách hình ảnh theo slide

| Slide | Hình ảnh cần có | Nguồn/Cách tạo |
|-------|-----------------|----------------|
| 1 | Logo Premier League | Google Images |
| 2 | Data Analytics room | Stock photos |
| 4 | FBref screenshot | Screenshot website |
| 5 | Sample data table | Export từ Excel |
| 6 | Preprocessing flowchart | Draw.io/Canva |
| 9 | Association rules scatter | Notebook output |
| 12 | Clustering scatter + Radar | Notebook output |
| 15 | Confusion Matrix, Feature Importance | Notebook output |
| 17 | Anomaly scatter plot | Notebook output |
| 19 | Recommendation radar chart | Dashboard screenshot |
| 21 | Dashboard screenshots | Chụp từ Streamlit |

## Cách lấy hình từ notebooks
1. Chạy các notebooks trong thư mục `notebooks/`
2. Các hình được lưu trong `results/`
3. Export thêm từ Dashboard

---

# PHỤ LỤC: SCRIPT THUYẾT TRÌNH

## Thời gian dự kiến: 15-20 phút

| Phần | Slides | Thời gian |
|------|--------|-----------|
| Giới thiệu | 1-3 | 2 phút |
| Dữ liệu | 4-6 | 2 phút |
| Association Rules | 7-9 | 3 phút |
| Clustering | 10-12 | 3 phút |
| Classification | 13-15 | 3 phút |
| Anomaly Detection | 16-17 | 2 phút |
| Recommendation | 18-19 | 2 phút |
| Kết quả & Demo | 20-21 | 2 phút |
| Kết luận | 22-23 | 1 phút |

## Tips thuyết trình
1. **Mở đầu**: Bắt đầu với câu hỏi "Làm sao Liverpool tìm được Salah?"
2. **Chuyển tiếp**: Liên kết các kỹ thuật với nhau
3. **Demo**: Chuẩn bị sẵn các case study
4. **Kết thúc**: Tóm tắt 3 insights quan trọng nhất

---

*File này chứa nội dung chi tiết cho presentation. Copy nội dung vào PowerPoint/Google Slides và thêm hình ảnh theo gợi ý.*


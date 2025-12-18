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

## Gợi ý hình ảnh
- Logo Premier League
- Hình nền sân bóng đá

---

# SLIDE 2: BỐI CẢNH VÀ LÝ DO CHỌN ĐỀ TÀI

## Nội dung
### Bối cảnh
- Bóng đá hiện đại ngày càng phụ thuộc vào **phân tích dữ liệu**
- Các CLB lớn (Liverpool, Man City) đều có đội ngũ Data Analyst
- Premier League là giải đấu hấp dẫn nhất với dữ liệu phong phú

### Lý do chọn đề tài
- **Tính ứng dụng cao**: Hỗ trợ chiêu mộ cầu thủ, phân tích đối thủ
- **Dữ liệu đa dạng**: 100+ chỉ số thống kê (Goals, xG, Assists, Tackles...)
- **Phù hợp yêu cầu**: Áp dụng được Association Rules + các kỹ thuật khác

## Gợi ý hình ảnh
- Hình ảnh phòng phân tích dữ liệu CLB
- Screenshot: `results/position_distribution.png`

---

# SLIDE 3: MỤC TIÊU VÀ PHẠM VI PROJECT

## Nội dung
### 5 Mục tiêu chính

| # | Mục tiêu | Kỹ thuật |
|---|----------|----------|
| 1 | Tìm mẫu kết hợp trong chỉ số cầu thủ | FP-Growth |
| 2 | Phân nhóm cầu thủ theo phong cách | K-Means, Hierarchical |
| 3 | Phân loại vị trí, hiệu suất | Random Forest, Decision Tree |
| 4 | Phát hiện cầu thủ xuất sắc/bất thường | Isolation Forest, LOF |
| 5 | Gợi ý cầu thủ phù hợp | Cosine Similarity |

### Phạm vi
- **Giải đấu**: Premier League 2024-2025
- **Dữ liệu**: **574 cầu thủ**, 20 đội bóng, **100+ features**
- **Nguồn**: FBref.com

## Gợi ý hình ảnh
- Sơ đồ 5 mục tiêu dạng icon/flowchart

---

# SLIDE 4: NGUỒN DỮ LIỆU VÀ THU THẬP

## Nội dung
### Nguồn dữ liệu
- **Website**: FBref.com - Nguồn thống kê bóng đá uy tín
- **Giải đấu**: Premier League 2024-2025

### Phương pháp thu thập
- **Web Scraping** sử dụng Python
- Thư viện: `undetected_chromedriver`, `pandas.read_html`
- File: `get_data.py`, `get_full_data_2025.py`

### 4 Dataset chính

| Dataset | Mô tả | Records | Features |
|---------|-------|---------|----------|
| Players Full | Thống kê cầu thủ | 574 | 146 |
| Keepers | Thống kê thủ môn | ~40 | 50+ |
| Teams For | Chỉ số tấn công đội | 20 | 18 |
| Teams VS | Home/Away stats | 20 | 28 |

## Gợi ý hình ảnh
- Screenshot giao diện FBref
- Bảng mẫu dữ liệu

---

# SLIDE 5: CÁC CHỈ SỐ THỐNG KÊ QUAN TRỌNG

## Nội dung
### Chỉ số tấn công
- **Goals**: Số bàn thắng
- **Shots, SoT**: Số cú sút, sút trúng đích
- **xG (Expected Goals)**: Xác suất ghi bàn dựa trên vị trí sút

### Chỉ số kiến tạo
- **Assists**: Số đường kiến tạo
- **Key Passes**: Đường chuyền tạo cơ hội
- **xA (Expected Assists)**: Xác suất kiến tạo

### Chỉ số phòng ngự
- **Tackles, Interceptions**: Tranh cướp, chặn bóng
- **Blocks, Clearances**: Chặn sút, phá bóng

### Ý nghĩa xG
- Goals > xG: Cầu thủ dứt điểm tốt hơn kỳ vọng
- Goals < xG: Cầu thủ đang "xui" hoặc kỹ năng kém

## Gợi ý hình ảnh
- Infographic giải thích xG
- Biểu đồ phân bố Goals vs xG

---

# SLIDE 6: QUY TRÌNH XỬ LÝ DỮ LIỆU

## Nội dung
### Pipeline

```
Raw Data (Excel) → Clean → Transform → Feature Engineering → Ready for Analysis
```

### Các bước xử lý
1. **Xử lý missing values**: Điền 0 hoặc loại bỏ
2. **Chuyển đổi kiểu dữ liệu**: String → Numeric
3. **Chuẩn hóa tên cột**: Loại bỏ ký tự đặc biệt
4. **Feature Engineering**:
   - Tính Goals per 90 minutes
   - Tính Finishing Efficiency = Goals/xG
   - Tạo Position groups (FW, MF, DF, GK)

### Kết quả
- File: `players_processed.xlsx` (574 rows, 146 columns)
- File: `teams_processed.xlsx`

## Gợi ý hình ảnh
- Flowchart quy trình xử lý
- Screenshot code `data_preprocessing.py`

---

# SLIDE 7: ASSOCIATION RULE MINING - GIỚI THIỆU

## Nội dung
### Vấn đề cần giải quyết
- Tìm **mẫu (patterns)** trong dữ liệu cầu thủ
- Ví dụ: "Cầu thủ có xG thấp thường có Goals như thế nào?"

### Ứng dụng thực tế
- Phát hiện: "xG thấp + SoT% thấp → Goals thấp"
- Hiểu mối quan hệ giữa các chỉ số

### Thuật toán: FP-Growth
- **Ưu điểm**: Chỉ quét database 2 lần
- **Hiệu quả**: Không sinh candidate itemsets như Apriori

### Discretization
- Chia dữ liệu liên tục thành 3 mức: **Low, Medium, High**
- Sử dụng phương pháp **Quantile** (33%, 67%)

## Gợi ý hình ảnh
- Ví dụ trực quan về association rule
- Minh họa discretization

---

# SLIDE 8: ASSOCIATION RULES - LÝ THUYẾT

## Nội dung
### 3 Độ đo quan trọng

**1. Support** - Tần suất xuất hiện
$$Support(A) = \frac{\text{Số transactions chứa A}}{\text{Tổng transactions}}$$

**2. Confidence** - Độ tin cậy
$$Confidence(A \rightarrow B) = \frac{Support(A \cup B)}{Support(A)}$$

**3. Lift** - Độ nâng
$$Lift(A \rightarrow B) = \frac{Confidence}{Support(B)}$$

### Ý nghĩa Lift
- **Lift = 1**: A và B độc lập
- **Lift > 1**: Tương quan dương (hay đi cùng nhau)
- **Lift < 1**: Tương quan âm

## Gợi ý hình ảnh
- Công thức với ví dụ số cụ thể
- Biểu đồ Venn minh họa

---

# SLIDE 9: ASSOCIATION RULES - KẾT QUẢ THỰC TẾ

## Nội dung
### Thiết lập
- min_support = 0.1 (10%)
- min_confidence = 0.6 (60%)

### Kết quả: **200+ rules** được tìm thấy

### Top Rules với Confidence = 100%

| Antecedent | Consequent | Support | Confidence | Lift |
|------------|------------|---------|------------|------|
| xG=Low, SoT%=Low | Goals=Low | 27.5% | **100%** | 1.90 |
| xG=Low, SoT%=Low, Ast=Low | Goals=Low | 24.9% | **100%** | 1.90 |
| xG=Low, xAG=Low | Goals=Low | 27.9% | 99.4% | 1.89 |

### Nhận xét
- Cầu thủ có **xG thấp + SoT% thấp → 100% có Goals thấp**
- Mối quan hệ chặt giữa xG và Goals (Lift = 1.9)
- Patterns phù hợp với logic bóng đá thực tế

## Gợi ý hình ảnh
- Screenshot: `results/association_rules/player_rules_visualization.png`
- Bảng top 10 rules

---

# SLIDE 10: CLUSTERING - GIỚI THIỆU

## Nội dung
### Vấn đề
- Với **574 cầu thủ**, làm sao nhóm theo "phong cách chơi"?

### Ứng dụng
- **Tìm cầu thủ thay thế**: Cầu thủ cùng cụm có phong cách tương tự
- **Phân tích chiến thuật**: Mỗi cụm = 1 loại cầu thủ
- **So sánh**: Salah thuộc cụm nào?

### Thuật toán sử dụng
1. **K-Means**: Phân cụm theo centroid
2. **Hierarchical**: Phân cụm phân cấp (Ward's method)

### Tiền xử lý
- **Standardization**: Z-score normalization
- **100 features** được chọn cho clustering

## Gợi ý hình ảnh
- Minh họa clustering 2D
- Screenshot: `results/clustering/optimal_clusters.png`

---

# SLIDE 11: CLUSTERING - LÝ THUYẾT

## Nội dung
### K-Means Algorithm

**Bước 1**: Khởi tạo K centroids ngẫu nhiên
**Bước 2**: Gán mỗi điểm vào cụm có centroid gần nhất
**Bước 3**: Cập nhật centroid = trung bình các điểm
**Bước 4**: Lặp lại đến khi hội tụ

### Chọn K tối ưu
- **Elbow Method**: Tìm điểm "khuỷu tay" trên đồ thị Inertia
- **Silhouette Score**: Đo chất lượng phân cụm (-1 đến 1)

### Khoảng cách Euclidean
$$d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}$$

### Hierarchical Clustering
- **Ward's Method**: Gộp 2 cụm sao cho tăng variance ít nhất

## Gợi ý hình ảnh
- Animation minh họa K-Means
- Elbow curve plot

---

# SLIDE 12: CLUSTERING - KẾT QUẢ THỰC TẾ

## Nội dung
### K-Means Results
- **Số cụm tối ưu**: K = 2
- **Silhouette Score**: 0.359
- **Davies-Bouldin Score**: 1.360

### Hierarchical Results
- **Số cụm tối ưu**: K = 5
- **Silhouette Score**: 0.346

### Phân bố cầu thủ (K-Means)

| Cluster | Số cầu thủ | Đặc điểm | Ví dụ |
|---------|------------|----------|-------|
| 0 | 468 | Ít thi đấu, stats thấp | Dự bị, trẻ |
| 1 | 106 | Regular players, stats cao | Salah, Haaland, Saka |

### Hierarchical (5 cụm)
- Cluster 4: **Goal Scorers** - Haaland, Salah, Isak
- Cluster 1: **Playmakers** - De Bruyne, Ødegaard
- Cluster 2: **Defensive** - Van Dijk, Saliba

## Gợi ý hình ảnh
- Screenshot: `results/clustering/kmeans_visualization.png`
- Screenshot: `results/clustering/hierarchical_visualization.png`

---

# SLIDE 13: CLASSIFICATION - GIỚI THIỆU

## Nội dung
### 3 Bài toán phân loại

**1. Phân loại vị trí cầu thủ**
- Input: 100 features thống kê
- Output: FW, MF, DF, GK

**2. Dự đoán Top 4**
- Input: Chỉ số đội bóng
- Output: Top 4 hay không

**3. Phân loại hiệu suất**
- Input: Goals, xG, Assists...
- Output: Elite, Good, Average, Below Average

### Thuật toán
- **Random Forest**: Ensemble của nhiều Decision Trees
- **Decision Tree**: Cây quyết định (Information Gain/Gini)

## Gợi ý hình ảnh
- Sơ đồ 3 bài toán
- Minh họa Decision Tree

---

# SLIDE 14: CLASSIFICATION - LÝ THUYẾT

## Nội dung
### Decision Tree
- Chia dữ liệu theo điều kiện tối ưu
- Sử dụng **Entropy** hoặc **Gini Index**

**Entropy**:
$$H(S) = -\sum_{i=1}^{c} p_i \log_2(p_i)$$

### Random Forest
- **Bagging**: Train nhiều trees trên bootstrap samples
- **Feature Randomization**: Mỗi node xét m features ngẫu nhiên
- **Voting**: Kết hợp kết quả tất cả trees

### Tại sao Random Forest tốt hơn?
- Giảm overfitting
- Giảm variance
- Robust với noise

## Gợi ý hình ảnh
- Minh họa Random Forest ensemble
- Công thức Entropy với ví dụ

---

# SLIDE 15: CLASSIFICATION - KẾT QUẢ THỰC TẾ

## Nội dung
### 1. Phân loại vị trí cầu thủ (4 classes: DF, MF, FW, GK)

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| Random Forest | **78.3%** | 0.807 | 0.783 | 0.773 |
| Decision Tree | **79.1%** | 0.791 | 0.791 | 0.791 |

### Top Features quan trọng nhất
1. `shooting_Standard_Sh/90` (4.5%)
2. `possession_Touches_Def 3rd` (4.1%)
3. `possession_Touches_Def Pen` (3.7%)
4. `passing_types_Pass Types_TI` (3.6%)
5. `xG_per_90` (3.6%)

### Nhận xét
- Accuracy ~80% cho 4 classes là tốt
- Features phòng ngự (Def 3rd, Def Pen) quan trọng nhất
- Có thể dự đoán vị trí cầu thủ từ stats

## Gợi ý hình ảnh
- Screenshot: `results/classification/position_classification.png`
- Confusion Matrix
- Feature Importance bar chart

---

# SLIDE 16: ANOMALY DETECTION - GIỚI THIỆU

## Nội dung
### Anomaly là gì?
- Điểm dữ liệu **khác biệt đáng kể** so với phần còn lại

### Trong bóng đá, anomaly có thể là:
- **Positive**: Cầu thủ xuất sắc (Salah, Haaland)
- **Negative**: Hiệu suất kém bất thường
- **Interesting**: Phong cách độc đáo

### 2 Thuật toán sử dụng

**1. Isolation Forest**
- Ý tưởng: Outliers dễ bị "cô lập"
- Anomaly Score: 0 (normal) → 1 (anomaly)

**2. LOF (Local Outlier Factor)**
- So sánh mật độ cục bộ với láng giềng
- LOF > 1: Outlier

## Gợi ý hình ảnh
- Minh họa Isolation Forest
- Scatter plot với outliers highlighted

---

# SLIDE 17: ANOMALY DETECTION - KẾT QUẢ THỰC TẾ

## Nội dung
### Thiết lập
- **Contamination**: 10% (giả định 10% là outliers)

### Kết quả phát hiện

| Method | Số Outliers |
|--------|-------------|
| Isolation Forest | 58 |
| LOF | 58 |
| **Cả 2 methods** | **4** |

### Top Outliers (Cả 2 methods đều phát hiện)

| Cầu thủ | Đội | LOF Score | Lý do |
|---------|-----|-----------|-------|
| **Mohamed Salah** | Liverpool | 2.21 | 29 goals, 18 assists |
| **Cole Palmer** | Chelsea | 1.61 | Hiệu suất vượt trội |
| **Bryan Mbeumo** | Brentford | 1.71 | Goals + Assists cao |
| **Matheus Cunha** | Wolves | 1.64 | Phong cách độc đáo |

### Nhận xét
- Phát hiện đúng các **ngôi sao hàng đầu** giải
- LOF Score cao = khác biệt nhiều so với cầu thủ khác

## Gợi ý hình ảnh
- Screenshot: `results/anomaly_detection/player_anomalies.png`
- Bảng outliers với scores

---

# SLIDE 18: RECOMMENDATION SYSTEM - GIỚI THIỆU

## Nội dung
### 3 Loại gợi ý

**1. Similar Players**
- Tìm cầu thủ tương tự với cầu thủ cho trước
- Ví dụ: "Tìm cầu thủ giống Chris Wood"

**2. Team Needs**
- Gợi ý cầu thủ phù hợp với nhu cầu đội
- Ví dụ: "West Ham cần FW"

**3. Style-based**
- Gợi ý theo phong cách mong muốn
- Ví dụ: "Tìm cầu thủ có Goals=15, Assists=10"

### Phương pháp: Content-based Filtering
- Dựa trên **features** (chỉ số thống kê)
- Không cần dữ liệu user preferences

### Độ đo: Cosine Similarity
$$\cos(\theta) = \frac{A \cdot B}{\|A\| \times \|B\|}$$

## Gợi ý hình ảnh
- Sơ đồ hệ thống recommendation
- Minh họa Cosine Similarity

---

# SLIDE 19: RECOMMENDATION - KẾT QUẢ THỰC TẾ

## Nội dung
### 1. Similar Players - Tìm cầu thủ giống Chris Wood

| Cầu thủ | Đội | Similarity | Goals | Assists |
|---------|-----|------------|-------|---------|
| Jean-Philippe Mateta | Crystal Palace | **88.4%** | 14 | 2 |
| Alexander Isak | Newcastle | **85.9%** | 23 | 6 |
| Ollie Watkins | Aston Villa | **82.7%** | 16 | 8 |
| Raúl Jiménez | Fulham | **81.6%** | 12 | 3 |
| Yoane Wissa | Brentford | **78.9%** | 19 | 4 |

### 2. Team Needs - West Ham cần FW
- Gợi ý top FW không thuộc West Ham
- Dựa trên tổng hợp: Goals + Assists + xG + xA

### 3. Style-based
- Input: Goals=15, Assists=10, xG=12
- Output: Cầu thủ có profile gần nhất

## Gợi ý hình ảnh
- Screenshot: `results/recommendation_system/similar_players.png`
- Radar chart so sánh cầu thủ

---

# SLIDE 20: TỔNG HỢP KẾT QUẢ

## Nội dung
### Bảng tóm tắt 5 kỹ thuật

| Kỹ thuật | Thuật toán | Kết quả chính |
|----------|------------|---------------|
| Association Rules | FP-Growth | **200+ rules**, top confidence 100% |
| Clustering | K-Means, Hierarchical | **2-5 cụm**, Silhouette 0.35 |
| Classification | Random Forest | **Accuracy 78-79%** |
| Anomaly Detection | Isolation Forest, LOF | **4 outliers** (Salah, Palmer...) |
| Recommendation | Cosine Similarity | **Similarity 80%+** |

### 3 Insights quan trọng
1. **xG là chỉ số quan trọng nhất** để dự đoán Goals
2. **Cầu thủ phân thành 2-5 nhóm** rõ ràng theo phong cách
3. **Anomaly detection** phát hiện chính xác top performers

## Gợi ý hình ảnh
- Infographic tổng hợp
- Key numbers highlights

---

# SLIDE 21: DEMO DASHBOARD

## Nội dung
### Streamlit Dashboard
- **Chạy**: `streamlit run dashboard.py`

### 6 Tab chức năng
1. **Overview**: Tổng quan dữ liệu
2. **Association Rules**: Khám phá luật kết hợp
3. **Clustering**: Visualize các cụm
4. **Classification**: Dự đoán vị trí/hiệu suất
5. **Anomaly Detection**: Phát hiện outliers
6. **Recommendation System**: Gợi ý cầu thủ

### Demo scenarios
- Tìm cầu thủ tương tự Salah
- Gợi ý tiền đạo cho West Ham
- Xem anomaly scores của top players

## Gợi ý hình ảnh
- Screenshots các tab của Dashboard
- GIF demo tương tác (nếu có)

---

# SLIDE 22: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

## Nội dung
### Kết luận
- Áp dụng thành công **5 kỹ thuật Data Mining**
- **Association Rules**: 200+ patterns với confidence cao
- **Clustering**: Phân nhóm cầu thủ hiệu quả
- **Classification**: Accuracy ~80%
- **Anomaly Detection**: Phát hiện đúng ngôi sao
- **Recommendation**: Similarity 80%+

### Hạn chế
- Dữ liệu chỉ 1 mùa giải 2024-2025
- Thiếu dữ liệu tài chính (giá, lương)

### Hướng phát triển
1. Thêm dữ liệu nhiều mùa giải
2. Kết hợp dữ liệu chuyển nhượng
3. Dự đoán kết quả trận đấu
4. Real-time dashboard updates

## Gợi ý hình ảnh
- Roadmap hướng phát triển
- Summary infographic

---

# SLIDE 23: HỎI ĐÁP

## Nội dung

# Q&A

### Cảm ơn thầy/cô và các bạn đã lắng nghe!

**Source code**: [Link GitHub repository]

## Gợi ý hình ảnh
- Hình nền Premier League
- QR code link GitHub

---

# PHỤ LỤC: DANH SÁCH HÌNH ẢNH CẦN CHUẨN BỊ

## Từ thư mục results/

| Slide | File | Mô tả |
|-------|------|-------|
| 2 | `results/position_distribution.png` | Phân bố vị trí cầu thủ |
| 9 | `results/association_rules/player_rules_visualization.png` | Scatter plot rules |
| 10 | `results/clustering/optimal_clusters.png` | Elbow + Silhouette |
| 12 | `results/clustering/kmeans_visualization.png` | PCA clusters |
| 12 | `results/clustering/hierarchical_visualization.png` | Hierarchical |
| 15 | `results/classification/position_classification.png` | Confusion Matrix |
| 17 | `results/anomaly_detection/player_anomalies.png` | Outliers plot |
| 19 | `results/recommendation_system/similar_players.png` | Similar players |

---

# PHỤ LỤC: SCRIPT THUYẾT TRÌNH

## Thời gian dự kiến: 15-20 phút

| Phần | Slides | Thời gian |
|------|--------|-----------|
| Giới thiệu | 1-3 | 2 phút |
| Dữ liệu | 4-6 | 2 phút |
| Association Rules | 7-9 | 3 phút |
| Clustering | 10-12 | 3 phút |
| Classification | 13-15 | 2 phút |
| Anomaly Detection | 16-17 | 2 phút |
| Recommendation | 18-19 | 2 phút |
| Kết quả & Demo | 20-21 | 2 phút |
| Kết luận | 22-23 | 1 phút |

## Tips thuyết trình
1. **Mở đầu**: "Làm sao Liverpool tìm được Salah với giá rẻ?"
2. **Chuyển tiếp**: Liên kết các kỹ thuật với nhau
3. **Demo**: Chuẩn bị sẵn Dashboard đang chạy
4. **Kết thúc**: Nhấn mạnh 3 insights quan trọng nhất

---

*File này chứa nội dung chi tiết cho presentation với dữ liệu thực tế từ project.*
*Copy nội dung vào PowerPoint/Google Slides và thêm hình ảnh từ thư mục results/.*

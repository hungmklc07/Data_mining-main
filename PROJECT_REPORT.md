# BÁO CÁO LÝ THUYẾT DATA MINING
## Phân tích dữ liệu Premier League 2024-2025

---

## Mục lục
1. [Giới thiệu và Bối cảnh](#1-giới-thiệu-và-bối-cảnh)
2. [Association Rule Mining](#2-association-rule-mining)
3. [Clustering](#3-clustering)
4. [Classification](#4-classification)
5. [Anomaly Detection](#5-anomaly-detection)
6. [Recommendation System](#6-recommendation-system)

---

## 1. Giới thiệu và Bối cảnh

### 1.1 Vấn đề cần giải quyết

Trong bóng đá hiện đại, việc phân tích dữ liệu đóng vai trò quan trọng trong:
- **Chiêu mộ cầu thủ**: Tìm cầu thủ phù hợp với lối chơi của đội
- **Phân tích đối thủ**: Hiểu điểm mạnh/yếu để xây dựng chiến thuật
- **Đánh giá hiệu suất**: So sánh cầu thủ với kỳ vọng thống kê
- **Phát hiện tài năng**: Tìm cầu thủ tiềm năng bị đánh giá thấp

### 1.2 Dữ liệu sử dụng

Dữ liệu bao gồm các chỉ số thống kê của cầu thủ và đội bóng:
- **Chỉ số tấn công**: Goals, Assists, Shots, Shots on Target
- **Chỉ số kỳ vọng**: xG (Expected Goals), xA (Expected Assists)
- **Chỉ số phòng ngự**: Tackles, Interceptions, Blocks
- **Chỉ số chuyền bóng**: Passes, Pass Completion Rate

### 1.3 Các khái niệm thống kê quan trọng

#### Expected Goals (xG)
**xG** là xác suất một cú sút trở thành bàn thắng, dựa trên:
- Vị trí sút (khoảng cách, góc độ)
- Loại cú sút (chân, đầu)
- Tình huống (phản công, đá phạt)

**Ý nghĩa**:
- Nếu Goals > xG: Cầu thủ dứt điểm tốt hơn kỳ vọng
- Nếu Goals < xG: Cầu thủ đang "xui" hoặc kỹ năng dứt điểm kém

#### Expected Assists (xA)
**xA** là xác suất một đường chuyền dẫn đến bàn thắng, dựa trên chất lượng cơ hội tạo ra.

---

## 2. Association Rule Mining

### 2.1 Tại sao cần tìm luật kết hợp trong bóng đá?

**Vấn đề**: Chúng ta muốn tìm ra các **mẫu (patterns)** trong dữ liệu cầu thủ. Ví dụ:
- Cầu thủ có xG cao thường có bao nhiêu Shots?
- Tiền đạo ghi nhiều bàn thường có đặc điểm gì chung?
- Chỉ số nào thường đi cùng nhau?

**Ứng dụng thực tế**:
- Nếu phát hiện luật: "Cầu thủ có Shots on Target cao → thường có Goals cao", ta biết cần cải thiện độ chính xác sút
- Nếu phát hiện: "Tiền vệ có Passes cao VÀ xA cao → thường có Assists cao", ta biết cần tìm tiền vệ có cả hai đặc điểm

### 2.2 Các khái niệm cơ bản

#### Itemset
Một tập hợp các "item" (đặc điểm). Trong bóng đá, item có thể là:
- Goals = High
- Position = FW (Forward)
- xG = Medium

#### Association Rule
Một luật có dạng: **A → B** (Nếu A thì B)

Ví dụ: {Goals = High, xG = High} → {Position = FW}

Nghĩa là: "Nếu cầu thủ có Goals cao VÀ xG cao, thì có khả năng cao là Tiền đạo"

### 2.3 Các độ đo quan trọng

#### Support (Độ hỗ trợ)

**Định nghĩa**: Tần suất xuất hiện của itemset trong toàn bộ dữ liệu.

**Công thức**:
\[
Support(A) = \frac{\text{Số transactions chứa A}}{\text{Tổng số transactions}}
\]

**Ví dụ**: Có 500 cầu thủ, 75 cầu thủ có Goals = High
\[
Support(\text{Goals = High}) = \frac{75}{500} = 0.15 = 15\%
\]

**Ý nghĩa**: 15% cầu thủ có số bàn thắng cao

#### Confidence (Độ tin cậy)

**Định nghĩa**: Xác suất B xảy ra khi đã biết A xảy ra.

**Công thức**:
\[
Confidence(A \rightarrow B) = \frac{Support(A \cup B)}{Support(A)} = P(B|A)
\]

**Ví dụ**: 
- 75 cầu thủ có Goals = High
- Trong đó 60 cầu thủ là FW (Forward)

\[
Confidence(\text{Goals = High} \rightarrow \text{FW}) = \frac{60}{75} = 0.8 = 80\%
\]

**Ý nghĩa**: 80% cầu thủ ghi nhiều bàn là Tiền đạo

#### Lift (Độ nâng)

**Định nghĩa**: Đo mức độ tương quan giữa A và B so với khi chúng độc lập.

**Công thức**:
\[
Lift(A \rightarrow B) = \frac{Confidence(A \rightarrow B)}{Support(B)} = \frac{P(A \cap B)}{P(A) \times P(B)}
\]

**Giải thích giá trị Lift**:
- **Lift = 1**: A và B độc lập (không liên quan)
- **Lift > 1**: A và B có tương quan dương (xuất hiện cùng nhau nhiều hơn ngẫu nhiên)
- **Lift < 1**: A và B có tương quan âm (ít xuất hiện cùng nhau)

**Ví dụ**:
- Support(FW) = 100/500 = 0.2 (20% cầu thủ là FW)
- Confidence(Goals = High → FW) = 0.8

\[
Lift = \frac{0.8}{0.2} = 4
\]

**Ý nghĩa**: Cầu thủ ghi nhiều bàn có khả năng là FW cao gấp 4 lần so với ngẫu nhiên.

### 2.4 Thuật toán FP-Growth

#### Tại sao không dùng Apriori?

**Apriori** hoạt động theo nguyên tắc:
1. Sinh tất cả candidate itemsets có độ dài k
2. Quét database để đếm support
3. Loại bỏ itemsets có support < min_support
4. Lặp lại với k+1

**Vấn đề của Apriori**:
- Phải quét database nhiều lần (tốn thời gian)
- Sinh ra quá nhiều candidate itemsets

#### FP-Growth hoạt động như thế nào?

**Ý tưởng chính**: Nén dữ liệu vào cấu trúc cây (FP-Tree), chỉ cần quét database 2 lần.

**Bước 1**: Quét database lần 1 - đếm tần suất mỗi item

**Bước 2**: Xây dựng FP-Tree
- Sắp xếp items theo tần suất giảm dần
- Chèn từng transaction vào cây
- Các transaction có prefix chung sẽ chia sẻ nhánh

**Bước 3**: Khai thác frequent itemsets từ FP-Tree
- Không cần sinh candidate
- Sử dụng conditional pattern base

**Ưu điểm**:
- Chỉ quét database 2 lần
- Không sinh candidate itemsets
- Hiệu quả với dữ liệu lớn

### 2.5 Discretization (Rời rạc hóa)

**Vấn đề**: Association Rules yêu cầu dữ liệu categorical, nhưng Goals, xG là số liên tục.

**Giải pháp**: Chia thành các khoảng (bins)

**Phương pháp Quantile**:
- **Low**: Dưới 33% (phân vị thứ 33)
- **Medium**: 33% - 67%
- **High**: Trên 67%

**Ví dụ**: Nếu Goals có phân bố:
- Q33 = 2 bàn
- Q67 = 8 bàn

Thì:
- Goals ≤ 2 → "Goals = Low"
- 2 < Goals ≤ 8 → "Goals = Medium"
- Goals > 8 → "Goals = High"

---

## 3. Clustering

### 3.1 Tại sao cần phân cụm cầu thủ?

**Vấn đề**: Với hàng trăm cầu thủ, làm sao nhóm họ theo "phong cách chơi"?

**Ứng dụng**:
- **Tìm cầu thủ thay thế**: Cầu thủ cùng cụm có phong cách tương tự
- **Phân tích chiến thuật**: Mỗi cụm đại diện một loại cầu thủ (playmaker, goalscorer, defensive midfielder...)
- **So sánh**: Cầu thủ thuộc cụm nào trong các cầu thủ hàng đầu?

### 3.2 K-Means Clustering

#### Ý tưởng cơ bản

Chia n điểm dữ liệu thành K cụm sao cho:
- Mỗi điểm thuộc cụm có **centroid (tâm)** gần nhất
- Tổng khoảng cách từ các điểm đến centroid của cụm là **nhỏ nhất**

#### Khoảng cách Euclidean

Đo khoảng cách giữa hai điểm trong không gian nhiều chiều:

\[
d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}
\]

**Ví dụ 2 chiều** (Goals và Assists):
- Cầu thủ A: (10 goals, 5 assists)
- Cầu thủ B: (8 goals, 7 assists)

\[
d(A, B) = \sqrt{(10-8)^2 + (5-7)^2} = \sqrt{4 + 4} = 2.83
\]

#### Thuật toán Lloyd (K-Means chuẩn)

**Bước 1**: Khởi tạo K centroids ngẫu nhiên

**Bước 2**: Gán mỗi điểm vào cụm có centroid gần nhất
\[
C_i = \{x : \|x - \mu_i\| \leq \|x - \mu_j\| \text{ với mọi } j\}
\]

**Bước 3**: Cập nhật centroid = trung bình các điểm trong cụm
\[
\mu_i = \frac{1}{|C_i|} \sum_{x \in C_i} x
\]

**Bước 4**: Lặp lại Bước 2-3 cho đến khi hội tụ (centroids không đổi)

#### Hàm mục tiêu (Inertia)

K-Means tối thiểu hóa **Within-Cluster Sum of Squares (WCSS)**:

\[
J = \sum_{i=1}^{K} \sum_{x \in C_i} \|x - \mu_i\|^2
\]

**Ý nghĩa**: Tổng bình phương khoảng cách từ mỗi điểm đến centroid của cụm. Càng nhỏ = các cụm càng "chặt".

### 3.3 Hierarchical Clustering

#### Ý tưởng

Xây dựng **cây phân cấp (dendrogram)** thể hiện mối quan hệ giữa các điểm.

**Agglomerative (Bottom-up)**:
1. Bắt đầu: Mỗi điểm là một cụm
2. Lặp: Gộp 2 cụm gần nhất thành 1
3. Dừng: Khi chỉ còn 1 cụm

#### Các phương pháp đo khoảng cách giữa cụm (Linkage)

**Single Linkage** (Khoảng cách ngắn nhất):
\[
d(C_i, C_j) = \min_{x \in C_i, y \in C_j} d(x, y)
\]
- Ưu: Phát hiện cụm hình dạng dài
- Nhược: Dễ bị "chaining effect"

**Complete Linkage** (Khoảng cách xa nhất):
\[
d(C_i, C_j) = \max_{x \in C_i, y \in C_j} d(x, y)
\]
- Ưu: Tạo cụm compact
- Nhược: Nhạy với outliers

**Average Linkage** (Khoảng cách trung bình):
\[
d(C_i, C_j) = \frac{1}{|C_i| \cdot |C_j|} \sum_{x \in C_i} \sum_{y \in C_j} d(x, y)
\]

**Ward's Method** (Phương sai tối thiểu):
\[
d(C_i, C_j) = \sqrt{\frac{2|C_i||C_j|}{|C_i|+|C_j|}} \|\mu_i - \mu_j\|
\]
- Gộp 2 cụm sao cho tăng WCSS ít nhất
- Thường cho kết quả tốt nhất

### 3.4 Chọn số cụm K tối ưu

#### Elbow Method

**Ý tưởng**: Vẽ đồ thị Inertia theo K, tìm điểm "khuỷu tay" - nơi giảm Inertia chậm lại.

**Giải thích**: 
- K nhỏ: Inertia cao (cụm lớn, không chặt)
- K lớn: Inertia thấp (nhiều cụm nhỏ)
- Điểm tối ưu: Cân bằng giữa số cụm và độ chặt

#### Silhouette Score

**Định nghĩa**: Đo chất lượng phân cụm cho mỗi điểm.

Với mỗi điểm i:
- **a(i)** = Khoảng cách trung bình đến các điểm cùng cụm
- **b(i)** = Khoảng cách trung bình đến các điểm của cụm gần nhất khác

\[
s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
\]

**Giá trị**:
- s(i) ≈ 1: Điểm được phân cụm tốt (xa cụm khác, gần cụm mình)
- s(i) ≈ 0: Điểm nằm giữa 2 cụm
- s(i) ≈ -1: Điểm có thể bị phân sai cụm

**Silhouette Score tổng** = Trung bình s(i) của tất cả điểm. Chọn K có score cao nhất.

#### Davies-Bouldin Index

\[
DB = \frac{1}{K} \sum_{i=1}^{K} \max_{j \neq i} \frac{\sigma_i + \sigma_j}{d(\mu_i, \mu_j)}
\]

Trong đó:
- σᵢ = Độ phân tán của cụm i (trung bình khoảng cách đến centroid)
- d(μᵢ, μⱼ) = Khoảng cách giữa 2 centroids

**Ý nghĩa**: Càng nhỏ càng tốt (cụm compact và tách biệt)

### 3.5 Chuẩn hóa dữ liệu (Standardization)

**Vấn đề**: Goals có range 0-30, Passes có range 0-3000. Khoảng cách Euclidean sẽ bị chi phối bởi Passes.

**Giải pháp - Z-score Standardization**:
\[
z = \frac{x - \mu}{\sigma}
\]

Sau chuẩn hóa: mean = 0, std = 1 cho mọi feature.

---

## 4. Classification

### 4.1 Tại sao cần phân loại?

**Bài toán**: Dựa trên các chỉ số, dự đoán:
- Vị trí cầu thủ (FW, MF, DF, GK)
- Đội có vào Top 4 không?
- Hiệu suất cầu thủ (Elite, Good, Average, Below Average)

**Ứng dụng**:
- Xác định vị trí tối ưu cho cầu thủ trẻ
- Dự đoán kết quả cuối mùa
- Đánh giá hiệu suất khách quan

### 4.2 Decision Tree

#### Ý tưởng

Xây dựng cây quyết định bằng cách chia dữ liệu theo các điều kiện.

**Ví dụ đơn giản**:
```
                    [Goals > 10?]
                    /           \
                  Yes            No
                  /               \
            [FW: 80%]      [Tackles > 50?]
                            /           \
                          Yes            No
                          /               \
                    [DF: 70%]        [MF: 60%]
```

#### Entropy (Độ hỗn loạn)

Đo mức độ "hỗn loạn" của tập dữ liệu:

\[
H(S) = -\sum_{i=1}^{c} p_i \log_2(p_i)
\]

Trong đó pᵢ = tỷ lệ class i trong tập S.

**Ví dụ**: Tập có 60 FW, 40 MF
- p(FW) = 0.6, p(MF) = 0.4
- H = -0.6 log₂(0.6) - 0.4 log₂(0.4) ≈ 0.97

**Ý nghĩa**:
- H = 0: Tập hoàn toàn thuần nhất (chỉ 1 class)
- H = 1 (với 2 class): Tập hoàn toàn hỗn loạn (50-50)

#### Information Gain

Đo lượng thông tin thu được khi chia theo thuộc tính A:

\[
IG(S, A) = H(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} H(S_v)
\]

**Ý nghĩa**: Chọn thuộc tính có Information Gain cao nhất để chia.

**Ví dụ**: 
- Trước khi chia: H(S) = 0.97
- Sau khi chia theo "Goals > 10": 
  - Nhánh Yes: 50 FW, 5 MF → H = 0.35
  - Nhánh No: 10 FW, 35 MF → H = 0.74
- IG = 0.97 - (55/100 × 0.35 + 45/100 × 0.74) = 0.97 - 0.53 = 0.44

#### Gini Index (Thay thế cho Entropy)

\[
Gini(S) = 1 - \sum_{i=1}^{c} p_i^2
\]

**Ví dụ**: p(FW) = 0.6, p(MF) = 0.4
- Gini = 1 - (0.6² + 0.4²) = 1 - 0.52 = 0.48

**So sánh**: Gini tính toán nhanh hơn Entropy, kết quả tương tự.

### 4.3 Random Forest

#### Vấn đề của Decision Tree đơn

- **Overfitting**: Cây quá sâu sẽ học thuộc dữ liệu training
- **High variance**: Thay đổi nhỏ trong dữ liệu → cây khác hoàn toàn

#### Ý tưởng Ensemble

**"Wisdom of the crowd"**: Kết hợp nhiều models yếu thành model mạnh.

#### Bootstrap Aggregating (Bagging)

**Bước 1**: Tạo B tập dữ liệu bootstrap
- Lấy mẫu ngẫu nhiên **có hoàn lại** từ dữ liệu gốc
- Mỗi tập có cùng kích thước với dữ liệu gốc

**Bước 2**: Train B decision trees trên B tập dữ liệu

**Bước 3**: Dự đoán = **Majority voting** (class được nhiều cây chọn nhất)

#### Random Forest = Bagging + Feature Randomization

Ngoài Bagging, mỗi node chỉ xét **m features ngẫu nhiên** (thường m = √p với p features).

**Tại sao hiệu quả?**
- Giảm correlation giữa các cây
- Mỗi cây học các patterns khác nhau
- Kết hợp → giảm variance, tránh overfitting

### 4.4 Các độ đo đánh giá

#### Confusion Matrix

|  | Predicted Positive | Predicted Negative |
|--|--------------------|--------------------|
| **Actual Positive** | TP (True Positive) | FN (False Negative) |
| **Actual Negative** | FP (False Positive) | TN (True Negative) |

#### Accuracy

\[
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
\]

**Ý nghĩa**: Tỷ lệ dự đoán đúng tổng thể.

**Hạn chế**: Không phù hợp với dữ liệu mất cân bằng.

#### Precision

\[
Precision = \frac{TP}{TP + FP}
\]

**Ý nghĩa**: Trong số dự đoán là Positive, bao nhiêu % thực sự Positive?

**Ví dụ**: Trong 100 cầu thủ được dự đoán là FW, 85 đúng là FW → Precision = 85%

#### Recall (Sensitivity)

\[
Recall = \frac{TP}{TP + FN}
\]

**Ý nghĩa**: Trong số thực sự Positive, bao nhiêu % được phát hiện?

**Ví dụ**: Có 100 FW thực tế, model phát hiện được 80 → Recall = 80%

#### F1-Score

\[
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
\]

**Ý nghĩa**: Trung bình điều hòa của Precision và Recall. Cân bằng giữa hai độ đo.

---

## 5. Anomaly Detection

### 5.1 Tại sao phát hiện bất thường trong bóng đá?

**Anomaly (Outlier)** = Điểm dữ liệu khác biệt đáng kể so với phần còn lại.

**Trong bóng đá, anomaly có thể là**:
- **Positive**: Cầu thủ xuất sắc (Messi, Haaland) có chỉ số vượt trội
- **Negative**: Cầu thủ hiệu suất kém bất thường
- **Interesting**: Cầu thủ có phong cách độc đáo (thủ môn kiến tạo nhiều)

**Ứng dụng**:
- Phát hiện tài năng đặc biệt
- Tìm cầu thủ bị đánh giá thấp
- Phát hiện vấn đề (chấn thương, phong độ giảm)

### 5.2 Isolation Forest

#### Ý tưởng cốt lõi

**"Anomalies are few and different"** - Outliers ít và khác biệt, nên dễ bị **cô lập**.

**Quan sát**: Nếu chia dữ liệu ngẫu nhiên bằng các đường thẳng:
- Điểm bình thường: Cần nhiều lần chia mới bị cô lập
- Outlier: Chỉ cần ít lần chia đã bị cô lập

#### Thuật toán

**Bước 1**: Xây dựng Isolation Tree
- Chọn ngẫu nhiên 1 feature
- Chọn ngẫu nhiên 1 giá trị split trong range của feature
- Chia dữ liệu thành 2 nhánh
- Lặp lại cho đến khi mỗi điểm bị cô lập hoặc đạt max depth

**Bước 2**: Xây dựng nhiều Isolation Trees (forest)

**Bước 3**: Tính Anomaly Score

#### Path Length và Anomaly Score

**Path Length h(x)** = Số edges từ root đến node chứa x

**Average Path Length**:
\[
E[h(x)] = \frac{1}{t} \sum_{i=1}^{t} h_i(x)
\]
với t = số trees

**Normalization factor c(n)**:
\[
c(n) = 2H(n-1) - \frac{2(n-1)}{n}
\]
với H(k) = ln(k) + 0.5772... (Euler constant)

**Anomaly Score**:
\[
s(x, n) = 2^{-\frac{E[h(x)]}{c(n)}}
\]

**Giải thích score**:
- s ≈ 1: Anomaly (path length ngắn)
- s ≈ 0.5: Normal (path length trung bình)
- s < 0.5: Rất normal (path length dài)

#### Contamination Parameter

**Contamination** = Tỷ lệ outliers dự kiến trong dữ liệu.

Ví dụ: contamination = 0.1 → Giả định 10% dữ liệu là outliers

### 5.3 Local Outlier Factor (LOF)

#### Ý tưởng

So sánh **mật độ cục bộ** của một điểm với mật độ của các láng giềng.

- Nếu mật độ thấp hơn nhiều → Outlier
- Nếu mật độ tương đương → Normal

#### Các khái niệm

**k-distance(x)**: Khoảng cách đến láng giềng thứ k

**Reachability Distance**:
\[
reach\_dist_k(x, y) = \max(k\text{-}distance(y), d(x, y))
\]

**Ý nghĩa**: Khoảng cách "có điều chỉnh" - không nhỏ hơn k-distance của y.

**Local Reachability Density (LRD)**:
\[
LRD_k(x) = \frac{1}{\frac{\sum_{y \in N_k(x)} reach\_dist_k(x, y)}{|N_k(x)|}}
\]

**Ý nghĩa**: Nghịch đảo của trung bình reachability distance. Mật độ cao = LRD cao.

**Local Outlier Factor**:
\[
LOF_k(x) = \frac{\sum_{y \in N_k(x)} \frac{LRD_k(y)}{LRD_k(x)}}{|N_k(x)|}
\]

**Giải thích LOF**:
- LOF ≈ 1: Mật độ tương đương láng giềng (normal)
- LOF >> 1: Mật độ thấp hơn nhiều (outlier)
- LOF < 1: Mật độ cao hơn (very normal)

#### So sánh Isolation Forest vs LOF

| Tiêu chí | Isolation Forest | LOF |
|----------|------------------|-----|
| Ý tưởng | Cô lập | So sánh mật độ |
| Tốc độ | Nhanh (O(n log n)) | Chậm hơn (O(n²)) |
| Global vs Local | Global | Local |
| Phù hợp | Dữ liệu lớn | Dữ liệu có cụm mật độ khác nhau |

---

## 6. Recommendation System

### 6.1 Tại sao cần hệ thống gợi ý cầu thủ?

**Bài toán**:
- Tìm cầu thủ **tương tự** với cầu thủ hiện tại (thay thế khi chấn thương)
- Tìm cầu thủ **phù hợp** với nhu cầu đội bóng
- Tìm cầu thủ có **phong cách chơi** mong muốn

### 6.2 Content-based Filtering

#### Ý tưởng

Gợi ý dựa trên **nội dung** (features) của items, không dựa trên hành vi người dùng.

**Trong bóng đá**: So sánh các chỉ số thống kê của cầu thủ.

### 6.3 Cosine Similarity

#### Định nghĩa

Đo độ tương tự giữa 2 vectors dựa trên **góc** giữa chúng:

\[
\cos(\theta) = \frac{A \cdot B}{\|A\| \times \|B\|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \times \sqrt{\sum_{i=1}^{n} B_i^2}}
\]

#### Ý nghĩa hình học

- **cos(0°) = 1**: Hai vectors cùng hướng → hoàn toàn giống nhau
- **cos(90°) = 0**: Hai vectors vuông góc → không liên quan
- **cos(180°) = -1**: Hai vectors ngược hướng → hoàn toàn khác nhau

#### Tại sao dùng Cosine thay vì Euclidean?

**Euclidean** đo **khoảng cách tuyệt đối**:
- Cầu thủ A: (20 goals, 10 assists)
- Cầu thủ B: (10 goals, 5 assists)
- d(A,B) lớn vì A ghi nhiều hơn

**Cosine** đo **hướng** (tỷ lệ):
- A và B có tỷ lệ Goals:Assists = 2:1 giống nhau
- Cosine similarity cao

**Ứng dụng**: Tìm cầu thủ có "phong cách" tương tự, không phải "số lượng" tương tự.

#### Ví dụ tính toán

Cầu thủ X: (Goals=15, Assists=8, xG=12)
Cầu thủ Y: (Goals=10, Assists=6, xG=9)

\[
X \cdot Y = 15 \times 10 + 8 \times 6 + 12 \times 9 = 150 + 48 + 108 = 306
\]

\[
\|X\| = \sqrt{15^2 + 8^2 + 12^2} = \sqrt{225 + 64 + 144} = \sqrt{433} \approx 20.8
\]

\[
\|Y\| = \sqrt{10^2 + 6^2 + 9^2} = \sqrt{100 + 36 + 81} = \sqrt{217} \approx 14.7
\]

\[
\cos(X, Y) = \frac{306}{20.8 \times 14.7} = \frac{306}{305.76} \approx 1.0
\]

**Kết luận**: X và Y rất tương tự về phong cách (tỷ lệ các chỉ số gần như giống nhau).

### 6.4 Các phương pháp gợi ý

#### Similar Players

**Input**: Tên cầu thủ A
**Output**: Top N cầu thủ có cosine similarity cao nhất với A

**Quy trình**:
1. Chuẩn hóa features (StandardScaler)
2. Tính cosine similarity giữa A và tất cả cầu thủ
3. Sắp xếp theo similarity giảm dần
4. Trả về top N (loại bỏ chính A)

#### Team Needs Recommendation

**Input**: Tên đội, vị trí cần tìm
**Output**: Top N cầu thủ phù hợp nhất

**Quy trình**:
1. Loại bỏ cầu thủ đã thuộc đội
2. Lọc theo vị trí
3. Tính điểm tổng hợp = Goals + Assists + xG + xA
4. Trả về top N theo điểm

#### Style-based Recommendation

**Input**: Các chỉ số mong muốn (ví dụ: Goals=15, Assists=10)
**Output**: Top N cầu thủ phù hợp nhất

**Quy trình**:
1. Tạo vector "cầu thủ lý tưởng" từ input
2. Chuẩn hóa vector
3. Tính cosine similarity với tất cả cầu thủ
4. Trả về top N

### 6.5 Content-based vs Collaborative Filtering

| Tiêu chí | Content-based | Collaborative |
|----------|---------------|---------------|
| Dữ liệu cần | Features của items | Ratings/interactions của users |
| Cold start | Không vấn đề | Có vấn đề (user/item mới) |
| Diversity | Thấp (gợi ý tương tự) | Cao (dựa trên users khác) |
| Giải thích | Dễ (dựa trên features) | Khó (dựa trên users) |

**Trong project này**: Sử dụng **Content-based** vì:
- Có đầy đủ features của cầu thủ
- Không có dữ liệu về "preferences" của users
- Dễ giải thích: "Cầu thủ A tương tự B vì có Goals, Assists, xG gần nhau"

---

## 7. Tổng kết

### 7.1 Bảng tóm tắt các kỹ thuật

| Kỹ thuật | Mục đích | Thuật toán chính | Output |
|----------|----------|------------------|--------|
| Association Rules | Tìm mẫu kết hợp | FP-Growth | Rules (A → B) với Support, Confidence, Lift |
| Clustering | Nhóm cầu thủ tương tự | K-Means, Hierarchical | Cluster labels |
| Classification | Dự đoán class | Random Forest, Decision Tree | Predicted class + metrics |
| Anomaly Detection | Phát hiện bất thường | Isolation Forest, LOF | Outlier labels + scores |
| Recommendation | Gợi ý cầu thủ | Cosine Similarity | Ranked list of players |

### 7.2 Workflow tổng thể

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Raw Data   │────▶│   Clean &   │────▶│  Feature    │
│  (Excel)    │     │  Preprocess │     │  Engineering│
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
       ┌───────────────────────────────────────┼───────────────────────────────────────┐
       │                   │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼                   ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Association │     │  Clustering │     │Classification│     │  Anomaly   │     │Recommendation│
│   Rules     │     │             │     │             │     │  Detection │     │   System    │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
       │                   │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼                   ▼
   Patterns           Clusters            Predictions          Outliers          Recommendations
```

### 7.3 Ứng dụng thực tế

1. **Scout cầu thủ**: Dùng Recommendation System tìm cầu thủ phù hợp
2. **Phân tích đối thủ**: Dùng Clustering xem đối thủ thuộc nhóm nào
3. **Dự đoán**: Dùng Classification dự đoán Top 4
4. **Phát hiện tài năng**: Dùng Anomaly Detection tìm outliers positive
5. **Hiểu patterns**: Dùng Association Rules tìm mẫu thành công

---

*Báo cáo lý thuyết Data Mining - Premier League 2024-2025*

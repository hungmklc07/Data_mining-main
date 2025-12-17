# Data Mining Project - Premier League 2024-2025 Analysis

Dự án khai phá dữ liệu bóng đá Premier League 2024-2025 sử dụng 4 kỹ thuật Data Mining.

## Cấu trúc dự án

```
Data_mining-main/
├── data/                          # Dữ liệu đã clean
│   ├── Cleaned_FBref_Premier-League_2024-2025_Full_Merged.xlsx
│   ├── Cleaned_PL_2024-2025_Keepers_Full.xlsx
│   ├── Cleaned_PL_2024-2025_Teams_For.xlsx
│   └── Cleaned_PL_2024-2025_Teams_VS.xlsx
├── notebooks/                     # Jupyter notebooks
│   ├── 1_data_exploration.ipynb
│   ├── 2_association_rules.ipynb
│   ├── 3_clustering.ipynb
│   ├── 4_classification.ipynb
│   └── 5_anomaly_detection.ipynb
├── src/                           # Code Python modules
│   ├── data_preprocessing.py
│   ├── association_rules.py
│   ├── clustering.py
│   ├── classification.py
│   └── anomaly_detection.py
├── results/                       # Kết quả và visualizations
│   ├── association_rules/
│   ├── clustering/
│   ├── classification/
│   └── anomaly_detection/
├── requirements.txt
├── main.py
└── README.md
```

## Cài đặt

1. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

2. Đảm bảo các file dữ liệu đã được copy vào thư mục `data/`

## Sử dụng

### Chạy từng notebook

Chạy các notebook theo thứ tự:

1. **Data Exploration** (`notebooks/1_data_exploration.ipynb`)
   - Khám phá và tiền xử lý dữ liệu
   - Feature engineering
   - Lưu dữ liệu đã xử lý

2. **Association Rule Mining** (`notebooks/2_association_rules.ipynb`)
   - Discretize features
   - Áp dụng FP-Growth
   - Tìm association rules
   - Visualize rules

3. **Clustering** (`notebooks/3_clustering.ipynb`)
   - Phân cụm cầu thủ theo phong cách chơi
   - K-Means và Hierarchical Clustering
   - Tìm số cụm tối ưu
   - Visualize clusters

4. **Classification** (`notebooks/4_classification.ipynb`)
   - Phân loại vị trí cầu thủ
   - Phân loại đội bóng Top 4
   - Phân loại hiệu suất cầu thủ
   - Đánh giá models

5. **Anomaly Detection** (`notebooks/5_anomaly_detection.ipynb`)
   - Phát hiện cầu thủ bất thường
   - Phát hiện đội bóng bất thường
   - Isolation Forest và LOF
   - Visualize outliers

### Chạy main script

```bash
python main.py
```

Script này sẽ kiểm tra cấu trúc thư mục và hiển thị hướng dẫn.

## Kết quả

Tất cả kết quả sẽ được lưu trong thư mục `results/`:

- `results/association_rules/`: Association rules và visualizations
- `results/clustering/`: Cluster assignments và visualizations
- `results/classification/`: Model metrics và confusion matrices
- `results/anomaly_detection/`: Outlier lists và visualizations

## Các kỹ thuật sử dụng

1. **Association Rule Mining (FP-Growth)**
   - Tìm mẫu kết hợp giữa các chỉ số
   - Rules với Support, Confidence, Lift

2. **Clustering**
   - K-Means Clustering
   - Hierarchical Clustering
   - Elbow Method và Silhouette Score

3. **Classification**
   - Random Forest Classifier
   - Decision Tree Classifier
   - Metrics: Accuracy, Precision, Recall, F1

4. **Anomaly Detection**
   - Isolation Forest
   - Local Outlier Factor (LOF)

## Lưu ý

- Đảm bảo đã chạy notebook `1_data_exploration.ipynb` trước để tạo dữ liệu đã xử lý
- Các notebook có thể tự động load dữ liệu từ file gốc nếu chưa có file processed
- Kết quả sẽ được tự động lưu vào thư mục `results/`

## Tác giả

Dự án Data Mining - Môn Khai phá dữ liệu

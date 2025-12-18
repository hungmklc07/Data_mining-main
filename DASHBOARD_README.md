# Dashboard Hướng Dẫn Sử Dụng

## Cài đặt

1. Cài đặt Streamlit và Plotly (nếu chưa có):
```bash
pip install streamlit plotly
```

Hoặc cài tất cả dependencies:
```bash
pip install -r requirements.txt
```

## Chạy Dashboard

### Windows:
```bash
run_dashboard.bat
```

Hoặc:
```bash
streamlit run dashboard.py
```

### Linux/Mac:
```bash
chmod +x run_dashboard.sh
./run_dashboard.sh
```

Hoặc:
```bash
streamlit run dashboard.py
```

## Các tính năng Dashboard

### 1. 📊 Overview
- Tổng quan dữ liệu
- Phân bố vị trí cầu thủ
- Thống kê mô tả

### 2. 🔗 Association Rules
- Tùy chỉnh min_support và min_confidence
- Xem top association rules
- Visualization: Support vs Confidence scatter plot

### 3. 🎯 Clustering
- Chọn số cụm hoặc tự động tìm tối ưu
- K-Means hoặc Hierarchical Clustering
- PCA visualization
- Xem cầu thủ theo từng cụm

### 4. 📈 Classification
- **Tab 1**: Dự đoán vị trí cầu thủ
  - Train model và xem metrics
  - Feature importance
  - Confusion matrix
- **Tab 2**: Dự đoán đội bóng Top 4
- **Tab 3**: Phân loại hiệu suất cầu thủ

### 5. 🚨 Anomaly Detection
- Phát hiện outliers cho cầu thủ hoặc đội bóng
- Tùy chỉnh contamination
- Chọn phương pháp: Isolation Forest, LOF, hoặc cả hai
- Visualization scatter plot với outliers được đánh dấu

### 6. ⭐ Recommendation System (Tính năng chính)
- **Tab 1: Tìm cầu thủ tương tự**
  - Chọn cầu thủ từ dropdown
  - Tùy chỉnh số lượng gợi ý
  - Chọn có tìm cùng vị trí hay không
  - Xem danh sách và visualization

- **Tab 2: Gợi ý cho đội bóng**
  - Chọn đội bóng
  - Chọn vị trí cần tìm
  - Xem danh sách gợi ý với recommendation score

- **Tab 3: Tìm theo phong cách**
  - Nhập các chỉ số mong muốn (Goals, Assists, xG, xA)
  - Tìm cầu thủ phù hợp với phong cách đó
  - Visualization similarity scores

- **Tab 4: Player Profile**
  - Chọn cầu thủ để xem profile chi tiết
  - Thông tin cơ bản và chỉ số quan trọng
  - Visualization stats

## Lưu ý

- Dashboard sẽ tự động load dữ liệu từ `data/players_processed.xlsx` hoặc từ file gốc nếu chưa có
- Tất cả kết quả được cache để tăng tốc độ
- Có thể tương tác trực tiếp với các biểu đồ Plotly (zoom, pan, hover)

## Troubleshooting

Nếu gặp lỗi:
1. Đảm bảo đã cài đủ dependencies: `pip install -r requirements.txt`
2. Kiểm tra file dữ liệu có trong thư mục `data/`
3. Chạy notebook `1_data_exploration.ipynb` trước để tạo file processed


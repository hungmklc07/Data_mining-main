"""
Main script để chạy tất cả các phân tích Data Mining
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')

def main():
    print("="*70)
    print("DATA MINING PROJECT - PREMIER LEAGUE 2024-2025")
    print("="*70)
    print("\nDự án này thực hiện 5 kỹ thuật khai phá dữ liệu:")
    print("1. Association Rule Mining (FP-Growth)")
    print("2. Clustering (K-Means & Hierarchical)")
    print("3. Classification (Random Forest & Decision Tree)")
    print("4. Anomaly Detection (Isolation Forest & LOF)")
    print("5. Recommendation System (Content-based & Similarity-based)")
    print("\n" + "="*70)
    
    # Kiểm tra cấu trúc thư mục
    print("\n📁 Kiểm tra cấu trúc thư mục...")
    required_dirs = ['data', 'notebooks', 'src', 'results']
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            print(f"⚠️ Thư mục {dir_name} chưa tồn tại. Đang tạo...")
            os.makedirs(dir_name, exist_ok=True)
        else:
            print(f"✅ {dir_name}/")
    
    required_subdirs = [
        'results/association_rules',
        'results/clustering',
        'results/classification',
        'results/anomaly_detection',
        'results/recommendation_system'
    ]
    for dir_name in required_subdirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
    
    print("\n" + "="*70)
    print("HƯỚNG DẪN SỬ DỤNG:")
    print("="*70)
    print("\n1. Chạy các notebook theo thứ tự:")
    print("   - notebooks/1_data_exploration.ipynb")
    print("   - notebooks/2_association_rules.ipynb")
    print("   - notebooks/3_clustering.ipynb")
    print("   - notebooks/4_classification.ipynb")
    print("   - notebooks/5_anomaly_detection.ipynb")
    print("   - notebooks/6_recommendation_system.ipynb")
    print("\n2. Hoặc chạy từng notebook trong Jupyter:")
    print("   jupyter notebook notebooks/1_data_exploration.ipynb")
    print("\n3. Kết quả sẽ được lưu trong thư mục results/")
    print("\n" + "="*70)
    print("✅ Cấu trúc dự án đã sẵn sàng!")
    print("="*70)

if __name__ == "__main__":
    main()



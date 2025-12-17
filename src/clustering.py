"""
Module thực hiện Clustering (K-Means và Hierarchical)
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')

def select_features_for_clustering(df, feature_keywords=None, exclude_cols=None):
    """
    Chọn các features phù hợp cho clustering
    
    Parameters:
    -----------
    df : DataFrame
        Dữ liệu gốc
    feature_keywords : list
        Danh sách keywords để tìm features (mặc định: các chỉ số quan trọng)
    exclude_cols : list
        Danh sách cột cần loại bỏ
    
    Returns:
    --------
    List of feature columns
    """
    if exclude_cols is None:
        exclude_cols = ['Player', 'Nation', 'Pos', 'Squad', 'Born', 'Team', 'Age']
    
    if feature_keywords is None:
        feature_keywords = ['gls', 'ast', 'xg', 'xa', 'sh', 'sot', 'pass', 'tkl', 'touches', 
                           'prg', 'sca', 'gca', 'int', 'blocks', 'carries']
    
    # Lấy các cột số
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Lọc các cột có chứa keywords
    selected_cols = []
    for col in numeric_cols:
        if col not in exclude_cols:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in feature_keywords):
                # Loại bỏ các cột có '%' hoặc 'category' (đã được discretize)
                if '%' not in col and 'category' not in col_lower:
                    selected_cols.append(col)
    
    return selected_cols

def find_optimal_clusters(X, max_k=10, method='kmeans'):
    """
    Tìm số cụm tối ưu sử dụng Elbow Method và Silhouette Score
    
    Parameters:
    -----------
    X : array-like
        Dữ liệu đã scale
    max_k : int
        Số cụm tối đa để thử
    method : str
        'kmeans' hoặc 'hierarchical'
    
    Returns:
    --------
    dict với kết quả
    """
    inertias = []
    silhouette_scores = []
    davies_bouldin_scores = []
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        if method == 'kmeans':
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
        else:
            model = AgglomerativeClustering(n_clusters=k)
        
        labels = model.fit_predict(X)
        
        if method == 'kmeans':
            inertias.append(model.inertia_)
        
        silhouette_scores.append(silhouette_score(X, labels))
        davies_bouldin_scores.append(davies_bouldin_score(X, labels))
    
    # Tìm k tối ưu dựa trên silhouette score (cao nhất)
    optimal_k = k_range[np.argmax(silhouette_scores)]
    
    results = {
        'k_range': list(k_range),
        'inertias': inertias if method == 'kmeans' else None,
        'silhouette_scores': silhouette_scores,
        'davies_bouldin_scores': davies_bouldin_scores,
        'optimal_k': optimal_k
    }
    
    return results

def perform_kmeans_clustering(X, n_clusters=None, find_optimal=True, max_k=10):
    """
    Thực hiện K-Means clustering
    
    Parameters:
    -----------
    X : array-like
        Dữ liệu đã scale
    n_clusters : int
        Số cụm (nếu None sẽ tự tìm)
    find_optimal : bool
        Có tìm số cụm tối ưu không
    max_k : int
        Số cụm tối đa khi tìm optimal
    
    Returns:
    --------
    dict với kết quả
    """
    if find_optimal or n_clusters is None:
        print("🔍 Đang tìm số cụm tối ưu...")
        optimal_results = find_optimal_clusters(X, max_k=max_k, method='kmeans')
        n_clusters = optimal_results['optimal_k']
        print(f"✅ Số cụm tối ưu: {n_clusters} (Silhouette Score: {optimal_results['silhouette_scores'][n_clusters-2]:.3f})")
    else:
        optimal_results = None
    
    # Thực hiện clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    
    # Tính các metrics
    silhouette = silhouette_score(X, labels)
    davies_bouldin = davies_bouldin_score(X, labels)
    
    results = {
        'model': kmeans,
        'labels': labels,
        'n_clusters': n_clusters,
        'silhouette_score': silhouette,
        'davies_bouldin_score': davies_bouldin,
        'optimal_results': optimal_results,
        'centers': kmeans.cluster_centers_
    }
    
    return results

def perform_hierarchical_clustering(X, n_clusters=None, find_optimal=True, max_k=10, linkage='ward'):
    """
    Thực hiện Hierarchical Clustering
    
    Parameters:
    -----------
    X : array-like
        Dữ liệu đã scale
    n_clusters : int
        Số cụm
    find_optimal : bool
        Có tìm số cụm tối ưu không
    max_k : int
        Số cụm tối đa
    linkage : str
        Linkage method ('ward', 'complete', 'average')
    
    Returns:
    --------
    dict với kết quả
    """
    if find_optimal or n_clusters is None:
        print("🔍 Đang tìm số cụm tối ưu...")
        optimal_results = find_optimal_clusters(X, max_k=max_k, method='hierarchical')
        n_clusters = optimal_results['optimal_k']
        print(f"✅ Số cụm tối ưu: {n_clusters} (Silhouette Score: {optimal_results['silhouette_scores'][n_clusters-2]:.3f})")
    else:
        optimal_results = None
    
    # Thực hiện clustering
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = hierarchical.fit_predict(X)
    
    # Tính metrics
    silhouette = silhouette_score(X, labels)
    davies_bouldin = davies_bouldin_score(X, labels)
    
    results = {
        'model': hierarchical,
        'labels': labels,
        'n_clusters': n_clusters,
        'silhouette_score': silhouette,
        'davies_bouldin_score': davies_bouldin,
        'optimal_results': optimal_results
    }
    
    return results

def analyze_clusters(df, labels, feature_cols):
    """
    Phân tích đặc điểm của từng cụm
    
    Parameters:
    -----------
    df : DataFrame
        Dữ liệu gốc
    labels : array
        Cluster labels
    feature_cols : list
        Danh sách features đã sử dụng
    
    Returns:
    --------
    DataFrame với thống kê từng cụm
    """
    df_clustered = df.copy()
    df_clustered['Cluster'] = labels
    
    # Tính thống kê cho từng cụm
    cluster_stats = df_clustered.groupby('Cluster')[feature_cols].mean()
    
    # Thêm số lượng cầu thủ trong mỗi cụm
    cluster_counts = df_clustered['Cluster'].value_counts().sort_index()
    cluster_stats['Count'] = cluster_counts.values
    
    return cluster_stats

def reduce_dimensions_for_visualization(X, n_components=2):
    """
    Giảm chiều dữ liệu bằng PCA để visualize
    
    Parameters:
    -----------
    X : array-like
        Dữ liệu
    n_components : int
        Số chiều sau khi giảm
    
    Returns:
    --------
    X_reduced : array
        Dữ liệu đã giảm chiều
    pca : PCA object
    """
    pca = PCA(n_components=n_components, random_state=42)
    X_reduced = pca.fit_transform(X)
    
    return X_reduced, pca



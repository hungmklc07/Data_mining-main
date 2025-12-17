"""
Module thực hiện Anomaly Detection (Isolation Forest và LOF)
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def detect_anomalies_isolation_forest(X, contamination=0.1, random_state=42):
    """
    Phát hiện anomalies sử dụng Isolation Forest
    
    Parameters:
    -----------
    X : array-like
        Dữ liệu
    contamination : float
        Tỷ lệ outliers dự kiến (0-1)
    random_state : int
        Random seed
    
    Returns:
    --------
    dict với kết quả
    """
    # Scale dữ liệu
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Áp dụng Isolation Forest
    iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
    predictions = iso_forest.fit_predict(X_scaled)
    
    # -1 là outlier, 1 là inlier
    outlier_mask = predictions == -1
    
    results = {
        'model': iso_forest,
        'predictions': predictions,
        'outlier_mask': outlier_mask,
        'n_outliers': outlier_mask.sum(),
        'scaler': scaler
    }
    
    return results

def detect_anomalies_lof(X, n_neighbors=20, contamination=0.1):
    """
    Phát hiện anomalies sử dụng Local Outlier Factor (LOF)
    
    Parameters:
    -----------
    X : array-like
        Dữ liệu
    n_neighbors : int
        Số neighbors
    contamination : float
        Tỷ lệ outliers dự kiến
    
    Returns:
    --------
    dict với kết quả
    """
    # Scale dữ liệu
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Áp dụng LOF
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    predictions = lof.fit_predict(X_scaled)
    
    # -1 là outlier, 1 là inlier
    outlier_mask = predictions == -1
    
    results = {
        'model': lof,
        'predictions': predictions,
        'outlier_mask': outlier_mask,
        'n_outliers': outlier_mask.sum(),
        'outlier_scores': -lof.negative_outlier_factor_,  # Chuyển thành positive scores
        'scaler': scaler
    }
    
    return results

def analyze_player_anomalies(df, feature_keywords=None, contamination=0.1):
    """
    Phân tích anomalies cho cầu thủ
    
    Parameters:
    -----------
    df : DataFrame
        Dữ liệu cầu thủ
    feature_keywords : list
        Keywords để chọn features
    contamination : float
        Tỷ lệ outliers
    
    Returns:
    --------
    dict với kết quả
    """
    if feature_keywords is None:
        feature_keywords = ['gls', 'ast', 'xg', 'xa', 'sh', 'sot', 'pass', 'tkl', 'touches']
    
    # Chọn features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['Player', 'Nation', 'Pos', 'Squad', 'Born', 'Age', 'Team']
    
    feature_cols = []
    for col in numeric_cols:
        if col not in exclude_cols:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in feature_keywords):
                if '%' not in col and 'category' not in col_lower:
                    feature_cols.append(col)
    
    if len(feature_cols) == 0:
        print("⚠️ Không tìm thấy features phù hợp")
        return None
    
    # Chuẩn bị dữ liệu
    X = df[feature_cols].fillna(0)
    
    # Isolation Forest
    iso_results = detect_anomalies_isolation_forest(X, contamination=contamination)
    
    # LOF
    lof_results = detect_anomalies_lof(X, contamination=contamination)
    
    # Tạo dataframe với kết quả
    df_results = df.copy()
    df_results['IsolationForest_Outlier'] = iso_results['outlier_mask']
    df_results['LOF_Outlier'] = lof_results['outlier_mask']
    df_results['LOF_Score'] = lof_results['outlier_scores']
    
    # Cầu thủ được cả 2 methods phát hiện là outlier
    df_results['Both_Methods_Outlier'] = (iso_results['outlier_mask'] & lof_results['outlier_mask'])
    
    results = {
        'isolation_forest': iso_results,
        'lof': lof_results,
        'df_with_anomalies': df_results,
        'feature_cols': feature_cols
    }
    
    return results

def analyze_team_anomalies(teams_df, contamination=0.2):
    """
    Phân tích anomalies cho đội bóng
    
    Parameters:
    -----------
    teams_df : DataFrame
        Dữ liệu đội bóng
    contamination : float
        Tỷ lệ outliers
    
    Returns:
    --------
    dict với kết quả
    """
    # Chọn features
    feature_keywords = ['gf', 'ga', 'xg', 'xga', 'pts', 'gd']
    numeric_cols = teams_df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['Squad', 'Team']
    
    feature_cols = []
    for col in numeric_cols:
        if col not in exclude_cols:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in feature_keywords):
                if '%' not in col and 'category' not in col_lower:
                    feature_cols.append(col)
    
    if len(feature_cols) == 0:
        print("⚠️ Không tìm thấy features phù hợp")
        return None
    
    # Chuẩn bị dữ liệu
    X = teams_df[feature_cols].fillna(0)
    
    # Isolation Forest
    iso_results = detect_anomalies_isolation_forest(X, contamination=contamination)
    
    # LOF
    lof_results = detect_anomalies_lof(X, contamination=contamination)
    
    # Tạo dataframe với kết quả
    df_results = teams_df.copy()
    df_results['IsolationForest_Outlier'] = iso_results['outlier_mask']
    df_results['LOF_Outlier'] = lof_results['outlier_mask']
    df_results['LOF_Score'] = lof_results['outlier_scores']
    df_results['Both_Methods_Outlier'] = (iso_results['outlier_mask'] & lof_results['outlier_mask'])
    
    results = {
        'isolation_forest': iso_results,
        'lof': lof_results,
        'df_with_anomalies': df_results,
        'feature_cols': feature_cols
    }
    
    return results



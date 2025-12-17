"""
Module thực hiện Classification (Random Forest và Decision Tree)
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def prepare_classification_data(df, target_col, feature_keywords=None, exclude_cols=None):
    """
    Chuẩn bị dữ liệu cho classification
    
    Parameters:
    -----------
    df : DataFrame
        Dữ liệu gốc
    target_col : str
        Tên cột target
    feature_keywords : list
        Keywords để chọn features
    exclude_cols : list
        Các cột cần loại bỏ
    
    Returns:
    --------
    X, y : Features và target
    feature_names : List tên features
    """
    if exclude_cols is None:
        exclude_cols = ['Player', 'Nation', 'Squad', 'Born', 'Team']
    
    if feature_keywords is None:
        feature_keywords = ['gls', 'ast', 'xg', 'xa', 'sh', 'sot', 'pass', 'tkl', 'touches', 
                           'prg', 'sca', 'gca', 'int', 'blocks', 'carries']
    
    # Chọn features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = []
    
    for col in numeric_cols:
        if col not in exclude_cols and col != target_col:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in feature_keywords):
                if '%' not in col and 'category' not in col_lower:
                    feature_cols.append(col)
    
    # Lọc dữ liệu
    df_clean = df[[target_col] + feature_cols].dropna()
    
    X = df_clean[feature_cols].fillna(0)
    y = df_clean[target_col]
    
    return X, y, feature_cols

def classify_player_position(df, min_samples_per_class=10):
    """
    Phân loại vị trí cầu thủ dựa trên các chỉ số
    
    Parameters:
    -----------
    df : DataFrame
        Dữ liệu cầu thủ
    min_samples_per_class : int
        Số mẫu tối thiểu mỗi class
    
    Returns:
    --------
    dict với kết quả
    """
    if 'Pos' not in df.columns:
        print("⚠️ Không có cột vị trí (Pos)")
        return None
    
    # Lọc các vị trí có đủ mẫu
    pos_counts = df['Pos'].value_counts()
    valid_positions = pos_counts[pos_counts >= min_samples_per_class].index.tolist()
    
    if len(valid_positions) == 0:
        print("⚠️ Không có vị trí nào có đủ mẫu")
        return None
    
    df_filtered = df[df['Pos'].isin(valid_positions)].copy()
    print(f"📊 Phân loại vị trí với {len(valid_positions)} classes: {valid_positions}")
    
    # Chuẩn bị dữ liệu
    X, y, feature_cols = prepare_classification_data(df_filtered, 'Pos')
    
    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    # Train models
    results = {}
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    
    results['random_forest'] = {
        'model': rf,
        'predictions': y_pred_rf,
        'y_test': y_test,
        'y_train': y_train,
        'label_encoder': le,
        'feature_names': feature_cols
    }
    
    # Decision Tree
    dt = DecisionTreeClassifier(random_state=42, max_depth=10)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    
    results['decision_tree'] = {
        'model': dt,
        'predictions': y_pred_dt,
        'y_test': y_test,
        'y_train': y_train,
        'label_encoder': le,
        'feature_names': feature_cols
    }
    
    return results

def classify_team_top4(teams_df):
    """
    Phân loại đội bóng vào Top 4
    
    Parameters:
    -----------
    teams_df : DataFrame
        Dữ liệu đội bóng
    
    Returns:
    --------
    dict với kết quả
    """
    # Tìm cột Points
    pts_cols = [c for c in teams_df.columns if 'pts' in c.lower() and 'category' not in c.lower()]
    
    if len(pts_cols) == 0:
        print("⚠️ Không tìm thấy cột Points")
        return None
    
    pts_col = pts_cols[0]
    
    # Tạo target: Top 4 (1) hoặc không (0)
    teams_df = teams_df.copy()
    teams_df['Top4'] = (teams_df[pts_col].rank(ascending=False) <= 4).astype(int)
    
    print(f"📊 Phân loại Top 4: {teams_df['Top4'].sum()} đội trong Top 4")
    
    # Chuẩn bị dữ liệu
    feature_keywords = ['gf', 'ga', 'xg', 'xga', 'pts', 'gd']
    X, y, feature_cols = prepare_classification_data(teams_df, 'Top4', feature_keywords=feature_keywords)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Train models
    results = {}
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    
    results['random_forest'] = {
        'model': rf,
        'predictions': y_pred_rf,
        'y_test': y_test,
        'feature_names': feature_cols
    }
    
    # Decision Tree
    dt = DecisionTreeClassifier(random_state=42, max_depth=5)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    
    results['decision_tree'] = {
        'model': dt,
        'predictions': y_pred_dt,
        'y_test': y_test,
        'feature_names': feature_cols
    }
    
    return results

def classify_player_performance(df):
    """
    Phân loại hiệu suất cầu thủ: Elite, Good, Average, Below Average
    
    Parameters:
    -----------
    df : DataFrame
        Dữ liệu cầu thủ
    
    Returns:
    --------
    dict với kết quả
    """
    # Tính điểm tổng hợp
    score_cols = []
    for keyword in ['gls', 'ast', 'xg', 'xa']:
        cols = [c for c in df.columns if keyword in c.lower() and 'category' not in c.lower() and 'per' not in c.lower()]
        if cols:
            score_cols.append(cols[0])
    
    if len(score_cols) == 0:
        print("⚠️ Không tìm thấy các chỉ số phù hợp")
        return None
    
    df = df.copy()
    df['Performance_Score'] = df[score_cols].sum(axis=1)
    
    # Phân loại dựa trên quantiles
    q1 = df['Performance_Score'].quantile(0.25)
    q2 = df['Performance_Score'].quantile(0.50)
    q3 = df['Performance_Score'].quantile(0.75)
    
    def classify_perf(score):
        if score >= q3:
            return 'Elite'
        elif score >= q2:
            return 'Good'
        elif score >= q1:
            return 'Average'
        else:
            return 'Below_Average'
    
    df['Performance_Class'] = df['Performance_Score'].apply(classify_perf)
    
    print(f"📊 Phân loại hiệu suất:")
    print(df['Performance_Class'].value_counts())
    
    # Chuẩn bị dữ liệu
    feature_keywords = ['gls', 'ast', 'xg', 'xa', 'sh', 'sot', 'pass', 'tkl']
    X, y, feature_cols = prepare_classification_data(df, 'Performance_Class', feature_keywords=feature_keywords)
    
    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    # Train models
    results = {}
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    
    results['random_forest'] = {
        'model': rf,
        'predictions': y_pred_rf,
        'y_test': y_test,
        'label_encoder': le,
        'feature_names': feature_cols
    }
    
    # Decision Tree
    dt = DecisionTreeClassifier(random_state=42, max_depth=10)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    
    results['decision_tree'] = {
        'model': dt,
        'predictions': y_pred_dt,
        'y_test': y_test,
        'label_encoder': le,
        'feature_names': feature_cols
    }
    
    return results

def evaluate_classification(results, model_name):
    """
    Đánh giá kết quả classification
    
    Parameters:
    -----------
    results : dict
        Kết quả từ model
    model_name : str
        Tên model
    
    Returns:
    --------
    dict với các metrics
    """
    y_test = results['y_test']
    y_pred = results['predictions']
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }
    
    return metrics

def get_feature_importance(model, feature_names, top_n=15):
    """
    Lấy feature importance từ model
    
    Parameters:
    -----------
    model : Model object
    feature_names : list
        Tên các features
    top_n : int
        Số features top
    
    Returns:
    --------
    DataFrame với feature importance
    """
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)
        return importance_df
    return None



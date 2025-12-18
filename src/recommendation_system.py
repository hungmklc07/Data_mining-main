"""
Module thực hiện Recommendation System cho bóng đá
"""
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

def prepare_player_features(df, feature_keywords=None):
    """
    Chuẩn bị features cho recommendation system
    
    Parameters:
    -----------
    df : DataFrame
        Dữ liệu cầu thủ
    feature_keywords : list
        Keywords để chọn features
    
    Returns:
    --------
    X_scaled : array
        Features đã được scale
    feature_cols : list
        Tên các features
    """
    if feature_keywords is None:
        feature_keywords = ['gls', 'ast', 'xg', 'xa', 'sh', 'sot', 'pass', 'tkl', 'touches', 
                           'prg', 'sca', 'gca', 'int', 'blocks', 'carries']
    
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
        return None, None
    
    # Chuẩn bị dữ liệu
    X = df[feature_cols].fillna(0)
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, feature_cols, scaler

def find_similar_players(df, player_name, n_recommendations=5, same_position=True):
    """
    Tìm cầu thủ tương tự dựa trên chỉ số
    
    Parameters:
    -----------
    df : DataFrame
        Dữ liệu cầu thủ
    player_name : str
        Tên cầu thủ cần tìm tương tự
    n_recommendations : int
        Số lượng gợi ý
    same_position : bool
        Chỉ tìm cầu thủ cùng vị trí
    
    Returns:
    --------
    DataFrame với các cầu thủ tương tự
    """
    if 'Player' not in df.columns:
        print("⚠️ Không có cột Player")
        return None
    
    # Tìm cầu thủ
    player_mask = df['Player'].str.contains(player_name, case=False, na=False)
    if not player_mask.any():
        print(f"⚠️ Không tìm thấy cầu thủ: {player_name}")
        return None
    
    player_idx = df[player_mask].index[0]
    player_data = df.iloc[player_idx]
    
    # Chuẩn bị features
    X_scaled, feature_cols, scaler = prepare_player_features(df)
    if X_scaled is None:
        return None
    
    # Tính similarity
    player_vector = X_scaled[player_idx].reshape(1, -1)
    similarities = cosine_similarity(player_vector, X_scaled)[0]
    
    # Tạo DataFrame với similarities
    df_similar = df.copy()
    df_similar['Similarity'] = similarities
    
    # Lọc: loại bỏ chính cầu thủ đó
    df_similar = df_similar[df_similar.index != player_idx]
    
    # Lọc theo vị trí nếu cần
    if same_position and 'Pos' in df.columns:
        player_pos = player_data['Pos']
        if pd.notna(player_pos):
            df_similar = df_similar[df_similar['Pos'] == player_pos]
    
    # Sắp xếp theo similarity
    df_similar = df_similar.sort_values('Similarity', ascending=False)
    
    # Lấy top recommendations
    recommendations = df_similar.head(n_recommendations).copy()
    
    # Thêm thông tin hữu ích
    result_cols = ['Player', 'Pos', 'Squad', 'Similarity']
    if 'shooting_Standard_Gls' in recommendations.columns:
        result_cols.append('shooting_Standard_Gls')
    if 'passing_Ast' in recommendations.columns:
        result_cols.append('passing_Ast')
    
    available_cols = [c for c in result_cols if c in recommendations.columns]
    recommendations = recommendations[available_cols]
    
    return recommendations, player_data

def recommend_players_by_team_needs(df, team_name, position=None, n_recommendations=5):
    """
    Gợi ý cầu thủ cho đội bóng dựa trên nhu cầu
    
    Parameters:
    -----------
    df : DataFrame
        Dữ liệu cầu thủ
    team_name : str
        Tên đội bóng
    position : str
        Vị trí cần tìm (FW, MF, DF, GK)
    n_recommendations : int
        Số lượng gợi ý
    
    Returns:
    --------
    DataFrame với các cầu thủ được gợi ý
    """
    # Lọc cầu thủ không thuộc đội này
    if 'Squad' in df.columns:
        df_available = df[df['Squad'] != team_name].copy()
    else:
        df_available = df.copy()
    
    # Lọc theo vị trí nếu có
    if position and 'Pos' in df_available.columns:
        df_available = df_available[df_available['Pos'] == position]
    
    # Tính điểm dựa trên các chỉ số quan trọng
    score_cols = []
    
    # Goals
    goal_cols = [c for c in df_available.columns if 'gls' in c.lower() and 'category' not in c.lower()]
    if goal_cols:
        score_cols.append(goal_cols[0])
    
    # Assists
    assist_cols = [c for c in df_available.columns if 'ast' in c.lower() and 'category' not in c.lower()]
    if assist_cols:
        score_cols.append(assist_cols[0])
    
    # xG
    xg_cols = [c for c in df_available.columns if 'xg' in c.lower() and 'xga' not in c.lower() and 'category' not in c.lower()]
    if xg_cols:
        score_cols.append(xg_cols[0])
    
    # xA
    xa_cols = [c for c in df_available.columns if 'xa' in c.lower() and 'category' not in c.lower()]
    if xa_cols:
        score_cols.append(xa_cols[0])
    
    if len(score_cols) == 0:
        print("⚠️ Không tìm thấy các chỉ số phù hợp")
        return None
    
    # Tính điểm tổng hợp
    df_available['Recommendation_Score'] = df_available[score_cols].sum(axis=1)
    
    # Sắp xếp theo điểm
    recommendations = df_available.nlargest(n_recommendations, 'Recommendation_Score').copy()
    
    # Chọn cột hiển thị
    result_cols = ['Player', 'Pos', 'Squad', 'Recommendation_Score'] + score_cols[:4]
    available_cols = [c for c in result_cols if c in recommendations.columns]
    recommendations = recommendations[available_cols]
    
    return recommendations

def recommend_players_by_style(df, target_features_dict, n_recommendations=5):
    """
    Gợi ý cầu thủ dựa trên phong cách chơi mong muốn
    
    Parameters:
    -----------
    df : DataFrame
        Dữ liệu cầu thủ
    target_features_dict : dict
        Dictionary với các chỉ số mong muốn
        Ví dụ: {'shooting_Standard_Gls': 10, 'passing_Ast': 5}
    n_recommendations : int
        Số lượng gợi ý
    
    Returns:
    --------
    DataFrame với các cầu thủ được gợi ý
    """
    # Chuẩn bị features
    X_scaled, feature_cols, scaler = prepare_player_features(df)
    if X_scaled is None:
        return None
    
    # Tạo vector mục tiêu
    target_vector = np.zeros(len(feature_cols))
    for feature, value in target_features_dict.items():
        if feature in feature_cols:
            idx = feature_cols.index(feature)
            target_vector[idx] = value
    
    # Scale vector mục tiêu
    target_vector_scaled = scaler.transform(target_vector.reshape(1, -1))
    
    # Tính similarity
    similarities = cosine_similarity(target_vector_scaled, X_scaled)[0]
    
    # Tạo DataFrame
    df_recommend = df.copy()
    df_recommend['Similarity'] = similarities
    
    # Sắp xếp
    recommendations = df_recommend.nlargest(n_recommendations, 'Similarity').copy()
    
    # Chọn cột hiển thị
    result_cols = ['Player', 'Pos', 'Squad', 'Similarity'] + list(target_features_dict.keys())
    available_cols = [c for c in result_cols if c in recommendations.columns]
    recommendations = recommendations[available_cols]
    
    return recommendations

def create_player_profile(df, player_name):
    """
    Tạo profile chi tiết của cầu thủ
    
    Parameters:
    -----------
    df : DataFrame
        Dữ liệu cầu thủ
    player_name : str
        Tên cầu thủ
    
    Returns:
    --------
    dict với thông tin profile
    """
    player_mask = df['Player'].str.contains(player_name, case=False, na=False)
    if not player_mask.any():
        return None
    
    player_data = df[player_mask].iloc[0]
    
    profile = {
        'Player': player_data.get('Player', 'N/A'),
        'Position': player_data.get('Pos', 'N/A'),
        'Team': player_data.get('Squad', 'N/A'),
        'Age': player_data.get('Age', 'N/A'),
        'Nation': player_data.get('Nation', 'N/A')
    }
    
    # Thêm các chỉ số quan trọng
    important_stats = {}
    stat_mapping = {
        'Goals': ['gls', 'goals'],
        'Assists': ['ast', 'assist'],
        'xG': ['xg'],
        'xA': ['xa'],
        'Shots': ['sh'],
        'Shots on Target': ['sot']
    }
    
    for stat_name, keywords in stat_mapping.items():
        for col in df.columns:
            col_lower = col.lower()
            if any(kw in col_lower for kw in keywords) and 'category' not in col_lower:
                if col in player_data.index:
                    value = player_data[col]
                    if pd.notna(value) and value != 0:
                        important_stats[stat_name] = value
                        break
    
    profile['Stats'] = important_stats
    
    return profile



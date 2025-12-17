"""
Module xử lý và tiền xử lý dữ liệu Premier League
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load tất cả các file dữ liệu đã clean"""
    # Sửa đường dẫn để tìm từ thư mục gốc của project
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data")
    
    players_df = pd.read_excel(os.path.join(data_path, "Cleaned_FBref_Premier-League_2024-2025_Full_Merged.xlsx"))
    keepers_df = pd.read_excel(os.path.join(data_path, "Cleaned_PL_2024-2025_Keepers_Full.xlsx"))
    teams_for_df = pd.read_excel(os.path.join(data_path, "Cleaned_PL_2024-2025_Teams_For.xlsx"))
    teams_vs_df = pd.read_excel(os.path.join(data_path, "Cleaned_PL_2024-2025_Teams_VS.xlsx"))
    
    return {
        'players': players_df,
        'keepers': keepers_df,
        'teams_for': teams_for_df,
        'teams_vs': teams_vs_df
    }

def explore_data(df, name="Dataset"):
    """Khám phá dữ liệu cơ bản"""
    print(f"\n{'='*60}")
    print(f"EXPLORING: {name}")
    print(f"{'='*60}")
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\nMissing values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Percentage': missing_pct
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
    if len(missing_df) > 0:
        print(missing_df.head(10))
    else:
        print("No missing values!")
    
    print(f"\nData types:")
    print(df.dtypes.value_counts())
    
    return missing_df

def get_numeric_columns(df, exclude_cols=None):
    """Lấy danh sách các cột số"""
    if exclude_cols is None:
        exclude_cols = []
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Loại bỏ các cột không phải số thực sự
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]
    return numeric_cols

def get_categorical_columns(df, exclude_cols=None):
    """Lấy danh sách các cột phân loại"""
    if exclude_cols is None:
        exclude_cols = []
    
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in exclude_cols]
    return cat_cols

def feature_engineering_players(df):
    """Tạo các features mới cho cầu thủ"""
    df = df.copy()
    
    # Tính toán các chỉ số per 90 minutes
    if '90s_shooting' in df.columns and df['90s_shooting'].sum() > 0:
        # Goals per 90
        if 'shooting_Standard_Gls' in df.columns:
            df['Goals_per_90'] = df['shooting_Standard_Gls'] / df['90s_shooting'].replace(0, np.nan)
        
        # Assists per 90
        if 'passing_Ast' in df.columns:
            df['Assists_per_90'] = df['passing_Ast'] / df['90s_passing'].replace(0, np.nan)
        
        # xG per 90
        if 'shooting_Expected_xG' in df.columns:
            df['xG_per_90'] = df['shooting_Expected_xG'] / df['90s_shooting'].replace(0, np.nan)
        
        # xA per 90
        if 'passing_Expected_xA' in df.columns:
            df['xA_per_90'] = df['passing_Expected_xA'] / df['90s_passing'].replace(0, np.nan)
    
    # Tính toán hiệu suất finishing
    if 'shooting_Standard_Gls' in df.columns and 'shooting_Expected_xG' in df.columns:
        df['Finishing_Efficiency'] = df['shooting_Standard_Gls'] - df['shooting_Expected_xG']
    
    # Tính toán hiệu suất tạo cơ hội
    if 'passing_Ast' in df.columns and 'passing_Expected_xA' in df.columns:
        df['Assist_Efficiency'] = df['passing_Ast'] - df['passing_Expected_xA']
    
    # Tổng hợp chỉ số tấn công
    attack_cols = []
    if 'shooting_Standard_Gls' in df.columns:
        attack_cols.append('shooting_Standard_Gls')
    if 'passing_Ast' in df.columns:
        attack_cols.append('passing_Ast')
    if 'shooting_Expected_xG' in df.columns:
        attack_cols.append('shooting_Expected_xG')
    if 'passing_Expected_xA' in df.columns:
        attack_cols.append('passing_Expected_xA')
    
    if len(attack_cols) > 0:
        df['Total_Attack_Contribution'] = df[attack_cols].sum(axis=1)
    
    # Tổng hợp chỉ số phòng thủ
    defense_cols = []
    if 'defense_Tackles_Tkl' in df.columns:
        defense_cols.append('defense_Tackles_Tkl')
    if 'defense_Int' in df.columns:
        defense_cols.append('defense_Int')
    if 'defense_Blocks_Blocks' in df.columns:
        defense_cols.append('defense_Blocks_Blocks')
    
    if len(defense_cols) > 0:
        df['Total_Defense_Contribution'] = df[defense_cols].sum(axis=1)
    
    return df

def feature_engineering_teams(df_for, df_vs):
    """Tạo các features mới cho đội bóng"""
    df_for = df_for.copy()
    df_vs = df_vs.copy()
    
    # Merge For và VS
    if 'Squad' in df_for.columns and 'Squad' in df_vs.columns:
        df_merged = pd.merge(df_for, df_vs, on='Squad', how='inner', suffixes=('_For', '_VS'))
        
        # Tính toán goal difference
        if 'GF' in df_merged.columns and 'GA' in df_merged.columns:
            df_merged['GD'] = df_merged['GF'] - df_merged['GA']
        
        # Tính toán xG difference
        if 'xG' in df_merged.columns and 'xGA' in df_merged.columns:
            df_merged['xGD'] = df_merged['xG'] - df_merged['xGA']
        
        # Tính toán hiệu suất (Points vs Expected)
        # Có thể tính dựa trên xG và xGA
        
        return df_merged
    
    return df_for

def prepare_data_for_analysis(df, target_cols=None):
    """Chuẩn bị dữ liệu cho phân tích"""
    df = df.copy()
    
    # Loại bỏ các cột không cần thiết
    exclude_cols = ['Player', 'Nation', 'Born', 'Squad', 'Team']
    if target_cols:
        exclude_cols = [c for c in exclude_cols if c not in target_cols]
    
    # Xử lý missing values - điền bằng median cho số, mode cho phân loại
    numeric_cols = get_numeric_columns(df, exclude_cols)
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    cat_cols = get_categorical_columns(df, exclude_cols)
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown', inplace=True)
    
    return df



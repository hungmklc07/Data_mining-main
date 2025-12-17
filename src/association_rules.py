"""
Module thực hiện Association Rule Mining sử dụng FP-Growth
"""
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
warnings.filterwarnings('ignore')

def discretize_continuous_features(df, feature_cols, n_bins=3, labels=None):
    """
    Chuyển đổi các biến liên tục thành các itemset rời rạc
    
    Parameters:
    -----------
    df : DataFrame
        Dữ liệu gốc
    feature_cols : list
        Danh sách các cột cần discretize
    n_bins : int
        Số lượng bins (mặc định 3: Low, Medium, High)
    labels : list
        Tên labels cho các bins (mặc định: ['Low', 'Medium', 'High'])
    
    Returns:
    --------
    DataFrame với các cột đã được discretize
    """
    df_discrete = df.copy()
    
    if labels is None:
        labels = ['Low', 'Medium', 'High']
    
    for col in feature_cols:
        if col in df_discrete.columns:
            # Bỏ qua các giá trị NaN hoặc 0
            non_zero = df_discrete[col].replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(non_zero) > 0:
                # Tính quantiles
                q1 = non_zero.quantile(0.33)
                q2 = non_zero.quantile(0.67)
                
                # Discretize
                conditions = [
                    df_discrete[col] <= q1,
                    (df_discrete[col] > q1) & (df_discrete[col] <= q2),
                    df_discrete[col] > q2
                ]
                df_discrete[col + '_category'] = np.select(conditions, labels, default='Low')
            else:
                df_discrete[col + '_category'] = 'Low'
    
    return df_discrete

def create_transaction_dataset(df, feature_cols):
    """
    Tạo transaction dataset từ DataFrame đã discretize
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame đã discretize
    feature_cols : list
        Danh sách các cột category cần sử dụng
    
    Returns:
    --------
    List of transactions
    """
    transactions = []
    
    for idx, row in df.iterrows():
        transaction = []
        for col in feature_cols:
            if col in df.columns:
                value = str(row[col])
                if pd.notna(value) and value != 'nan' and value != '':
                    # Tạo item dạng "FeatureName=Value"
                    item = f"{col}={value}"
                    transaction.append(item)
        if len(transaction) > 0:
            transactions.append(transaction)
    
    return transactions

def apply_fpgrowth(transactions, min_support=0.1):
    """
    Áp dụng FP-Growth để tìm frequent itemsets
    
    Parameters:
    -----------
    transactions : list
        List of transactions
    min_support : float
        Minimum support threshold (0-1)
    
    Returns:
    --------
    DataFrame với frequent itemsets
    """
    # Chuyển đổi transactions thành format phù hợp với mlxtend
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_transactions = pd.DataFrame(te_ary, columns=te.columns_)
    
    # Áp dụng FP-Growth
    frequent_itemsets = fpgrowth(df_transactions, min_support=min_support, use_colnames=True)
    
    return frequent_itemsets, df_transactions

def generate_rules(frequent_itemsets, metric="confidence", min_threshold=0.6):
    """
    Tạo association rules từ frequent itemsets
    
    Parameters:
    -----------
    frequent_itemsets : DataFrame
        Frequent itemsets từ FP-Growth
    metric : str
        Metric để đánh giá rules (confidence, lift, etc.)
    min_threshold : float
        Minimum threshold cho metric
    
    Returns:
    --------
    DataFrame với association rules
    """
    if len(frequent_itemsets) == 0:
        return pd.DataFrame()
    
    rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)
    
    # Sắp xếp theo confidence và lift
    if len(rules) > 0:
        rules = rules.sort_values(['confidence', 'lift'], ascending=False)
    
    return rules

def analyze_player_performance_patterns(df, min_support=0.15, min_confidence=0.6):
    """
    Phân tích mẫu chỉ số cầu thủ dẫn đến hiệu suất cao
    
    Parameters:
    -----------
    df : DataFrame
        Dữ liệu cầu thủ
    min_support : float
        Minimum support
    min_confidence : float
        Minimum confidence
    
    Returns:
    --------
    Tuple: (frequent_itemsets, rules, discretized_df)
    """
    # Chọn các features quan trọng
    feature_cols = []
    
    # Tìm các cột liên quan đến goals, assists, xG, xA
    goal_cols = [c for c in df.columns if any(kw in c.lower() for kw in ['gls', 'goals']) and 'category' not in c.lower()]
    assist_cols = [c for c in df.columns if any(kw in c.lower() for kw in ['ast', 'assist']) and 'category' not in c.lower()]
    xg_cols = [c for c in df.columns if 'xg' in c.lower() and 'category' not in c.lower() and 'per' not in c.lower()]
    xa_cols = [c for c in df.columns if 'xa' in c.lower() and 'category' not in c.lower() and 'per' not in c.lower()]
    sot_cols = [c for c in df.columns if 'sot' in c.lower() and '%' in c.lower() and 'category' not in c.lower()]
    
    # Chọn cột đầu tiên tìm được cho mỗi loại
    if goal_cols:
        feature_cols.append(goal_cols[0])
    if assist_cols:
        feature_cols.append(assist_cols[0])
    if xg_cols:
        feature_cols.append(xg_cols[0])
    if xa_cols:
        feature_cols.append(xa_cols[0])
    if sot_cols:
        feature_cols.append(sot_cols[0])
    
    # Thêm Position nếu có
    if 'Pos' in df.columns:
        feature_cols.append('Pos')
    
    # Lọc các cột có trong DataFrame
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    if len(feature_cols) == 0:
        print("⚠️ Không tìm thấy các cột phù hợp để phân tích")
        return None, None, None
    
    print(f"📊 Đang phân tích với {len(feature_cols)} features: {feature_cols[:5]}...")
    
    # Discretize
    numeric_cols = [c for c in feature_cols if c != 'Pos']
    df_discrete = discretize_continuous_features(df, numeric_cols, n_bins=3)
    
    # Tạo category columns list
    category_cols = [c + '_category' for c in numeric_cols]
    if 'Pos' in feature_cols:
        category_cols.append('Pos')
    
    # Tạo transactions
    transactions = create_transaction_dataset(df_discrete, category_cols)
    
    if len(transactions) == 0:
        print("⚠️ Không tạo được transactions")
        return None, None, None
    
    # Áp dụng FP-Growth
    frequent_itemsets, df_transactions = apply_fpgrowth(transactions, min_support=min_support)
    
    if len(frequent_itemsets) == 0:
        print("⚠️ Không tìm thấy frequent itemsets với min_support này. Hãy thử giảm min_support.")
        return None, None, None
    
    # Generate rules
    rules = generate_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    
    return frequent_itemsets, rules, df_discrete

def analyze_team_patterns(teams_df, min_support=0.3, min_confidence=0.7):
    """
    Phân tích mẫu tấn công/phòng thủ của đội bóng
    
    Parameters:
    -----------
    teams_df : DataFrame
        Dữ liệu đội bóng
    min_support : float
        Minimum support
    min_confidence : float
        Minimum confidence
    
    Returns:
    --------
    Tuple: (frequent_itemsets, rules, discretized_df)
    """
    # Chọn các features quan trọng
    feature_cols = []
    
    # Tìm các cột liên quan
    gf_cols = [c for c in teams_df.columns if 'gf' in c.lower() and 'category' not in c.lower()]
    ga_cols = [c for c in teams_df.columns if 'ga' in c.lower() and 'category' not in c.lower()]
    xg_cols = [c for c in teams_df.columns if 'xg' in c.lower() and 'xga' not in c.lower() and 'category' not in c.lower()]
    xga_cols = [c for c in teams_df.columns if 'xga' in c.lower() and 'category' not in c.lower()]
    pts_cols = [c for c in teams_df.columns if 'pts' in c.lower() and 'category' not in c.lower()]
    
    # Chọn cột đầu tiên tìm được
    for col_list in [gf_cols, ga_cols, xg_cols, xga_cols, pts_cols]:
        if col_list:
            feature_cols.append(col_list[0])
    
    feature_cols = [c for c in feature_cols if c in teams_df.columns]
    
    if len(feature_cols) == 0:
        print("⚠️ Không tìm thấy các cột phù hợp")
        return None, None, None
    
    print(f"📊 Đang phân tích đội bóng với {len(feature_cols)} features...")
    
    # Discretize
    df_discrete = discretize_continuous_features(teams_df, feature_cols, n_bins=3)
    
    # Tạo category columns
    category_cols = [c + '_category' for c in feature_cols]
    
    # Tạo transactions
    transactions = create_transaction_dataset(df_discrete, category_cols)
    
    if len(transactions) == 0:
        return None, None, None
    
    # Áp dụng FP-Growth
    frequent_itemsets, df_transactions = apply_fpgrowth(transactions, min_support=min_support)
    
    if len(frequent_itemsets) == 0:
        print("⚠️ Không tìm thấy frequent itemsets. Hãy thử giảm min_support.")
        return None, None, None
    
    # Generate rules
    rules = generate_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    
    return frequent_itemsets, rules, df_discrete



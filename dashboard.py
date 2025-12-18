"""
Streamlit Dashboard cho Data Mining Project - Premier League 2024-2025
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Thêm src vào path
sys.path.append('src')

# Import các modules
from data_preprocessing import load_data, feature_engineering_players, prepare_data_for_analysis
from association_rules import analyze_player_performance_patterns
from clustering import (
    select_features_for_clustering, perform_kmeans_clustering,
    perform_hierarchical_clustering, reduce_dimensions_for_visualization,
    analyze_clusters
)
from classification import (
    classify_player_position, evaluate_classification,
    get_feature_importance, classify_team_top4, classify_player_performance
)
from anomaly_detection import analyze_player_anomalies, analyze_team_anomalies
from recommendation_system import (
    find_similar_players, recommend_players_by_team_needs,
    recommend_players_by_style, create_player_profile
)

# Cấu hình trang
st.set_page_config(
    page_title="Premier League Data Mining Dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_all_data():
    """Load và cache dữ liệu"""
    try:
        players_df = pd.read_excel('data/players_processed.xlsx')
        teams_df = pd.read_excel('data/teams_processed.xlsx')
    except:
        data = load_data()
        players_df = feature_engineering_players(data['players'])
        players_df = prepare_data_for_analysis(players_df)
        
        from data_preprocessing import feature_engineering_teams
        teams_merged = feature_engineering_teams(data['teams_for'], data['teams_vs'])
        if teams_merged is not None:
            teams_df = prepare_data_for_analysis(teams_merged, target_cols=['Squad'])
        else:
            teams_df = None
    
    return players_df, teams_df

# Load dữ liệu
players_df, teams_df = load_all_data()

# Sidebar navigation
st.sidebar.title("⚽ Premier League Data Mining")
st.sidebar.markdown("---")

page = st.sidebar.selectbox(
    "Chọn phần demo:",
    [
        "📊 Overview",
        "🔗 Association Rules",
        "🎯 Clustering",
        "📈 Classification",
        "🚨 Anomaly Detection",
        "⭐ Recommendation System"
    ]
)

# ==================== OVERVIEW PAGE ====================
if page == "📊 Overview":
    st.markdown('<h1 class="main-header">📊 Data Overview</h1>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Players", len(players_df))
    with col2:
        st.metric("Total Teams", players_df['Squad'].nunique() if 'Squad' in players_df.columns else 0)
    with col3:
        st.metric("Total Features", len(players_df.columns))
    with col4:
        if 'Pos' in players_df.columns:
            st.metric("Positions", players_df['Pos'].nunique())
    
    st.markdown("---")
    
    # Phân bố vị trí
    if 'Pos' in players_df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Phân bố vị trí cầu thủ")
            pos_counts = players_df['Pos'].value_counts()
            fig = px.pie(values=pos_counts.values, names=pos_counts.index, 
                        title="Position Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Top 10 đội bóng (số cầu thủ)")
            if 'Squad' in players_df.columns:
                squad_counts = players_df['Squad'].value_counts().head(10)
                fig = px.bar(x=squad_counts.values, y=squad_counts.index, 
                           orientation='h', title="Players per Team")
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
    
    # Thống kê mô tả
    st.subheader("Thống kê mô tả - Các chỉ số quan trọng")
    numeric_cols = players_df.select_dtypes(include=[np.number]).columns.tolist()
    important_cols = [c for c in numeric_cols if any(kw in c.lower() 
        for kw in ['gls', 'ast', 'xg', 'xa', 'sh', 'sot'])][:10]
    
    if important_cols:
        st.dataframe(players_df[important_cols].describe(), use_container_width=True)

# ==================== ASSOCIATION RULES PAGE ====================
elif page == "🔗 Association Rules":
    st.markdown('<h1 class="main-header">🔗 Association Rule Mining</h1>', unsafe_allow_html=True)
    
    st.info("Sử dụng FP-Growth để tìm các mẫu kết hợp giữa các chỉ số cầu thủ")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        min_support = st.slider("Min Support", 0.05, 0.5, 0.15, 0.05, key="ar_min_support")
        min_confidence = st.slider("Min Confidence", 0.3, 0.9, 0.6, 0.05, key="ar_min_confidence")
        run_analysis = st.button("🔍 Tìm Association Rules", key="ar_button")
    
    if run_analysis:
        with st.spinner("Đang phân tích..."):
            frequent_itemsets, rules, _ = analyze_player_performance_patterns(
                players_df, min_support=min_support, min_confidence=min_confidence
            )
        
        if rules is not None and len(rules) > 0:
            st.success(f"✅ Tìm thấy {len(rules)} association rules!")
            
            # Hiển thị top rules
            st.subheader("Top 20 Association Rules")
            top_rules = rules.head(20).copy()
            
            # Format rules để hiển thị
            display_rules = []
            for idx, row in top_rules.iterrows():
                antecedents = list(row['antecedents'])
                consequents = list(row['consequents'])
                display_rules.append({
                    'Rule': f"{', '.join(antecedents)} → {', '.join(consequents)}",
                    'Support': f"{row['support']:.3f}",
                    'Confidence': f"{row['confidence']:.3f}",
                    'Lift': f"{row['lift']:.3f}"
                })
            
            rules_df = pd.DataFrame(display_rules)
            st.dataframe(rules_df, use_container_width=True, hide_index=True)
            
            # Visualization
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=top_rules['support'],
                    y=top_rules['confidence'],
                    mode='markers',
                    marker=dict(
                        size=top_rules['lift']*10,
                        color=top_rules['lift'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Lift")
                    ),
                    text=[f"Rule {i+1}" for i in range(len(top_rules))],
                    hovertemplate='Support: %{x}<br>Confidence: %{y}<br>Lift: %{marker.color}<extra></extra>'
                ))
                fig.update_layout(
                    title="Association Rules: Support vs Confidence",
                    xaxis_title="Support",
                    yaxis_title="Confidence"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                top_10 = top_rules.head(10)
                fig = px.bar(
                    x=top_10['confidence'],
                    y=[f"Rule {i+1}" for i in range(len(top_10))],
                    orientation='h',
                    title="Top 10 Rules by Confidence"
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Không tìm thấy rules với tham số này. Hãy thử giảm min_support hoặc min_confidence.")

# ==================== CLUSTERING PAGE ====================
elif page == "🎯 Clustering":
    st.markdown('<h1 class="main-header">🎯 Clustering Analysis</h1>', unsafe_allow_html=True)
    
    st.info("Phân cụm cầu thủ theo phong cách chơi sử dụng K-Means và Hierarchical Clustering")
    
    col1, col2 = st.columns(2)
    with col1:
        n_clusters = st.slider("Số cụm", 2, 10, 4, key="clust_n_clusters")
        method = st.selectbox("Phương pháp", ["K-Means", "Hierarchical"], key="clust_method")
    
    with col2:
        show_optimal = st.checkbox("Tự động tìm số cụm tối ưu", value=True, key="clust_optimal")
        run_clustering = st.button("🔍 Thực hiện Clustering", key="clust_button")
    
    if run_clustering:
        with st.spinner("Đang phân cụm..."):
            # Chọn features
            feature_cols = select_features_for_clustering(players_df)
            X = players_df[feature_cols].fillna(0)
            
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Lưu feature_cols để dùng sau
            st.session_state['feature_cols'] = feature_cols
            
            # Thực hiện clustering
            if method == "K-Means":
                results = perform_kmeans_clustering(
                    X_scaled, n_clusters=n_clusters if not show_optimal else None,
                    find_optimal=show_optimal, max_k=10
                )
            else:
                results = perform_hierarchical_clustering(
                    X_scaled, n_clusters=n_clusters if not show_optimal else None,
                    find_optimal=show_optimal, max_k=10
                )
        
        if results:
            st.success(f"✅ Phân cụm thành công với {results['n_clusters']} cụm!")
            st.metric("Silhouette Score", f"{results['silhouette_score']:.3f}")
            
            # Thêm cluster labels
            players_clustered = players_df.copy()
            players_clustered['Cluster'] = results['labels']
            
            # Lưu vào session state
            st.session_state['players_clustered'] = players_clustered
            st.session_state['clustering_results'] = results
            
            # Visualization với PCA
            X_pca, pca = reduce_dimensions_for_visualization(X_scaled, n_components=2)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.scatter(
                    x=X_pca[:, 0],
                    y=X_pca[:, 1],
                    color=results['labels'],
                    hover_data=[players_clustered['Player'].values if 'Player' in players_clustered.columns else None],
                    title=f"{method} Clustering (PCA Visualization)",
                    labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
                           'y': f'PC2 ({pca.explained_variance_ratio_[1]:.1%})',
                           'color': 'Cluster'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'Pos' in players_clustered.columns:
                    cluster_pos = pd.crosstab(players_clustered['Cluster'], players_clustered['Pos'])
                    fig = px.bar(
                        cluster_pos,
                        title="Distribution of Positions in Each Cluster",
                        labels={'value': 'Count', 'index': 'Cluster'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Hiển thị cầu thủ theo cụm
            st.subheader("Cầu thủ theo cụm")
            selected_cluster = st.selectbox("Chọn cụm để xem", range(results['n_clusters']), key="clust_select_cluster")
            cluster_players = players_clustered[players_clustered['Cluster'] == selected_cluster]
            
            # Phân tích đặc điểm cụm
            from clustering import analyze_clusters
            cluster_stats = analyze_clusters(players_df, results['labels'], feature_cols)
            st.subheader(f"Đặc điểm Cụm {selected_cluster}")
            if selected_cluster in cluster_stats.index:
                st.dataframe(cluster_stats.loc[[selected_cluster]], use_container_width=True)
            
            if 'Player' in cluster_players.columns:
                display_cols = ['Player', 'Pos', 'Squad']
                if 'shooting_Standard_Gls' in cluster_players.columns:
                    display_cols.append('shooting_Standard_Gls')
                if 'passing_Ast' in cluster_players.columns:
                    display_cols.append('passing_Ast')
                
                available_cols = [c for c in display_cols if c in cluster_players.columns]
                st.dataframe(cluster_players[available_cols], use_container_width=True)

# ==================== CLASSIFICATION PAGE ====================
elif page == "📈 Classification":
    st.markdown('<h1 class="main-header">📈 Classification</h1>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Dự đoán vị trí", "Dự đoán Top 4", "Phân loại hiệu suất"])
    
    with tab1:
        st.subheader("Dự đoán vị trí cầu thủ dựa trên chỉ số")
        
        if st.button("🔍 Train Model", key="class_pos_button"):
            with st.spinner("Đang train model..."):
                results = classify_player_position(players_df, min_samples_per_class=10)
            
            if results:
                rf_metrics = evaluate_classification(results['random_forest'], 'Random Forest')
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{rf_metrics['accuracy']:.3f}")
                with col2:
                    st.metric("Precision", f"{rf_metrics['precision']:.3f}")
                with col3:
                    st.metric("Recall", f"{rf_metrics['recall']:.3f}")
                with col4:
                    st.metric("F1 Score", f"{rf_metrics['f1']:.3f}")
                
                # Feature importance
                importance = get_feature_importance(
                    results['random_forest']['model'],
                    results['random_forest']['feature_names'],
                    top_n=10
                )
                
                if importance is not None:
                    fig = px.bar(
                        importance,
                        x='importance',
                        y='feature',
                        orientation='h',
                        title="Top 10 Feature Importance"
                    )
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                
                # Confusion Matrix
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(
                    results['random_forest']['y_test'],
                    results['random_forest']['predictions']
                )
                le = results['random_forest']['label_encoder']
                
                fig = px.imshow(
                    cm,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=le.classes_,
                    y=le.classes_,
                    title="Confusion Matrix"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Dự đoán đội bóng vào Top 4")
        
        if teams_df is not None:
            if st.button("🔍 Train Model (Top 4)", key="class_top4_button"):
                with st.spinner("Đang train..."):
                    results = classify_team_top4(teams_df)
                
                if results:
                    rf_metrics = evaluate_classification(results['random_forest'], 'Random Forest')
                    st.metric("Accuracy", f"{rf_metrics['accuracy']:.3f}")
                    
                    from sklearn.metrics import confusion_matrix
                    cm = confusion_matrix(
                        results['random_forest']['y_test'],
                        results['random_forest']['predictions']
                    )
                    fig = px.imshow(
                        cm,
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=['Not Top 4', 'Top 4'],
                        y=['Not Top 4', 'Top 4'],
                        title="Confusion Matrix - Top 4 Prediction"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Không có dữ liệu đội bóng")
    
    with tab3:
        st.subheader("Phân loại hiệu suất cầu thủ")
        
        if st.button("🔍 Phân loại hiệu suất", key="class_perf_button"):
            with st.spinner("Đang phân tích..."):
                results = classify_player_performance(players_df)
            
            if results:
                rf_metrics = evaluate_classification(results['random_forest'], 'Random Forest')
                st.metric("Accuracy", f"{rf_metrics['accuracy']:.3f}")

# ==================== ANOMALY DETECTION PAGE ====================
elif page == "🚨 Anomaly Detection":
    st.markdown('<h1 class="main-header">🚨 Anomaly Detection</h1>', unsafe_allow_html=True)
    
    st.info("Phát hiện cầu thủ và đội bóng có chỉ số bất thường")
    
    col1, col2 = st.columns(2)
    with col1:
        contamination = st.slider("Contamination", 0.05, 0.3, 0.1, 0.05, key="anom_contamination")
        detection_type = st.selectbox("Loại phát hiện", ["Cầu thủ", "Đội bóng"], key="anom_type")
    
    with col2:
        method = st.selectbox("Phương pháp", ["Isolation Forest", "LOF", "Cả hai"], key="anom_method")
        run_detection = st.button("🔍 Phát hiện Anomalies", key="anom_button")
    
    if run_detection:
        if detection_type == "Cầu thủ":
            with st.spinner("Đang phân tích..."):
                anomalies = analyze_player_anomalies(players_df, contamination=contamination)
            
            if anomalies:
                df_anomalies = anomalies['df_with_anomalies']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Isolation Forest Outliers", anomalies['isolation_forest']['n_outliers'])
                with col2:
                    st.metric("LOF Outliers", anomalies['lof']['n_outliers'])
                with col3:
                    st.metric("Cả 2 methods", df_anomalies['Both_Methods_Outlier'].sum())
                
                # Hiển thị outliers
                if method in ["Isolation Forest", "Cả hai"]:
                    iso_outliers = df_anomalies[df_anomalies['IsolationForest_Outlier']]
                    if 'Player' in iso_outliers.columns and len(iso_outliers) > 0:
                        st.subheader("Outliers - Isolation Forest")
                        display_cols = ['Player', 'Pos', 'Squad']
                        if 'shooting_Standard_Gls' in iso_outliers.columns:
                            display_cols.append('shooting_Standard_Gls')
                        available_cols = [c for c in display_cols if c in iso_outliers.columns]
                        st.dataframe(iso_outliers[available_cols], use_container_width=True)
                
                if method in ["LOF", "Cả hai"]:
                    lof_outliers = df_anomalies[df_anomalies['LOF_Outlier']].nlargest(10, 'LOF_Score')
                    if 'Player' in lof_outliers.columns and len(lof_outliers) > 0:
                        st.subheader("Top Outliers - LOF (Highest Scores)")
                        display_cols = ['Player', 'Pos', 'Squad', 'LOF_Score']
                        available_cols = [c for c in display_cols if c in lof_outliers.columns]
                        st.dataframe(lof_outliers[available_cols], use_container_width=True)
                
                # Visualization
                feature_cols = anomalies['feature_cols']
                if len(feature_cols) >= 2:
                    col1, col2 = feature_cols[0], feature_cols[1]
                    
                    normal = df_anomalies[~df_anomalies['IsolationForest_Outlier']]
                    outliers = df_anomalies[df_anomalies['IsolationForest_Outlier']]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=normal[col1],
                        y=normal[col2],
                        mode='markers',
                        name='Normal',
                        marker=dict(color='blue', size=5, opacity=0.5)
                    ))
                    fig.add_trace(go.Scatter(
                        x=outliers[col1],
                        y=outliers[col2],
                        mode='markers',
                        name='Outliers',
                        marker=dict(color='red', size=10, symbol='x')
                    ))
                    fig.update_layout(
                        title="Anomaly Detection - Isolation Forest",
                        xaxis_title=col1,
                        yaxis_title=col2
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        else:  # Đội bóng
            if teams_df is not None:
                with st.spinner("Đang phân tích..."):
                    anomalies = analyze_team_anomalies(teams_df, contamination=contamination)
                
                if anomalies:
                    df_anomalies = anomalies['df_with_anomalies']
                    st.metric("Outliers", anomalies['isolation_forest']['n_outliers'])
                    
                    outliers = df_anomalies[df_anomalies['IsolationForest_Outlier']]
                    if 'Squad' in outliers.columns:
                        st.subheader("Đội bóng bất thường")
                        st.dataframe(outliers[['Squad']], use_container_width=True)

# ==================== RECOMMENDATION SYSTEM PAGE ====================
elif page == "⭐ Recommendation System":
    st.markdown('<h1 class="main-header">⭐ Recommendation System</h1>', unsafe_allow_html=True)
    
    st.info("Hệ thống gợi ý cầu thủ với nhiều tính năng tương tác")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Tìm cầu thủ tương tự",
        "⚽ Gợi ý cho đội bóng",
        "🎨 Tìm theo phong cách",
        "👤 Player Profile"
    ])
    
    with tab1:
        st.subheader("Tìm cầu thủ tương tự")
        
        # Tìm danh sách cầu thủ
        if 'Player' in players_df.columns:
            player_list = sorted(players_df['Player'].unique().tolist())
            selected_player = st.selectbox("Chọn cầu thủ", player_list, key="rec_similar_player")
            
            col1, col2 = st.columns(2)
            with col1:
                n_recommendations = st.slider("Số lượng gợi ý", 5, 20, 10, key="rec_similar_n")
            with col2:
                same_position = st.checkbox("Chỉ tìm cùng vị trí", value=True, key="rec_similar_pos")
            
            if st.button("🔍 Tìm cầu thủ tương tự", key="rec_similar_button"):
                with st.spinner("Đang tìm..."):
                    recommendations, player_info = find_similar_players(
                        players_df, selected_player,
                        n_recommendations=n_recommendations,
                        same_position=same_position
                    )
                
                if recommendations is not None and len(recommendations) > 0:
                    st.success(f"✅ Tìm thấy {len(recommendations)} cầu thủ tương tự!")
                    
                    # Hiển thị thông tin cầu thủ gốc
                    if player_info is not None:
                        st.info(f"**Cầu thủ gốc:** {player_info.get('Player', 'N/A')} | "
                               f"Vị trí: {player_info.get('Pos', 'N/A')} | "
                               f"Đội: {player_info.get('Squad', 'N/A')}")
                    
                    # Bảng kết quả
                    st.dataframe(recommendations, use_container_width=True)
                    
                    # Visualization
                    top_5 = recommendations.head(5)
                    fig = px.bar(
                        top_5,
                        x='Similarity',
                        y='Player',
                        orientation='h',
                        title="Top 5 Cầu thủ tương tự nhất",
                        color='Similarity',
                        color_continuous_scale='Blues'
                    )
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Không tìm thấy cầu thủ tương tự")
    
    with tab2:
        st.subheader("Gợi ý cầu thủ cho đội bóng")
        
        if 'Squad' in players_df.columns:
            team_list = sorted(players_df['Squad'].unique().tolist())
            selected_team = st.selectbox("Chọn đội bóng", team_list, key="rec_team_team")
            
            position = st.selectbox("Vị trí cần tìm", ["Tất cả", "FW", "MF", "DF", "GK"], key="rec_team_pos")
            n_recommendations = st.slider("Số lượng gợi ý", 5, 15, 10, key="rec_team_n")
            
            if st.button("🔍 Tìm gợi ý", key="rec_team_button"):
                with st.spinner("Đang tìm..."):
                    recommendations = recommend_players_by_team_needs(
                        players_df,
                        selected_team,
                        position=position if position != "Tất cả" else None,
                        n_recommendations=n_recommendations
                    )
                
                if recommendations is not None and len(recommendations) > 0:
                    st.success(f"✅ Tìm thấy {len(recommendations)} gợi ý!")
                    st.dataframe(recommendations, use_container_width=True)
                    
                    # Visualization
                    fig = px.bar(
                        recommendations.head(10),
                        x='Recommendation_Score',
                        y='Player',
                        orientation='h',
                        title="Top 10 Gợi ý",
                        color='Recommendation_Score',
                        color_continuous_scale='Greens'
                    )
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Không tìm thấy gợi ý phù hợp")
    
    with tab3:
        st.subheader("Tìm cầu thủ theo phong cách chơi")
        
        st.markdown("Nhập các chỉ số mong muốn để tìm cầu thủ phù hợp:")
        
        # Tìm các cột có sẵn
        available_cols = [c for c in players_df.columns if any(kw in c.lower() 
            for kw in ['gls', 'ast', 'xg', 'xa', 'sh', 'sot', 'pass', 'tkl'])]
        
        col1, col2 = st.columns(2)
        target_features = {}
        
        with col1:
            if len(available_cols) > 0:
                goal_col = [c for c in available_cols if 'gls' in c.lower()][0] if any('gls' in c.lower() for c in available_cols) else None
                assist_col = [c for c in available_cols if 'ast' in c.lower()][0] if any('ast' in c.lower() for c in available_cols) else None
                
                if goal_col:
                    target_features[goal_col] = st.number_input("Goals", min_value=0.0, value=float(players_df[goal_col].quantile(0.75)), key="rec_style_goals")
                if assist_col:
                    target_features[assist_col] = st.number_input("Assists", min_value=0.0, value=float(players_df[assist_col].quantile(0.75)), key="rec_style_assists")
        
        with col2:
            xg_col = [c for c in available_cols if 'xg' in c.lower() and 'xga' not in c.lower()][0] if any('xg' in c.lower() and 'xga' not in c.lower() for c in available_cols) else None
            xa_col = [c for c in available_cols if 'xa' in c.lower()][0] if any('xa' in c.lower() for c in available_cols) else None
            
            if xg_col:
                target_features[xg_col] = st.number_input("xG", min_value=0.0, value=float(players_df[xg_col].quantile(0.75)), key="rec_style_xg")
            if xa_col:
                target_features[xa_col] = st.number_input("xA", min_value=0.0, value=float(players_df[xa_col].quantile(0.75)), key="rec_style_xa")
        
        n_recommendations = st.slider("Số lượng gợi ý", 5, 20, 10, key="rec_style_n")
        
        if st.button("🔍 Tìm cầu thủ phù hợp", key="rec_style_button"):
            if len(target_features) > 0:
                with st.spinner("Đang tìm..."):
                    recommendations = recommend_players_by_style(
                        players_df, target_features, n_recommendations=n_recommendations
                    )
                
                if recommendations is not None and len(recommendations) > 0:
                    st.success(f"✅ Tìm thấy {len(recommendations)} cầu thủ phù hợp!")
                    st.dataframe(recommendations, use_container_width=True)
                    
                    # Visualization
                    fig = px.bar(
                        recommendations.head(10),
                        x='Similarity',
                        y='Player',
                        orientation='h',
                        title="Top 10 Cầu thủ phù hợp với phong cách",
                        color='Similarity',
                        color_continuous_scale='Viridis'
                    )
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Không tìm thấy cầu thủ phù hợp")
            else:
                st.warning("Vui lòng nhập ít nhất một chỉ số")
    
    with tab4:
        st.subheader("Player Profile")
        
        if 'Player' in players_df.columns:
            player_list = sorted(players_df['Player'].unique().tolist())
            selected_player = st.selectbox("Chọn cầu thủ để xem profile", player_list, key="rec_profile_player")
            
            if st.button("👤 Xem Profile", key="rec_profile_button"):
                with st.spinner("Đang tải..."):
                    profile = create_player_profile(players_df, selected_player)
                
                if profile:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### Thông tin cơ bản")
                        for key, value in profile.items():
                            if key != 'Stats':
                                st.write(f"**{key}:** {value}")
                    
                    with col2:
                        if 'Stats' in profile and len(profile['Stats']) > 0:
                            st.markdown("### Chỉ số quan trọng")
                            stats_df = pd.DataFrame(list(profile['Stats'].items()), columns=['Stat', 'Value'])
                            st.dataframe(stats_df, use_container_width=True, hide_index=True)
                            
                            # Visualization
                            fig = px.bar(
                                stats_df,
                                x='Value',
                                y='Stat',
                                orientation='h',
                                title=f"Stats của {profile.get('Player', 'N/A')}",
                                color='Value',
                                color_continuous_scale='Blues'
                            )
                            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                            st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Data Mining Project - Premier League 2024-2025 | "
    "Association Rules | Clustering | Classification | Anomaly Detection | Recommendation System"
    "</div>",
    unsafe_allow_html=True
)


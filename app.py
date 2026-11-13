import streamlit as st
import pandas as pd
import numpy as np

from uye1_veri_isleme import load_and_clean, fig_histogram, fig_correlation, fig_treemap
from src.visualizations import (
    GEO_METRICS,
    draw_choropleth_map,
    draw_sankey_diagram,
    draw_tag_network,
)
from src.ml_models import (
    RF_CONTINUOUS_FEATURES,
    draw_parallel_coordinates,
    draw_pca_3d_clusters,
    draw_rf_decision_surface,
    run_kmeans_pca_pipeline,
    run_random_forest_pipeline,
)

st.set_page_config(
    page_title="YouTube Trending Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_data(show_spinner="Loading & cleaning 5-country YouTube dataset...")
def get_clean_data():
    df = load_and_clean("data")

    if "trending_date" in df.columns and "publish_time" in df.columns:
        t_date = pd.to_datetime(df["trending_date"], format="%y.%d.%m", errors="coerce", utc=True)
        p_date = pd.to_datetime(df["publish_time"], errors="coerce", utc=True)
        days_diff = (t_date - p_date).dt.days.fillna(2).clip(lower=0)
        df["days_to_trend"] = days_diff
    else:
        df["days_to_trend"] = 2

    if "category_name" in df.columns and "category" not in df.columns:
        df["category"] = df["category_name"]

    return df


df = get_clean_data()


@st.cache_data(show_spinner=False)
def get_cached_eda_figs(data: pd.DataFrame, metric: str):
    return (
        fig_histogram(data, metric),
        fig_correlation(data),
        fig_treemap(data),
    )


@st.cache_data(show_spinner=False)
def get_cached_adv_figs(
    data: pd.DataFrame,
    geo_metric: str,
    sankey_top_n: int,
    net_top_tags: int,
    net_sizing_mode: str,
):
    fig_map, geo_stats = draw_choropleth_map(data, metric_key=geo_metric)
    fig_sankey, sankey_stats = draw_sankey_diagram(data, top_n_categories=sankey_top_n)
    fig_network, net_stats = draw_tag_network(
        data,
        top_n_tags=net_top_tags,
        min_cooccurrence=5,
        sizing_mode=net_sizing_mode,
    )
    return fig_map, geo_stats, fig_sankey, sankey_stats, fig_network, net_stats


# --- SIDEBAR CONTROLS (3 Distinct Interactive Control Types) ---
st.sidebar.header("Dashboard Filters")

# 1. Multiselect: Countries
available_countries = sorted(df["country_code"].unique())
selected_countries = st.sidebar.multiselect(
    "Select Countries",
    options=available_countries,
    default=available_countries
)

# 2. Range Slider: View Count
min_views = int(df["views"].min())
max_views = int(df["views"].max())
view_range = st.sidebar.slider(
    "View Count Range",
    min_value=min_views,
    max_value=max_views,
    value=(min_views, max_views),
    step=max(1, (max_views - min_views) // 100)
)

# 3. Checkbox: High-Impact Viral Speed Filter
fast_trend_only = st.sidebar.checkbox(
    "Fast-Trending Videos Only (<3 Days)",
    value=False,
    help="Filters videos that reached the trending list within 48 hours of publishing."
)

# --- FILTERING ---
filtered_df = df[
    (df["country_code"].isin(selected_countries)) &
    (df["views"] >= view_range[0]) &
    (df["views"] <= view_range[1])
]

if fast_trend_only:
    filtered_df = filtered_df[filtered_df["days_to_trend"] <= 2]

# --- HEADER & KPI METRICS ---
st.title("YouTube Trending Video Analytics")
st.markdown(
    "Interactive exploratory analysis, machine learning insights, and multi-dimensional visualizations across global markets.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Filtered Videos", f"{len(filtered_df):,}")
col2.metric("Average Views", f"{filtered_df['views'].mean():,.0f}" if not filtered_df.empty else "0")
col3.metric("Average Likes", f"{filtered_df['likes'].mean():,.0f}" if not filtered_df.empty else "0")
col4.metric("Top Category", filtered_df["category"].mode()[0] if not filtered_df.empty else "N/A")

st.divider()

# --- NAVIGATION TABS ---
tab_options = [
    "1. Exploratory Data Analysis (Fatih AKSUT)",
    "2. Machine Learning Models (Yusuf YALCIN)",
    "3. Advanced Visualizations (Ali Oktay OZKAN)"
]
active_tab = st.radio("Navigation", tab_options, horizontal=True, label_visibility="collapsed")

if filtered_df.empty:
    st.warning("No videos match the current filter criteria. Please adjust your sidebar filters.")
else:
    # --- TAB 1: MEMBER 1 EDA (1 Top + 2 Bottom Grid) ---
    if active_tab == tab_options[0]:
        st.subheader("Distribution & Metric Correlations")

        metric_choice = st.selectbox(
            "Select Distribution Metric",
            ["views", "likes"],
            format_func=lambda x: "Views" if x == "views" else "Likes"
        )
        f_hist, f_corr, f_tree = get_cached_eda_figs(filtered_df, metric_choice)
        st.plotly_chart(f_hist, use_container_width=True)

        with st.expander("📌 Distribution Insights & Statistical Notes"):
            st.markdown(
                """
- **Insight:** Both views and likes exhibit strong right-skewness across all five markets, where a small fraction of viral videos captures the majority of engagement.

- **Marginal Boxplot:** Highlights median values and interquartile ranges (IQR) per country after 99th-percentile outlier filtering.

                """
            )

        st.divider()

        col_left, col_right = st.columns(2)
        with col_left:
            st.plotly_chart(f_corr, use_container_width=True)
            with st.expander("📌 Correlation Matrix Notes"):
                st.markdown(
                    """
- **Insight:** Strong positive correlation exists between `Views`, `Likes` (~0.78), and `Comments` (~0.65), whereas title length and uppercase character counts show near-zero linear impact on view volume.

                    """
                )
        with col_right:
            st.plotly_chart(f_tree, use_container_width=True)
            with st.expander("📌 Hierarchical Treemap Notes"):
                st.markdown(
                    """
- **Insight:** `Entertainment`, `Sports`, and `Film & Animation` dominate cumulative trending views, led by major broadcast and studio channels.

                    """
                )

    # --- TAB 2: MEMBER 2 ML (1 Top + 2 Bottom Grid + Interactive ML Controls) ---
    elif active_tab == tab_options[1]:
        st.subheader("Behavioral Clustering (K-Means + PCA) & Non-Linear Virality Prediction (Random Forest)")

        if len(filtered_df) < 40:
            st.warning(
                "Please broaden your sidebar filters to include at least 40 videos for reliable Machine Learning training and cross-validation.")
        else:
            # --- INTERACTIVE ML HYPERPARAMETER & FEATURE CONTROLS ---
            ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1.2, 1.3, 1.3, 1.4])
            with ctrl_col1:
                n_clusters = st.slider(
                    "K-Means Clusters (k)",
                    min_value=2,
                    max_value=6,
                    value=4,
                    help="Adjust the number of behavioral video archetypes discovered by K-Means.",
                )
            with ctrl_col2:
                rf_x_feat = st.selectbox(
                    "RF Surface X-Axis",
                    options=list(RF_CONTINUOUS_FEATURES.keys()),
                    index=0,
                    format_func=lambda k: RF_CONTINUOUS_FEATURES[k],
                )
            with ctrl_col3:
                rf_y_options = [k for k in RF_CONTINUOUS_FEATURES.keys() if k != rf_x_feat]
                rf_y_feat = st.selectbox(
                    "RF Surface Y-Axis",
                    options=rf_y_options,
                    index=0,
                    format_func=lambda k: RF_CONTINUOUS_FEATURES[k],
                )
            with ctrl_col4:
                rf_view_mode = st.radio(
                    "Random Forest Diagnostic Mode",
                    options=["surface", "importance"],
                    format_func=lambda m: "2D Decision Surface" if m == "surface" else "Feature Importance",
                    horizontal=True,
                )

            # Run cached ML pipelines
            kmeans_res = run_kmeans_pca_pipeline(filtered_df, n_clusters=n_clusters)
            rf_res = run_random_forest_pipeline(filtered_df)

            # --- ML MODEL EVALUATION KPI ROW ---
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric(
                "K-Means Silhouette Score",
                f"{kmeans_res['silhouette']:.3f}",
                help="Measures cluster cohesion and separation in 7D standardized space.",
            )
            m_col2.metric(
                "3D PCA Variance Captured",
                f"{kmeans_res['total_var']:.1f}%",
                help="Percentage of total 7-dimensional behavioral variance retained in 3D projection.",
            )
            m_col3.metric(
                "RF Classifier ROC-AUC",
                f"{rf_res['roc_auc']:.3f}",
                f"Acc: {rf_res['accuracy'] * 100:.1f}%",
                help="Out-of-sample test ROC-AUC and Accuracy for predicting High-Engagement Virality.",
            )
            m_col4.metric(
                "RF Regressor R² (Engagement)",
                f"{rf_res['reg_r2']:.3f}",
                f"Median Thr: {rf_res['median_eng']:.2f}%",
                help="Out-of-sample variance explained by Random Forest Regressor using pre-publication & velocity features.",
            )

            # --- CHART 1 (TOP FULL-WIDTH): PARALLEL COORDINATES ---
            fig_parcoords = draw_parallel_coordinates(kmeans_res)
            st.plotly_chart(fig_parcoords, use_container_width=True)
            with st.expander("📌 Parallel Coordinates & K-Means Archetype Insights"):
                st.markdown(
                    """
- **Analytical Insight:** Unsupervised **K-Means clustering** on log-normalized metrics uncovers distinct behavioral video archetypes (e.g., *Mega-Viral Blockbusters*, *High-Engagement Community*, *Fast-Burn Click-Drivers*, and *SEO Tag-Heavy Performers*).

- **Interactive Brushing:** Drag vertically along any axis to isolate specific behavioral regimes and observe how metadata attributes (`Title Length`, `Uppercase %`, `Tag Count`) co-vary with audience reach.
                    """
                )

            st.divider()

            # --- BOTTOM ROW: 3D PCA MANIFOLD (LEFT) + RANDOM FOREST SURFACE (RIGHT) ---
            col_left, col_right = st.columns(2)
            with col_left:
                fig_pca3d = draw_pca_3d_clusters(kmeans_res)
                st.plotly_chart(fig_pca3d, use_container_width=True)
                with st.expander("📌 3D PCA Manifold & Dimensionality Reduction Notes"):
                    st.markdown(
                        f"""
- **Methodology:** Projects the 7-dimensional standardized feature space onto its top 3 orthogonal Principal Components, preserving **{kmeans_res['total_var']:.1f}%** of total variance.

- **Centroid Geometry:** White diamond markers denote the exact geometric center of each K-Means cluster in 3D latent space. Rotate, zoom, and hover over individual videos to inspect outlier trajectories.
                        """
                    )

            with col_right:
                fig_rf = draw_rf_decision_surface(
                    rf_res,
                    x_feat=rf_x_feat,
                    y_feat=rf_y_feat,
                    view_mode=rf_view_mode,
                )
                st.plotly_chart(fig_rf, use_container_width=True)
                with st.expander("📌 Random Forest Non-Linear Virality & Explainability Notes"):
                    st.markdown(
                        f"""
- **Why Random Forest Beats Linear Correlation:** While Tab 1 showed near-zero Pearson linear correlation for title/tag metadata, our **Random Forest** (**ROC-AUC: {rf_res['roc_auc']:.3f}**) captures **non-linear threshold effects** and multi-feature interactions without data leakage.

- **Decision Surface:** Contour isolines map the predicted probability of achieving High-Engagement Virality (`>= {rf_res['median_eng']:.2f}%` engagement rate) across `{RF_CONTINUOUS_FEATURES[rf_x_feat]}` and `{RF_CONTINUOUS_FEATURES[rf_y_feat]}`.
                        """
                    )

    # --- TAB 3: MEMBER 3 ADVANCED VISUALIZATIONS (Matching 1 Top + 2 Bottom Grid + Graph/Geo Controls) ---
    elif active_tab == tab_options[2]:
        st.subheader("Geospatial Reach, Categorical Velocity Flow & Graph-Theoretic Tag Network")

        # --- INTERACTIVE ADVANCED VISUALIZATION CONTROLS ---
        adv_col1, adv_col2, adv_col3, adv_col4 = st.columns([1.4, 1.1, 1.1, 1.4])
        with adv_col1:
            geo_metric = st.selectbox(
                "Geospatial Map Metric",
                options=list(GEO_METRICS.keys()),
                index=0,
                format_func=lambda k: GEO_METRICS[k]["label"],
                help="Select the regional performance indicator projected onto the Choropleth map.",
            )
        with adv_col2:
            sankey_top_n = st.slider(
                "Sankey Top Categories",
                min_value=5,
                max_value=10,
                value=8,
                help="Number of dominant content categories traced through the velocity-to-engagement flow.",
            )
        with adv_col3:
            net_top_tags = st.slider(
                "Network Top Tags (Nodes)",
                min_value=15,
                max_value=35,
                value=22,
                help="Number of top recurring tags included in the Fruchterman-Reingold graph topology.",
            )
        with adv_col4:
            net_sizing_mode = st.radio(
                "Network Node Sizing Metric",
                options=["frequency", "degree", "betweenness"],
                format_func=lambda m: (
                    "Frequency" if m == "frequency" else ("Degree Centrality" if m == "degree" else "Betweenness")
                ),
                horizontal=True,
                help="Scale node radii by raw occurrence count, Degree Centrality, or Betweenness Centrality (bridge tags).",
            )

        fig_map, geo_stats, fig_sankey, sankey_stats, fig_network, net_stats = get_cached_adv_figs(
            filtered_df,
            geo_metric=geo_metric,
            sankey_top_n=sankey_top_n,
            net_top_tags=net_top_tags,
            net_sizing_mode=net_sizing_mode,
        )

        # --- ADVANCED ANALYTICS KPI ROW ---
        a_col1, a_col2, a_col3, a_col4 = st.columns(4)
        a_col1.metric(
            "Leading Market (Selected Metric)",
            geo_stats["lead_country"],
            geo_stats["lead_val_fmt"],
            help=f"Top-performing country for {geo_stats['metric_label']}.",
        )
        a_col2.metric(
            "Fast-Trend Conversion (<3d)",
            f"{sankey_stats['fast_pct']:.1f}%",
            f"{sankey_stats['fast_high_eng_pct']:.1f}% High-Eng",
            help="Share of videos trending within 48 hours and their conversion rate into High Engagement.",
        )
        a_col3.metric(
            "Tag Network Density",
            f"{net_stats['density']:.3f}",
            f"{net_stats['num_nodes']} Nodes / {net_stats['num_edges']} Edges",
            help="Graph density = 2E / [V*(V-1)], measuring tag co-occurrence interconnectedness.",
        )
        a_col4.metric(
            "Top Bridge Tag (Betweenness)",
            net_stats["top_bridge_tag"],
            f"Centrality: {net_stats['top_bridge_score']:.3f}",
            help="Tag with the highest shortest-path Betweenness Centrality connecting disparate content categories.",
        )

        # --- CHART 1 (TOP FULL-WIDTH): CHOROPLETH MAP ---
        st.plotly_chart(fig_map, use_container_width=True)
        with st.expander("📌 Geospatial Insights & Multi-Metric Projection Notes"):
            st.markdown(
                """
- **Multi-Dimensional Comparison:** Dynamically switches between volume (`Average Views`), audience responsiveness (`Engagement Rate %`, `Comments / 1K Views`), and viral speed (`Avg Days to Trend`) across US, GB, DE, FR, and RU markets.

- **Cartographic Design:** Natural Earth projection focused on the Northern Hemisphere (`lataxis_range=[22, 78]`) with rich custom hover tooltips displaying 6 cross-market indicators simultaneously.
                """
            )

        st.divider()

        # --- BOTTOM ROW: SANKEY FLOW (LEFT) + SEMANTIC TAG NETWORK (RIGHT) ---
        col_left, col_right = st.columns(2)
        with col_left:
            st.plotly_chart(fig_sankey, use_container_width=True)
            with st.expander("📌 Sankey Flow & Viral Velocity Notes"):
                st.markdown(
                    f"""
- **Multi-Stage Transition Matrix:** Maps how the top **{sankey_top_n}** categories funnel into **Fast Trend (<3d)** vs. **Slow Trend (3d+)** regimes and subsequently convert into **High** vs. **Standard Engagement** tiers (split at median engagement rate).

- **Analytical Finding:** Currently **{sankey_stats['fast_pct']:.1f}%** of filtered videos achieve rapid trending velocity (`<= 2` days), with **{sankey_stats['fast_high_eng_pct']:.1f}%** of those sustaining above-median audience engagement.
                    """
                )

        with col_right:
            st.plotly_chart(fig_network, use_container_width=True)
            with st.expander("📌 Graph Theory & Centrality Topology Notes"):
                st.markdown(
                    f"""
- **Force-Directed Topology:** Computed via NetworkX **Fruchterman-Reingold** (`k=0.85`, 80 iterations) over co-occurring video tags (`weight >= 5`), colored by each tag's dominant video category.

- **Graph Centrality Diagnostics:** Toggle between **Tag Frequency**, **Degree Centrality** ($C_D(v)$), and **Betweenness Centrality** ($C_B(v)$) to identify semantic bridge tags like `{net_stats['top_bridge_tag']}` that link distinct creator communities.
                    """
                )
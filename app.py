import streamlit as st
import pandas as pd
import numpy as np

from uye1_veri_isleme import load_and_clean, fig_histogram, fig_correlation, fig_treemap
from src.visualizations import draw_choropleth_map, draw_sankey_diagram, draw_tag_network
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
    "1. Exploratory Data Analysis (Member 1)",
    "2. Machine Learning Models (Member 2)",
    "3. Advanced Visualizations (Member 3)"
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
        st.plotly_chart(fig_histogram(filtered_df, metric_choice), width="stretch")
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
            st.plotly_chart(fig_correlation(filtered_df), width="stretch")
            with st.expander("📌 Correlation Matrix Notes"):
                st.markdown(
                    """
- **Insight:** Strong positive correlation exists between `Views`, `Likes` (~0.78), and `Comments` (~0.65), whereas title length and uppercase character counts show near-zero linear impact on view volume.

                    """
                )
        with col_right:
            st.plotly_chart(fig_treemap(filtered_df), width="stretch")
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
                "Please broaden your sidebar filters to include at least 40 videos for machine learning training and evaluation.")
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
                help="Out-of-sample variance explained by Random Forest Regressor using the selected video metadata features.",
            )

            # --- CHART 1 (TOP FULL-WIDTH): PARALLEL COORDINATES ---
            fig_parcoords = draw_parallel_coordinates(kmeans_res)
            st.plotly_chart(fig_parcoords, width="stretch")
            with st.expander("📌 Video Group Insights"):
                st.markdown(
                    """
**What it shows:** K-Means groups videos with similar features.

**Key insight:** Groups differ in views, engagement, and title style.

**How to use:** Select a range on an axis to explore the groups.
                    """
                )

            st.divider()

            # --- BOTTOM CHARTS: PCA HEADER ALIGNED WITH THE RF SELECTORS ---
            # Left title starts at the same height as the two selectors on the right.
            # Keep Plotly's interactive cluster legend (users can toggle clusters).
            rf_country, rf_category = None, None
            col_left, col_right = st.columns(2, gap="large")

            with col_left:
                pca_title = f"3D PCA Clusters ({kmeans_res['total_var']:.1f}% Variance Explained)"
                st.markdown(
                    f'<div style="font-size:22px;font-weight:700;line-height:1.3;'
                    f'padding-top:2px;">{pca_title}</div>',
                    unsafe_allow_html=True,
                )
                fig_pca3d = draw_pca_3d_clusters(kmeans_res)
                fig_pca3d.update_layout(
                    title=None,  # Show title above chart to avoid Plotly toolbar overlap.
                    height=570,
                    margin=dict(l=10, r=10, t=92, b=25),
                    legend=dict(
                        orientation="h", x=0, y=1.04,
                        yanchor="bottom", font=dict(size=15),
                    ),
                    scene=dict(domain=dict(x=[0.0, 1.0], y=[0.02, 0.91])),
                )
                st.plotly_chart(fig_pca3d, width="stretch")

            with col_right:
                if rf_view_mode == "surface":
                    country_col, category_col = st.columns(2)
                    with country_col:
                        rf_country = st.selectbox(
                            "Reference Country",
                            sorted(rf_res["ml_df"]["country_code"].dropna().unique()),
                            key="rf_reference_country",
                        )
                    available_categories = sorted(
                        rf_res["ml_df"].loc[
                            rf_res["ml_df"]["country_code"] == rf_country, "category"
                        ].dropna().unique()
                    )
                    with category_col:
                        rf_category = st.selectbox(
                            "Reference Category",
                            available_categories,
                            key="rf_reference_category",
                        )

                fig_rf = draw_rf_decision_surface(
                    rf_res,
                    x_feat=rf_x_feat,
                    y_feat=rf_y_feat,
                    view_mode=rf_view_mode,
                    country_code=rf_country,
                    category=rf_category,
                )
                if rf_view_mode == "surface":
                    fig_rf.update_layout(title=dict(text="RF High-Engagement Probability"))
                fig_rf.update_layout(height=520)
                st.plotly_chart(fig_rf, width="stretch")

            # Short PCA guide in its own row, keeping both expanders aligned.
            pca_guide_col, _ = st.columns(2, gap="large")
            with pca_guide_col:
                st.caption("PC1 = Engagement  |  PC2 = Views & Time  |  PC3 = Tags & Titles")

            # Put both note expanders in a NEW ROW rather than inside the chart
            # columns. This keeps them aligned when chart heights differ.
            notes_left, notes_right = st.columns(2, gap="large")
            with notes_left:
                with st.expander("📌 3D PCA Insights"):
                    st.markdown(
                        f"""
**What it shows:** PCA displays video groups in 3D.

**Key insight:** The three axes explain **{kmeans_res['total_var']:.1f}%** of the data's variation.

**How to use:** Rotate the chart or hover over points. White diamonds show group centers.
                        """
                    )

            with notes_right:
                with st.expander("📌 Random Forest Insights"):
                    st.markdown(
                        f"""
**What it shows:** Random Forest estimates high engagement from video features.

**Key insight:** Its test ROC-AUC is **{rf_res['roc_auc']:.3f}**. It predicts above-median engagement (**{rf_res['median_eng']:.2f}%**), not whether a video becomes trending.

**How to use:** Select features, country, and category to explore predictions. Feature Importance shows which inputs the model uses most.
                        """
                    )

    # --- TAB 3: MEMBER 3 ADVANCED VISUALIZATIONS (Matching 1 Top + 2 Bottom Grid) ---
    elif active_tab == tab_options[2]:
        st.subheader("Global Trending Reach (Geospatial)")
        fig_map = draw_choropleth_map(filtered_df)
        st.plotly_chart(fig_map, width="stretch")
        with st.expander("📌 Geospatial Insights & Technical Notes"):
            st.markdown(
                """
- **Insight:** Compares average trending view volume across US, GB, DE, FR, and RU markets.

- **Projection:** Natural Earth projection with ISO-3 country mapping and discrete view scales.

                """
            )

        st.divider()

        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("Categorical Flow: Velocity to Engagement")
            fig_sankey = draw_sankey_diagram(filtered_df)
            st.plotly_chart(fig_sankey, width="stretch")
            with st.expander("📌 Sankey Flow Insights & Technical Notes"):
                st.markdown(
                    """
- **Insight:** Traces how top content categories transition into fast (<3d) or slow (3d+) trending speeds and their resulting engagement tiers.

                    """
                )

        with col_right:
            st.subheader("Semantic Tag Relationship Network")
            fig_network = draw_tag_network(filtered_df)
            st.plotly_chart(fig_network, width="stretch")
            with st.expander("📌 Network Graph Insights & Technical Notes"):
                st.markdown(
                    """
- **Insight:** Uncovers clusters of real YouTube tags that frequently co-occur on trending videos across dominant categories.

- **Layout Algorithm:** NetworkX Fruchterman-Reingold force-directed layout.

                    """
                )

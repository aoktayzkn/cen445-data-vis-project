import streamlit as st
import pandas as pd
import numpy as np

from uye1_veri_isleme import load_and_clean, fig_histogram, fig_correlation, fig_treemap
from src.visualizations import draw_choropleth_map, draw_sankey_diagram, draw_tag_network

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
st.markdown("Interactive exploratory analysis, machine learning insights, and multi-dimensional visualizations across global markets.")

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
        st.plotly_chart(fig_histogram(filtered_df, metric_choice), use_container_width=True)
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
            st.plotly_chart(fig_correlation(filtered_df), use_container_width=True)
            with st.expander("📌 Correlation Matrix Notes"):
                st.markdown(
                    """
                    - **Insight:** Strong positive correlation exists between `Views`, `Likes` (~0.78), and `Comments` (~0.65), whereas title length and uppercase character counts show near-zero linear impact on view volume.
                    """
                )
        with col_right:
            st.plotly_chart(fig_treemap(filtered_df), use_container_width=True)
            with st.expander("📌 Hierarchical Treemap Notes"):
                st.markdown(
                    """
                    - **Insight:** `Entertainment`, `Sports`, and `Film & Animation` dominate cumulative trending views, led by major broadcast and studio channels.
                    """
                )

    # --- TAB 2: MEMBER 2 ML ---
    elif active_tab == tab_options[1]:
        st.info("Machine Learning models (Clustering / Classification / Regression) by Member 2 will be integrated here.")

    # --- TAB 3: MEMBER 3 ADVANCED VISUALIZATIONS (Matching 1 Top + 2 Bottom Grid) ---
    elif active_tab == tab_options[2]:
        st.subheader("Global Trending Reach (Geospatial)")
        fig_map = draw_choropleth_map(filtered_df)
        st.plotly_chart(fig_map, use_container_width=True)
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
            st.plotly_chart(fig_sankey, use_container_width=True)
            with st.expander("📌 Sankey Flow Insights & Technical Notes"):
                st.markdown(
                    """
                    - **Insight:** Traces how top content categories transition into fast (<3d) or slow (3d+) trending speeds and their resulting engagement tiers.
                    """
                )

        with col_right:
            st.subheader("Semantic Tag Relationship Network")
            fig_network = draw_tag_network(filtered_df)
            st.plotly_chart(fig_network, use_container_width=True)
            with st.expander("📌 Network Graph Insights & Technical Notes"):
                st.markdown(
                    """
                    - **Insight:** Uncovers clusters of real YouTube tags that frequently co-occur on trending videos across dominant categories.
                    - **Layout Algorithm:** NetworkX Fruchterman-Reingold force-directed layout.
                    """
                )
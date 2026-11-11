import streamlit as st
import pandas as pd
import numpy as np
from src.visualizations import draw_choropleth_map, draw_sankey_diagram, draw_tag_network

# Page configuration
st.set_page_config(
    page_title="YouTube Trending Analytics Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Mock data
@st.cache_data
def load_mock_data():
    countries = ["US", "GB", "DE", "FR", "CA"]
    categories = ["Music", "Entertainment", "Gaming", "Education", "Comedy"]
    
    np.random.seed(42)
    data = {
        "country": np.random.choice(countries, 500),
        "category": np.random.choice(categories, 500),
        "views": np.random.randint(10000, 5000000, 500),
        "likes": np.random.randint(500, 250000, 500),
        "comment_count": np.random.randint(50, 50000, 500),
        "comments_disabled": np.random.choice([False, True], 500, p=[0.9, 0.1]),
        "days_to_trend": np.random.randint(1, 10, 500)
    }
    return pd.DataFrame(data)

df = load_mock_data()

# Sidebar filter panel
st.sidebar.title("Filter Panel")

selected_countries = st.sidebar.multiselect(
    "Select Countries:",
    options=sorted(df["country"].unique()),
    default=sorted(df["country"].unique())
)

min_views, max_views = int(df["views"].min()), int(df["views"].max())
views_range = st.sidebar.slider(
    "Views Range:",
    min_value=min_views,
    max_value=max_views,
    value=(min_views, max_views),
    step=50000
)

hide_disabled_comments = st.sidebar.checkbox(
    "Exclude Comments-Disabled Videos",
    value=True
)

# Data filtering
filtered_df = df[
    (df["country"].isin(selected_countries)) &
    (df["views"] >= views_range[0]) &
    (df["views"] <= views_range[1])
]

if hide_disabled_comments:
    filtered_df = filtered_df[~filtered_df["comments_disabled"]]

# Main header & metrics
st.title("🎬 YouTube Trending Videos Analytics Dashboard")
st.caption("CEN445 Introduction to Data Visualization — Term Project")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Filtered Videos", f"{len(filtered_df):,}")
with col2:
    st.metric("Average Views", f"{int(filtered_df['views'].mean() if len(filtered_df) > 0 else 0):,}")
with col3:
    st.metric("Selected Countries", f"{len(selected_countries)}")

st.divider()

# Navigation tabs
tab_options = [
    "📊 Exploratory Data Analysis (Fatih Aksüt)",
    "🤖 Machine Learning Insights (Yusuf Yalçın)",
    "🌐 Advanced Analytics & Flows (Ali Oktay Özkan)"
]

active_tab = st.radio(
    "Navigation",
    options=tab_options,
    horizontal=True,
    label_visibility="collapsed",
    key="active_dashboard_tab"
)

if active_tab == tab_options[0]:
    st.info("Exploratory visualizations will be placed here: Treemap, Correlation Heatmap, and Distribution plots.")

elif active_tab == tab_options[1]:
    st.info("Machine learning models and charts will be placed here: K-Means Clustering and Parallel Coordinates.")

elif active_tab == tab_options[2]:
    st.subheader("Geospatial Distributions, Flows & Tag Networks")
    
    if len(filtered_df) > 0 and len(selected_countries) > 0:
        fig_map = draw_choropleth_map(filtered_df)
        st.plotly_chart(fig_map, use_container_width=True)
        
        with st.expander("📌 Chart Insights & Technical Notes"):
            st.markdown(
                """
                - **Insight:** Compares average engagement and view volumes across trending videos per target country.
                - **Interactivity:** Supports hover tooltips with formatted metrics, continuous zoom, and dynamic recalculation via sidebar filters.
                - **Projection:** Natural Earth projection with standardized oceanic and continental basemap styling.
                """
            )
        st.divider()
        
        # 2. Sankey Diagram
        st.subheader("Categorical Flow: Velocity to Engagement")
        fig_sankey = draw_sankey_diagram(filtered_df)
        st.plotly_chart(fig_sankey, use_container_width=True)
        
        with st.expander("📌 Sankey Flow Insights & Technical Notes"):
            st.markdown(
                """
                - **Insight:** Traces how content categories branch into trending velocity (fast vs. slow) and ultimately lead to high or standard viewer engagement.
                - **Interactivity:** Hover over nodes and links to inspect exact video flow counts and percentage distributions dynamically driven by sidebar filters.
                """
            )
        st.divider()

        # 3. Tag Co-occurrence Network
        st.subheader("Semantic Tag Relationship Network")
        fig_network = draw_tag_network(filtered_df)
        st.plotly_chart(fig_network, use_container_width=True)

        with st.expander("📌 Network Graph Insights & Technical Notes"):
            st.markdown(
                """
                - **Insight:** Uncovers clusters of tags that frequently appear together on trending videos, revealing algorithmic metadata patterns.
                - **Interactivity:** Densely connected hub tags are sized and colored by network degree. Hover reveals co-occurrence strength.
                - **Layout Algorithm:** NetworkX Spring layout (Fruchterman-Reingold force-directed placement).
                """
            )
    else:
        st.warning("No data matches the selected filter criteria. Please adjust sidebar filters.")
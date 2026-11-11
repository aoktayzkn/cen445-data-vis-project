import plotly.express as px
import pandas as pd
import numpy as np

COUNTRY_MAP = {
    "US": "USA",
    "GB": "GBR",
    "DE": "DEU",
    "FR": "FRA",
    "CA": "CAN"
}

def draw_choropleth_map(df: pd.DataFrame):
    country_summary = (
        df.groupby("country")
        .agg(avg_views=("views", "mean"), video_count=("views", "count"))
        .reset_index()
    )
    
    country_summary["iso_alpha"] = country_summary["country"].map(COUNTRY_MAP)
    
    min_v = country_summary["avg_views"].min()
    max_v = country_summary["avg_views"].max()
    
    if pd.isna(min_v) or pd.isna(max_v) or min_v == max_v:
        tickvals = [min_v if not pd.isna(min_v) else 0]
        ticktext = [f"{tickvals[0]/1e6:.2f}M"]
    else:
        tickvals = np.linspace(min_v, max_v, 5).tolist()
        ticktext = [f"{v/1e6:.2f}M" for v in tickvals]
    
    # Modernized color palette
    fig = px.choropleth(
        country_summary,
        locations="iso_alpha",
        color="avg_views",
        hover_name="country",
        hover_data={"iso_alpha": False, "avg_views": ":,.0f", "video_count": ":,"},
        color_continuous_scale="YlOrRd",
        title="Average Video Views by Target Country"
    )
    
    fig.update_geos(
        showframe=False,
        showcoastlines=True,
        coastlinecolor="#64748b",
        coastlinewidth=0.7,
        showland=True,
        landcolor="#334155",
        showocean=True,
        oceancolor="#1e293b",
        showlakes=True,
        lakecolor="#1e293b",
        projection_type="natural earth",
        bgcolor="rgba(0,0,0,0)"
    )
    
    fig.data[0].colorbar.update(
        title="Avg Views",
        thickness=14,
        len=0.75,
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext
    )
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=50, b=10),
        font=dict(family="sans-serif", size=12)
    )

    return fig

import plotly.graph_objects as go

def draw_sankey_diagram(df: pd.DataFrame):
    # Segment data into analytical tiers
    temp_df = df.copy()
    
    # 1. Trending Speed Segmentation
    temp_df["trend_speed"] = temp_df["days_to_trend"].apply(
        lambda x: "Fast Trend (<3d)" if x <= 2 else "Slow Trend (3d+)"
    )
    
    # 2. Engagement Rate Segmentation (Likes per View)
    engagement_rate = temp_df["likes"] / (temp_df["views"] + 1)
    median_engagement = engagement_rate.median()
    temp_df["engagement_level"] = engagement_rate.apply(
        lambda x: "High Engagement" if x >= median_engagement else "Standard Engagement"
    )
    
    # Flow 1: Category -> Trending Speed
    flow1 = temp_df.groupby(["category", "trend_speed"]).size().reset_index(name="count")
    flow1.columns = ["source_label", "target_label", "count"]
    
    # Flow 2: Trending Speed -> Engagement Level
    flow2 = temp_df.groupby(["trend_speed", "engagement_level"]).size().reset_index(name="count")
    flow2.columns = ["source_label", "target_label", "count"]
    
    flows = pd.concat([flow1, flow2], ignore_index=True)
    
    # Unique nodes mapping
    all_nodes = list(pd.unique(flows[["source_label", "target_label"]].values.ravel("K")))
    node_to_idx = {name: idx for idx, name in enumerate(all_nodes)}
    
    sources = [node_to_idx[s] for s in flows["source_label"]]
    targets = [node_to_idx[t] for t in flows["target_label"]]
    values = flows["count"].tolist()
    
    # Node colors
    node_colors = []
    for node in all_nodes:
        if "Engagement" in node:
            node_colors.append("#22c55e" if "High" in node else "#94a3b8")
        elif "Trend" in node:
            node_colors.append("#38bdf8" if "Fast" in node else "#f59e0b")
        else:
            node_colors.append("#ef4444")
            
    # Build Sankey Figure
    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=18,
                    thickness=16,
                    line=dict(color="#0f172a", width=0.5),
                    label=all_nodes,
                    color=node_colors
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color="rgba(148, 163, 184, 0.22)"
                )
            )
        ]
    )
    
    fig.update_layout(
        title="Content Velocity & Audience Engagement Flow",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=50, b=10),
        font=dict(family="sans-serif", size=12, color="#e2e8f0")
    )
    
    return fig

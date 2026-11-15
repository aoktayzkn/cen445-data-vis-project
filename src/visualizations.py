import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import networkx as nx
from itertools import combinations
from collections import Counter

COUNTRY_MAP = {
    "US": "USA",
    "GB": "GBR",
    "DE": "DEU",
    "FR": "FRA",
    "RU": "RUS"
}

GEO_METRICS = {
    "avg_views": {
        "label": "Average Views",
        "colorbar": "Avg Views",
        "scale": "YlOrRd",
        "fmt": lambda v: f"{v/1e6:.2f}M" if v >= 1e6 else f"{v/1e3:.0f}K",
    },
    "engagement_rate": {
        "label": "Engagement Rate (Likes + Comments %)",
        "colorbar": "Engage %",
        "scale": "Viridis",
        "fmt": lambda v: f"{v:.2f}%",
    },
    "comments_per_1k": {
        "label": "Comment Intensity (per 1K Views)",
        "colorbar": "Comm / 1K",
        "scale": "Plasma",
        "fmt": lambda v: f"{v:.1f}",
    },
    "avg_days_to_trend": {
        "label": "Trending Velocity (Avg Days to Trend)",
        "colorbar": "Days",
        "scale": "Tealrose",
        "fmt": lambda v: f"{v:.1f}d",
    },
}


def draw_choropleth_map(df: pd.DataFrame, metric_key: str = "avg_views"):
    if metric_key not in GEO_METRICS:
        metric_key = "avg_views"
    m_cfg = GEO_METRICS[metric_key]

    temp = df.copy()
    views_safe = temp["views"].replace(0, np.nan)
    comments_col = temp["comment_count"] if "comment_count" in temp.columns else pd.Series(0, index=temp.index)
    days_col = temp["days_to_trend"] if "days_to_trend" in temp.columns else pd.Series(2.0, index=temp.index)

    temp["_eng_rate"] = ((temp["likes"] + comments_col) / views_safe).fillna(0.0) * 100.0
    temp["_comm_1k"] = (comments_col / views_safe).fillna(0.0) * 1000.0
    temp["_days"] = days_col.fillna(2.0)

    group_col = "country_code" if "country_code" in temp.columns else "country"
    group_keys = ["country_code", "country"] if ("country_code" in temp.columns and "country" in temp.columns) else [group_col]
    hover_col = "country" if "country" in group_keys else group_col

    country_summary = (
        temp.groupby(group_keys)
        .agg(
            avg_views=("views", "mean"),
            engagement_rate=("_eng_rate", "mean"),
            comments_per_1k=("_comm_1k", "mean"),
            avg_days_to_trend=("_days", "mean"),
            video_count=("views", "count"),
            top_category=("category", lambda s: s.mode().iloc[0] if not s.mode().empty else "N/A"),
        )
        .reset_index()
    )

    country_summary["iso_alpha"] = country_summary[group_col].map(COUNTRY_MAP)

    min_v = country_summary[metric_key].min()
    max_v = country_summary[metric_key].max()

    if pd.isna(min_v) or pd.isna(max_v) or min_v == max_v:
        tickvals = [min_v if not pd.isna(min_v) else 0.0]
        ticktext = [m_cfg["fmt"](tickvals[0])]
    else:
        tickvals = np.linspace(min_v, max_v, 5).tolist()
        ticktext = [m_cfg["fmt"](v) for v in tickvals]

    fig = px.choropleth(
        country_summary,
        locations="iso_alpha",
        color=metric_key,
        hover_name=hover_col,
        custom_data=[
            "avg_views",
            "engagement_rate",
            "comments_per_1k",
            "avg_days_to_trend",
            "video_count",
            "top_category",
        ],
        color_continuous_scale=m_cfg["scale"],
        title=f"Geospatial Distribution: {m_cfg['label']}",
    )

    fig.update_traces(
        hovertemplate=(
            "<b>%{hovertext}</b><br><br>"
            "Avg Views: %{customdata[0]:,.0f}<br>"
            "Engagement Rate: %{customdata[1]:.2f}%<br>"
            "Comments / 1K Views: %{customdata[2]:.2f}<br>"
            "Avg Days to Trend: %{customdata[3]:.2f} days<br>"
            "Dominant Category: %{customdata[5]}<br>"
            "Trending Videos: %{customdata[4]:,}<extra></extra>"
        )
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
        lataxis_range=[22, 78],
        bgcolor="rgba(0,0,0,0)",
    )

    fig.data[0].colorbar.update(
        title=m_cfg["colorbar"],
        thickness=14,
        len=0.75,
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext,
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=50, b=10),
        font=dict(family="sans-serif", size=12),
    )

    if not country_summary.empty:
        if metric_key == "avg_days_to_trend":
            best_row = country_summary.loc[country_summary[metric_key].idxmin()]
        else:
            best_row = country_summary.loc[country_summary[metric_key].idxmax()]
        lead_country = str(best_row[hover_col])
        lead_val_fmt = m_cfg["fmt"](float(best_row[metric_key]))
    else:
        lead_country = "N/A"
        lead_val_fmt = "0"

    geo_stats = {
        "lead_country": lead_country,
        "lead_val_fmt": lead_val_fmt,
        "metric_label": m_cfg["label"],
    }

    return fig, geo_stats


def draw_sankey_diagram(df: pd.DataFrame, top_n_categories: int = 8):
    temp_df = df.copy()

    top_cats = temp_df["category"].value_counts().nlargest(top_n_categories).index
    temp_df["clean_category"] = temp_df["category"].where(
        temp_df["category"].isin(top_cats), "Other Categories"
    )

    temp_df["trend_speed"] = temp_df["days_to_trend"].apply(
        lambda x: "Fast Trend (<3d)" if x <= 2 else "Slow Trend (3d+)"
    )

    engagement_rate = temp_df["likes"] / (temp_df["views"] + 1)
    median_engagement = engagement_rate.median()
    temp_df["engagement_level"] = engagement_rate.apply(
        lambda x: "High Engagement" if x >= median_engagement else "Standard Engagement"
    )

    flow1 = temp_df.groupby(["clean_category", "trend_speed"]).size().reset_index(name="count")
    flow1.columns = ["source_label", "target_label", "count"]

    flow2 = temp_df.groupby(["trend_speed", "engagement_level"]).size().reset_index(name="count")
    flow2.columns = ["source_label", "target_label", "count"]

    flows = pd.concat([flow1, flow2], ignore_index=True)

    all_nodes = list(pd.unique(flows[["source_label", "target_label"]].values.ravel("K")))
    node_to_idx = {name: idx for idx, name in enumerate(all_nodes)}

    sources = [node_to_idx[s] for s in flows["source_label"]]
    targets = [node_to_idx[t] for t in flows["target_label"]]
    values = flows["count"].tolist()

    node_colors = []
    for node in all_nodes:
        if "Engagement" in node:
            node_colors.append("#22c55e" if "High" in node else "#94a3b8")
        elif "Trend" in node:
            node_colors.append("#38bdf8" if "Fast" in node else "#f59e0b")
        else:
            node_colors.append("#ef4444")

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=18,
                    thickness=16,
                    line=dict(color="#0f172a", width=0.5),
                    label=all_nodes,
                    color=node_colors,
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color="rgba(148, 163, 184, 0.22)",
                ),
            )
        ]
    )

    fig.update_layout(
        title=f"Content Velocity & Engagement Flow (Top {top_n_categories} Categories)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=50, b=10),
        font=dict(family="sans-serif", size=12),
    )

    total_vids = len(temp_df)
    fast_mask = temp_df["trend_speed"] == "Fast Trend (<3d)"
    fast_pct = (fast_mask.sum() / total_vids * 100.0) if total_vids > 0 else 0.0
    fast_high_eng_pct = (
        ((fast_mask & (temp_df["engagement_level"] == "High Engagement")).sum() / max(1, fast_mask.sum())) * 100.0
    )

    sankey_stats = {
        "fast_pct": fast_pct,
        "fast_high_eng_pct": fast_high_eng_pct,
    }

    return fig, sankey_stats


def draw_tag_network(
    df: pd.DataFrame,
    top_n_tags: int = 22,
    min_cooccurrence: int = 5,
    sizing_mode: str = "frequency",
):
    palette = ["#ef4444", "#f59e0b", "#3b82f6", "#10b981", "#8b5cf6", "#ec4899", "#06b6d4", "#84cc16"]

    tag_counts = Counter()
    tag_cat_counter = {}
    video_tag_lists = []

    for tags_raw, cat in zip(df["tags"].astype(str), df["category"].astype(str)):
        parsed = [
            t.strip().lower()
            for t in tags_raw.replace('"', "").split("|")
            if t.strip() and t.strip() != "[none]" and len(t.strip()) > 1
        ]
        unique_tags = list(set(parsed))
        video_tag_lists.append(unique_tags)
        for t in unique_tags:
            tag_counts[t] += 1
            if t not in tag_cat_counter:
                tag_cat_counter[t] = Counter()
            tag_cat_counter[t][cat] += 1

    frequent_tags = {tag for tag, _ in tag_counts.most_common(top_n_tags)}

    dominant_categories = {}
    for tag in frequent_tags:
        dominant_categories[tag] = tag_cat_counter[tag].most_common(1)[0][0]

    unique_cats = sorted(set(dominant_categories.values()))
    category_colors = {cat: palette[i % len(palette)] for i, cat in enumerate(unique_cats)}

    edge_counts = Counter()
    for tags in video_tag_lists:
        filtered = [t for t in tags if t in frequent_tags]
        if len(filtered) >= 2:
            for pair in combinations(sorted(filtered), 2):
                edge_counts[pair] += 1

    G = nx.Graph()
    for tag in frequent_tags:
        G.add_node(tag, count=tag_counts[tag], category=dominant_categories[tag])

    for (t1, t2), weight in edge_counts.items():
        if weight >= min_cooccurrence:
            G.add_edge(t1, t2, weight=weight)

    G.remove_nodes_from(list(nx.isolates(G)))
    if len(G.nodes) == 0:
        fig = go.Figure()
        fig.update_layout(
            title="Semantic Tag Network (Insufficient Co-occurrences)",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        return fig, {
            "density": 0.0,
            "top_bridge_tag": "N/A",
            "top_bridge_score": 0.0,
            "num_nodes": 0,
            "num_edges": 0,
        }

    # Graf Teorisi Merkezilik (Centrality) Hesaplamaları
    deg_centrality = nx.degree_centrality(G)
    bet_centrality = nx.betweenness_centrality(G, weight="weight")
    net_density = nx.density(G)

    top_bridge_node = max(bet_centrality, key=bet_centrality.get) if bet_centrality else "N/A"
    top_bridge_score = bet_centrality.get(top_bridge_node, 0.0)

    pos = nx.spring_layout(G, seed=42, k=0.85, iterations=80)

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1.0, color="rgba(148, 163, 184, 0.25)"),
        hoverinfo="none",
        mode="lines",
        showlegend=False,
    )

    if sizing_mode == "degree":
        raw_vals = [deg_centrality[n] for n in G.nodes()]
        mode_title = "Sized by Degree Centrality"
    elif sizing_mode == "betweenness":
        raw_vals = [bet_centrality[n] for n in G.nodes()]
        mode_title = "Sized by Betweenness Centrality"
    else:
        raw_vals = [G.nodes[n]["count"] for n in G.nodes()]
        mode_title = "Sized by Tag Frequency"

    min_v, max_v = min(raw_vals), max(raw_vals)

    node_x = []
    node_y = []
    node_text = []
    node_size = []
    node_color = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        degree = G.degree[node]
        freq = G.nodes[node]["count"]
        cat = G.nodes[node]["category"]
        d_cent = deg_centrality[node]
        b_cent = bet_centrality[node]

        val = d_cent if sizing_mode == "degree" else (b_cent if sizing_mode == "betweenness" else freq)
        norm_size = 24 if max_v == min_v else 16 + ((val - min_v) / (max_v - min_v)) * 20
        node_size.append(norm_size)
        node_color.append(category_colors[cat])
        node_text.append(
            f"<b>#{node}</b><br>"
            f"Dominant Category: {cat}<br>"
            f"Video Occurrences: {freq:,}<br>"
            f"Connected Tags (Degree): {degree} ({d_cent:.2f})<br>"
            f"Bridge Score (Betweenness): {b_cent:.3f}"
        )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=[f"#{n}" for n in G.nodes()],
        textposition="top center",
        textfont=dict(family="sans-serif", size=11),
        hoverinfo="text",
        hovertext=node_text,
        showlegend=False,
        marker=dict(
            color=node_color,
            size=node_size,
            line=dict(width=1.5, color="#0f172a"),
        ),
    )

    legend_traces = []
    for cat in unique_cats:
        legend_traces.append(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=cat,
                marker=dict(size=10, color=category_colors[cat]),
                showlegend=True,
            )
        )

    fig = go.Figure(
        data=[edge_trace, node_trace] + legend_traces,
        layout=go.Layout(
            title=dict(
                text=f"Semantic Tag Network ({mode_title})",
                font=dict(size=14),
                y=0.96,
                x=0,
                xanchor="left",
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
                font=dict(size=10),
            ),
            hovermode="closest",
            margin=dict(b=20, l=10, r=10, t=85),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            font=dict(family="sans-serif", size=12),
        ),
    )

    net_stats = {
        "density": net_density,
        "top_bridge_tag": f"#{top_bridge_node}",
        "top_bridge_score": top_bridge_score,
        "num_nodes": len(G.nodes),
        "num_edges": len(G.edges),
    }

    return fig, net_stats
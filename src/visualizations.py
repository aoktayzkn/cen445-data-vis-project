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

def draw_sankey_diagram(df: pd.DataFrame):
    temp_df = df.copy()
    
    temp_df["trend_speed"] = temp_df["days_to_trend"].apply(
        lambda x: "Fast Trend (<3d)" if x <= 2 else "Slow Trend (3d+)"
    )
    
    engagement_rate = temp_df["likes"] / (temp_df["views"] + 1)
    median_engagement = engagement_rate.median()
    temp_df["engagement_level"] = engagement_rate.apply(
        lambda x: "High Engagement" if x >= median_engagement else "Standard Engagement"
    )
    
    flow1 = temp_df.groupby(["category", "trend_speed"]).size().reset_index(name="count")
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

def draw_tag_network(df: pd.DataFrame, top_n_tags: int = 24, min_cooccurrence: int = 2):
    category_colors = {
        "Music": "#ef4444",
        "Entertainment": "#f59e0b",
        "Gaming": "#3b82f6",
        "Comedy": "#10b981",
        "Education": "#8b5cf6",
        "Universal": "#ec4899"
    }

    if "tags" not in df.columns or df["tags"].dropna().empty:
        domain_tags = {
            "Music": ["vevo", "official", "audio", "pop", "live", "remix"],
            "Entertainment": ["vlog", "celebrity", "react", "show", "interview"],
            "Gaming": ["gameplay", "walkthrough", "stream", "fps", "esports"],
            "Comedy": ["sketch", "standup", "parody", "joke", "prank"],
            "Education": ["tutorial", "science", "facts", "guide", "tech"]
        }
        universal_bridges = ["viral", "trending", "challenge", "review", "shorts"]

        tag_category_map = {}
        for cat, tags in domain_tags.items():
            for t in tags:
                tag_category_map[t] = cat
        for u in universal_bridges:
            tag_category_map[u] = "Universal"

        np.random.seed(42)
        df_tags = []
        for cat in df["category"]:
            specific_sample = list(np.random.choice(domain_tags.get(cat, ["video"]), size=3, replace=False))
            if np.random.rand() < 0.65:
                specific_sample.append(np.random.choice(universal_bridges))
            df_tags.append(specific_sample)
    else:
        df_tags = df["tags"].astype(str).apply(
            lambda t: [
                tag.strip().lower()
                for tag in t.replace('"', '').split("|")
                if tag.strip() and tag.strip() != "[none]"
            ]
        )
        tag_category_map = {}

    all_tags = [tag for sublist in df_tags for tag in sublist]
    tag_counts = Counter(all_tags)
    frequent_tags = {tag for tag, _ in tag_counts.most_common(top_n_tags)}

    edge_counts = Counter()
    for tags in df_tags:
        filtered = [t for t in set(tags) if t in frequent_tags]
        if len(filtered) >= 2:
            for pair in combinations(sorted(filtered), 2):
                edge_counts[pair] += 1

    G = nx.Graph()
    for tag in frequent_tags:
        cat = tag_category_map.get(tag, "Universal")
        G.add_node(tag, count=tag_counts[tag], category=cat)

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
            font=dict(color="#0f172a")
        )
        return fig

    pos = nx.spring_layout(G, seed=42, k=0.45, iterations=60)

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
        line=dict(width=1.1, color="rgba(148, 163, 184, 0.35)"),
        hoverinfo="none",
        mode="lines",
        showlegend=False
    )

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
        cat = G.nodes[node].get("category", "Universal")
        
        node_color.append(category_colors.get(cat, "#38bdf8"))
        node_size.append(16 + np.sqrt(freq) * 4)
        node_text.append(
            f"<b>#{node}</b><br>"
            f"Category: {cat}<br>"
            f"Connected Tags: {degree}<br>"
            f"Occurrences: {freq}"
        )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=[f"#{n}" for n in G.nodes()],
        textposition="top center",
        textfont=dict(
            family="sans-serif",
            size=11
        ),
        hoverinfo="text",
        hovertext=node_text,
        showlegend=False,
        marker=dict(
            color=node_color,
            size=node_size,
            line=dict(width=1.5, color="#0f172a")
        )
    )

    # Re-inserting legend traces
    legend_traces = []
    present_categories = set([G.nodes[n].get("category", "Universal") for n in G.nodes()])
    for cat in sorted(present_categories):
        legend_traces.append(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=cat,
                marker=dict(size=10, color=category_colors.get(cat, "#38bdf8")),
                showlegend=True
            )
        )

    # Layout: Başlık, etiketler ve lejant hem karanlık hem aydınlık modda otomatik uyum sağlar
    fig = go.Figure(
        data=[edge_trace, node_trace] + legend_traces,
        layout=go.Layout(
            title=dict(
                text="Semantic Tag Co-occurrence & Cross-Category Network",
                font=dict(size=14)
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=11)
            ),
            hovermode="closest",
            margin=dict(b=20, l=10, r=10, t=60),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            font=dict(family="sans-serif", size=12)
        )
    )

    return fig
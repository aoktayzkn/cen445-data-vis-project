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

def draw_choropleth_map(df: pd.DataFrame):
    group_col = "country_code" if "country_code" in df.columns else "country"
    
    if "country_code" in df.columns and "country" in df.columns:
        country_summary = (
            df.groupby(["country_code", "country"])
            .agg(avg_views=("views", "mean"), video_count=("views", "count"))
            .reset_index()
        )
        hover_col = "country"
    else:
        country_summary = (
            df.groupby(group_col)
            .agg(avg_views=("views", "mean"), video_count=("views", "count"))
            .reset_index()
        )
        hover_col = group_col
        
    country_summary["iso_alpha"] = country_summary[group_col].map(COUNTRY_MAP)
    
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
        hover_name=hover_col,
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
        lataxis_range=[22, 78],  # Kuzey yarımküre odaklı temiz kadraj
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
    
    # En popüler 8 kategoriyi koru, kalanları 'Other' altında grupla (görsel karmaşayı önler)
    top_cats = temp_df["category"].value_counts().nlargest(8).index
    temp_df["clean_category"] = temp_df["category"].where(temp_df["category"].isin(top_cats), "Other Categories")
    
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
        title="Content Velocity & Audience Engagement Flow (Top Categories)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=50, b=10),
        font=dict(family="sans-serif", size=12)
    )
    
    return fig

def draw_tag_network(df: pd.DataFrame, top_n_tags: int = 22, min_cooccurrence: int = 5):
    palette = ["#ef4444", "#f59e0b", "#3b82f6", "#10b981", "#8b5cf6", "#ec4899", "#06b6d4", "#84cc16"]
    
    # Gerçek etiketleri ve ait oldukları baskın kategorileri çıkar
    tag_counts = Counter()
    tag_cat_counter = {}
    video_tag_lists = []
    
    for tags_raw, cat in zip(df["tags"].astype(str), df["category"].astype(str)):
        parsed = [
            t.strip().lower()
            for t in tags_raw.replace('"', '').split("|")
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
    
    # Her etikete en çok geçtiği baskın kategoriyi ata
    dominant_categories = {}
    for tag in frequent_tags:
        dominant_categories[tag] = tag_cat_counter[tag].most_common(1)[0][0]
        
    unique_cats = sorted(set(dominant_categories.values()))
    category_colors = {cat: palette[i % len(palette)] for i, cat in enumerate(unique_cats)}

    # Birliktelik (co-occurrence) hesapla
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
            plot_bgcolor="rgba(0,0,0,0)"
        )
        return fig

    # Düğümleri birbirinden uzaklaştıran daha geniş yayılım (k=0.85)
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
        showlegend=False
    )

    # Düğüm boyutlarını gerçek frekansa göre 16px - 34px arasına normalize et
    freqs = [G.nodes[n]["count"] for n in G.nodes()]
    min_f, max_f = min(freqs), max(freqs)

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
        
        norm_size = 24 if max_f == min_f else 16 + ((freq - min_f) / (max_f - min_f)) * 18
        node_size.append(norm_size)
        node_color.append(category_colors[cat])
        node_text.append(
            f"<b>#{node}</b><br>"
            f"Dominant Category: {cat}<br>"
            f"Connected Tags: {degree}<br>"
            f"Video Occurrences: {freq:,}"
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
            line=dict(width=1.5, color="#0f172a")
        )
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
                showlegend=True
            )
        )

    fig = go.Figure(
        data=[edge_trace, node_trace] + legend_traces,
        layout=go.Layout(
            title=dict(
                text="Semantic Tag Co-occurrence & Dominant Category Network",
                font=dict(size=14),
                y=0.96,
                x=0,
                xanchor="left"
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
                font=dict(size=10)
            ),
            hovermode="closest",
            margin=dict(b=20, l=10, r=10, t=85),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            font=dict(family="sans-serif", size=12)
        )
    )

    return fig
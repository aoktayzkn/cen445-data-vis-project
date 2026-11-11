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
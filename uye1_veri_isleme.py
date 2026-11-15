"""
Üye 1 - Veri Ön İşleme, Metrik Analizi ve Görselleştirmeler
Veri seti: Kaggle "Trending YouTube Video Statistics" (datasnaek/youtube-new)
Ülkeler : GB, FR, US, RU, DE   ->  {XX}videos.csv + {XX}_category_id.json

Klasör düzeni (varsayılan):
    proje/
      data/GBvideos.csv, GB_category_id.json, FRvideos.csv, ...
      uye1_veri_isleme.py

Kullanım:
  - Doğrudan çalıştırma : python uye1_veri_isleme.py   (grafikleri tarayıcıda açar)
  - Streamlit'te        : from uye1_veri_isleme import load_and_clean, fig_histogram, fig_correlation, fig_treemap
"""

import json
import os

import pandas as pd
import plotly.express as px

DATA_DIR = "data"  # CSV/JSON dosyalarını proje ana klasörüne koyduysan "." yap
COUNTRIES = {
    "GB": "Great Britain",
    "FR": "France",
    "US": "USA",
    "RU": "Russia",
    "DE": "Germany",
}


# ---------------------------------------------------------------------------
# 1) VERİ ÖN İŞLEME
# ---------------------------------------------------------------------------
def load_categories(json_path: str, country_code: str) -> pd.DataFrame:
    """JSON dosyasından category_id -> category_name eşleşmesini çıkarır."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    rows = [
        {"category_id": int(item["id"]), "category_name": item["snippet"]["title"]}
        for item in data["items"]
    ]
    cats = pd.DataFrame(rows)
    cats["country_code"] = country_code
    return cats


def load_country(code: str, data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Tek bir ülkenin CSV'sini okuyup kategori isimleriyle birleştirir."""
    csv_path = os.path.join(data_dir, f"{code}videos.csv")
    json_path = os.path.join(data_dir, f"{code}_category_id.json")

    # Bozuk karakterler (RU/DE/FR) okumayı durdurmasın diye errors='replace'
    df = pd.read_csv(csv_path, encoding="utf-8", encoding_errors="replace")
    df["country_code"] = code
    df["country"] = COUNTRIES[code]

    cats = load_categories(json_path, code)
    return df.merge(cats, on=["country_code", "category_id"], how="left")


def remove_outliers_percentile(df: pd.DataFrame, cols, upper_q: float = 0.99) -> pd.DataFrame:
    """Verilen sütunlarda üst yüzdelik dilimin (varsayılan %99) üzerindeki satırları siler."""
    mask = pd.Series(True, index=df.index)
    for col in cols:
        mask &= df[col] <= df[col].quantile(upper_q)
    return df[mask]


def load_and_clean(data_dir: str = DATA_DIR, outlier_q: float = 0.99) -> pd.DataFrame:
    """5 ülkeyi birleştirir, eksik/aykırı değerleri temizler, yeni özellikleri ekler."""
    df = pd.concat([load_country(c, data_dir) for c in COUNTRIES], ignore_index=True)
    df["category_name"] = df["category_name"].fillna("Unknown")

    # --- Eksik değerler ---
    # Analiz için kritik sütunlarda eksik varsa satırı sil
    df = df.dropna(subset=["title", "views", "likes", "comment_count", "tags"])
    # Açıklama metin alanıdır; eksikse boş string ile doldur
    if "description" in df.columns:
        df["description"] = df["description"].fillna("")

    # --- Tekrarlayan kayıtlar ---
    # Aynı video aynı ülkede birden fazla gün trend olabilir; son (en yüksek) kaydı tut
    df = df.drop_duplicates(subset=["country_code", "video_id"], keep="last")

    # --- Aykırı değerler (views, likes, comment_count) ---
    df = remove_outliers_percentile(df, ["views", "likes", "comment_count"], upper_q=outlier_q)

    # --- 2) METRİK ANALİZİ: yeni özellikler ---
    df["title_upper_count"] = df["title"].apply(lambda t: sum(c.isupper() for c in t))
    df["tag_count"] = df["tags"].apply(
        lambda t: 0 if t.strip() == "[none]" else len(t.split("|"))
    )
    df["title_length"] = df["title"].str.len()

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3) GÖRSELLEŞTİRMELER (Plotly)
# ---------------------------------------------------------------------------
def fig_histogram(df: pd.DataFrame, column: str = "views"):
    """Distribution histogram of views/likes colored by country."""
    labels = {"views": "Views", "likes": "Likes"}
    metric_label = labels.get(column, column.capitalize())
    
    fig = px.histogram(
        df,
        x=column,
        color="country",
        nbins=50,
        barmode="overlay",
        opacity=0.65,
        marginal="box",
        title=f"{metric_label} Distribution Across Countries",
        labels={column: metric_label, "country": "Country"},
    )
    fig.update_layout(
        yaxis_title="Video Count",
        bargap=0.05,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def fig_correlation(df: pd.DataFrame):
    """Correlation heatmap across numerical and title/tag features."""
    cols = {
        "views": "Views",
        "likes": "Likes",
        "comment_count": "Comments",
        "title_upper_count": "Uppercase Count",
        "tag_count": "Tag Count",
        "title_length": "Title Length",
    }
    corr = df[list(cols)].corr().rename(index=cols, columns=cols).round(2)
    fig = px.imshow(
        corr,
        text_auto=True,
        zmin=-1,
        zmax=1,
        color_continuous_scale="RdBu_r",
        aspect="auto",
        title="Feature Correlation Heatmap",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


import textwrap

def fig_treemap(df: pd.DataFrame, top_categories: int = 8, top_channels: int = 4):
    """Hierarchical treemap of total views by Top Categories -> Top Channels."""
    top_cats = (
        df.groupby("category_name")["views"]
        .sum()
        .nlargest(top_categories)
        .index
    )
    filtered = df[df["category_name"].isin(top_cats)].copy()
    
    grouped = filtered.groupby(["category_name", "channel_title"], as_index=False)["views"].sum()
    grouped = (
        grouped.sort_values("views", ascending=False)
        .groupby("category_name")
        .head(top_channels)
    )
    
    # Dar dikey kutulara da sığması için 9 karakterde bir satır kır (kelime ortasından bölme)
    def wrap_label(name: str) -> str:
        clean = name if len(name) <= 20 else name[:18] + ".."
        return "<br>".join(textwrap.wrap(clean, width=9, break_long_words=False))
    
    grouped["wrapped_channel"] = grouped["channel_title"].apply(wrap_label)
    
    fig = px.treemap(
        grouped,
        path=[px.Constant("All Categories"), "category_name", "wrapped_channel"],
        values="views",
        color="views",
        color_continuous_scale="YlOrRd",
        custom_data=["channel_title"],
        title="Total View Share by Top Categories & Channels",
        labels={"views": "Total Views"},
    )
    fig.update_traces(
        maxdepth=3,
        textposition="middle center",
        tiling=dict(pad=2),
        marker=dict(line=dict(width=1.2, color="#1e293b")),
        hovertemplate="<b>%{customdata[0]}</b><br>Total Views: %{value:,.0f}<extra></extra>",
        root_color="rgba(0,0,0,0)"
    )
    fig.update_layout(
        margin=dict(t=50, l=5, r=5, b=5),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig

# ---------------------------------------------------------------------------
# Doğrudan çalıştırma
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    data = load_and_clean()
    print(f"Temizlenmiş veri: {data.shape[0]} satır, {data.shape[1]} sütun")
    print(data.groupby("country").size())
    print(data[["title", "country", "category_name", "views", "title_upper_count",
                "tag_count", "title_length"]].head())

    fig_histogram(data, "views").show()
    fig_correlation(data).show()
    fig_treemap(data).show()

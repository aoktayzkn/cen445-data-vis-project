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
    df["category_name"] = df["category_name"].fillna("Bilinmeyen")

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
    """Temel: İzlenme/beğeni dağılımı histogramı (ülkelere göre renkli)."""
    labels = {"views": "İzlenme", "likes": "Beğeni"}
    fig = px.histogram(
        df, x=column, color="country", nbins=50, barmode="overlay", opacity=0.65,
        marginal="box",
        title=f"{labels.get(column, column)} Sayısı Dağılımı",
        labels={column: labels.get(column, column), "country": "Ülke"},
    )
    fig.update_layout(yaxis_title="Video Sayısı", bargap=0.05)
    return fig


def fig_correlation(df: pd.DataFrame):
    """Gelişmiş: Korelasyon ısı haritası."""
    cols = {
        "views": "İzlenme",
        "likes": "Beğeni",
        "comment_count": "Yorum Sayısı",
        "title_upper_count": "Başlık Büyük Harf",
        "tag_count": "Etiket Sayısı",
        "title_length": "Başlık Uzunluğu",
    }
    corr = df[list(cols)].corr().rename(index=cols, columns=cols).round(2)
    fig = px.imshow(
        corr, text_auto=True, zmin=-1, zmax=1,
        color_continuous_scale="RdBu_r", aspect="auto",
        title="Metrikler Arası Korelasyon Isı Haritası",
    )
    return fig


def fig_treemap(df: pd.DataFrame, top_channels: int = 5):
    """Gelişmiş: Kategori -> Kanal hiyerarşisinde toplam izlenme treemap'i."""
    grouped = df.groupby(["category_name", "channel_title"], as_index=False)["views"].sum()
    # Her kategoride en çok izlenen N kanalı göster (grafiği okunabilir tutar)
    grouped = (
        grouped.sort_values("views", ascending=False)
        .groupby("category_name")
        .head(top_channels)
    )
    fig = px.treemap(
        grouped, path=[px.Constant("Tüm Kategoriler"), "category_name", "channel_title"],
        values="views", color="views", color_continuous_scale="YlOrRd",
        title="Kategorilere Göre Toplam İzlenme Payı (Treemap)",
    )
    fig.update_traces(hovertemplate="<b>%{label}</b><br>İzlenme: %{value:,.0f}<extra></extra>")
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

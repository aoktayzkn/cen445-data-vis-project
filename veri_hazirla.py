"""
Veri küçültme scripti (bir kez çalıştırılır).

raw_data/ klasöründeki orijinal Kaggle dosyalarını okur, GitHub'a uygun küçük
dosyalar üretip data/ klasörüne yazar. uye1_veri_isleme.py data/ klasörünü okur.

Yapılanlar:
  1) Aynı video_id'nin tekrar eden kayıtlarından sonuncusu tutulur
  2) Kullanılmayan ağır sütunlar (description, thumbnail_link) atılır
  3) Her ülkeden en fazla N_PER_COUNTRY satır rastgele örneklenir (sabit seed)
  4) Kategori JSON dosyaları olduğu gibi kopyalanır

Kullanım: python veri_hazirla.py
"""

import os
import shutil

import pandas as pd

RAW_DIR = "raw_data"      # Kaggle'dan indirilen orijinal dosyalar (GitHub'a YÜKLEME)
OUT_DIR = "data"          # küçültülmüş dosyalar (GitHub'a yüklenecek)
COUNTRIES = ["GB", "FR", "US", "RU", "DE"]
N_PER_COUNTRY = 5000
DROP_COLS = ["description", "thumbnail_link"]
SEED = 42

os.makedirs(OUT_DIR, exist_ok=True)

for code in COUNTRIES:
    src = os.path.join(RAW_DIR, f"{code}videos.csv")
    df = pd.read_csv(src, encoding="utf-8", encoding_errors="replace")

    df = df.drop_duplicates(subset="video_id", keep="last")
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df = df.sample(n=min(N_PER_COUNTRY, len(df)), random_state=SEED)

    dst = os.path.join(OUT_DIR, f"{code}videos.csv")
    df.to_csv(dst, index=False, encoding="utf-8")
    size_mb = os.path.getsize(dst) / 1024 / 1024
    print(f"{code}: {len(df)} satır, {size_mb:.1f} MB")

    shutil.copy(
        os.path.join(RAW_DIR, f"{code}_category_id.json"),
        os.path.join(OUT_DIR, f"{code}_category_id.json"),
    )

print("Bitti. data/ klasörünü kontrol et.")

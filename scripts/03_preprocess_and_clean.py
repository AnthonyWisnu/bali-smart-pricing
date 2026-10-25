"""
=============================================================================
MODUL 03: PEMBERSIHAN DATA, REKAYASA FITUR RASIO & GEOSPASIAL HAVERSINE
Bali Smart Villa & Homestay Pricing Engine (Pure ML Pipeline)
=============================================================================
Fungsi:
1. Membaca dataset master (bali_airbnb_enriched_master.csv).
2. Menghapus duplikasi listing berdasarkan room_id dan menyaring anomali
   harga ekstrem / sewa tahunan (Rp 150.000 <= price_idr <= Rp 10.000.000
   dan price_per_bedroom <= Rp 4.000.000).
3. Merekayasa fitur rasio kapasitas struktural objektif (guests_per_bedroom,
   baths_per_bedroom, beds_per_bedroom, total_rooms_index).
4. Menghitung total fasilitas mewah objektif (total_luxury_amenities)
   tanpa pembobotan manual subjektif.
5. Menghitung jarak geospasial fisik Haversine ke 6 sentra pariwisata utama
   dan bandara di Bali.
6. Melakukan pembagian Stratified Train-Test Split (80% Latih : 20% Uji)
   berdasarkan kabupaten.
=============================================================================
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_CSV = os.path.join(BASE_DIR, "data", "processed", "bali_airbnb_enriched_master.csv")
TRAIN_CSV = os.path.join(BASE_DIR, "data", "processed", "dataset_train.csv")
TEST_CSV = os.path.join(BASE_DIR, "data", "processed", "dataset_test.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

TOURISM_HUBS = {
    'canggu_seminyak': (-8.6650, 115.1450),
    'ubud_center': (-8.5069, 115.2625),
    'uluwatu': (-8.8149, 115.0884),
    'sanur': (-8.6913, 115.2628),
    'kintamani': (-8.2423, 115.3752),
    'airport': (-8.7467, 115.1668)
}


def haversine_km(lat1, lon1, lat2, lon2):
    r_earth = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2.0 * r_earth * np.arcsin(np.sqrt(a))


def main():
    print("=" * 75)
    print(" [MODUL 03] PREPROCESSING & REKAYASA FITUR DATA-DRIVEN MURNI")
    print("=" * 75)

    os.makedirs(MODELS_DIR, exist_ok=True)
    df = pd.read_csv(INPUT_CSV)
    initial_rows = len(df)

    # 1. Deduplikasi berdasarkan room_id
    df = df.drop_duplicates(subset=['room_id']).copy()
    dedup_rows = len(df)

    # 2. Penyaringan batas wajar tarif 1 malam (Rp 150rb s.d. Rp 10 Juta & <= Rp 4 Juta/kamar)
    df = df[df['price_idr'].notnull()].copy()
    df['price_per_bedroom'] = df['price_idr'] / df['bedrooms'].clip(lower=1)
    df_clean = df[
        (df['price_idr'] >= 150_000) &
        (df['price_idr'] <= 10_000_000) &
        (df['price_per_bedroom'] <= 4_000_000)
    ].copy().reset_index(drop=True)

    print(f"[*] Total observasi awal dari Modul 02          : {initial_rows} baris")
    print(f"[*] Duplikasi identik dibersihkan               : {initial_rows - dedup_rows} baris")
    print(f"[*] Total penginapan unik                       : {dedup_rows} unit")
    print(f"[*] Penginapan pasar reguler dipertahankan      : {len(df_clean)} / {dedup_rows} unit ({len(df_clean)/dedup_rows*100:.1f}%)")
    print(f"[*] Median tarif pasar reguler (1 malam)        : Rp {df_clean['price_idr'].median():,.0f} / malam")

    # 3. Rekayasa Fitur Rasio Kapasitas & Total Fasilitas Mewah Objektif
    df_clean['guests_per_bedroom'] = df_clean['accommodates'] / df_clean['bedrooms']
    df_clean['baths_per_bedroom'] = df_clean['bathrooms'] / df_clean['bedrooms']
    df_clean['beds_per_bedroom'] = df_clean['beds'] / df_clean['bedrooms']
    df_clean['total_rooms_index'] = df_clean['bedrooms'] + df_clean['bathrooms']
    
    # Hitungan objektif tanpa pembobotan manual subjektif
    df_clean['total_luxury_amenities'] = (
        df_clean['has_private_pool'] +
        df_clean['has_pool'] +
        df_clean['has_bathtub'] +
        df_clean['has_jacuzzi'] +
        df_clean['has_service_staff']
    )

    # 4. Rekayasa Fitur Jarak Geospasial Haversine ke Sentra Wisata & Bandara
    for hub_name, (h_lat, h_lon) in TOURISM_HUBS.items():
        df_clean[f'dist_{hub_name}_km'] = haversine_km(df_clean['latitude'], df_clean['longitude'], h_lat, h_lon)

    df_clean['min_dist_tourism_hub_km'] = df_clean[[
        'dist_canggu_seminyak_km', 'dist_ubud_center_km', 'dist_uluwatu_km', 'dist_sanur_km'
    ]].min(axis=1)

    # 5. Transformasi Logaritmik Target
    df_clean['log_price'] = np.log1p(df_clean['price_idr'])

    # 6. Stratified Train-Test Split (80% Train, 20% Test)
    train_df, test_df = train_test_split(
        df_clean,
        test_size=0.20,
        random_state=42,
        stratify=df_clean['regency']
    )
    train_df = train_df.copy().reset_index(drop=True)
    test_df = test_df.copy().reset_index(drop=True)

    feature_cols = [
        'rating', 'is_new_listing', 'review_count', 'is_room_only',
        'latitude', 'longitude', 'accommodates', 'bedrooms', 'beds', 'bathrooms',
        'is_superhost', 'has_self_checkin', 'has_private_pool', 'has_pool',
        'has_bathtub', 'has_parking', 'has_workspace',
        'nlp_luxury', 'nlp_beach_ocean', 'nlp_nature_view', 'nlp_budget_cozy',
        'has_jacuzzi', 'has_service_staff',
        'guests_per_bedroom', 'baths_per_bedroom', 'beds_per_bedroom',
        'total_rooms_index', 'total_luxury_amenities',
        'dist_canggu_seminyak_km', 'dist_ubud_center_km', 'dist_uluwatu_km',
        'dist_sanur_km', 'dist_kintamani_km', 'dist_airport_km',
        'min_dist_tourism_hub_km',
        'regency', 'sub_district', 'property_type'
    ]
    meta_cols = ['room_id', 'name', 'price_idr', 'log_price']

    train_out = train_df[meta_cols + feature_cols]
    test_out = test_df[meta_cols + feature_cols]

    train_out.to_csv(TRAIN_CSV, index=False)
    test_out.to_csv(TEST_CSV, index=False)

    # Simpan master bersih untuk keperluan peta eksplorasi aplikasi
    df_master_clean = pd.concat([train_out, test_out], axis=0, ignore_index=True)
    df_master_clean.to_csv(INPUT_CSV, index=False)

    print(f"\n[SUKSES] Preprocessing & Rekayasa Fitur Data-Driven Murni Selesai:")
    print(f" -> Data Latih (Train Set 80%)   : {len(train_out)} baris ({TRAIN_CSV})")
    print(f" -> Data Uji   (Test Set 20%)    : {len(test_out)} baris ({TEST_CSV})")
    print(f" -> Total Fitur Prediktor Murni  : {len(feature_cols)} fitur (Tanpa Target Lag / Bumbu Heuristik)")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()

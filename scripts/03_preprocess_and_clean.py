"""
=============================================================================
MODUL 03: PEMBERSIHAN DATA, HAVERSINE GEOSPASIAL & SPATIAL HEDONIC IDW
Bali Smart Villa & Homestay Pricing Engine
=============================================================================
Fungsi:
1. Membaca dataset hasil penggabungan (bali_airbnb_enriched_master.csv).
2. Menghapus duplikasi listing berdasarkan room_id dan menyaring anomali
   harga ekstrem / sewa tahunan (Rp 150.000 <= price_idr <= Rp 10.000.000
   dan price_per_bedroom <= Rp 4.000.000).
3. Merekayasa fitur rasio kapasitas struktural, skor fasilitas mewah komposit
   (luxury_amenity_score), serta jarak Haversine ke 6 pusat wisata utama Bali.
4. Melakukan pembagian Stratified Train-Test Split (80% Latih : 20% Uji).
5. Menghitung fitur Out-of-Fold Spatial Hedonic Price-Per-Bedroom IDW
   (spatial_knn_log_price, spatial_knn_median_log_price, spatial_knn_log_ppb,
   dan expected_spatial_bedroom_log_price) yang di-fit murni pada Data Latih
   (0% Data Leakage).
=============================================================================
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

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

SPATIAL_SCALE_VEC = np.array([12.0, 12.0, 1.8, 3.5, 0.6])
K_NEIGHBORS = 12


def haversine_km(lat1, lon1, lat2, lon2):
    r_earth = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2.0 * r_earth * np.arcsin(np.sqrt(a))


def main():
    print("=" * 75)
    print(" [MODUL 03] PREPROCESSING, HAVERSINE & SPATIAL HEDONIC IDW (0% LEAKAGE)")
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

    # 3. Rekayasa Fitur Rasio Kapasitas & Skor Fasilitas Mewah Komposit
    df_clean['guests_per_bedroom'] = df_clean['accommodates'] / df_clean['bedrooms']
    df_clean['baths_per_bedroom'] = df_clean['bathrooms'] / df_clean['bedrooms']
    df_clean['beds_per_bedroom'] = df_clean['beds'] / df_clean['bedrooms']
    df_clean['total_rooms_index'] = df_clean['bedrooms'] + df_clean['bathrooms']
    df_clean['luxury_amenity_score'] = (
        df_clean['has_private_pool'] * 2 +
        df_clean['has_bathtub'] +
        df_clean['has_jacuzzi'] * 2 +
        df_clean['nlp_luxury'] * 2 +
        df_clean['nlp_beach_ocean'] * 2 +
        df_clean['has_service_staff']
    )

    # 4. Rekayasa Fitur Jarak Geospasial Haversine ke Pusat Wisata
    for hub_name, (h_lat, h_lon) in TOURISM_HUBS.items():
        df_clean[f'dist_{hub_name}_km'] = haversine_km(df_clean['latitude'], df_clean['longitude'], h_lat, h_lon)

    df_clean['min_dist_tourism_hub_km'] = df_clean[[
        'dist_canggu_seminyak_km', 'dist_ubud_center_km', 'dist_uluwatu_km', 'dist_sanur_km'
    ]].min(axis=1)

    # 5. Transformasi Logaritmik Target
    df_clean['log_price'] = np.log1p(df_clean['price_idr'])

    # 6. Stratified Train-Test Split (80% Train, 20% Test) SEBELUM perhitungan Spatial KNN
    train_df, test_df = train_test_split(
        df_clean,
        test_size=0.20,
        random_state=42,
        stratify=df_clean['regency']
    )
    train_df = train_df.copy().reset_index(drop=True)
    test_df = test_df.copy().reset_index(drop=True)

    # 7. Spatial-Capacity KNN + Spatial Hedonic Price-Per-Bedroom IDW (Murni dari Data Latih -> 0% Data Leakage!)
    space_cols = ['latitude', 'longitude', 'bedrooms', 'is_room_only', 'luxury_amenity_score']
    X_space_train = train_df[space_cols].values * SPATIAL_SCALE_VEC
    X_space_test = test_df[space_cols].values * SPATIAL_SCALE_VEC

    nn_model = NearestNeighbors(n_neighbors=16, metric='euclidean')
    nn_model.fit(X_space_train)

    dists_tr, idx_tr = nn_model.kneighbors(X_space_train)
    dists_te, idx_te = nn_model.kneighbors(X_space_test, n_neighbors=15)

    tr_log_ppb = np.log1p(train_df['price_idr'].values / train_df['bedrooms'].values)

    w_tr = 1.0 / (dists_tr[:, 1:13] + 0.15)
    w_tr = w_tr / np.sum(w_tr, axis=1, keepdims=True)
    w_te = 1.0 / (dists_te[:, :12] + 0.15)
    w_te = w_te / np.sum(w_te, axis=1, keepdims=True)

    train_df['spatial_knn_log_price'] = np.sum(train_df['log_price'].values[idx_tr[:, 1:13]] * w_tr, axis=1)
    train_df['spatial_knn_median_log_price'] = np.median(train_df['log_price'].values[idx_tr[:, 1:13]], axis=1)
    train_df['spatial_knn_log_ppb'] = np.sum(tr_log_ppb[idx_tr[:, 1:13]] * w_tr, axis=1)
    train_df['expected_spatial_bedroom_log_price'] = np.log1p(np.expm1(train_df['spatial_knn_log_ppb']) * train_df['bedrooms'])

    test_df['spatial_knn_log_price'] = np.sum(train_df['log_price'].values[idx_te[:, :12]] * w_te, axis=1)
    test_df['spatial_knn_median_log_price'] = np.median(train_df['log_price'].values[idx_te[:, :12]], axis=1)
    test_df['spatial_knn_log_ppb'] = np.sum(tr_log_ppb[idx_te[:, :12]] * w_te, axis=1)
    test_df['expected_spatial_bedroom_log_price'] = np.log1p(np.expm1(test_df['spatial_knn_log_ppb']) * test_df['bedrooms'])

    # Simpan objek referensi KNN agar aplikasi kalkulator dapat menghitung harga tetangga secara real-time
    knn_ref_path = os.path.join(MODELS_DIR, "spatial_knn_reference.joblib")
    joblib.dump({
        'nn_model': nn_model,
        'scale_vec': SPATIAL_SCALE_VEC,
        'space_cols': space_cols,
        'train_log_prices': train_df['log_price'].values,
        'train_log_ppb': tr_log_ppb,
        'k_neighbors': K_NEIGHBORS
    }, knn_ref_path)

    feature_cols = [
        'rating', 'is_new_listing', 'review_count', 'is_room_only',
        'latitude', 'longitude', 'accommodates', 'bedrooms', 'beds', 'bathrooms',
        'is_superhost', 'has_self_checkin', 'has_private_pool', 'has_pool',
        'has_bathtub', 'has_parking', 'has_workspace',
        'nlp_luxury', 'nlp_beach_ocean', 'nlp_nature_view', 'nlp_budget_cozy',
        'has_jacuzzi', 'has_service_staff',
        'guests_per_bedroom', 'baths_per_bedroom', 'beds_per_bedroom',
        'total_rooms_index', 'luxury_amenity_score',
        'dist_canggu_seminyak_km', 'dist_ubud_center_km', 'dist_uluwatu_km',
        'dist_sanur_km', 'dist_kintamani_km', 'dist_airport_km',
        'min_dist_tourism_hub_km',
        'spatial_knn_log_price', 'spatial_knn_median_log_price',
        'spatial_knn_log_ppb', 'expected_spatial_bedroom_log_price',
        'regency', 'sub_district', 'property_type'
    ]
    meta_cols = ['room_id', 'name', 'price_idr', 'log_price']

    train_out = train_df[meta_cols + feature_cols]
    test_out = test_df[meta_cols + feature_cols]

    train_out.to_csv(TRAIN_CSV, index=False)
    test_out.to_csv(TEST_CSV, index=False)

    # Simpan pula master bersih untuk keperluan peta eksplorasi aplikasi
    df_master_clean = pd.concat([train_out, test_out], axis=0, ignore_index=True)
    df_master_clean.to_csv(INPUT_CSV, index=False)

    print(f"\n[SUKSES] Preprocessing & Rekayasa Fitur Spatial Hedonic IDW Selesai:")
    print(f" -> Data Latih (Train Set 80%)   : {len(train_out)} baris ({TRAIN_CSV})")
    print(f" -> Data Uji   (Test Set 20%)    : {len(test_out)} baris ({TEST_CSV})")
    print(f" -> Referensi KNN Spasial IDW    : {knn_ref_path} (Murni Data Latih, 0% Leakage)")
    print(f" -> Total Fitur Pemodelan        : {len(feature_cols)} fitur prediktor")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()

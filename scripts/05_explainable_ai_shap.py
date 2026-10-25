"""
=============================================================================
MODUL 05: EXPLAINABLE ARTIFICIAL INTELLIGENCE (XAI) BERBASIS SHAP
Bali Smart Villa & Homestay Pricing Engine
=============================================================================
Fungsi:
1. Memuat model terbaik CatBoost yang telah dilatih.
2. Menghitung SHAP Values (Shapley Additive exPlanations) untuk mengukur
   kontribusi marginal setiap atribut terhadap penentuan tarif sewa harian.
3. Memetakan seluruh variabel teknis menjadi Bahasa Indonesia alami yang
   mudah dipahami masyarakat umum dan pemilik penginapan.
4. Menghasilkan visualisasi beresolusi tinggi (300 DPI):
   - reports/shap_feature_importance.png
   - reports/shap_summary_beeswarm.png
   - reports/shap_waterfall_example.png
=============================================================================
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from catboost import CatBoostRegressor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
TEST_CSV = os.path.join(BASE_DIR, "data", "processed", "dataset_test.csv")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

FEATURE_LABEL_MAP = {
    'property_type': 'Tipe Penginapan (Villa/Homestay)',
    'has_bathtub': 'Fasilitas Bak Mandi (Bathtub)',
    'bedrooms': 'Jumlah Kamar Tidur',
    'accommodates': 'Kapasitas Tamu (Orang)',
    'longitude': 'Koordinat Bujur Lokasi',
    'latitude': 'Koordinat Lintang Lokasi',
    'has_pool': 'Kolam Renang (Pribadi/Bersama)',
    'has_private_pool': 'Kolam Renang Pribadi',
    'sub_district': 'Kawasan / Kecamatan Wisata',
    'regency': 'Kabupaten / Kota di Bali',
    'beds': 'Jumlah Tempat Tidur',
    'bathrooms': 'Jumlah Kamar Mandi',
    'is_room_only': 'Lingkup Sewa Kamar Pribadi vs Unit Utuh',
    'baths_per_bedroom': 'Rasio Kamar Mandi per Kamar Tidur',
    'beds_per_bedroom': 'Rasio Ranjang per Kamar Tidur',
    'guests_per_bedroom': 'Kepadatan Tamu per Kamar',
    'total_rooms_index': 'Skala Total Ruangan Properti',
    'is_superhost': 'Status Pelayanan Superhost',
    'has_parking': 'Area Parkir Kendaraan Gratis',
    'has_workspace': 'Ruang / Meja Kerja Khusus',
    'has_self_checkin': 'Layanan Check-in Mandiri',
    'nlp_luxury': 'Desain Eksklusif / Villa Mewah',
    'nlp_beach_ocean': 'Dekat Pantai / Pemandangan Laut',
    'nlp_nature_view': 'Pemandangan Sawah / Lembah / Gunung',
    'nlp_budget_cozy': 'Kategori Ekonomis / Transit',
    'dist_canggu_seminyak_km': 'Jarak ke Sentra Canggu & Seminyak (km)',
    'dist_ubud_center_km': 'Jarak ke Pusat Wisata Ubud (km)',
    'dist_uluwatu_km': 'Jarak ke Kawasan Tebing Uluwatu (km)',
    'dist_sanur_km': 'Jarak ke Pantai Sanur (km)',
    'dist_kintamani_km': 'Jarak ke Kawasan Kaldera Kintamani (km)',
    'dist_airport_km': 'Jarak ke Bandara Ngurah Rai (km)',
    'min_dist_tourism_hub_km': 'Jarak ke Pusat Wisata Terdekat (km)',
    'spatial_knn5_log_price': 'Rata-rata Harga 5 Penginapan Terdekat (Mikro)',
    'spatial_knn12_log_price': 'Rata-rata Harga 12 Penginapan Terdekat (Kawasan)',
    'spatial_knn_log_price': 'Rata-rata Harga Pasar Tetangga Sekitar',
    'spatial_knn_median_log_price': 'Harga Tengah (Median) Properti Tetangga',
    'spatial_knn_log_ppb': 'Tarif Pasar per Kamar di Lingkungan Sekitar',
    'expected_spatial_bedroom_log_price': 'Ekspektasi Harga Spasial Sesuai Jumlah Kamar',
    'luxury_amenity_score': 'Skor Kelengkapan Fasilitas Mewah',
    'total_luxury_amenities': 'Total Fasilitas Mewah',
    'has_jacuzzi': 'Fasilitas Jacuzzi / Hot Tub Pribadi',
    'has_service_staff': 'Layanan Staf / Sarapan / Butler',
    'review_count': 'Jumlah Ulasan Tamu',
    'rating': 'Skor Kepuasan Tamu (Rating)',
    'is_new_listing': 'Status Properti Baru Terdaftar'
}


def main():
    print("=" * 75)
    print(" [MODUL 05] ANALISIS EXPLAINABLE AI (SHAP) - BAHASA ALAMI RAMAH PUBLIK")
    print("=" * 75)

    os.makedirs(REPORTS_DIR, exist_ok=True)

    meta = joblib.load(os.path.join(MODELS_DIR, "feature_metadata.joblib"))
    cat_cols = meta["cat_cols"]
    num_cols = meta["num_cols"]
    all_feature_cols = num_cols + cat_cols

    model_path = os.path.join(MODELS_DIR, "catboost_pricing_model.cbm")
    model = CatBoostRegressor()
    model.load_model(model_path)
    print(f"[*] Model CatBoost berhasil dimuat dari: {model_path}")

    test_df = pd.read_csv(TEST_CSV)
    X_test = test_df[all_feature_cols].copy()
    for c in cat_cols:
        X_test[c] = X_test[c].astype(str)

    print(f"[*] Jumlah sampel evaluasi SHAP: {len(X_test)} data uji")

    print("[*] Menghitung nilai kontribusi Shapley (TreeExplainer)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    natural_names = [FEATURE_LABEL_MAP.get(c, c) for c in all_feature_cols]
    shap_values.feature_names = natural_names

    # 1. Visualisasi Bar Feature Importance
    plt.figure(figsize=(10, 8))
    shap.plots.bar(shap_values, max_display=12, show=False)
    plt.title("Faktor & Fasilitas Paling Berpengaruh terhadap Tarif Sewa Harian di Bali", fontsize=12, fontweight='bold', pad=15)
    plt.xlabel("Rata-rata Tingkat Pengaruh terhadap Nilai Sewa", fontsize=10)
    plt.tight_layout()
    fi_path = os.path.join(REPORTS_DIR, "shap_feature_importance.png")
    plt.savefig(fi_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Visualisasi Feature Importance tersimpan ke: {fi_path}")

    # 2. Visualisasi Beeswarm Summary
    plt.figure(figsize=(11, 8))
    shap.plots.beeswarm(shap_values, max_display=12, show=False)
    plt.title("Arah Pengaruh Lokasi & Fasilitas terhadap Tarif Sewa Harian di Bali", fontsize=12, fontweight='bold', pad=15)
    plt.xlabel("Pengaruh terhadap Harga (Kanan = Menaikkan Tarif, Kiri = Tarif Ekonomis)", fontsize=10)
    plt.tight_layout()
    bee_path = os.path.join(REPORTS_DIR, "shap_summary_beeswarm.png")
    plt.savefig(bee_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Visualisasi Beeswarm Summary tersimpan ke: {bee_path}")

    # 3. Visualisasi Waterfall Kasus Riil
    plt.figure(figsize=(10, 7))
    sample_idx = 0
    shap.plots.waterfall(shap_values[sample_idx], max_display=10, show=False)
    plt.title("Contoh Perjalanan Penentuan Tarif Harian pada 1 Penginapan di Bali", fontsize=12, fontweight='bold', pad=15)
    plt.xlabel("Kontribusi Penyesuaian Nilai Fasilitas & Lokasi", fontsize=10)
    plt.tight_layout()
    wf_path = os.path.join(REPORTS_DIR, "shap_waterfall_example.png")
    plt.savefig(wf_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Visualisasi Waterfall Kasus Riil tersimpan ke: {wf_path}")

    print("\n" + "=" * 75)
    print(" [SUKSES] Visualisasi SHAP Bahasa Alami berhasil dibuat!")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()

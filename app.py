"""
=============================================================================
BALI SMART PRICING ENGINE - WEB APPLICATION (STREAMLIT)
Sistem Penaksir Tarif Sewa Wajar Villa & Homestay di Bali Berbasis ML Murni
=============================================================================
Pengembang:
- I Kadek Adi Sunetra (2305551053)
- Anthony Wisnu Jati (2305551138)
Dosen Pengampu:
- Yogiswara Dharma Putra, S.TI., M.T.
Program Studi Teknologi Informasi, Fakultas Teknik, Universitas Udayana
=============================================================================
"""

import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from catboost import CatBoostRegressor

# -----------------------------------------------------------------------------
# 1. KONFIGURASI HALAMAN
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Kalkulator Tarif Sewa Villa & Homestay Bali",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .app-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 24px 30px;
        margin-bottom: 24px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    .app-header h1 {
        color: #ffffff !important;
        font-size: 1.7rem !important;
        font-weight: 800 !important;
        margin: 0 !important;
        letter-spacing: -0.02em;
    }
    .app-header p {
        color: #94a3b8 !important;
        font-size: 0.92rem !important;
        margin: 6px 0 0 0 !important;
    }

    .badge-bar {
        display: flex;
        gap: 10px;
        margin-top: 14px;
        flex-wrap: wrap;
    }
    .badge-item {
        background: #1e293b;
        color: #38bdf8;
        border: 1px solid #0284c7;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 20px;
    }
    .badge-item-green {
        background: #064e3b;
        color: #34d399;
        border: 1px solid #059669;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 20px;
    }

    .price-box {
        background: linear-gradient(135deg, #064e3b 0%, #022c22 100%);
        border: 1.5px solid #10b981;
        border-radius: 14px;
        padding: 24px;
        text-align: center;
        margin-bottom: 22px;
        box-shadow: 0 8px 24px rgba(16, 185, 129, 0.15);
    }
    .price-box-label {
        font-size: 0.95rem;
        font-weight: 700;
        color: #6ee7b7;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .price-box-amount {
        font-size: 2.5rem;
        font-weight: 800;
        color: #ffffff;
        margin: 8px 0;
        letter-spacing: -0.02em;
    }
    .price-box-range {
        font-size: 0.88rem;
        color: #a7f3d0;
        font-weight: 500;
    }

    .stat-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 14px 18px;
        text-align: center;
    }
    .stat-card-title {
        font-size: 0.75rem;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stat-card-value {
        font-size: 1.15rem;
        color: #f8fafc;
        font-weight: 700;
        margin-top: 4px;
    }

    .form-section-title {
        font-size: 0.88rem;
        font-weight: 700;
        color: #38bdf8;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 12px;
        border-bottom: 1px solid #334155;
        padding-bottom: 6px;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 2px solid #334155;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        border-radius: 8px 8px 0 0;
        gap: 6px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #94a3b8;
        font-weight: 600;
        font-size: 0.92rem;
    }
    .stTabs [aria-selected="true"] {
        color: #38bdf8 !important;
        border-bottom: 3px solid #38bdf8 !important;
        background-color: rgba(56, 189, 248, 0.08) !important;
    }

    .figure-caption {
        font-size: 0.85rem;
        color: #cbd5e1;
        font-style: italic;
        margin-top: 6px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. DEFINISI SENTRA WISATA & PEMUATAN MODEL & METADATA
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "bali_airbnb_enriched_master.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "catboost_pricing_model.cbm")
META_PATH = os.path.join(BASE_DIR, "models", "feature_metadata.joblib")
BENCHMARK_PATH = os.path.join(BASE_DIR, "reports", "model_benchmark_results.csv")
TIER_PATH = os.path.join(BASE_DIR, "reports", "tier_accuracy_breakdown.csv")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")


def haversine_km(lat1, lon1, lat2, lon2):
    r_earth = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return float(2.0 * r_earth * np.arcsin(np.sqrt(a)))


@st.cache_data
def load_dataset():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        df['sub_district'] = df['sub_district'].fillna('Unknown').astype(str)
        df['regency'] = df['regency'].fillna('Unknown').astype(str)
        df['property_type'] = df['property_type'].fillna('Villa').astype(str)
        return df
    return pd.DataFrame()


@st.cache_resource
def load_pricing_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(META_PATH):
        cb = CatBoostRegressor()
        cb.load_model(MODEL_PATH)
        meta = joblib.load(META_PATH)
        return cb, meta
    return None, None


df_master = load_dataset()
cb_model, meta = load_pricing_model()

# -----------------------------------------------------------------------------
# 3. HEADER UTAMA
# -----------------------------------------------------------------------------
total_units = len(df_master) if not df_master.empty else 1000
st.markdown(f"""
<div class="app-header">
    <h1>Kalkulator Tarif Sewa Wajar Villa & Homestay Bali</h1>
    <p>Sistem Inferensi Harga Sewa Berbasis Data-Driven Machine Learning (Benchmark 3 Model Utama)</p>
    <div class="badge-bar">
        <span class="badge-item-green">CatBoost Champion Model</span>
        <span class="badge-item">Dataset Empiris: {total_units:,} Properti Riil</span>
        <span class="badge-item">4 Kategori Alami Bali</span>
        <span class="badge-item">38 Fitur Murni (0% Data Leakage)</span>
    </div>
</div>
""", unsafe_allow_html=True)

if cb_model is None or meta is None:
    st.error("Model CatBoost atau berkas metadata belum tersedia. Silakan jalankan modul pelatihan di scripts/04_train_and_benchmark_models.py.")
    st.stop()

# Koordinat median per kecamatan untuk penentuan titik otomatis
coord_lookup = df_master.groupby(['regency', 'sub_district'])[['latitude', 'longitude']].median().reset_index()

# -----------------------------------------------------------------------------
# 4. TAB NAVIGASI UTAMA
# -----------------------------------------------------------------------------
tab_calc, tab_map, tab_bench, tab_xai = st.tabs([
    "Cek Estimasi Harga Properti",
    "Peta Persebaran Lokasi",
    "Akurasi Model Prediksi",
    "Faktor Penentu Harga (SHAP)"
])

# =============================================================================
# TAB 1: FORMULIR INFERENSI TARIF SEWA WAJAR (ML MURNI)
# =============================================================================
with tab_calc:
    st.markdown("### Simulasi Karakteristik Properti & Penaksiran Tarif Sewa Wajar")
    st.caption("Masukkan spesifikasi fisik, lokasi, dan fasilitas penginapan. Model Machine Learning CatBoost akan menaksir tarif sewa wajar per malam secara objektif.")

    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container(border=True):
            st.markdown('<div class="form-section-title">1. Lokasi & Tipe Properti</div>', unsafe_allow_html=True)

            regencies = sorted(df_master['regency'].unique().tolist())
            def_reg_idx = regencies.index("Gianyar") if "Gianyar" in regencies else 0
            sel_regency = st.selectbox("Kabupaten / Kota:", regencies, index=def_reg_idx)

            sub_list = sorted(df_master[df_master['regency'] == sel_regency]['sub_district'].unique().tolist())
            def_sub_idx = sub_list.index("Ubud") if "Ubud" in sub_list else 0
            sel_subdistrict = st.selectbox("Kecamatan / Kawasan Wisata:", sub_list, index=def_sub_idx)

            # 4 Kategori Alami Bali yang Seimbang
            type_options = ["Villa", "Private Room/Homestay", "Cottage/Bungalow", "Entire House"]
            type_labels = {
                "Villa": "Villa Pribadi",
                "Private Room/Homestay": "Kamar Sewa / Homestay",
                "Cottage/Bungalow": "Bungalow / Cottage Unik",
                "Entire House": "Rumah / Hunian Utuh"
            }
            sel_property_type = st.selectbox(
                "Tipe Penginapan:",
                type_options,
                format_func=lambda x: type_labels.get(x, x),
                index=0
            )

            is_room_selected = sel_property_type == "Private Room/Homestay"

            match_coords = coord_lookup[(coord_lookup['regency'] == sel_regency) & (coord_lookup['sub_district'] == sel_subdistrict)]
            if not match_coords.empty:
                cur_lat = float(match_coords['latitude'].values[0])
                cur_lon = float(match_coords['longitude'].values[0])
            else:
                cur_lat = -8.5069
                cur_lon = 115.2625

            st.caption(f"Titik Koordinat Pusat: {cur_lat:.4f}, {cur_lon:.4f}")

    with col2:
        with st.container(border=True):
            st.markdown('<div class="form-section-title">2. Kapasitas & Reputasi</div>', unsafe_allow_html=True)

            default_br = 1 if is_room_selected else 2
            default_acc = 2 if is_room_selected else 4
            default_baths = 1.0 if is_room_selected else 2.0

            col_cap1, col_cap2 = st.columns(2)
            with col_cap1:
                inp_accommodates = st.number_input("Kapasitas Tamu (Orang)", min_value=1, max_value=20, value=default_acc, step=1)
                inp_bedrooms = st.number_input("Jumlah Kamar Tidur", min_value=1, max_value=12, value=default_br, step=1)
            with col_cap2:
                inp_beds = st.number_input("Jumlah Tempat Tidur", min_value=1, max_value=16, value=default_br, step=1)
                inp_bathrooms = st.number_input("Jumlah Kamar Mandi", min_value=1.0, max_value=12.0, value=default_baths, step=0.5)

            inp_rating = st.slider("Skor Kepuasan Tamu (Rating)", min_value=3.5, max_value=5.0, value=4.88, step=0.01)
            inp_reviews = st.number_input("Jumlah Ulasan Tamu", min_value=0, max_value=1000, value=24, step=1)

    with col3:
        with st.container(border=True):
            st.markdown('<div class="form-section-title">3. Fasilitas & Karakteristik</div>', unsafe_allow_html=True)

            col_f1, col_f2 = st.columns(2)
            with col_f1:
                opt_private_pool = st.checkbox("Kolam Renang Pribadi", value=(not is_room_selected))
                opt_pool = st.checkbox("Akses Kolam Renang", value=True)
                opt_bathtub = st.checkbox("Bak Mandi (Bathtub)", value=(not is_room_selected))
                opt_jacuzzi = st.checkbox("Jacuzzi / Hot Tub Pribadi", value=False)
                opt_parking = st.checkbox("Area Parkir Gratis", value=True)
                opt_workspace = st.checkbox("Ruang / Meja Kerja", value=True)
                opt_self_checkin = st.checkbox("Check-in Mandiri", value=True)
            with col_f2:
                opt_superhost = st.checkbox("Status Superhost", value=True)
                opt_staff = st.checkbox("Layanan Staf / Sarapan", value=(not is_room_selected))
                opt_luxury = st.checkbox("Desain Mewah / Eksklusif", value=False)
                opt_beach = st.checkbox("Dekat Pantai / Laut", value=False)
                opt_nature = st.checkbox("Pemandangan Sawah / Alam", value=True)
                opt_budget = st.checkbox("Konsep Ekonomis / Sederhana", value=is_room_selected)

    # -------------------------------------------------------------------------
    # INFERENSI MODEL MACHINE LEARNING
    # -------------------------------------------------------------------------
    def build_feature_row(cfg):
        lat = cfg['latitude']
        lon = cfg['longitude']
        br = max(1, int(cfg['bedrooms']))
        bds = max(1, int(cfg['beds']))
        bth = max(1.0, float(cfg['bathrooms']))
        acc = max(1, int(cfg['accommodates']))
        room_only = int(cfg['is_room_only'])

        priv_pool = 0 if room_only == 1 else int(cfg['has_private_pool'])
        pool_any = int(cfg['has_pool'] or cfg['has_private_pool'])
        bathtub = int(cfg['has_bathtub'])
        jacuzzi = int(cfg.get('has_jacuzzi', 0))
        staff = int(cfg.get('has_service_staff', 0))
        nlp_lux = int(cfg['nlp_luxury'])
        nlp_bch = int(cfg['nlp_beach_ocean'])

        total_lux = priv_pool + pool_any + bathtub + jacuzzi + staff

        d_canggu = haversine_km(lat, lon, -8.6650, 115.1450)
        d_ubud = haversine_km(lat, lon, -8.5069, 115.2625)
        d_uluwatu = haversine_km(lat, lon, -8.8149, 115.0884)
        d_sanur = haversine_km(lat, lon, -8.6913, 115.2628)
        d_kintamani = haversine_km(lat, lon, -8.2423, 115.3752)
        d_airport = haversine_km(lat, lon, -8.7467, 115.1668)
        min_hub = min(d_canggu, d_ubud, d_uluwatu, d_sanur)

        row = {
            'rating': float(cfg['rating']),
            'is_new_listing': 1 if int(cfg['review_count']) == 0 else 0,
            'review_count': int(cfg['review_count']),
            'is_room_only': room_only,
            'latitude': lat,
            'longitude': lon,
            'accommodates': acc,
            'bedrooms': br,
            'beds': bds,
            'bathrooms': bth,
            'is_superhost': int(cfg['is_superhost']),
            'has_self_checkin': int(cfg['has_self_checkin']),
            'has_private_pool': priv_pool,
            'has_pool': pool_any,
            'has_bathtub': bathtub,
            'has_parking': int(cfg['has_parking']),
            'has_workspace': int(cfg['has_workspace']),
            'nlp_luxury': nlp_lux,
            'nlp_beach_ocean': nlp_bch,
            'nlp_nature_view': int(cfg['nlp_nature_view']),
            'nlp_budget_cozy': int(cfg['nlp_budget_cozy']),
            'has_jacuzzi': jacuzzi,
            'has_service_staff': staff,
            'guests_per_bedroom': acc / br,
            'baths_per_bedroom': bth / br,
            'beds_per_bedroom': bds / br,
            'total_rooms_index': br + bth,
            'total_luxury_amenities': total_lux,
            'dist_canggu_seminyak_km': d_canggu,
            'dist_ubud_center_km': d_ubud,
            'dist_uluwatu_km': d_uluwatu,
            'dist_sanur_km': d_sanur,
            'dist_kintamani_km': d_kintamani,
            'dist_airport_km': d_airport,
            'min_dist_tourism_hub_km': min_hub,
            'regency': str(cfg['regency']),
            'sub_district': str(cfg['sub_district']),
            'property_type': str(cfg['property_type'])
        }
        return row

    def hitung_tarif_dasar(cfg):
        full_row = build_feature_row(cfg)
        df_row = pd.DataFrame([full_row])
        for c in meta['cat_cols']:
            df_row[c] = df_row[c].astype(str)
        log_val = cb_model.predict(df_row[meta['all_features']])
        return float(np.expm1(log_val)[0])

    konfigurasi_aktif = {
        'accommodates': inp_accommodates,
        'bedrooms': 1 if (is_room_selected and inp_accommodates <= 2) else inp_bedrooms,
        'beds': inp_beds,
        'bathrooms': 1.0 if (is_room_selected and inp_accommodates <= 2) else inp_bathrooms,
        'is_room_only': 1 if is_room_selected else 0,
        'rating': inp_rating,
        'review_count': inp_reviews,
        'is_superhost': 1 if opt_superhost else 0,
        'has_private_pool': 1 if (opt_private_pool and not is_room_selected) else 0,
        'has_pool': 1 if (opt_pool or opt_private_pool) else 0,
        'has_bathtub': 1 if opt_bathtub else 0,
        'has_jacuzzi': 1 if opt_jacuzzi else 0,
        'has_service_staff': 1 if opt_staff else 0,
        'has_parking': 1 if opt_parking else 0,
        'has_workspace': 1 if opt_workspace else 0,
        'has_self_checkin': 1 if opt_self_checkin else 0,
        'nlp_luxury': 1 if opt_luxury else 0,
        'nlp_beach_ocean': 1 if opt_beach else 0,
        'nlp_nature_view': 1 if opt_nature else 0,
        'nlp_budget_cozy': 1 if opt_budget else 0,
        'latitude': cur_lat,
        'longitude': cur_lon,
        'regency': sel_regency,
        'sub_district': sel_subdistrict,
        'property_type': sel_property_type
    }

    # Prediksi tarif wajar murni dari model CatBoost
    tarif_wajar_ml = hitung_tarif_dasar(konfigurasi_aktif)
    toleransi_error = 433928.0  # MedAE CatBoost pada Data Uji
    batas_bawah = max(150000.0, tarif_wajar_ml - toleransi_error)
    batas_atas = tarif_wajar_ml + toleransi_error

    # Hitung jarak terdekat ke sentra wisata
    d_canggu = haversine_km(cur_lat, cur_lon, -8.6650, 115.1450)
    d_ubud = haversine_km(cur_lat, cur_lon, -8.5069, 115.2625)
    d_uluwatu = haversine_km(cur_lat, cur_lon, -8.8149, 115.0884)
    d_sanur = haversine_km(cur_lat, cur_lon, -8.6913, 115.2628)
    min_hub_km = min(d_canggu, d_ubud, d_uluwatu, d_sanur)

    total_lux_count = (
        konfigurasi_aktif['has_private_pool'] +
        konfigurasi_aktif['has_pool'] +
        konfigurasi_aktif['has_bathtub'] +
        konfigurasi_aktif['has_jacuzzi'] +
        konfigurasi_aktif['has_service_staff']
    )

    # -------------------------------------------------------------------------
    # KARTU HASIL PREDIKSI TARIF WAJAR
    # -------------------------------------------------------------------------
    st.markdown(f"""
    <div class="price-box">
        <div class="price-box-label">Estimasi Tarif Sewa Wajar per Malam (Inferensi Machine Learning Murni)</div>
        <div class="price-box-amount">Rp {tarif_wajar_ml:,.0f} <span style="font-size: 1.1rem; font-weight: 500; color: #a7f3d0;">/ malam</span></div>
        <div class="price-box-range">Rentang Pasar Wajar: Rp {batas_bawah:,.0f} s.d. Rp {batas_atas:,.0f} / malam (Toleransi MedAE Data Uji Rp {toleransi_error:,.0f})</div>
    </div>
    """, unsafe_allow_html=True)

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        eff_br = max(1, konfigurasi_aktif['bedrooms'])
        tarif_per_kamar = tarif_wajar_ml / eff_br
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-card-title">Tarif per Kamar Tidur</div>
            <div class="stat-card-value">Rp {tarif_per_kamar:,.0f}</div>
            <div style="font-size:0.75rem; color:#cbd5e1; margin-top:3px;">Berdasarkan {eff_br} kamar tidur aktif</div>
        </div>
        """, unsafe_allow_html=True)
    with col_s2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-card-title">Jarak Sentra Wisata Terdekat</div>
            <div class="stat-card-value">{min_hub_km:.1f} km</div>
            <div style="font-size:0.75rem; color:#cbd5e1; margin-top:3px;">Geospasial Haversine Riil</div>
        </div>
        """, unsafe_allow_html=True)
    with col_s3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-card-title">Fasilitas Mewah Aktif</div>
            <div class="stat-card-value">{total_lux_count} dari 5</div>
            <div style="font-size:0.75rem; color:#cbd5e1; margin-top:3px;">Pool, Tub, Jacuzzi, Staf</div>
        </div>
        """, unsafe_allow_html=True)
    with col_s4:
        if tarif_wajar_ml < 750000:
            kategori_teks = "Homestay / Ekonomis"
        elif tarif_wajar_ml <= 2500000:
            kategori_teks = "Villa Reguler Populer"
        elif tarif_wajar_ml <= 6000000:
            kategori_teks = "Villa Menengah"
        else:
            kategori_teks = "Villa Premium"

        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-card-title">Kelas Pasar Properti</div>
            <div class="stat-card-value">{kategori_teks}</div>
            <div style="font-size:0.75rem; color:#cbd5e1; margin-top:3px;">Klasifikasi Nilai Pasar Bali</div>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # TABEL SIMULASI NILAI TAMBAH FASILITAS (WHAT-IF ANALYSIS KUANTITATIF)
    # -------------------------------------------------------------------------
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("### Simulasi Nilai Tambah Fasilitas (What-If Analysis Murni Berbasis Model ML)")
    st.caption("Perubahan tarif sewa harian yang diproyeksikan langsung oleh model CatBoost jika fasilitas penginapan ditambahkan:")

    c_pool_on = konfigurasi_aktif.copy(); c_pool_on['has_private_pool'] = 1; c_pool_on['has_pool'] = 1; c_pool_on['is_room_only'] = 0
    c_pool_off = konfigurasi_aktif.copy(); c_pool_off['has_private_pool'] = 0; c_pool_off['has_pool'] = 0
    t_pool_on = hitung_tarif_dasar(c_pool_on)
    t_pool_off = hitung_tarif_dasar(c_pool_off)
    delta_pool = abs(t_pool_on - t_pool_off)

    c_tub_on = konfigurasi_aktif.copy(); c_tub_on['has_bathtub'] = 1
    c_tub_off = konfigurasi_aktif.copy(); c_tub_off['has_bathtub'] = 0
    t_tub_on = hitung_tarif_dasar(c_tub_on)
    t_tub_off = hitung_tarif_dasar(c_tub_off)
    delta_tub = abs(t_tub_on - t_tub_off)

    c_lux_on = konfigurasi_aktif.copy(); c_lux_on['nlp_luxury'] = 1
    c_lux_off = konfigurasi_aktif.copy(); c_lux_off['nlp_luxury'] = 0
    t_lux_on = hitung_tarif_dasar(c_lux_on)
    t_lux_off = hitung_tarif_dasar(c_lux_off)
    delta_lux = abs(t_lux_on - t_lux_off)

    c_sh_on = konfigurasi_aktif.copy(); c_sh_on['is_superhost'] = 1
    c_sh_off = konfigurasi_aktif.copy(); c_sh_off['is_superhost'] = 0
    t_sh_on = hitung_tarif_dasar(c_sh_on)
    t_sh_off = hitung_tarif_dasar(c_sh_off)
    delta_sh = abs(t_sh_on - t_sh_off)

    tabel_saran = [
        {
            "Parameter Fasilitas Diuji": "Kolam Renang Pribadi (Private Pool)",
            "Tarif Tanpa Fitur": f"Rp {t_pool_off:,.0f}",
            "Proyeksi Tarif Baru": f"Rp {t_pool_on:,.0f}",
            "Nilai Tambah Prediksi ML (Delta IDR)": f"+ Rp {delta_pool:,.0f} / malam",
            "Persentase Kenaikan (%)": f"{(delta_pool / max(1, t_pool_off)) * 100:.1f}%"
        },
        {
            "Parameter Fasilitas Diuji": "Bak Mandi (Bathtub) Kamar Utama",
            "Tarif Tanpa Fitur": f"Rp {t_tub_off:,.0f}",
            "Proyeksi Tarif Baru": f"Rp {t_tub_on:,.0f}",
            "Nilai Tambah Prediksi ML (Delta IDR)": f"+ Rp {delta_tub:,.0f} / malam",
            "Persentase Kenaikan (%)": f"{(delta_tub / max(1, t_tub_off)) * 100:.1f}%"
        },
        {
            "Parameter Fasilitas Diuji": "Penataan Desain Interior Eksklusif / Luxury",
            "Tarif Tanpa Fitur": f"Rp {t_lux_off:,.0f}",
            "Proyeksi Tarif Baru": f"Rp {t_lux_on:,.0f}",
            "Nilai Tambah Prediksi ML (Delta IDR)": f"+ Rp {delta_lux:,.0f} / malam",
            "Persentase Kenaikan (%)": f"{(delta_lux / max(1, t_lux_off)) * 100:.1f}%"
        },
        {
            "Parameter Fasilitas Diuji": "Lencana Reputasi Pelayanan Superhost",
            "Tarif Tanpa Fitur": f"Rp {t_sh_off:,.0f}",
            "Proyeksi Tarif Baru": f"Rp {t_sh_on:,.0f}",
            "Nilai Tambah Prediksi ML (Delta IDR)": f"+ Rp {delta_sh:,.0f} / malam",
            "Persentase Kenaikan (%)": f"{(delta_sh / max(1, t_sh_off)) * 100:.1f}%"
        }
    ]
    st.table(pd.DataFrame(tabel_saran))

# =============================================================================
# TAB 2: PETA PERSEBARAN LOKASI (OPENSTREETMAP - BEBAS WATERMARK)
# =============================================================================
with tab_map:
    st.markdown("### Peta Persebaran Geospasial Penginapan di Seluruh Bali")
    st.caption("Visualisasi sebaran 1.000 listing penginapan bersih di 9 Kabupaten/Kota Bali berbasis OpenStreetMap publik:")

    col_m1, col_m2 = st.columns([1, 3])
    with col_m1:
        filter_reg = st.multiselect("Filter Kabupaten / Kota:", regencies, default=[])
        filter_type = st.multiselect("Filter Tipe Properti:", type_options, format_func=lambda x: type_labels.get(x, x), default=[])
        max_price_view = st.slider("Batas Tarif Tertinggi (Juta Rp):", min_value=1.0, max_value=10.0, value=10.0, step=0.5)

    df_view = df_master.copy()
    if filter_reg:
        df_view = df_view[df_view['regency'].isin(filter_reg)]
    if filter_type:
        df_view = df_view[df_view['property_type'].isin(filter_type)]
    df_view = df_view[df_view['price_idr'] <= (max_price_view * 1e6)]

    with col_m2:
        st.markdown(f"Menampilkan **{len(df_view):,}** dari {len(df_master):,} penginapan:")
        fig_map = px.scatter_mapbox(
            df_view,
            lat="latitude",
            lon="longitude",
            color="price_idr",
            size="accommodates",
            hover_name="name",
            hover_data={
                "sub_district": True,
                "regency": True,
                "price_idr": ":,.0f",
                "bedrooms": True,
                "rating": True,
                "latitude": False,
                "longitude": False
            },
            color_continuous_scale="Viridis",
            range_color=[150000, 6000000],
            zoom=9,
            center={"lat": -8.45, "lon": 115.15},
            mapbox_style="open-street-map",
            height=580
        )
        fig_map.update_layout(
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            coloraxis_colorbar=dict(
                title="Tarif / Malam (Rp)",
                tickfont=dict(color="#f8fafc"),
                title_font=dict(color="#f8fafc")
            ),
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_map, use_container_width=True)

# =============================================================================
# TAB 3: AKURASI MODEL PREDIKSI (BENCHMARK 3 MODEL UTAMA)
# =============================================================================
with tab_bench:
    st.markdown("### Evaluasi Ilmiah Akurasi 3 Model Machine Learning")
    st.caption("Hasil pengujian independen pada 20% Data Uji terisolasi (0% Data Leakage) dengan 38 fitur murni.")

    if os.path.exists(BENCHMARK_PATH):
        bench_data = pd.read_csv(BENCHMARK_PATH)
        cb_row = bench_data[bench_data['Model'].str.contains('CatBoost')].iloc[0]
        med_ae_val = float(cb_row['Test_MedAE_IDR'])
        mae_val = float(cb_row['Test_MAE_IDR'])
        r2_val = float(cb_row['Test_R2'])
        mdape_val = float(cb_row['Test_MdAPE_Pct'])
    else:
        med_ae_val, mae_val, r2_val, mdape_val = 433928.0, 762318.0, 0.7168, 32.80

    col_acc1, col_acc2, col_acc3, col_acc4 = st.columns(4)
    with col_acc1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-card-title">Skor Determinasi (Test R2)</div>
            <div class="stat-card-value" style="color: #a7f3d0;">{r2_val * 100:.2f}%</div>
            <div style="font-size:0.78rem; color:#cbd5e1; margin-top:3px;">CatBoost Champion Model</div>
        </div>
        """, unsafe_allow_html=True)
    with col_acc2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-card-title">Nilai Tengah Selisih (MedAE)</div>
            <div class="stat-card-value" style="color: #38bdf8;">Rp {med_ae_val:,.0f}</div>
            <div style="font-size:0.78rem; color:#cbd5e1; margin-top:3px;">Selisih tipikal per malam di Bali</div>
        </div>
        """, unsafe_allow_html=True)
    with col_acc3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-card-title">Rata-rata Selisih Absolut (MAE)</div>
            <div class="stat-card-value" style="color: #10b981;">Rp {mae_val:,.0f}</div>
            <div style="font-size:0.78rem; color:#cbd5e1; margin-top:3px;">Mean Absolute Error Data Uji</div>
        </div>
        """, unsafe_allow_html=True)
    with col_acc4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-card-title">Tingkat Kesalahan Relatif (MdAPE)</div>
            <div class="stat-card-value" style="color: #fef08a;">{mdape_val:.1f}%</div>
            <div style="font-size:0.78rem; color:#cbd5e1; margin-top:3px;">Akurasi relatif {100.0 - mdape_val:.1f}% pada Data Uji</div>
        </div>
        """, unsafe_allow_html=True)

    if os.path.exists(BENCHMARK_PATH):
        tampil_bench = bench_data.copy()
        tampil_bench = tampil_bench.rename(columns={
            'Model': 'Algoritma Machine Learning',
            'CV_R2_Mean': 'Skor 5-Fold CV (R2)',
            'Test_R2': 'Skor Data Uji (R2)',
            'Test_MedAE_IDR': 'Nilai Tengah Meleset (MedAE)',
            'Test_MAE_IDR': 'Rata-rata Meleset (MAE)',
            'Test_RMSE_IDR': 'RMSE (Rupiah)',
            'Test_MdAPE_Pct': 'Persentase Selisih (MdAPE)'
        })
        if 'CV_MAE_IDR' in tampil_bench.columns:
            tampil_bench = tampil_bench.drop(columns=['CV_MAE_IDR'])

        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown("#### Tabel Komparasi 3 Model Machine Learning Utama:")
        st.dataframe(
            tampil_bench.style.format({
                "Skor 5-Fold CV (R2)": "{:.4f}",
                "Skor Data Uji (R2)": "{:.4f}",
                "Nilai Tengah Meleset (MedAE)": "Rp {:,.0f}",
                "Rata-rata Meleset (MAE)": "Rp {:,.0f}",
                "RMSE (Rupiah)": "Rp {:,.0f}",
                "Persentase Selisih (MdAPE)": "{:.2f}%"
            }),
            use_container_width=True
        )

        col_tb1, col_tb2 = st.columns(2)
        with col_tb1:
            fig_r2 = go.Figure(go.Bar(
                x=bench_data['Model'],
                y=bench_data['Test_R2'],
                marker_color=['#10b981', '#38bdf8', '#a855f7'],
                text=bench_data['Test_R2'].apply(lambda x: f"{x:.4f}"),
                textposition='auto',
                textfont=dict(color='#ffffff', size=13)
            ))
            fig_r2.update_layout(
                template="plotly_dark",
                title=dict(text="Perbandingan Skor Determinasi R2 pada Data Uji", font=dict(color="#f8fafc", size=14)),
                yaxis_title="Skor R2",
                yaxis=dict(gridcolor="#334155", color="#cbd5e1"),
                xaxis=dict(color="#cbd5e1"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=320,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig_r2, use_container_width=True)

        with col_tb2:
            fig_err = go.Figure()
            fig_err.add_trace(go.Bar(
                name='Nilai Tengah Selisih (MedAE)',
                x=bench_data['Model'],
                y=bench_data['Test_MedAE_IDR'] / 1000,
                marker_color='#38bdf8',
                text=(bench_data['Test_MedAE_IDR'] / 1000).apply(lambda x: f"Rp {x:,.0f} rb"),
                textposition='auto',
                textfont=dict(color='#ffffff', size=11)
            ))
            fig_err.add_trace(go.Bar(
                name='Rata-rata Selisih (MAE)',
                x=bench_data['Model'],
                y=bench_data['Test_MAE_IDR'] / 1000,
                marker_color='#fb7185',
                text=(bench_data['Test_MAE_IDR'] / 1000).apply(lambda x: f"Rp {x:,.0f} rb"),
                textposition='auto',
                textfont=dict(color='#ffffff', size=11)
            ))
            fig_err.update_layout(
                template="plotly_dark",
                barmode='group',
                title=dict(text="Perbandingan Error MedAE vs MAE (Ribu Rupiah)", font=dict(color="#f8fafc", size=14)),
                yaxis_title="Selisih (Ribu Rp)",
                yaxis=dict(gridcolor="#334155", color="#cbd5e1"),
                xaxis=dict(color="#cbd5e1"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=320,
                margin=dict(l=20, r=20, t=50, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_err, use_container_width=True)

    if os.path.exists(TIER_PATH):
        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown("#### Rincian Akurasi Berdasarkan Kelas Pasar Penginapan (Model CatBoost pada Data Uji):")
        tier_df = pd.read_csv(TIER_PATH)
        tampil_tier = tier_df.rename(columns={
            "Kelas_Penginapan": "Kelas Pasar",
            "Jumlah_Sampel_Uji": "Sampel Uji",
            "Rata_Rata_Aktual_IDR": "Rata-rata Pasar Aktual",
            "Rata_Rata_Prediksi_IDR": "Rata-rata Tebakan Model",
            "MAE_IDR": "Rata-rata Selisih (MAE)",
            "MedAE_IDR": "Nilai Tengah Selisih (MedAE)",
            "MdAPE_Pct": "Persentase Error (MdAPE)",
            "Karakteristik_Properti": "Jenis Properti"
        })
        st.dataframe(
            tampil_tier.style.format({
                "Rata-rata Pasar Aktual": "Rp {:,.0f}",
                "Rata-rata Tebakan Model": "Rp {:,.0f}",
                "Rata-rata Selisih (MAE)": "Rp {:,.0f}",
                "Nilai Tengah Selisih (MedAE)": "Rp {:,.0f}",
                "Persentase Error (MdAPE)": "{:.1f}%"
            }),
            use_container_width=True
        )

# =============================================================================
# TAB 4: FAKTOR PENENTU HARGA (EXPLAINABLE AI - SHAP)
# =============================================================================
with tab_xai:
    st.markdown("### Analisis Faktor Penentu Harga Sewa (Explainable AI - SHAP)")
    st.caption("Interpretasi saintifik dari model CatBoost mengenai kontribusi marjinal Shapley pada atribut fisik dan lokasi:")

    col_x1, col_x2 = st.columns(2)
    with col_x1:
        with st.container(border=True):
            img_imp = os.path.join(REPORTS_DIR, "shap_feature_importance.png")
            if os.path.exists(img_imp):
                st.image(img_imp, width="stretch")
                st.markdown('<div class="figure-caption">Figure 1: Mean Absolute SHAP Value (Global Feature Importance Across Test Instances)</div>', unsafe_allow_html=True)

    with col_x2:
        with st.container(border=True):
            img_bee = os.path.join(REPORTS_DIR, "shap_summary_beeswarm.png")
            if os.path.exists(img_bee):
                st.image(img_bee, width="stretch")
                st.markdown('<div class="figure-caption">Figure 2: Beeswarm Distribution of SHAP Values across Feature Dimensions</div>', unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    with st.container(border=True):
        img_wf = os.path.join(REPORTS_DIR, "shap_waterfall_example.png")
        if os.path.exists(img_wf):
            st.image(img_wf, width="stretch")
            st.markdown('<div class="figure-caption">Figure 3: Local Attribution (Waterfall Plot) for a Single Prediction Instance</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. FOOTER
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="display: flex; justify-content: space-between; color: #cbd5e1; font-size: 0.85rem; padding: 10px 0;">
    <div><b>Kalkulator Tarif Sewa Villa & Homestay Bali</b> | Penelitian Machine Learning</div>
    <div>Pengembang: I Kadek Adi Sunetra (2305551053) & Anthony Wisnu Jati (2305551138)</div>
    <div>Dosen Pengampu: Yogiswara Dharma Putra, S.TI., M.T. (Universitas Udayana)</div>
</div>
""", unsafe_allow_html=True)

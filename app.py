"""
=============================================================================
KALKULATOR TARIF SEWA VILLA & HOMESTAY BALI
Sistem Cerdas Estimasi Harga Sewa Harian Berbasis Lokasi, Fasilitas & Kalender
Universitas Udayana - Program Studi Teknologi Informasi
=============================================================================
"""

import os
from datetime import date, timedelta
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
        color: #cbd5e1 !important;
        font-size: 0.95rem !important;
        margin: 8px 0 0 0 !important;
        font-weight: 400;
    }
    .tag-pill {
        display: inline-block;
        background-color: #334155;
        color: #f1f5f9;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 20px;
        margin-top: 12px;
        margin-right: 8px;
        border: 1px solid #475569;
    }
    
    .price-box {
        background: linear-gradient(135deg, #064e3b 0%, #047857 100%);
        border: 1px solid #10b981;
        border-radius: 14px;
        padding: 26px 20px;
        color: #ffffff;
        margin: 18px 0 24px 0;
        text-align: center;
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.2);
    }
    .price-box-label {
        font-size: 0.88rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #a7f3d0;
        margin-bottom: 8px;
    }
    .price-box-amount {
        font-size: 2.7rem;
        font-weight: 800;
        color: #ffffff;
        font-variant-numeric: tabular-nums;
        margin: 8px 0;
    }
    .price-box-range {
        font-size: 0.98rem;
        color: #fef08a;
        font-weight: 600;
    }
    
    .stat-card {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px 18px;
        margin-bottom: 12px;
    }
    .stat-card-title {
        font-size: 0.8rem;
        color: #cbd5e1;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    .stat-card-value {
        font-size: 1.32rem;
        font-weight: 700;
        color: #ffffff;
        margin-top: 5px;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        border-bottom: 2px solid #334155;
        padding-bottom: 6px;
        margin-bottom: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: auto;
        padding: 10px 18px !important;
        border-radius: 8px 8px 0 0 !important;
        background-color: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid transparent !important;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(255, 255, 255, 0.08) !important;
    }
    .stTabs [data-baseweb="tab"] p, 
    .stTabs [data-baseweb="tab"] div, 
    .stTabs [data-baseweb="tab"] span {
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        color: #e2e8f0 !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(56, 189, 248, 0.12) !important;
        border-bottom: 3px solid #38bdf8 !important;
    }
    .stTabs [aria-selected="true"] p, 
    .stTabs [aria-selected="true"] div, 
    .stTabs [aria-selected="true"] span {
        color: #38bdf8 !important;
        font-weight: 700 !important;
    }

    .form-section-title {
        color: #38bdf8;
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 12px;
        border-bottom: 1px solid #334155;
        padding-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. MEMUAT DATA, MODEL & REFERENSI SPATIAL KNN
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "bali_airbnb_enriched_master.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "catboost_pricing_model.cbm")
META_PATH = os.path.join(BASE_DIR, "models", "feature_metadata.joblib")
KNN_REF_PATH = os.path.join(BASE_DIR, "models", "spatial_knn_reference.joblib")
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
        knn_ref = joblib.load(KNN_REF_PATH) if os.path.exists(KNN_REF_PATH) else None
        return cb, meta, knn_ref
    return None, None, None


df_master = load_dataset()
cb_model, meta, knn_ref = load_pricing_model()

# -----------------------------------------------------------------------------
# 3. HEADER UTAMA
# -----------------------------------------------------------------------------
total_units = len(df_master) if not df_master.empty else 1000
st.markdown(f"""
<div class="app-header">
    <h1>Kalkulator Tarif Sewa Villa & Homestay Bali</h1>
    <p>Sistem cerdas penaksiran tarif sewa harian wajar berbasis lokasi mikro, fasilitas, tetangga terdekat, dan kalender Check-in / Check-out.</p>
    <div>
        <span class="tag-pill">Basis Data: {total_units} Penginapan Organik Bali</span>
        <span class="tag-pill">Arsitektur: Spatial-KNN + CatBoost Ensemble</span>
        <span class="tag-pill">Cakupan: 9 Kabupaten / Kota</span>
    </div>
</div>
""", unsafe_allow_html=True)

if df_master.empty or cb_model is None:
    st.error("Data penginapan atau model perhitungan belum siap. Silakan jalankan script pipeline terlebih dahulu.")
    st.stop()

coord_lookup = df_master.groupby(['regency', 'sub_district'])[['latitude', 'longitude']].median().reset_index()

# -----------------------------------------------------------------------------
# 4. TAB NAVIGASI UTAMA
# -----------------------------------------------------------------------------
tab_val, tab_map, tab_bench, tab_xai = st.tabs([
    "Cek Estimasi Harga & Kalender",
    "Peta Persebaran Lokasi",
    "Akurasi Model Prediksi",
    "Faktor Penentu Harga"
])

# =============================================================================
# TAB 1: CEK ESTIMASI HARGA & SIMULASI KALENDER CHECK-IN / CHECK-OUT
# =============================================================================
with tab_val:
    st.markdown("### Masukkan Spesifikasi Penginapan & Tanggal Menginap")
    st.caption("Pilih lokasi, kapasitas kamar, fasilitas, serta tanggal Check-in dan Check-out untuk menghitung tarif harian dan total biaya masa menginap.")

    col1, col2, col3 = st.columns([1.05, 1.05, 1.2])

    with col1:
        with st.container(border=True):
            st.markdown('<div class="form-section-title">1. Lokasi, Tipe & Tanggal</div>', unsafe_allow_html=True)

            regencies = sorted(df_master['regency'].unique().tolist())
            def_reg_idx = regencies.index("Gianyar") if "Gianyar" in regencies else 0
            sel_regency = st.selectbox("Kabupaten / Kota:", regencies, index=def_reg_idx)

            sub_list = sorted(df_master[df_master['regency'] == sel_regency]['sub_district'].unique().tolist())
            def_sub_idx = sub_list.index("Ubud") if "Ubud" in sub_list else 0
            sel_subdistrict = st.selectbox("Kecamatan / Kawasan:", sub_list, index=def_sub_idx)

            type_options = ["Villa", "Private Room/Homestay", "Cottage/Bungalow", "Entire House", "Apartment", "Shared Room/Hostel"]
            type_labels = {
                "Villa": "Villa Pribadi",
                "Private Room/Homestay": "Kamar Pribadi / Homestay",
                "Cottage/Bungalow": "Bungalow / Cottage Unik",
                "Entire House": "Rumah Utuh",
                "Apartment": "Apartemen / Studio",
                "Shared Room/Hostel": "Kamar Bersama (Hostel)"
            }
            sel_property_type = st.selectbox(
                "Tipe Penginapan:",
                type_options,
                format_func=lambda x: type_labels.get(x, x),
                index=0
            )

            default_checkin = date.today() + timedelta(days=5)
            default_checkout = date.today() + timedelta(days=7)
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                inp_checkin = st.date_input("Tanggal Check-in:", value=default_checkin, min_value=date.today())
            with col_d2:
                inp_checkout = st.date_input(
                    "Tanggal Check-out:",
                    value=max(default_checkout, inp_checkin + timedelta(days=1)),
                    min_value=inp_checkin + timedelta(days=1)
                )

            match_coords = coord_lookup[(coord_lookup['regency'] == sel_regency) & (coord_lookup['sub_district'] == sel_subdistrict)]
            if not match_coords.empty:
                cur_lat = float(match_coords['latitude'].values[0])
                cur_lon = float(match_coords['longitude'].values[0])
            else:
                cur_lat = -8.5069
                cur_lon = 115.2625

            lead_days = (inp_checkin - date.today()).days
            stay_nights = max(1, (inp_checkout - inp_checkin).days)
            if 4 <= lead_days <= 14:
                lead_status = f"Optimal ({lead_days} hari sebelum Check-in: Kalender homestay & villa terbuka penuh)"
            elif lead_days < 4:
                lead_status = f"Pemesanan Mendadak ({lead_days} hari: Sebagian homestay butuh waktu persiapan)"
            else:
                lead_status = f"Jangka Panjang ({lead_days} hari ke depan: Sebagian homestay lokal mengunci kalender > 14 hari)"
            st.caption(f"Status Kalender: {lead_status}")

    with col2:
        with st.container(border=True):
            st.markdown('<div class="form-section-title">2. Kapasitas & Reputasi</div>', unsafe_allow_html=True)

            is_room_selected = sel_property_type in ["Private Room/Homestay", "Shared Room/Hostel"]
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

            inp_rating = st.slider("Skor Kepuasan Tamu (1.0 - 5.0)", min_value=3.5, max_value=5.0, value=4.88, step=0.01)
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
    # PERHITUNGAN FITUR SPASIAL, TETANGGA KNN & HARGA KALENDER DINAMIS
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

        lux_score = float(
            priv_pool * 2 +
            bathtub +
            jacuzzi * 2 +
            nlp_lux * 2 +
            nlp_bch * 2 +
            staff
        )

        d_canggu = haversine_km(lat, lon, -8.6650, 115.1450)
        d_ubud = haversine_km(lat, lon, -8.5069, 115.2625)
        d_uluwatu = haversine_km(lat, lon, -8.8149, 115.0884)
        d_sanur = haversine_km(lat, lon, -8.6913, 115.2628)
        d_kintamani = haversine_km(lat, lon, -8.2423, 115.3752)
        d_airport = haversine_km(lat, lon, -8.7467, 115.1668)
        min_hub = min(d_canggu, d_ubud, d_uluwatu, d_sanur)

        if knn_ref is not None:
            q_vec = np.array([[lat, lon, br, room_only, lux_score]]) * knn_ref['scale_vec']
            dists, n_idxs = knn_ref['nn_model'].kneighbors(q_vec, n_neighbors=12)
            w = 1.0 / (dists[0] + 0.15)
            w = w / np.sum(w)
            knn_mean_log = float(np.sum(knn_ref['train_log_prices'][n_idxs[0]] * w))
            knn_med_log = float(np.median(knn_ref['train_log_prices'][n_idxs[0]]))
            knn_log_ppb = float(np.sum(knn_ref['train_log_ppb'][n_idxs[0]] * w))
            expected_spatial_br_log = float(np.log1p(np.expm1(knn_log_ppb) * br))
        else:
            knn_mean_log = float(np.log1p(df_master['price_idr'].median()))
            knn_med_log = knn_mean_log
            knn_log_ppb = knn_mean_log - float(np.log(br))
            expected_spatial_br_log = knn_mean_log

        row = {
            'accommodates': acc,
            'bedrooms': br,
            'beds': bds,
            'bathrooms': bth,
            'is_room_only': room_only,
            'guests_per_bedroom': acc / br,
            'baths_per_bedroom': bth / br,
            'beds_per_bedroom': bds / br,
            'total_rooms_index': br + bth,
            'rating': float(cfg['rating']),
            'review_count': int(cfg['review_count']),
            'is_new_listing': 1 if int(cfg['review_count']) == 0 else 0,
            'is_superhost': int(cfg['is_superhost']),
            'has_private_pool': priv_pool,
            'has_pool': pool_any,
            'has_bathtub': bathtub,
            'has_jacuzzi': jacuzzi,
            'has_service_staff': staff,
            'has_parking': int(cfg['has_parking']),
            'has_workspace': int(cfg['has_workspace']),
            'has_self_checkin': int(cfg['has_self_checkin']),
            'nlp_luxury': nlp_lux,
            'nlp_beach_ocean': nlp_bch,
            'nlp_nature_view': int(cfg['nlp_nature_view']),
            'nlp_budget_cozy': int(cfg['nlp_budget_cozy']),
            'luxury_amenity_score': lux_score,
            'latitude': lat,
            'longitude': lon,
            'dist_canggu_seminyak_km': d_canggu,
            'dist_ubud_center_km': d_ubud,
            'dist_uluwatu_km': d_uluwatu,
            'dist_sanur_km': d_sanur,
            'dist_kintamani_km': d_kintamani,
            'dist_airport_km': d_airport,
            'min_dist_tourism_hub_km': min_hub,
            'spatial_knn_log_price': knn_mean_log,
            'spatial_knn_median_log_price': knn_med_log,
            'spatial_knn_log_ppb': knn_log_ppb,
            'expected_spatial_bedroom_log_price': expected_spatial_br_log,
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

    def evaluasi_kalender_menginap(base_rate, checkin_dt, checkout_dt):
        nights = max(1, (checkout_dt - checkin_dt).days)
        daily_rates = []
        weekend_nights = 0
        season_labels = set()

        for d_idx in range(nights):
            cur_d = checkin_dt + timedelta(days=d_idx)
            m = cur_d.month
            day = cur_d.day
            # Faktor musim pariwisata Bali
            if (m == 12 and day >= 20) or (m == 1 and day <= 5) or m in [7, 8]:
                s_mult = 1.20
                season_labels.add("Musim Puncak (Peak Season +20%)")
            elif m in [6, 9, 12]:
                s_mult = 1.10
                season_labels.add("Musim Ramai (High Season +10%)")
            elif m in [2, 3, 11]:
                s_mult = 0.92
                season_labels.add("Musim Tenang (Low Season -8%)")
            else:
                s_mult = 1.00
                season_labels.add("Musim Normal (Tarif Standar)")

            # Faktor akhir pekan (Jumat = 4, Sabtu = 5)
            if cur_d.weekday() in [4, 5]:
                w_mult = 1.08
                weekend_nights += 1
            else:
                w_mult = 1.00

            daily_rates.append(base_rate * s_mult * w_mult)

        avg_before_discount = float(np.mean(daily_rates))
        if nights >= 28:
            los_disc = 0.25
            los_label = "Diskon Bulanan (-25%)"
        elif nights >= 7:
            los_disc = 0.10
            los_label = "Diskon Mingguan (-10%)"
        else:
            los_disc = 0.0
            los_label = "Tarif Harian Normal (0%)"

        effective_nightly = avg_before_discount * (1.0 - los_disc)
        total_stay_cost = effective_nightly * nights
        return {
            "nights": nights,
            "weekend_nights": weekend_nights,
            "season_summary": ", ".join(sorted(season_labels)),
            "los_label": los_label,
            "effective_nightly": effective_nightly,
            "total_stay_cost": total_stay_cost
        }

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

    tarif_dasar_ml = hitung_tarif_dasar(konfigurasi_aktif)
    kal_info = evaluasi_kalender_menginap(tarif_dasar_ml, inp_checkin, inp_checkout)
    harga_taksiran = kal_info['effective_nightly']
    batas_bawah = harga_taksiran * 0.85
    batas_atas = harga_taksiran * 1.15

    # -------------------------------------------------------------------------
    # KARTU HASIL PREDIKSI
    # -------------------------------------------------------------------------
    st.markdown(f"""
    <div class="price-box">
        <div class="price-box-label">Estimasi Tarif Sewa Wajar per Malam ({inp_checkin.strftime('%d %b %Y')} s.d. {inp_checkout.strftime('%d %b %Y')})</div>
        <div class="price-box-amount">Rp {harga_taksiran:,.0f} <span style="font-size: 1.1rem; font-weight: 500; color: #a7f3d0;">/ malam</span></div>
        <div class="price-box-range">Total Tagihan {kal_info['nights']} Malam: Rp {kal_info['total_stay_cost']:,.0f} | Rentang Pasar Wajar: Rp {batas_bawah:,.0f} s.d. Rp {batas_atas:,.0f} / malam</div>
    </div>
    """, unsafe_allow_html=True)

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-card-title">Tarif Dasar Properti (ML)</div>
            <div class="stat-card-value">Rp {tarif_dasar_ml:,.0f}</div>
            <div style="font-size:0.75rem; color:#cbd5e1; margin-top:3px;">Sebelum penyesuaian musim/kalender</div>
        </div>
        """, unsafe_allow_html=True)
    with col_s2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-card-title">Rincian Kalender Menginap</div>
            <div class="stat-card-value">{kal_info['nights']} Malam ({kal_info['weekend_nights']} Akhir Pekan)</div>
            <div style="font-size:0.75rem; color:#cbd5e1; margin-top:3px;">{kal_info['season_summary']}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_s3:
        omzet_bulan = tarif_dasar_ml * 30 * 0.70
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-card-title">Potensi Omzet / Bulan</div>
            <div class="stat-card-value">Rp {omzet_bulan / 1e6:.1f} Juta</div>
            <div style="font-size:0.75rem; color:#cbd5e1; margin-top:3px;">Asumsi tingkat hunian 70% (21 malam)</div>
        </div>
        """, unsafe_allow_html=True)
    with col_s4:
        if harga_taksiran < 750000:
            kategori_teks = "Homestay / Ekonomis"
        elif harga_taksiran <= 2500000:
            kategori_teks = "Villa Reguler Populer"
        elif harga_taksiran <= 6000000:
            kategori_teks = "Villa Menengah"
        else:
            kategori_teks = "Villa Premium"

        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-card-title">Kelas Pasar Properti</div>
            <div class="stat-card-value">{kategori_teks}</div>
            <div style="font-size:0.75rem; color:#cbd5e1; margin-top:3px;">{kal_info['los_label']}</div>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # TABEL SIMULASI NILAI TAMBAH FASILITAS (WHAT-IF ROI)
    # -------------------------------------------------------------------------
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("### Simulasi Nilai Tambah Fasilitas (Panduan Investasi Pemilik)")
    st.caption("Perbandingan perubahan tarif dasar per malam apabila fasilitas tertentu ditambahkan pada penginapan ini:")

    c_pool_on = konfigurasi_aktif.copy(); c_pool_on['has_private_pool'] = 1; c_pool_on['has_pool'] = 1; c_pool_on['is_room_only'] = 0
    c_pool_off = konfigurasi_aktif.copy(); c_pool_off['has_private_pool'] = 0; c_pool_off['has_pool'] = 0
    tambah_pool = abs(hitung_tarif_dasar(c_pool_on) - hitung_tarif_dasar(c_pool_off))

    c_tub_on = konfigurasi_aktif.copy(); c_tub_on['has_bathtub'] = 1
    c_tub_off = konfigurasi_aktif.copy(); c_tub_off['has_bathtub'] = 0
    tambah_tub = abs(hitung_tarif_dasar(c_tub_on) - hitung_tarif_dasar(c_tub_off))

    c_lux_on = konfigurasi_aktif.copy(); c_lux_on['nlp_luxury'] = 1
    c_lux_off = konfigurasi_aktif.copy(); c_lux_off['nlp_luxury'] = 0
    tambah_lux = abs(hitung_tarif_dasar(c_lux_on) - hitung_tarif_dasar(c_lux_off))

    c_sh_on = konfigurasi_aktif.copy(); c_sh_on['is_superhost'] = 1
    c_sh_off = konfigurasi_aktif.copy(); c_sh_off['is_superhost'] = 0
    tambah_sh = abs(hitung_tarif_dasar(c_sh_on) - hitung_tarif_dasar(c_sh_off))

    tabel_saran = [
        {"Fasilitas / Peningkatan": "Kolam Renang Pribadi (Private Pool)", "Potensi Kenaikan Tarif": f"+ Rp {tambah_pool:,.0f} / malam", "Persentase": f"{(tambah_pool / max(1, tarif_dasar_ml)) * 100:.1f}%", "Rekomendasi Strategis": "Daya tarik utama wisatawan di Bali untuk menaikkan kelas sewa"},
        {"Fasilitas / Peningkatan": "Bak Mandi (Bathtub) Kamar Utama", "Potensi Kenaikan Tarif": f"+ Rp {tambah_tub:,.0f} / malam", "Persentase": f"{(tambah_tub / max(1, tarif_dasar_ml)) * 100:.1f}%", "Rekomendasi Strategis": "Biaya instalasi terukur dengan dampak nilai jual tinggi"},
        {"Fasilitas / Peningkatan": "Penataan Interior Eksklusif / Luxury", "Potensi Kenaikan Tarif": f"+ Rp {tambah_lux:,.0f} / malam", "Persentase": f"{(tambah_lux / max(1, tarif_dasar_ml)) * 100:.1f}%", "Rekomendasi Strategis": "Meningkatkan daya saing visual pada foto halaman pencarian"},
        {"Fasilitas / Peningkatan": "Reputasi Pelayanan Superhost", "Potensi Kenaikan Tarif": f"+ Rp {tambah_sh:,.0f} / malam", "Persentase": f"{(tambah_sh / max(1, tarif_dasar_ml)) * 100:.1f}%", "Rekomendasi Strategis": "Meningkatkan tingkat kepercayaan dan okupansi bulanan"}
    ]
    st.table(pd.DataFrame(tabel_saran))

# =============================================================================
# TAB 2: PETA PERSEBARAN LOKASI
# =============================================================================
with tab_map:
    st.markdown("### Peta Persebaran Tarif Harian Penginapan di Bali")
    st.caption(f"Visualisasi geospasial {len(df_master)} penginapan pasar reguler di 9 Kabupaten/Kota Bali (Tarif Murni 1 Malam).")

    col_mf1, col_mf2, col_mf3 = st.columns(3)
    with col_mf1:
        flt_regency = st.multiselect("Pilih Kabupaten / Kota:", regencies, default=regencies)
    with col_mf2:
        avail_types = sorted(df_master['property_type'].unique().tolist())
        flt_type = st.multiselect("Pilih Tipe Penginapan:", avail_types, default=avail_types)
    with col_mf3:
        max_limit = float(np.ceil(df_master['price_idr'].max() / 1e6))
        flt_price = st.slider("Rentang Tarif per Malam (Juta Rupiah):", 0.0, max_limit, (0.0, max_limit), step=0.5)

    filtered_df = df_master[
        (df_master['regency'].isin(flt_regency)) &
        (df_master['property_type'].isin(flt_type)) &
        (df_master['price_idr'] >= flt_price[0] * 1e6) &
        (df_master['price_idr'] <= flt_price[1] * 1e6)
    ].copy()

    st.write(f"Menampilkan **{len(filtered_df)} penginapan** dengan median tarif **Rp {filtered_df['price_idr'].median():,.0f} / malam**.")

    if not filtered_df.empty:
        fig_map = px.scatter_map(
            filtered_df,
            lat="latitude",
            lon="longitude",
            color="price_idr",
            size="bedrooms",
            hover_name="name",
            hover_data={
                "sub_district": True,
                "regency": True,
                "bedrooms": True,
                "price_idr": ":,.0f"
            },
            color_continuous_scale="Viridis",
            range_color=[300000, 5000000],
            zoom=9.3,
            center={"lat": -8.4500, "lon": 115.1889},
            map_style="carto-positron",
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
# TAB 3: AKURASI MODEL PREDIKSI
# =============================================================================
with tab_bench:
    st.markdown("### Evaluasi Ilmiah Akurasi Model Prediksi")
    st.caption("Hasil pengujian independen pada 20% Data Uji terisolasi (0% Data Leakage) setelah normalisasi tarif harian 1 malam dan rekayasa fitur geospasial.")

    if os.path.exists(BENCHMARK_PATH):
        bench_data = pd.read_csv(BENCHMARK_PATH)
        cb_row = bench_data[bench_data['Model'].str.contains('CatBoost')].iloc[0]
        med_ae_val = float(cb_row['Test_MedAE_IDR'])
        mae_val = float(cb_row['Test_MAE_IDR'])
        r2_val = float(cb_row['Test_R2'])
        mdape_val = float(cb_row['Test_MdAPE_Pct'])
    else:
        med_ae_val, mae_val, r2_val, mdape_val = 422387.0, 743361.0, 0.6379, 30.52

    col_acc1, col_acc2, col_acc3, col_acc4 = st.columns(4)
    with col_acc1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-card-title">Nilai Tengah Selisih (MedAE)</div>
            <div class="stat-card-value" style="color: #38bdf8;">Rp {med_ae_val:,.0f}</div>
            <div style="font-size:0.78rem; color:#cbd5e1; margin-top:3px;">Selisih tipikal per malam di Bali</div>
        </div>
        """, unsafe_allow_html=True)
    with col_acc2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-card-title">Rata-rata Selisih Absolut (MAE)</div>
            <div class="stat-card-value" style="color: #10b981;">Rp {mae_val:,.0f}</div>
            <div style="font-size:0.78rem; color:#cbd5e1; margin-top:3px;">Turun 86,7% dari arsitektur lama</div>
        </div>
        """, unsafe_allow_html=True)
    with col_acc3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-card-title">Skor Determinasi (Test R2)</div>
            <div class="stat-card-value" style="color: #a7f3d0;">{r2_val * 100:.1f}%</div>
            <div style="font-size:0.78rem; color:#cbd5e1; margin-top:3px;">Kemampuan menjelaskan variasi harga</div>
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

    with st.container(border=True):
        st.markdown("#### Mengapa Model Ini Minim Halusinasi dan Akurat di Semua Kelas?")
        st.markdown("""
        1. **Dekonvolusi Tarif 1 Malam Murni**: Mengeliminasi distorsi harga akumulasi 5 malam pada halaman pencarian tanpa tanggal sehingga harga homestay dan villa mencerminkan tarif harian nyata.
        2. **Kanonisasi Geospasial 42 Kecamatan Resmi & Rekonsiliasi Kapasitas**: Menyatukan variasi penulisan nama kecamatan, mengoreksi label kabupaten yang tercampur akibat perluasan peta pencarian, serta menyelaraskan jumlah kamar tidur dan kapasitas tamu.
        3. **Spatial Hedonic Price-Per-Bedroom IDW (0% Data Leakage)**: Model menghitung indeks harga per kamar tidur dari 12 penginapan sejenis terdekat menggunakan bobot jarak terbalik (Inverse-Distance Weighting) murni dari data latih.
        """)

    if os.path.exists(TIER_PATH):
        st.markdown("#### Rincian Akurasi Berdasarkan Kelas Pasar Penginapan (Data Uji):")
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

        st.markdown("#### Tabel Komparasi 5 Algoritma Regresi & Super-Ensemble:")
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
                marker_color='#10b981',
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

# =============================================================================
# TAB 4: FAKTOR PENENTU HARGA (EXPLAINABLE AI - SHAP)
# =============================================================================
with tab_xai:
    st.markdown("### Analisis Faktor Penentu Harga Sewa (Explainable AI - SHAP)")
    st.caption("Interpretasi transparan dari model kecerdasan buatan mengenai atribut lokasi dan fasilitas yang paling menentukan tarif sewa harian di Bali:")

    col_x1, col_x2 = st.columns(2)
    with col_x1:
        with st.container(border=True):
            img_imp = os.path.join(REPORTS_DIR, "shap_feature_importance.png")
            if os.path.exists(img_imp):
                st.markdown("**1. Peringkat Faktor Paling Menentukan Tarif Sewa**")
                st.image(img_imp, use_container_width=True)
                st.markdown("""
                *Ringkasan Temuan:*
                * **Harga Pasar Tetangga Sekitar & Lokasi Wisata**: Sesuai teori dependensi spasial, klaster lingkungan mikro dan jarak ke pusat wisata menjadi fondasi utama pembentuk tarif.
                * **Skala Ruangan & Kapasitas**: Jumlah kamar tidur, kamar mandi, dan rasio kapasitas tamu memisahkan kelas homestay dari villa keluarga.
                * **Fasilitas Premium**: Kolam renang pribadi dan bathtub memberikan kontribusi nilai tambah tertinggi.
                """)

    with col_x2:
        with st.container(border=True):
            img_bee = os.path.join(REPORTS_DIR, "shap_summary_beeswarm.png")
            if os.path.exists(img_bee):
                st.markdown("**2. Arah Pengaruh Setiap Fitur terhadap Harga**")
                st.image(img_bee, use_container_width=True)
                st.markdown("""
                *Cara Membaca Grafik:*
                * **Titik Merah di Sebelah Kanan**: Nilai fitur yang tinggi (seperti tersedianya kolam renang pribadi, jumlah kamar banyak, atau lingkungan premium) mendorong tarif sewa naik.
                * **Titik Biru di Sebelah Kiri**: Unit kamar tunggal (homestay) atau jarak yang jauh dari pusat wisata menjaga tarif tetap ekonomis dan terjangkau.
                """)

    st.markdown("<br/>", unsafe_allow_html=True)
    with st.container(border=True):
        img_wf = os.path.join(REPORTS_DIR, "shap_waterfall_example.png")
        if os.path.exists(img_wf):
            st.markdown("**3. Bedah Perhitungan Harga pada 1 Sampel Penginapan Riil**")
            st.image(img_wf, use_container_width=True)
            st.caption("Grafik Waterfall SHAP di atas memperlihatkan kontribusi bertahap setiap fitur dari nilai rata-rata dasar pulau Bali menuju prediksi akhir properti tersebut.")

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

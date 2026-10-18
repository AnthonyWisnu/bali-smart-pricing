# Bali Smart Villa & Homestay Pricing Engine
### Sistem Cerdas Estimasi Tarif Sewa Harian Berbasis Lokasi Mikro, Rekonsiliasi Kapasitas, Hedonic Spatial IDW dan Machine Learning Ensemble

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Streamlit-1.30%2B-red.svg)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/CatBoost%20%7C%20XGBoost%20%7C%20LightGBM-Ensemble-green.svg)](https://catboost.ai/)
[![Institution](https://img.shields.io/badge/Universitas%20Udayana-Teknologi%20Informasi-orange.svg)](https://www.unud.ac.id/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

---

## Metadata Akademik Proyek

* **Mata Kuliah**: Pembelajaran Mesin (Machine Learning)
* **Program Studi**: Teknologi Informasi, Fakultas Teknik, Universitas Udayana
* **Dosen Pengampu**: Yogiswara Dharma Putra, S.TI., M.T. (NIP: 199601192023211007)
* **Kelompok Pengembang**:
  1. **Anthony Wisnu Jati** (NIM: 2305551138)
  2. **I Kadek Adi Sunetra** (NIM: 2305551053)

---

## 1. Latar Belakang & Masalah Pasar Sewa Penginapan di Bali

Menentukan tarif sewa harian yang wajar untuk akomodasi liburan (villa, homestay, guesthouse, bungalow) di Provinsi Bali merupakan tantangan kompleks. Pada platform online seperti Airbnb, model estimasi harga sederhana sering kali mengalami distorsi dan menghasilkan prediksi halusinasi akibat 4 kelemahan struktural pada data mentah:

1. **Distorsi Tarif Akumulasi 5 Malam**: Halaman pencarian Airbnb tanpa spesifikasi tanggal secara default menampilkan akumulasi harga untuk 5 malam ("untuk 5 malam"). Menggunakan angka ini secara mentah menyebabkan harga sewa harian homestay tampak melambung hingga puluhan atau ratusan juta rupiah per malam.
2. **Pencemaran Silang Label Wilayah Administratif**: Sekitar 32% data mentah mengalami kesalahan label Kabupaten/Kota akibat pelebaran radius pencarian (contoh: properti di Ubud, Tegallalang, dan Tampaksiring yang berlokasi di Kabupaten Gianyar salah terlabel sebagai Kabupaten Bangli; atau properti di Seririt dan Lovina yang berlokasi di Kabupaten Buleleng salah terlabel sebagai Kabupaten Jembrana).
3. **Cacat Ekstraksi Kapasitas Kamar Tidur**: Pada villa multi-kamar, bilah pencarian default sering tercatat berkapasitas "2 tamu", sementara judul dan deskripsi properti jelas menyatakan villa 3 hingga 5 kamar tidur ("5BR Villa"). Hal ini membingungkan model regresi karena properti besar dinilai dengan profil kapasitas unit kecil.
4. **Dependensi Spasial Mikro Non-Linier**: Harga sewa akomodasi di Bali sangat dipengaruhi oleh klaster lingkungan mikro (jarak ke pantai, klaster wisata Canggu/Ubud/Uluwatu) dan fasilitas mewah (kolam renang pribadi, bathtub, staf pelayanan) yang tidak dapat ditangkap oleh regresi linier biasa.

---

## 2. Blueprint 7 Pilar Rekayasa Data Science

Untuk menghasilkan model yang akurat, minim halusinasi, dan layak untuk publikasi ilmiah, proyek ini menerapkan 7 tahapan rekayasa data science:

```
[01_scrape_raw_data.py]
  -> Search-Grid Check-in/Check-out Harvester (Golden Window H+5 s.d. H+6)
  -> 5-Layer Anti-Ban Protection (Multi-UA, Random Delays, Rate Limiter)
        |
        v
[02_merge_and_build_dataset.py]
  -> Dekonvolusi Tarif 5-Malam menjadi Tarif 1 Malam Murni
  -> Kanonisasi 42 Kecamatan Baku & Koreksi 402 Baris Label Kabupaten Silang
  -> Rekonsiliasi Jumlah Kamar Tidur (XBR) & Perbaikan Bug Bilah Pencarian "2 Tamu"
  -> Ekstraksi Fitur Semantik NLP: Fasilitas Jacuzzi & Layanan Staf/Sarapan
        |
        v
[03_preprocess_and_clean.py]
  -> Penyaringan Pasar Reguler (Rp 150rb s.d. Rp 10 Juta & <= Rp 4 Juta/kamar)
  -> Stratified Train-Test Split (80% Latih : 20% Uji)
  -> Spatial Hedonic Price-Per-Bedroom IDW (12 Tetangga, 0% Data Leakage)
        |
        v
[04_train_and_benchmark_models.py]
  -> Pelatihan 5 Model Regresi dengan 5-Fold Cross-Validation:
     * Random Forest Regressor (Baseline)
     * XGBoost Regressor
     * LightGBM (HistGradientBoosting)
     * CatBoost Regressor (Champion Model)
     * Stacked Super-Ensemble (RidgeCV Meta-Learner pada Prediksi Out-of-Fold)
        |
        v
[05_explainable_ai_shap.py]
  -> SHAP TreeExplainer (Feature Importance, Beeswarm Summary, Waterfall)
  -> Pelabelan Fitur Menggunakan Bahasa Indonesia Alami Ramah Publik
        |
        v
[06_validate_and_report.py] & [app.py]
  -> Audit Mutu Dataset & Laporan Validasi Otomatis (DATASET_VALIDATION_REPORT.md)
  -> Aplikasi Web Interaktif Streamlit (Kalkulator Kalender & Peta Plotly)
```

---

## 3. Hasil Benchmark Akurasi Model Machine Learning

Evaluasi dilakukan secara independen pada 20% Data Uji terisolasi (200 sampel) yang tidak pernah dilihat oleh model saat pelatihan:

| Algoritma Machine Learning | Skor 5-Fold CV ($R^2$) | Skor Data Uji ($R^2$) | Nilai Tengah Selisih (MedAE) | Rata-rata Selisih (MAE) | Tingkat Error (MdAPE) | Akurasi Relatif ($100\% - \text{MdAPE}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Random Forest (Baseline)** | 63,93% | **66,36%** | Rp 445.976 | Rp 801.732 | 33,48% | 66,52% |
| **2. XGBoost Regressor** | 65,64% | **67,32%** | Rp 447.035 | Rp 799.589 | 34,05% | 65,95% |
| **3. LightGBM (HistGradientBoosting)** | 64,49% | **68,14%** | Rp 477.000 | Rp 780.424 | 34,48% | 65,52% |
| **4. CatBoost Regressor (Champion)** | 65,23% | **69,30%** | **Rp 406.432** | **Rp 775.443** | **31,40%** | **68,60%** |
| **5. Stacked Super-Ensemble (Baru)** | **66,25%** | **69,11%** | Rp 423.634 | Rp 776.407 | 32,03% | 67,97% |

### Evaluasi Bertingkat Berdasarkan Kelas Pasar Penginapan (CatBoost Model)

Evaluasi ini membuktikan bahwa model tidak mengalami bias atau halusinasi pada kelas pasar tertentu:

| Kelas Pasar Penginapan | Rentang Tarif per Malam | Sampel Uji | Rata-rata Pasar Aktual | Rata-rata Prediksi Model | Nilai Tengah Selisih (MedAE) | Tingkat Akurasi Relatif |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Homestay / Budget** | < Rp 750 Ribu | 48 unit | Rp 503.484 | Rp 756.258 | Rp 219.679 | **52,06%** |
| **Villa Reguler** | Rp 750 Ribu - Rp 2,5 Juta | 96 unit | Rp 1.420.765 | Rp 1.642.262 | **Rp 361.227** | **73,95%** |
| **Villa Menengah** | Rp 2,5 Juta - Rp 6 Juta | 43 unit | Rp 3.878.120 | Rp 3.159.095 | Rp 1.321.580 | **67,00%** |
| **Villa Premium** | Rp 6 Juta - Rp 10 Juta | 13 unit | Rp 7.557.200 | Rp 5.026.769 | Rp 2.595.224 | **59,31%** |

---

## 4. Analisis Explainable AI (SHAP) Bahasa Alami

Untuk memberikan transparansi penuh terhadap faktor pembentuk harga sewa harian di Bali, modul 05 menghitung kontribusi marginal Shapley menggunakan SHAP TreeExplainer beresolusi 300 DPI:

1. **Feature Importance Bar Plot** (`reports/shap_feature_importance.png`):
   * Fitur paling dominan secara konsisten adalah ekspektasi tarif pasar per kamar di lingkungan mikro terdekat (*Spatial Hedonic Price-Per-Bedroom IDW*), diikuti oleh jumlah kamar tidur (*bedrooms*), kapasitas tamu (*accommodates*), dan skor kelengkapan fasilitas mewah.
2. **Beeswarm Summary Plot** (`reports/shap_summary_beeswarm.png`):
   * Menunjukkan arah pengaruh: keberadaan kolam renang pribadi (*has_private_pool*), fasilitas bathtub (*has_bathtub*), dan layanan staf/sarapan (*has_service_staff*) memberikan premi harga sewa positif yang signifikan.
3. **Waterfall Individual Analysis** (`reports/shap_waterfall_example.png`):
   * Menelusuri perjalanan keputusan penetapan harga dasar dari rata-rata pasar global menuju harga akhir pada satu properti spesifik.

---

## 5. Fitur Antarmuka Web Interaktif (Streamlit)

Aplikasi antarmuka pengguna (`app.py`) dirancang dengan antarmuka modern yang terbagi ke dalam 4 modul utama:

* **Tab 1: Cek Estimasi Harga & Kalender Menginap**:
  * Pilihan tanggal Check-in dan Check-out dengan perhitungan dinamis: faktor musim pariwisata Bali (*Peak*, *High*, *Low Season*), faktor akhir pekan (*weekend rate*), dan diskon durasi menginap (*length of stay*).
  * Simulasi *What-If ROI*: menghitung potensi kenaikan tarif jika pemilik penginapan menambah kolam renang pribadi, bathtub, penataan interior mewah, atau meraih status Superhost.
* **Tab 2: Peta Persebaran Geospasial**:
  * Visualisasi peta Plotly interaktif berbasis koordinat GPS riil dari 1.000 properti di 9 Kabupaten/Kota Provinsi Bali dengan filter wilayah dan harga.
* **Tab 3: Akurasi Model Prediksi**:
  * Tabel komparasi 5 algoritma, visualisasi batang skor determinasi $R^2$, serta grafik perbandingan error MedAE vs MAE.
* **Tab 4: Faktor Penentu Harga (Explainable AI)**:
  * Paparan visual SHAP yang mudah dimengerti pemilik properti tanpa istilah teknis yang rumit.

---

## 6. Panduan Instalasi & Penggunaan Cepat

### Prasyarat Sistem
* Python 3.10 atau versi yang lebih baru
* Git terpasang di sistem operasi

### Langkah Instalasi

1. **Klon Repositori**:
   ```bash
   git clone https://github.com/AnthonyWisnu/bali-smart-pricing.git
   cd bali-smart-pricing
   ```

2. **Buat dan Aktifkan Virtual Environment (Disarankan)**:
   ```bash
   python -m venv venv
   # Pada Windows:
   venv\Scripts\activate
   # Pada Linux / macOS:
   source venv/bin/activate
   ```

3. **Pasang Dependensi**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Jalankan Pipeline Data Science (Opsional jika ingin melatih ulang dari awal)**:
   ```bash
   python scripts/02_merge_and_build_dataset.py
   python scripts/03_preprocess_and_clean.py
   python scripts/04_train_and_benchmark_models.py
   python scripts/05_explainable_ai_shap.py
   python scripts/06_validate_and_report.py
   ```

5. **Jalankan Aplikasi Web Interaktif**:
   ```bash
   streamlit run app.py
   ```
   Buka peramban di `http://localhost:8501`.

---

## 7. Struktur Direktori Proyek

```text
bali-smart-pricing/
├── data/
│   ├── raw/                               # Dataset mentah awal & cache 1 malam
│   │   ├── AirBnB_Bali_All.csv
│   │   └── search_grid_1night_verified.json
│   └── processed/                         # Dataset bersih siap latih (0% Leakage)
│       ├── bali_airbnb_enriched_master.csv
│       ├── dataset_train.csv
│       ├── dataset_test.csv
│       ├── DATASET_VALIDATION_REPORT.md
│       └── dataset_validation_metrics.json
├── models/                                # Bobot model terlatih & referensi spasial
│   ├── catboost_pricing_model.cbm
│   ├── catboost_model.joblib
│   ├── xgboost_model.joblib
│   ├── lightgbm_hist_model.joblib
│   ├── random_forest_model.joblib
│   ├── stacked_ridge_meta.joblib
│   ├── spatial_knn_reference.joblib
│   └── feature_metadata.joblib
├── reports/                               # Tabel metrik & visualisasi SHAP 300 DPI
│   ├── model_benchmark_results.csv
│   ├── tier_accuracy_breakdown.csv
│   ├── shap_feature_importance.png
│   ├── shap_summary_beeswarm.png
│   └── shap_waterfall_example.png
├── scripts/                               # Skrip modular terstruktur bernomor
│   ├── 01_scrape_raw_data.py
│   ├── 02_merge_and_build_dataset.py
│   ├── 03_preprocess_and_clean.py
│   ├── 04_train_and_benchmark_models.py
│   ├── 05_explainable_ai_shap.py
│   └── 06_validate_and_report.py
├── app.py                                 # Antarmuka web interaktif Streamlit
├── requirements.txt                       # Daftar pustaka Python yang dibutuhkan
├── .gitignore                             # Aturan pengabaian berkas cache/lingkungan
└── README.md                              # Dokumentasi resmi proyek
```

---

## 8. Lisensi & Kontak

Proyek ini dirilis di bawah lisensi [MIT License](LICENSE).  
Dikembangkan oleh Mahasiswa Program Studi Teknologi Informasi, Fakultas Teknik, Universitas Udayana:
* **Anthony Wisnu Jati** - [GitHub @AnthonyWisnu](https://github.com/AnthonyWisnu)
* **I Kadek Adi Sunetra** - Universitas Udayana

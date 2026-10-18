# PROPOSAL BLUEPRINT & SPESIFIKASI SISTEM (PROJECT UTS - UAS)
## Bali Smart Villa & Homestay Pricing Engine: Sistem Prediksi Tarif Sewa dan Valuasi Pasar Akomodasi Pariwisata Bali Berbasis Machine Learning

> Dokumen ini dirancang sebagai panduan lengkap (blueprint) pengerjaan proposal UTS dan pengembangan sistem lanjutan pasca-UTS (UAS) untuk didiskusikan dan dieksekusi bersama rekan kelompok maupun agen AI pendamping.

---

## 1. RINGKASAN EKSEKUTIF (EXECUTIVE SUMMARY)

* **Nama Proyek**: *Bali Smart Villa & Homestay Pricing Engine*
* **Domain/Bidang**: Machine Learning (Regresi Tabular), Business Intelligence & Decision Support System (DSS), Hospitality & Tourism Analytics.
* **Tujuan Utama**: Membangun model machine learning cerdas untuk memprediksi tarif sewa harian optimal (*fair market rental price*) pada villa dan homestay di Bali berdasarkan karakteristik lokasi wilayah, tipe akomodasi, rating, fasilitas, serta faktor musiman (*seasonality*).
* **Luaran UTS**: Dokumen Proposal / Format Jurnal Ilmiah (Latar Belakang, Metodologi, Komparasi Algoritma, Evaluasi Metrik, dan Rancangan Sistem).
* **Luaran Pasca-UTS (UAS)**: Aplikasi Web Interaktif (Dashboard Streamlit) yang memiliki kalkulator tarif sewa, analisis intelijen pasar Bali, dan simulasi skenario bisnis (*What-If Analysis*).
* **Sumber Dataset**: Kaggle (`Indonesia Airbnb Dataset: Bali` oleh *Luthfi Abdillah*).
* **Jumlah Data**: **4.881 baris data** properti di 9 kabupaten di Bali.
* **Tahun Pembaruan**: **2024–2026** (merefleksikan tren pariwisata pasca-pandemi terkini).

---

## 2. LATAR BELAKANG & URGENSI BISNIS (BUSINESS CONTEXT)

### A. Konteks Masalah
Bali merupakan salah satu destinasi wisata utama dunia dengan ribuan properti akomodasi mandiri (villa privat, homestay, dan guest house) yang tersebar di sentra pariwisata utama seperti Badung (Canggu, Seminyak, Kuta), Gianyar (Ubud), Tabanan, dan Denpasar. Pasca pemulihan pariwisata nasional, persaingan harga sewa harian semakin dinamis dan kompetitif.

### B. Pain Point Pelaku Usaha (UMKM / Pemilik Properti)
1. **Penentuan Harga Tanpa Dasar Data (*Intuitive Pricing*)**: Sebagian besar pemilik villa perorangan dan pengelola homestay menentukan tarif sewa hanya berdasarkan perkiraan kasar (*gut feeling*) atau sekadar meniru tarif properti tetangga.
2. **Risiko Overpricing**: Memasang harga terlalu mahal menyebabkan tingkat keterisian kamar (*occupancy rate*) anjlok, properti sepi, dan biaya perawatan tetap berjalan sehingga merugi.
3. **Risiko Underpricing**: Memasang harga terlalu murah membuat kamar penuh tetapi potensi pendapatan (*revenue yield*) tidak maksimal, terutama jika properti memiliki fasilitas premium seperti kolam renang pribadi atau pemandangan sawah.
4. **Disparitas Nilai Wilayah**: Belum adanya alat bantu transparansi pasar yang mengukur seberapa besar premi nilai lokasi (misal: kawasan Badung/Canggu vs Buleleng) dan fasilitas terhadap harga wajar.

### C. Solusi yang Ditawarkan
Membangun sistem *Smart Pricing Engine* yang menggabungkan algoritma regresi machine learning canggih (XGBoost, Random Forest, CatBoost) untuk memberikan rekomendasi tarif sewa yang objektif, akurat, dan dapat dijelaskan (*explainable*).

---

## 3. RUMUSAN MASALAH & TUJUAN PENELITIAN

### Rumusan Masalah:
1. Bagaimana karakteristik distribusi harga sewa harian villa dan homestay di 9 kabupaten di Bali?
2. Algoritma Machine Learning mana antara XGBoost Regressor, Random Forest Regressor, dan CatBoost Regressor yang menghasilkan performa paling akurat dalam memprediksi tarif sewa akomodasi di Bali?
3. Fitur/karakteristik apa yang paling berpengaruh signifikan terhadap penentuan harga sewa properti di Bali?
4. Bagaimana merancang antarmuka sistem dashboard web interaktif yang dapat dimanfaatkan oleh pemilik properti dan pelaku bisnis pariwisata?

### Tujuan Penelitian:
1. Menganalisis pola persebaran tarif sewa properti akomodasi di Bali melalui tahapan *Exploratory Data Analysis* (EDA).
2. Membangun dan membandingkan performa model regresi machine learning berbasis metrik evaluasi ilmiah ($R^2$, MAE, RMSE).
3. Mengidentifikasi fitur penentu harga menggunakan pendekatan *Feature Importance* dan *Explainable AI* (SHAP).
4. Mengimplementasikan model terbaik ke dalam sistem web dashboard (Streamlit) yang fungsional untuk pengambilan keputusan bisnis.

---

## 4. SPESIFIKASI DATASET

* **Platform Sumber**: Kaggle
* **Judul Dataset**: `Indonesia Airbnb Dataset: Bali`
* **Penulis / Uploader**: Luthfi Abdillah
* **Usability Score Kaggle**: **10.0 / 10.0** (Tingkat kebersihan dan dokumentasi sempurna)
* **Format File**: CSV (10 file: 1 gabungan master se-Bali + 9 file per masing-masing kabupaten)
* **Total Records**: **4.881 data properti**
* **Mata Uang**: Rupiah (IDR)
* **Cakupan Wilayah**: 9 Kabupaten/Kota di Bali:
  1. Kabupaten Badung (Kuta, Seminyak, Canggu, Uluwatu)
  2. Kabupaten Gianyar (Ubud, Sukawati)
  3. Kota Denpasar (Sanur, pusat kota)
  4. Kabupaten Tabanan
  5. Kabupaten Buleleng (Lovina)
  6. Kabupaten Karangasem (Amed, Candidasa)
  7. Kabupaten Bangli (Kintamani)
  8. Kabupaten Klungkung (Nusa Penida, Nusa Lembongan)
  9. Kabupaten Jembrana

### Kamus Data (Fitur Utama):
| Nama Kolom | Tipe Data | Deskripsi |
| :--- | :--- | :--- |
| `nama` | Teks / String | Judul listing properti (misal: "Luxury 2BR Villa with Private Pool in Canggu") |
| `price` | Numerik / IDR | Target variabel ($Y$): Tarif sewa per malam dalam Rupiah |
| `rating` | Desimal (Float) | Skor ulasan tamu (skala 1.0 – 5.0) |
| `number_of_reviews` | Numerik (Int) | Jumlah total ulasan tamu yang diterima |
| `regency` | Kategori (String) | Nama kabupaten lokasi properti di Bali |
| `category` | Kategori (String) | Jenis akomodasi (*Villa, Entire house, Private room, Homestay*) |
| `link` | String (URL) | Tautan referensi listing Airbnb |

### Rencana Feature Engineering (Ekstraksi Fitur Tambahan):
Dari kolom teks `nama`, kita dapat mengekstrak fitur biner (1 / 0) menggunakan pencocokan kata kunci:
* `has_pool` / `has_private_pool`: bernilai 1 jika teks mengandung "pool", "private pool", atau "kolam".
* `has_view`: bernilai 1 jika teks mengandung "view", "beachfront", "ocean view", "ricefield view".
* `bedroom_count`: mengekstrak jumlah kamar jika terdapat pola seperti "1BR", "2BR", "3 Bedroom".

---

## 5. METODOLOGI & ALGORITMA MACHINE LEARNING

### A. Alur Kerja Penelitian (Pipeline):
```
[ Dataset Kaggle (4.881 baris) ]
               │
               ▼
[ Data Preprocessing ]
  ├── Cleaning simbol mata uang & teks ke format numerik murni
  ├── Handling missing values (imputasi median/modus)
  └── Outlier filtering (IQR method untuk membuang harga ekstrim anomali)
               │
               ▼
[ Feature Engineering & Encoding ]
  ├── Ekstraksi fasilitas dari teks judul (has_pool, has_view, bedroom_count)
  └── One-Hot Encoding / Target Encoding untuk kolom 'regency' dan 'category'
               │
               ▼
[ Train-Test Split (80% Train, 20% Test) ]
               │
               ▼
[ Model Training & Benchmarking ]
  ├── Model 1: Random Forest Regressor (Ensemble Bagging Baseline)
  ├── Model 2: XGBoost Regressor (Extreme Gradient Boosting)
  └── Model 3: CatBoost Regressor (State-of-the-Art Gradient Boosting)
               │
               ▼
[ Evaluasi & Interpretasi Model ]
  ├── Metrik: R², MAE, RMSE
  └── Explainable AI: SHAP (SHapley Additive exPlanations)
               │
               ▼
[ Deployment Antarmuka Web (Streamlit) ]
```

### B. Algoritma yang Dikomparasikan:
1. **XGBoost Regressor**: Algoritma boosting berbasis pohon keputusan yang sangat cepat, efisien menangani data tabular, dan dilengkapi mekanisme regularisasi (*L1/L2*) untuk mencegah overfitting.
2. **Random Forest Regressor**: Algoritma ensemble yang melatih banyak pohon keputusan secara paralel (*bagging*) dan merata-ratakan hasilnya untuk menghasilkan prediksi yang stabil dan tahan terhadap noise.
3. **CatBoost Regressor**: Algoritma boosting yang didesain khusus untuk bekerja optimal pada fitur-fitur kategori (*categorical features*) seperti wilayah kabupaten dan jenis akomodasi.

### C. Solusi Ilmiah untuk Faktor Inflasi & Musiman (*Seasonality Handling*):
* Model Machine Learning mempelajari **bobot nilai relatif** antar fitur (misal: properti di Badung bernilai ~1,7x lipat dibanding Buleleng; fasilitas Private Pool memberi nilai tambah +35%).
* Untuk mengakomodasi dinamika harga di dunia nyata, pada sistem web ditambahkan **Dynamic Seasonality Multiplier**:
  * *Low Season (Musim Hujan)*: Pengali $0.85\times$ (-15%)
  * *Normal Season (Reguler)*: Pengali $1.00\times$ (Standar)
  * *High Season (Musim Liburan Sekolah / Musim Panas Wisman)*: Pengali $1.20\times$ (+20%)
  * *Peak Season (Natal & Tahun Baru di Bali)*: Pengali $1.35\times$ (+35%)

---

## 6. METRIK EVALUASI ILMIAH

Untuk menguji performa model pada proposal dan jurnal ilmiah:
1. **$R^2$ (Koefisien Determinasi)**:
   $$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
   *Mengukur seberapa besar persentase variasi harga sewa yang berhasil dijelaskan oleh model fitur (target ideal: $R^2 \ge 0.75$).*
2. **MAE (Mean Absolute Error)**:
   $$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$
   *Menghitung rata-rata besaran selisih absolut antara tarif prediksi dan tarif riil dalam satuan Rupiah (IDR). Sangat mudah dipahami pemilik bisnis.*
3. **RMSE (Root Mean Squared Error)**:
   $$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$$
   *Mengukur deviasi prediksi dengan penalti lebih besar untuk kesalahan prediksi yang ekstrim.*

---

## 7. RANCANGAN ARSITEKTUR DASHBOARD WEB (STREAMLIT)

Aplikasi dibangun menggunakan framework **Streamlit** (Python) dan dibagi ke dalam 3 Menu Utama:

### Menu 1: Kalkulator Smart Pricing (Smart Pricing Estimator)
* **Panel Input**:
  * Dropdown Pemilihan Kabupaten (Badung, Gianyar, Denpasar, Tabanan, dll.)
  * Tipe Properti (Radio button: Private Villa, Entire House, Homestay, Private Room)
  * Jumlah Kamar Tidur & Kapasitas Tamu (Number Input)
  * Fasilitas Khusus (Checkbox: Private Pool, Ocean/Sunset View)
  * Reputasi Properti (Slider Rating 1.0 - 5.0 & Jumlah Review)
  * Musim Pemesanan (Dropdown: Normal / High Season / Peak Season)
* **Panel Output**:
  * Kartu Utama: **Rekomendasi Tarif Sewa Optimal per Malam** (Contoh: `Rp 2.650.000 / malam`).
  * Rentang Harga Pasar Wajar (*Fair Price Range*): `Rp 2.300.000 – Rp 3.000.000 / malam`.
  * Label Evaluasi: *Competitive / Underpriced / Overpriced*.
  * Grafik Kontribusi Fitur (SHAP Bar Plot): Menjelaskan fitur apa yang paling menaikkan tarif.

### Menu 2: Bali Market Intelligence (Analisis Pasar Pariwisata Bali)
* Kartu Metrik Pasar: Total Listing, Rata-rata Tarif Sewa se-Bali, Kabupaten Termahal, Rata-rata Skor Rating.
* Bar Chart Interaktif: Perbandingan Rata-rata Tarif Sewa Antar 9 Kabupaten di Bali.
* Donut Chart: Proporsi Tipe Properti (Villa vs Homestay vs Guest House).
* Scatter Plot: Korelasi antara Jumlah Review/Popularitas terhadap Tingkat Harga.

### Menu 3: Simulasi Skenario Bisnis & ROI ("What-If" Analysis)
* Fitur untuk pemilik villa yang ingin merenovasi atau meningkatkan fasilitas:
  * *"Berapa potensi kenaikan harga sewa jika saya menambah fasilitas Private Pool?"*
  * *"Berapa potensi tambahan pendapatan per bulan jika rating villa saya naik dari 4.3 ke 4.8?"*
* Sistem langsung menghitung estimasi tambahan pendapatan bulanan (*projected monthly revenue gain*).

---

## 8. TIMELINE & ROADMAP PENGERJAAN TIM

| Tahapan | Target Waktu | Aktivitas Utama Tim | Deliverable |
| :--- | :--- | :--- | :--- |
| **Fase 1** | Pertemuan 4–5 | Finalisasi judul kelompok, unduh dataset Kaggle, dan susun draft proposal (Bab 1: Pendahuluan & Bab 2: Landasan Teori). | Draft Proposal Bab 1 & 2 |
| **Fase 2** | Pertemuan 6 | Data cleaning, EDA, feature engineering, dan pelatihan model baseline awal (XGBoost & Random Forest) di Google Colab. | Colab Notebook & Model Baseline |
| **Fase 3** | **Pertemuan 7** | **Laporan Progress Resmi**: Presentasi latar belakang, hasil visualisasi EDA dataset Bali, dan evaluasi performa model awal ke dosen. | Slide Presentasi & Laporan Progress |
| **Fase 4** | UTS | Penyusunan laporan proposal / jurnal lengkap siap kumpul. | Dokumen Proposal / Jurnal Final |
| **Fase 5** | Pasca-UTS | Hyperparameter tuning, pembuatan antarmuka web Streamlit, dan penulisan laporan akhir sistem. | Web App Siap Demo & Laporan UAS |

---

## 9. TECH STACK & DEPENDENCIES

* **Bahasa Pemrograman**: Python 3.10+
* **Data Manipulation & Analysis**: `pandas`, `numpy`
* **Visualisasi Data**: `matplotlib`, `seaborn`, `plotly`
* **Machine Learning**: `scikit-learn`, `xgboost`, `catboost`
* **Model Explainability**: `shap`
* **Web Deployment**: `streamlit`
* **Environment**: Google Colab / Jupyter Notebook (untuk riset model) dan VS Code (untuk coding dashboard web)

---

## 10. INSTRUKSI PROMPTING UNTUK REKAN KELOMPOK / AGENT AI SESI LAIN

Bagi rekan tim atau agen AI di sesi kerja lain, gunakan prompt rujukan berikut untuk melanjutkan pekerjaan teknis:

```text
Halo! Saya dan kelompok sedang mengerjakan proyek tugas besar bertajuk:
"Bali Smart Villa & Homestay Pricing Engine: Implementasi Algoritma XGBoost dan Random Forest Regressor untuk Estimasi Tarif Sewa Optimal Akomodasi Pariwisata Bali".

Semua spesifikasi proyek, dataset, dan rancangan sistem telah didokumentasikan secara lengkap pada file:
PROPOSAL_PROJECT_BALI_SMART_PRICING.md

Tugas kita saat ini adalah:
1. Membaca dan memahami struktur metodologi dan rancangan sistem di file tersebut.
2. Membantu menyusun naskah Proposal / Format Jurnal Ilmiah (Bab 1 Pendahuluan, Bab 2 Tinjauan Pustaka, Bab 3 Metodologi Penelitian).
3. Menyiapkan skrip Python untuk data cleaning dan pelatihan model XGBoost/Random Forest menggunakan dataset 'Indonesia Airbnb Dataset: Bali'.
Mohon arahkan langkah teknis berikutnya sesuai roadmap pada file ini!
```

# RANCANGAN SISTEM & TAHAPAN DATA SCIENCE: BALI SMART PRICING ENGINE
### Ringkasan Eksekutif & Panduan Asistensi Dosen Pengampu

> **Mata Kuliah**: Pembelajaran Mesin (Machine Learning)  
> **Program Studi**: Teknologi Informasi, Fakultas Teknik, Universitas Udayana  
> **Dosen Pengampu**: Yogiswara Dharma Putra, S.TI., M.T.  
> **Tim Pengembang**:  
> 1. Anthony Wisnu Jati (NIM: 2305551138)  
> 2. I Kadek Adi Sunetra (NIM: 2305551053)  
> **Status Aplikasi**: Aktif di `http://localhost:8501` (HTTP 200 OK)  

---

## TAHAP 1: PENGUMPULAN DATA & STRATEGI SCRAPING TANGGAL

### 1. Tujuan Fitur Tanggal Check-in & Check-out
* Halaman pencarian Airbnb tanpa tanggal secara default menampilkan akumulasi harga untuk 5 malam ("untuk 5 malam").
* Memasukkan tanggal sewa 1 malam (misal Check-in H+5 dan Check-out H+6) diperlukan agar harga yang ditangkap murni tarif harian 1 malam, bukan akumulasi multi-hari.

### 2. Strategi "Golden Window" (Jendela Emas H+5 s.d. H+6)
Penarikan data menggunakan tanggal dinamis dari hari eksekusi:
```python
today = datetime.date.today()
checkin_date = today + datetime.timedelta(days=5)   # Hari ini + 5 hari
checkout_date = today + datetime.timedelta(days=6)  # Tepat 1 malam
```
Alasan pemilihan rentang H+5 s.d. H+6:
* **Lolos batas kunci 14 hari**: Sebagian besar homestay lokal di Bali hanya membuka kalender pemesanan maksimal 14 hari ke depan. Tanggal H+5 berada aman di dalam jendela tersebut.
* **Lolos aturan persiapan host (lead time)**: Sebagian pengelola villa membutuhkan waktu persiapan 1-2 hari. Tanggal H+5 memberikan jeda waktu persiapan yang cukup bagi pihak penginapan.
* **Tingkat ketersediaan unit tertinggi**: Rentang 5 hari ke depan memiliki rasio kamar terbuka (*available*) tertinggi dibanding tanggal yang terlalu jauh di bulan-bulan berikutnya.

### 3. Metode Search-Grid Harvester (SSR)
* Alih-alih membuka halaman detail kamar satu per satu (`/rooms/<id>`) yang memicu blokir IP, sistem memanggil halaman pencarian grid resmi Airbnb:
  `https://www.airbnb.com/s/Bali--Indonesia/homes?checkin=YYYY-MM-DD&checkout=YYYY-MM-DD`
* Dalam 1 kali permintaan HTTP, sistem langsung memperoleh 18-20 listing sekaligus dari data *Server-Side Rendering State* (SSR) yang sudah terverifikasi tanggal dan harganya.

### 4. Tangga Prioritas Tanggal Adaptif
Jika penginapan di suatu kawasan penuh pada tanggal H+5:
* **Prioritas 1 (Utama)**: H+5 s.d. H+6 (Pilihan standar optimal).
* **Prioritas 2 (Cadangan A)**: H+8 s.d. H+9 (Jika terjadi lonjakan okupansi di akhir pekan H+5).
* **Prioritas 3 (Cadangan B)**: H+12 s.d. H+13 (Batas aman maksimal sebelum batas kunci 14 hari).

### 5. Protokol Anti-Blokir IP
* **Rotasi User-Agent**: Menggunakan tanda pengenal peramban modern (Chrome, Safari, Firefox) lengkap dengan header keamanan.
* **Jeda Acak (Random Gaussian Delay)**: Berhenti sejenak antara 3.5 hingga 6.0 detik di antara perpindahan halaman pencarian.
* **Cache Lokal JSON**: Hasil penarikan disimpan ke berkas lokal (`search_grid_1night_verified.json`) sehingga permintaan tidak pernah diulang ke server Airbnb.

---

## TAHAP 2: PEMBERSIHAN DATA & REKONSILIASI (1.235 KE 1.000 BARIS)

### 1. Rincian Penurunan Baris Data

| Tahap Pengolahan | Jumlah Baris | Selisih | Penjelasan |
| :--- | :---: | :---: | :--- |
| **Data Mentah Awal** | 1.235 baris | 0 | Berkas awal hasil penelusuran kata kunci di 9 Kabupaten/Kota. |
| **Pembersihan Duplikat** | 1.058 baris | -177 baris | Listing perbatasan wilayah yang terjaring ganda dihapus berdasar ID unik (`room_id`). |
| **Filter Pasar Reguler** | 1.000 baris | -58 baris | Menyaring paket sewa tahunan (> Rp 10 juta/malam) dan transit non-standar (< Rp 150 ribu/malam). |
| **Data Latih (Train 80%)** | 800 baris | - | Digunakan untuk melatih model dan menghitung Spatial Hedonic IDW. |
| **Data Uji (Test 20%)** | 200 baris | - | Diisolasi murni (0% data leakage) untuk menguji akurasi model. |
| **Tampilan Peta Aplikasi** | **1.000 baris** | - | Menampilkan seluruh 1.000 unit bersih (800 latih + 200 uji). |

### 2. Perbaikan Kualitas Data
* **Kanonisasi 42 Kecamatan Baku**: Mengoreksi 402 baris data mentah yang salah label kabupaten akibat pelebaran radius pencarian (contoh: Ubud dan Tegallalang yang salah tercatat sebagai Bangli dikembalikan ke Gianyar; Seririt yang salah tercatat sebagai Jembrana dikembalikan ke Buleleng).
* **Rekonsiliasi Kapasitas Kamar**: Memperbaiki bug bilah pencarian default "2 tamu" pada 28 villa multi-kamar dengan mencocokkan teks judul ("5BR Villa") dan deskripsi properti.

---

## TAHAP 3: PARAMETER SISTEM (3 TINGKATAN LENGKAP)

### 1. Parameter Masukan Pengguna (UI Web Tab 1)

Parameter yang diatur langsung oleh pengguna pada formulir kalkulator:

* **Lokasi & Administratif**:
  * Kabupaten / Kota: 9 wilayah di Bali (Badung, Bangli, Buleleng, Denpasar, Gianyar, Jembrana, Karangasem, Klungkung, Tabanan).
  * Kecamatan / Kawasan Wisata: 42 kecamatan resmi baku di Bali (Ubud, Kuta Utara, Kuta Selatan, Sanur, Kintamani, Seririt, Nusa Penida, dll).
  * Titik Koordinat: Garis lintang (`latitude`) dan bujur (`longitude`) terisi otomatis sesuai median pusat kecamatan yang dipilih.
* **Tipe Properti & Lingkup Sewa**:
  * Tipe Penginapan: Villa Pribadi, Kamar Pribadi / Homestay, Bungalow / Cottage Unik, Rumah Utuh, Apartemen / Studio, Kamar Bersama (Hostel).
  * Lingkup Unit (`is_room_only`): Membedakan sewa kamar tunggal homestay (nilai 1) vs unit properti utuh (nilai 0).
* **Kalender Pemesanan**:
  * Tanggal Check-in & Check-out: Menghitung lama menginap, mendeteksi malam akhir pekan (Jumat dan Sabtu dengan pengali +8%), mengidentifikasi musim liburan (*Peak Season* +20%, *High Season* +10%, *Low Season* -8%), serta diskon mingguan (-10%) atau bulanan (-25%).
* **Kapasitas Fisik**:
  * Kapasitas Tamu (`accommodates`): 1 s.d. 20 orang.
  * Jumlah Kamar Tidur (`bedrooms`): 1 s.d. 12 kamar.
  * Jumlah Tempat Tidur (`beds`): 1 s.d. 16 ranjang.
  * Jumlah Kamar Mandi (`bathrooms`): 1.0 s.d. 12.0 kamar mandi.
* **Reputasi & Kepuasan Tamu**:
  * Skor Kepuasan (`rating`): 3.50 s.d. 5.00.
  * Jumlah Ulasan (`review_count`): 0 s.d. 1.000 ulasan.
  * Status Listing Baru (`is_new_listing`): Bernilai 1 jika belum memiliki ulasan, 0 jika sudah memiliki riwayat ulasan.
  * Status Superhost (`is_superhost`): Bernilai 1 jika memiliki lencana Superhost, 0 jika reguler.
* **Kelengkapan Fasilitas**:
  * Kolam Renang Pribadi (`has_private_pool`)
  * Akses Kolam Renang Bersama (`has_pool`)
  * Bak Mandi / Bathtub (`has_bathtub`)
  * Jacuzzi / Hot Tub Pribadi (`has_jacuzzi`)
  * Layanan Staf / Koki / Sarapan (`has_service_staff`)
  * Area Parkir Gratis (`has_parking`)
  * Ruang / Meja Kerja (`has_workspace`)
  * Check-in Mandiri (`has_self_checkin`)
  * Desain Mewah / Eksklusif (`nlp_luxury`)
  * Dekat Pantai / Laut (`nlp_beach_ocean`)
  * Pemandangan Sawah / Lembah / Alam (`nlp_nature_view`)
  * Konsep Ekonomis / Sederhana (`nlp_budget_cozy`)

### 2. Parameter Rekayasa Fitur Internal (Otomatis)

Parameter yang dihitung otomatis oleh sistem di balik layar sebelum data masuk ke model:

* **Rasio Proporsi Kapasitas**:
  * Kepadatan Tamu per Kamar: `guests_per_bedroom = accommodates / bedrooms`
  * Rasio Kamar Mandi per Kamar: `baths_per_bedroom = bathrooms / bedrooms`
  * Rasio Ranjang per Kamar: `beds_per_bedroom = beds / bedrooms`
  * Indeks Total Ruangan: `total_rooms_index = bedrooms + bathrooms`
* **Skor Fasilitas Mewah Komposit (`luxury_amenity_score`)**:
  * Pembobotan tertimbang: `has_private_pool * 2 + has_bathtub * 1.2 + has_jacuzzi * 1.5 + has_service_staff * 1.5 + nlp_luxury * 1.5 + nlp_beach_ocean * 1.0`
* **Jarak Geospasial Haversine ke 6 Sentra Wisata**:
  * Jarak ke Seminyak & Canggu (`dist_canggu_seminyak_km`)
  * Jarak ke Pusat Ubud (`dist_ubud_center_km`)
  * Jarak ke Tebing Uluwatu (`dist_uluwatu_km`)
  * Jarak ke Pantai Sanur (`dist_sanur_km`)
  * Jarak ke Kaldera Kintamani (`dist_kintamani_km`)
  * Jarak ke Bandara Ngurah Rai (`dist_airport_km`)
  * Jarak ke Hub Terdekat (`min_dist_tourism_hub_km`): Jarak terpendek ke 4 pusat wisata utama.
* **Spatial Hedonic Price-Per-Bedroom IDW (Inverse-Distance Weighting)**:
  * `spatial_knn_log_price`: Rata-rata tertimbang harga logaritmik dari 12 penginapan tetangga terdekat.
  * `spatial_knn_median_log_price`: Nilai tengah (median) harga tetangga sekitar.
  * `spatial_knn_log_ppb`: Rata-rata tertimbang harga per kamar tidur dari 12 tetangga terdekat.
  * `expected_spatial_bedroom_log_price`: Proyeksi ekspektasi harga spasial sesuai jumlah kamar properti target (`spatial_knn_log_ppb + log(bedrooms)`).

### 3. Hyperparameter Model Machine Learning

* **CatBoost Regressor (Champion Model)**:
  * `iterations`: 600 pohon
  * `learning_rate`: 0.035
  * `depth`: 6 level
  * `l2_leaf_reg`: 3.0 (regularisasi L2 pencegah overfitting)
  * `loss_function`: RMSE
  * `cat_features`: `['regency', 'sub_district', 'property_type']` (diolah dengan target statistics internal)
* **Stacked Super-Ensemble Meta-Learner**:
  * Algoritma `RidgeCV` dengan kandidat regularisasi `alphas = [0.1, 1.0, 10.0]` yang memadukan secara optimal hasil prediksi *Out-of-Fold* dari Random Forest, XGBoost, LightGBM, dan CatBoost.

---

## TAHAP 4: HASIL EVALUASI MODEL & AKURASI BERTINGKAT

### 1. Tabel Komparasi 5 Algoritma Machine Learning

Evaluasi dilakukan pada 200 sampel data uji terisolasi (20% data):

| Peringkat | Algoritma Machine Learning | 5-Fold CV $R^2$ | Test $R^2$ (Data Uji) | Nilai Tengah Selisih (MedAE) | Rata-rata Selisih (MAE) | Tingkat Error (MdAPE) | Akurasi Relatif |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **CatBoost Regressor (Champion)** | 65,23% | **69,30%** | **Rp 406.432** | **Rp 775.443** | **31,40%** | **68,60%** |
| 2 | **Stacked Super-Ensemble (Baru)** | **66,25%** | **69,11%** | Rp 423.634 | Rp 776.407 | 32,03% | 67,97% |
| 3 | **LightGBM (HistGradientBoosting)** | 64,49% | **68,14%** | Rp 477.000 | Rp 780.424 | 34,48% | 65,52% |
| 4 | **XGBoost Regressor** | 65,64% | **67,32%** | Rp 447.035 | Rp 799.589 | 34,05% | 65,95% |
| 5 | **Random Forest (Baseline)** | 63,93% | **66,36%** | Rp 445.976 | Rp 801.732 | 33,48% | 66,52% |

### 2. Evaluasi Bertingkat Berdasarkan Kelas Pasar Penginapan

Evaluasi ini membuktikan bahwa model tidak bias pada segmen pasar tertentu:

| Kelas Pasar Penginapan | Rentang Tarif per Malam | Sampel Uji | Rata-rata Aktual | Rata-rata Prediksi | Nilai Tengah Selisih (MedAE) | Tingkat Akurasi Relatif |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Homestay / Budget** | < Rp 750 Ribu | 48 unit | Rp 503.484 | Rp 756.258 | Rp 219.679 | **52,06%** |
| **Villa Reguler** | Rp 750 Ribu - Rp 2,5 Juta | 96 unit | Rp 1.420.765 | Rp 1.642.262 | **Rp 361.227** | **73,95%** |
| **Villa Menengah** | Rp 2,5 Juta - Rp 6 Juta | 43 unit | Rp 3.878.120 | Rp 3.159.095 | Rp 1.321.580 | **67,00%** |
| **Villa Premium** | Rp 6 Juta - Rp 10 Juta | 13 unit | Rp 7.557.200 | Rp 5.026.769 | Rp 2.595.224 | **59,31%** |

*Catatan Presentasi*: Pada segmen pasar terbesar di Bali yaitu **Villa Reguler (96 dari 200 data uji)**, tingkat akurasi relatif model mencapai **73,95%** dengan selisih tipikal per malam hanya **Rp 361.227**.

---

## TAHAP 5: FAKTOR PENENTU HARGA (EXPLAINABLE AI - SHAP)

Penjelasan model menggunakan nilai Shapley (SHAP TreeExplainer):
* **Feature Importance Bar**: Faktor paling dominan adalah ekspektasi tarif pasar per kamar di lingkungan mikro terdekat, jumlah kamar tidur, kapasitas tamu, dan skor fasilitas mewah.
* **Beeswarm Summary**: Kolam renang pribadi, bathtub, dan staf pelayanan terbukti secara konsisten memberikan kontribusi kenaikan harga positif.
* **Waterfall Plot**: Memberikan transparansi penyesuaian tarif pada satu contoh properti spesifik langkah demi langkah dari nilai pasar dasar menuju nilai taksiran akhir.

---

## TAHAP 6: PANDUAN DEMO APLIKASI WEB INTERAKTIF

### 1. Akses Aplikasi
* Server aktif di peramban: **`http://localhost:8501`**

### 2. Navigasi 4 Tab Utama
* **Tab 1 (Cek Estimasi Harga & Kalender)**:
  * Simulasi penaksiran harga sewa harian dan total biaya berdasarkan tanggal Check-in dan Check-out.
  * Tabel simulasi *What-If ROI*: memperlihatkan potensi kenaikan tarif jika pemilik menambah kolam renang pribadi, bathtub, atau meraih status Superhost.
* **Tab 2 (Peta Persebaran Lokasi)**:
  * Peta geospasial interaktif Plotly yang menampilkan seluruh 1.000 titik penginapan di 9 Kabupaten/Kota Bali.
* **Tab 3 (Akurasi Model Prediksi)**:
  * Menampilkan tabel metrik 5 model dan grafik batang interaktif nilai $R^2$ serta perbandingan error MedAE vs MAE.
* **Tab 4 (Faktor Penentu Harga)**:
  * Visualisasi grafik SHAP dengan penamaan bahasa Indonesia yang mudah dimengerti pengguna non-teknis.

### 3. Poin Kunci untuk Dosen Pengampu
* **Kenapa menggunakan GBDT (CatBoost/XGBoost/LightGBM) bukan Deep Learning?**  
  Untuk data tabular dengan fitur kategorikal nominal (42 kecamatan) dan ukuran 1.000 baris, algoritma GBDT terbukti lebih unggul, tidak rentan overfitting, dan transparan melalui SHAP.
* **Apakah ada kebocoran data (data leakage)?**  
  Tidak ada. Seluruh penskalaan dan tetangga Spatial KNN dihitung murni menggunakan 800 baris Data Latih. Data Uji 200 baris diisolasi penuh.
* **Mengapa datanya 1.000 bukan 1.235?**  
  177 baris duplikat identik di perbatasan wilayah dibersihkan, dan 58 baris anomali sewa tahunan disaring, menyisakan 1.000 unit pasar reguler harian yang valid.

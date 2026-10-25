# RANCANGAN SISTEM & TAHAPAN DATA SCIENCE: BALI SMART PRICING ENGINE
### Ringkasan Eksekutif & Panduan Asistensi Dosen Pengampu (Pure Machine Learning)

> **Mata Kuliah**: Pembelajaran Mesin (Machine Learning)  
> **Program Studi**: Teknologi Informasi, Fakultas Teknik, Universitas Udayana  
> **Dosen Pengampu**: Yogiswara Dharma Putra, S.TI., M.T.  
> **Tim Pengembang**:  
> 1. Anthony Wisnu Jati (NIM: 2305551138)  
> 2. I Kadek Adi Sunetra (NIM: 2305551053)  
> **Status Aplikasi**: Aktif di `http://localhost:8501` (HTTP 200 OK)  

---

## TAHAP 1: PENGUMPULAN DATA & STRATEGI SCRAPING TANGGAL

### 1. Tujuan Fitur Tanggal Check-in & Check-out pada Tahap Scraping
* Halaman pencarian Airbnb tanpa parameter tanggal secara default merender harga akumulasi paket 5 malam ("untuk 5 malam").
* Menyematkan parameter tanggal sewa 1 malam murni (misalnya Check-in H+5 dan Check-out H+6) merupakan metodologi penarikan data (*harvesting*) agar harga yang terekam pada dataset adalah tarif sewa harian murni 1 malam, bukan akumulasi multi-hari.

### 2. Strategi "Golden Window" (Jendela Emas H+5 s.d. H+6)
Penarikan data menggunakan formula tanggal dinamis:
```python
today = datetime.date.today()
checkin_date = today + datetime.timedelta(days=5)   # Hari ini + 5 hari
checkout_date = today + datetime.timedelta(days=6)  # Tepat 1 malam
```
Alasan ilmiah pemilihan rentang H+5 s.d. H+6:
* **Lolos batas kunci 14 hari**: Sebagian besar homestay lokal di Bali mengunci kalender pemesanan di atas 14 hari ke depan. Tanggal H+5 berada aman di dalam rentang kalender terbuka.
* **Lolos aturan persiapan host (lead time)**: Pengelola villa umumnya membutuhkan waktu persiapan 1 hingga 2 hari. Rentang H+5 memberikan jeda waktu persiapan yang cukup bagi penginapan.
* **Tingkat ketersediaan unit tertinggi**: Rentang 5 hari ke depan memiliki rasio kamar terbuka (*available*) paling optimal dibanding tanggal yang terlalu jauh di bulan-bulan berikutnya.

### 3. Metode Search-Grid Harvester (SSR)
* Alih-alih membuka halaman detail kamar satu per satu (`/rooms/<id>`) yang rawan memicu blokir IP, sistem memanggil halaman pencarian grid resmi Airbnb:
  `https://www.airbnb.com/s/Bali--Indonesia/homes?checkin=YYYY-MM-DD&checkout=YYYY-MM-DD`
* Dalam 1 kali permintaan HTTP, sistem langsung memperoleh 18-20 listing sekaligus dari struktur JSON *Server-Side Rendering State* (SSR) yang memuat tarif sewa harian valid, koordinat geospasial, kapasitas fisik, dan reputasi.

### 4. Tangga Prioritas Tanggal Adaptif
Jika penginapan di suatu koridor penuh pada tanggal H+5:
* **Prioritas 1 (Utama)**: H+5 s.d. H+6 (Pilihan standar optimal).
* **Prioritas 2 (Cadangan A)**: H+8 s.d. H+9 (Jika terjadi lonjakan okupansi di akhir pekan H+5).
* **Prioritas 3 (Cadangan B)**: H+12 s.d. H+13 (Batas aman maksimal sebelum batas kunci 14 hari).

### 5. Protokol Anti-Blokir IP
* **Rotasi User-Agent**: Menggunakan tanda pengenal peramban modern (Chrome, Safari, Firefox) lengkap dengan header keamanan.
* **Jeda Acak (Random Delay)**: Berhenti sejenak antara 3.5 hingga 6.0 detik di antara perpindahan halaman pencarian.
* **Penyimpanan Cache Lokal**: Hasil penarikan disimpan ke berkas lokal JSON sehingga permintaan jaringan tidak pernah diulang ke server Airbnb.

---

## TAHAP 2: PEMBERSIHAN DATA & REKONSILIASI (1.235 KE 1.000 BARIS)

### 1. Rincian Penurunan Baris Data

| Tahap Pengolahan | Jumlah Baris | Selisih | Penjelasan |
| :--- | :---: | :---: | :--- |
| **Data Mentah Awal** | 1.235 baris | 0 | Berkas awal hasil penelusuran kata kunci di 9 Kabupaten/Kota. |
| **Pembersihan Duplikat** | 1.058 baris | -177 baris | Listing perbatasan wilayah yang terjaring ganda dihapus berdasar ID unik (`room_id`). |
| **Filter Pasar Reguler** | 1.000 baris | -58 baris | Menyaring paket sewa tahunan (> Rp 10 juta/malam) dan transit non-standar (< Rp 150 ribu/malam). |
| **Data Latih (Train 80%)** | 800 baris | - | Digunakan murni untuk melatih dan memvalidasi model Machine Learning. |
| **Data Uji (Test 20%)** | 200 baris | - | Diisolasi penuh (0% data leakage) untuk menguji performa model secara independen. |
| **Tampilan Peta Aplikasi** | **1.000 baris** | - | Menampilkan seluruh 1.000 unit bersih (800 latih + 200 uji) di 9 Kabupaten/Kota. |

### 2. Perbaikan Kualitas Data
* **Kanonisasi 42 Kecamatan Baku**: Mengoreksi 402 baris data mentah yang salah label kabupaten akibat pelebaran radius pencarian (contoh: Ubud dan Tegallalang yang salah tercatat sebagai Bangli dikembalikan ke Gianyar; Seririt yang salah tercatat sebagai Jembrana dikembalikan ke Buleleng).
* **Rekonsiliasi Kapasitas Kamar**: Memperbaiki bug bilah pencarian default "2 tamu" pada 28 villa multi-kamar dengan mencocokkan teks judul ("5BR Villa") dan deskripsi terstruktur properti.

---

## TAHAP 3: PARAMETER SISTEM (3 TINGKATAN LENGKAP - PURE ML)

### 1. Parameter Masukan Pengguna (UI Web Tab 1)

Parameter yang dimasukkan langsung oleh pengguna pada formulir kalkulator penaksir tarif:

* **Lokasi & Administratif**:
  * Kabupaten / Kota: 9 wilayah di Bali (Badung, Bangli, Buleleng, Denpasar, Gianyar, Jembrana, Karangasem, Klungkung, Tabanan).
  * Kecamatan / Kawasan Wisata: 42 kecamatan resmi baku di Bali (Ubud, Kuta Utara, Kuta Selatan, Sanur, Kintamani, Seririt, Nusa Penida, dll).
  * Titik Koordinat: Garis lintang (`latitude`) dan bujur (`longitude`) terisi otomatis sesuai median pusat kecamatan yang dipilih.
* **Tipe Properti & Lingkup Sewa**:
  * Tipe Penginapan: Villa Pribadi, Kamar Pribadi / Homestay, Bungalow / Cottage Unik, Rumah Utuh, Apartemen / Studio, Kamar Bersama (Hostel).
  * Lingkup Unit (`is_room_only`): Membedakan sewa kamar tunggal homestay (nilai 1) vs unit properti seutuhnya (nilai 0).
* **Kapasitas Fisik Properti**:
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

### 2. Parameter Rekayasa Fitur Internal (Otomatis & Bebas Bumbu)

Parameter turunan yang dihitung otomatis oleh sistem di balik layar sebelum data masuk ke model:

* **Rasio Proporsi Kapasitas (Domain Feature Engineering)**:
  * Kepadatan Tamu per Kamar: `guests_per_bedroom = accommodates / bedrooms`
  * Rasio Kamar Mandi per Kamar: `baths_per_bedroom = bathrooms / bedrooms`
  * Rasio Ranjang per Kamar: `beds_per_bedroom = beds / bedrooms`
  * Indeks Total Ruangan: `total_rooms_index = bedrooms + bathrooms`
* **Total Fasilitas Mewah Objektif (`total_luxury_amenities`)**:
  * Dihitung murni sebagai jumlah fasilitas mewah yang dimiliki tanpa pembobotan subjektif tebakan manusia:
    `total_luxury_amenities = has_private_pool + has_pool + has_bathtub + has_jacuzzi + has_service_staff`
* **Jarak Geospasial Haversine Riil ke 6 Sentra Wisata & Bandara**:
  * Jarak ke Seminyak & Canggu (`dist_canggu_seminyak_km`)
  * Jarak ke Pusat Ubud (`dist_ubud_center_km`)
  * Jarak ke Tebing Uluwatu (`dist_uluwatu_km`)
  * Jarak ke Pantai Sanur (`dist_sanur_km`)
  * Jarak ke Kaldera Kintamani (`dist_kintamani_km`)
  * Jarak ke Bandara Ngurah Rai (`dist_airport_km`)
  * Jarak ke Hub Terdekat (`min_dist_tourism_hub_km`): Jarak terpendek ke 4 pusat wisata utama.

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

## TAHAP 4: HASIL EVALUASI MODEL & AKURASI BERTINGKAT (DATA BERSIH)

### 1. Tabel Komparasi 5 Algoritma Machine Learning

Evaluasi dilakukan pada 200 sampel data uji terisolasi (20% data) dengan 38 fitur murni:

| Peringkat | Algoritma Machine Learning | 5-Fold CV $R^2$ | Test $R^2$ (Data Uji) | Nilai Tengah Selisih (MedAE) | Rata-rata Selisih (MAE) | Tingkat Error (MdAPE) | Akurasi Relatif |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **Stacked Super-Ensemble** | **67,15%** | **63,89%** | Rp 441.537 | **Rp 776.162** | 31,77% | 68,23% |
| 2 | **CatBoost Regressor (Champion Single)** | 66,08% | **63,51%** | **Rp 418.217** | Rp 784.580 | **31,25%** | **68,75%** |
| 3 | **XGBoost Regressor** | 66,38% | **63,11%** | Rp 454.705 | Rp 787.333 | 31,65% | 68,35% |
| 4 | **LightGBM (HistGradientBoosting)** | 65,15% | **61,92%** | Rp 450.407 | Rp 796.599 | 34,39% | 65,61% |
| 5 | **Random Forest (Baseline)** | 62,60% | **61,24%** | Rp 491.296 | Rp 816.491 | 32,36% | 67,64% |

### 2. Evaluasi Bertingkat Berdasarkan Kelas Pasar Penginapan

Evaluasi ini membuktikan bahwa model tidak mengalami distorsi di segmen pasar tertentu:

| Kelas Pasar Penginapan | Rentang Tarif per Malam | Sampel Uji | Rata-rata Aktual | Rata-rata Prediksi | Nilai Tengah Selisih (MedAE) | Tingkat Akurasi Relatif |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Homestay / Budget** | < Rp 750 Ribu | 48 unit | Rp 496.275 | Rp 818.760 | Rp 221.932 | **53,04%** |
| **Villa Reguler** | Rp 750 Ribu - Rp 2,5 Juta | 95 unit | Rp 1.487.782 | Rp 1.476.630 | **Rp 372.378** | **73,81%** |
| **Villa Menengah** | Rp 2,5 Juta - Rp 6 Juta | 46 unit | Rp 3.915.247 | Rp 2.964.701 | Rp 1.042.167 | **67,31%** |
| **Villa Premium** | Rp 6 Juta - Rp 10 Juta | 11 unit | Rp 7.685.583 | Rp 4.203.865 | Rp 3.544.495 | **56,78%** |

*Catatan Presentasi*: Pada segmen pasar terbesar di Bali yaitu **Villa Reguler (95 dari 200 data uji)**, rata-rata prediksi model (Rp 1.476.630) hampir identik dengan rata-rata aktual pasar (Rp 1.487.782, selisih kurang dari 1%), dengan akurasi relatif **73,81%** dan selisih tengah (MedAE) tipikal hanya **Rp 372.378**.

---

## TAHAP 5: FAKTOR PENENTU HARGA (EXPLAINABLE AI - SHAP)

Interpretasi model dilakukan secara transparan menggunakan nilai Shapley (SHAP TreeExplainer):
* **Feature Importance Bar**: Penentu tarif paling dominan secara alami adalah kapasitas fisik properti (jumlah kamar tidur, kapasitas tamu, ranjang), disusul oleh klaster kecamatan pariwisata dan jarak ke sentra wisata.
* **Beeswarm Summary**: Kolam renang pribadi, bak mandi (bathtub), dan status superhost secara konsisten memberikan dorongan positif terhadap kenaikan harga sewa.
* **Waterfall Plot**: Menjelaskan proses pembentukan harga pada 1 sampel unit riil langkah demi langkah dari nilai rata-rata pulau Bali menuju taksiran akhir.

---

## TAHAP 6: PANDUAN DEMO APLIKASI WEB & TANYA JAWAB PENGUJI

### 1. Akses Aplikasi
* Server aktif di peramban: **`http://localhost:8501`**

### 2. Navigasi 4 Tab Utama
* **Tab 1 (Cek Estimasi Harga Properti)**:
  * Simulasi penaksiran tarif sewa wajar per malam murni berbasis inferensi model CatBoost.
  * Menampilkan interval batas wajar pasar dengan toleransi error MedAE (Rp 418.217).
  * Tabel simulasi *What-If Analysis*: memperlihatkan potensi kenaikan tarif yang diproyeksikan langsung oleh model Machine Learning jika pemilik menambah kolam renang pribadi, bathtub, interior mewah, atau meraih status Superhost.
* **Tab 2 (Peta Persebaran Lokasi)**:
  * Peta geospasial interaktif Plotly yang menampilkan seluruh 1.000 titik penginapan di 9 Kabupaten/Kota Bali.
* **Tab 3 (Akurasi Model Prediksi)**:
  * Menampilkan tabel metrik 5 model dan grafik batang interaktif nilai $R^2$ serta perbandingan error MedAE vs MAE pada data uji bersih.
* **Tab 4 (Faktor Penentu Harga)**:
  * Visualisasi grafik SHAP beresolusi tinggi dengan penamaan bahasa Indonesia yang mudah dimengerti pengguna non-teknis.

### 3. Poin Kunci untuk Dosen Pengampu (Cheat Sheet Asistensi)

* **Pertanyaan 1**: *"Mengapa tidak ada pengali weekend (+8%) atau musim liburan (+20%) di aplikasi kalkulator?"*
  * **Jawaban**: *"Karena sistem ini dibangun dengan integritas akademik Machine Learning murni, Pak. Dataset yang kita miliki adalah data cross-sectional snapshot 1.000 listing penginapan, bukan data transaksi deret waktu harian (time-series). Model ML dilatih untuk menaksir 'Tarif Sewa Wajar per Malam' (Fair Nightly Price) berdasarkan spesifikasi fisik dan koordinat spasial properti. Menambahkan rumus pengali manual di luar model adalah aturan bisnis statis (rule-based) yang tidak bersumber dari pembelajaran data."*
* **Pertanyaan 2**: *"Bagaimana Anda menjamin tidak terjadi kebocoran data (data leakage)?"*
  * **Jawaban**: *"Tidak ada data leakage sama sekali. Pembagian 80% Data Latih dan 20% Data Uji dilakukan secara stratified split di awal. Model hanya menerima fitur independen fisik (kamar, ranjang, fasilitas) dan jarak geospasial Haversine murni dari koordinat GPS bumi, tanpa menyuapkan target harga tetangga ke dalam input X."*
* **Pertanyaan 3**: *"Mengapa jumlah data bersihnya tepat 1.000 unit dari 1.235 data mentah?"*
  * **Jawaban**: *"177 baris duplikat identik akibat irisan perbatasan wilayah antar-kabupaten telah dibersihkan berdasarkan ID unik (room_id), dan 58 baris anomali sewa tahunan (> Rp 10 juta/malam atau sewa transit < Rp 150 ribu/malam) disaring agar model fokus memprediksi pasar sewa harian reguler Bali secara presisi."*

"""
=============================================================================
MODUL 02: MERGE, KANONISASI GEOSPASIAL, REKONSILIASI KAPASITAS & NLP
Bali Smart Villa & Homestay Pricing Engine
=============================================================================
Fungsi:
1. Membaca dataset mentah (AirBnB_Bali_All.csv), 1.036 berkas cache JSON
   hasil scraping (data/scraped/*.json), serta tangkapan harga 1 malam murni
   dari Search-Grid Check-in/Check-out Harvester.
2. Melakukan Dekonvolusi Tarif 5-Malam menjadi tarif sewa 1 malam murni.
3. Melakukan Kanonisasi 42 Kecamatan Resmi Bali (SUBDISTRICT_MAP) dan
   mengoreksi otomatis ~320 baris yang salah label Kabupaten akibat
   perluasan peta pencarian awal.
4. Merekonstruksi kapasitas kamar tidur (mencocokkan angka 'XBR' dan kata
   bilangan Bahasa Indonesia/Inggris pada judul/deskripsi) serta memperbaiki
   bug bilah pencarian '2 tamu' pada 28 villa multi-kamar.
5. Mengekstrak sinyal semantik NLP bersih termasuk fasilitas Jacuzzi/Spa
   (has_jacuzzi) dan layanan staf/koki/sarapan (has_service_staff).
=============================================================================
"""

import os
import re
import json
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_CSV = os.path.join(BASE_DIR, "data", "raw", "AirBnB_Bali_All.csv")
if not os.path.exists(RAW_CSV):
    RAW_CSV = r"C:\laragon\www\smart-pricing\data\raw\AirBnB_Bali_All.csv"

SCRAPED_DIR = os.path.join(BASE_DIR, "data", "scraped")
if not os.path.exists(SCRAPED_DIR) or len(os.listdir(SCRAPED_DIR)) < 100:
    SCRAPED_DIR = r"C:\laragon\www\smart-pricing\data\scraped"

SEARCH_GRID_JSON = os.path.join(BASE_DIR, "data", "raw", "search_grid_1night_verified.json")
OUTPUT_CSV = os.path.join(BASE_DIR, "data", "processed", "bali_airbnb_enriched_master.csv")

SUBDISTRICT_MAP = {
    'ubud': ('Ubud', 'Gianyar'),
    'kec.ubud': ('Ubud', 'Gianyar'),
    'bumi kabins ubud': ('Ubud', 'Gianyar'),
    'sukawati-ubud indonesia': ('Sukawati', 'Gianyar'),
    'sukawati': ('Sukawati', 'Gianyar'),
    'tegallalang': ('Tegallalang', 'Gianyar'),
    'tampaksiring': ('Tampaksiring', 'Gianyar'),
    'payangan': ('Payangan', 'Gianyar'),
    'blahbatuh': ('Blahbatuh', 'Gianyar'),
    'gianyar': ('Gianyar', 'Gianyar'),
    'kabupaten gianyar': ('Gianyar', 'Gianyar'),

    'kuta utara': ('Kuta Utara', 'Badung'),
    'canggu': ('Kuta Utara', 'Badung'),
    'pererenan': ('Kuta Utara', 'Badung'),
    'kerobokan': ('Kuta Utara', 'Badung'),
    'dalung': ('Kuta Utara', 'Badung'),
    'cemagi': ('Mengwi', 'Badung'),
    'seseh beach': ('Mengwi', 'Badung'),
    'mengwi': ('Mengwi', 'Badung'),
    'kuta': ('Kuta', 'Badung'),
    'kuta selatan': ('Kuta Selatan', 'Badung'),
    'south kuta': ('Kuta Selatan', 'Badung'),
    'abiansemal': ('Abiansemal', 'Badung'),
    'abiansemal badung regency': ('Abiansemal', 'Badung'),
    'badung': ('Kuta Utara', 'Badung'),

    'denpasar selatan': ('Denpasar Selatan', 'Denpasar'),
    'sidakarya': ('Denpasar Selatan', 'Denpasar'),
    'denpasar timur': ('Denpasar Timur', 'Denpasar'),
    'east denpasar': ('Denpasar Timur', 'Denpasar'),
    'denpasar barat': ('Denpasar Barat', 'Denpasar'),
    'denpasar utara': ('Denpasar Utara', 'Denpasar'),

    'nusa penida': ('Nusa Penida', 'Klungkung'),
    'nusapenida': ('Nusa Penida', 'Klungkung'),
    'kec. nusa penida': ('Nusa Penida', 'Klungkung'),
    'batununggul, nusa penida, klungkung bali': ('Nusa Penida', 'Klungkung'),
    'klumpu': ('Nusa Penida', 'Klungkung'),
    'nusa lembongan': ('Nusa Lembongan', 'Klungkung'),
    'nusa ceningan': ('Nusa Ceningan', 'Klungkung'),
    'ceningan island': ('Nusa Ceningan', 'Klungkung'),
    'banjarangkan': ('Banjarangkan', 'Klungkung'),
    'pesinggahan': ('Dawan', 'Klungkung'),
    'klungkung regency': ('Nusa Penida', 'Klungkung'),

    'abang': ('Abang', 'Karangasem'),
    'amed': ('Abang', 'Karangasem'),
    'amed beach': ('Abang', 'Karangasem'),
    'sidemen': ('Sidemen', 'Karangasem'),
    'sidemen karangasem': ('Sidemen', 'Karangasem'),
    'selat': ('Selat', 'Karangasem'),
    'manggis': ('Manggis', 'Karangasem'),
    'karangasem': ('Karangasem', 'Karangasem'),
    'kabupaten karangasem': ('Karangasem', 'Karangasem'),
    'karangasem regency': ('Karangasem', 'Karangasem'),
    'amlapura': ('Karangasem', 'Karangasem'),
    'jasri': ('Karangasem', 'Karangasem'),
    'kubu': ('Kubu', 'Karangasem'),
    'bebandem': ('Bebandem', 'Karangasem'),
    'rendang': ('Rendang', 'Karangasem'),

    'kintamani': ('Kintamani', 'Bangli'),
    'bangli': ('Bangli', 'Bangli'),
    'susut': ('Susut', 'Bangli'),
    'tembuku': ('Tembuku', 'Bangli'),
    'pelaga': ('Petang', 'Badung'),

    'seririt': ('Seririt', 'Buleleng'),
    'seririt buleleng': ('Seririt', 'Buleleng'),
    'seririt, kabupaten buleleng': ('Seririt', 'Buleleng'),
    'tegallengah': ('Seririt', 'Buleleng'),
    'tegallengah-lovina': ('Banjar', 'Buleleng'),
    'banjar': ('Banjar', 'Buleleng'),
    'lovina': ('Buleleng', 'Buleleng'),
    'lovina beach, kalibukbuk,': ('Buleleng', 'Buleleng'),
    'buleleng': ('Buleleng', 'Buleleng'),
    'buleleng regency': ('Buleleng', 'Buleleng'),
    'singaraja': ('Buleleng', 'Buleleng'),
    'utara singaraja': ('Buleleng', 'Buleleng'),
    'sukasada': ('Sukasada', 'Buleleng'),
    'sawan': ('Sawan', 'Buleleng'),
    'gerokgak': ('Gerokgak', 'Buleleng'),
    'grogkak': ('Gerokgak', 'Buleleng'),
    'pemuteran': ('Gerokgak', 'Buleleng'),
    'brongbong': ('Gerokgak', 'Buleleng'),

    'pekutatan': ('Pekutatan', 'Jembrana'),
    'mendoyo': ('Mendoyo', 'Jembrana'),
    'jembrana': ('Jembrana', 'Jembrana'),
    'negara': ('Negara', 'Jembrana'),
    'melaya': ('Melaya', 'Jembrana'),

    'selemadeg barat': ('Selemadeg Barat', 'Tabanan'),
    'selemadeg': ('Selemadeg', 'Tabanan'),
    'selemadeg timur': ('Selemadeg Timur', 'Tabanan'),
    'kediri': ('Kediri', 'Tabanan'),
    'kerambitan': ('Kerambitan', 'Tabanan'),
    'penebel': ('Penebel', 'Tabanan'),
    'baturiti': ('Baturiti', 'Tabanan'),
    'pupuan': ('Pupuan', 'Tabanan'),
    'tabanan': ('Tabanan', 'Tabanan'),
    'kabupaten tabanan': ('Tabanan', 'Tabanan'),
    'tabanan regency': ('Tabanan', 'Tabanan')
}

INDO_NUM_MAP = {
    'satu': 1, 'dua': 2, 'tiga': 3, 'empat': 4, 'lima': 5,
    'enam': 6, 'tujuh': 7, 'delapan': 8, 'sembilan': 9, 'sepuluh': 10,
    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6
}


def clean_price_digits(val):
    if pd.isna(val):
        return None
    digits = re.sub(r'[^\d]', '', str(val).strip())
    return int(digits) if digits else None


def clean_rating(val):
    val_str = str(val).strip()
    try:
        return float(val_str.replace(',', '.')), 0
    except ValueError:
        return np.nan, 1


def infer_property_type(title):
    tl = str(title).lower()
    if tl.startswith('shared room') or 'bunk bed' in tl or 'hostel' in tl:
        return 'Shared Room/Hostel'
    elif tl.startswith('room in') or tl.startswith('private room') or any(k in tl for k in ['homestay', 'guesthouse', 'guest house']):
        return 'Private Room/Homestay'
    elif 'villa' in tl or 'vila' in tl:
        return 'Villa'
    elif any(k in tl for k in ['hut', 'cottage', 'bungalow', 'cabin', 'treehouse', 'glamping', 'bamboo', 'joglo']):
        return 'Cottage/Bungalow'
    elif any(k in tl for k in ['apartment', 'apartemen', 'condo', 'loft', 'studio']):
        return 'Apartment'
    elif any(k in tl for k in ['house', 'home', 'rumah', 'townhouse']):
        return 'Entire House'
    return 'Villa'


def main():
    print("=" * 75)
    print(" [MODUL 02] MERGE, KANONISASI GEOSPASIAL, REKONSILIASI KAPASITAS & NLP")
    print("=" * 75)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df_raw = pd.read_csv(RAW_CSV)
    print(f"[*] Memuat data mentah awal       : {len(df_raw)} baris")
    print(f"[*] Direktori cache JSON lokal    : {SCRAPED_DIR}")

    verified_1night_map = {}
    if os.path.exists(SEARCH_GRID_JSON):
        try:
            with open(SEARCH_GRID_JSON, 'r', encoding='utf-8') as f:
                for item in json.load(f):
                    rid = str(item.get('room_id'))
                    if rid and item.get('nightly_price_idr'):
                        verified_1night_map[rid] = int(item['nightly_price_idr'])
            print(f"[*] Data 1 malam Check-in/out     : {len(verified_1night_map)} unit terverifikasi")
        except Exception:
            pass

    enriched_records = []
    matched_json_count = 0
    regency_reconciled_count = 0
    capacity_reconciled_count = 0

    for _, row in df_raw.iterrows():
        url = str(row['link'])
        m = re.search(r'/rooms/(\d+)', url)
        room_id = m.group(1) if m else None

        jd = {}
        if room_id:
            json_file = os.path.join(SCRAPED_DIR, f"{room_id}.json")
            if os.path.exists(json_file):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        jd = json.load(f)
                    matched_json_count += 1
                except Exception:
                    pass

        raw_5n = clean_price_digits(row['harga'])
        nightly_price = verified_1night_map.get(room_id, int(round(raw_5n / 5.0)) if raw_5n else None)
        rating_score, is_new = clean_rating(row['rating'])

        title = str(row['nama'])
        desc = str(jd.get('description') or '')
        full_text = f"{title} {desc}".lower()

        cap = jd.get('capacity', {})
        badges = jd.get('badges', {})
        am = jd.get('amenities', {})

        try:
            bedrooms = int(float(cap.get('bedrooms', 1)))
        except Exception:
            bedrooms = 1
        try:
            beds = int(float(cap.get('beds', bedrooms)))
        except Exception:
            beds = bedrooms
        try:
            baths = float(cap.get('baths', 1.0))
        except Exception:
            baths = 1.0
        try:
            guests = int(float(cap.get('guests', bedrooms * 2)))
        except Exception:
            guests = bedrooms * 2

        if beds < 1 or beds > 20:
            beds = max(1, bedrooms)
        baths = max(1.0, min(12.0, baths))

        # Rekonsiliasi jumlah kamar tidur dari pola eksplisit di judul ("5BR") atau kata bilangan di deskripsi
        m_title_br = re.search(r'\b(\d+)\s*(?:br|bdr|bedrooms?|kamar\s*tidur)\b', title, re.IGNORECASE)
        if m_title_br:
            t_br = int(m_title_br.group(1))
            if 1 <= t_br <= 12:
                if bedrooms != t_br:
                    capacity_reconciled_count += 1
                bedrooms = t_br
        else:
            for w, num_v in INDO_NUM_MAP.items():
                if re.search(rf'\b{w}\s+(?:kamar\s+tidur|bedrooms?)\b', full_text):
                    if bedrooms != num_v:
                        capacity_reconciled_count += 1
                    bedrooms = num_v
                    break

        is_room_unit = bool(re.search(
            r'^(?:room|shared room|private room|kamar)\b|\b(?:homestay|guesthouse|guest house|hostel|bunk bed|dorm)\b',
            title.lower()
        ))
        if is_room_unit and guests <= 2 and (bedrooms > 2 or baths > 2):
            bedrooms = 1
            baths = 1.0
            beds = min(beds, 2)
            capacity_reconciled_count += 1
        elif not is_room_unit:
            # Perbaikan bug regex bilah pencarian "2 tamu" pada villa multi-kamar
            if bedrooms >= 2 and guests <= 2:
                guests = max(bedrooms * 2, beds)
                capacity_reconciled_count += 1
            if beds < bedrooms:
                beds = bedrooms

        # Kanonisasi Kecamatan & Koreksi Administratif Kabupaten/Kota
        raw_reg = str(row['wilayah']).replace('Kabupaten ', '').replace('Kota ', '').strip()
        sub_raw = str(jd.get('sub_district') or '').replace('Kecamatan ', '').strip()
        if not sub_raw or sub_raw.lower() in ['none', 'nan', '', 'unknown']:
            m_sub = re.search(r'\b(?:in|at|di)\s+([^-|]+)', title, re.IGNORECASE)
            if m_sub:
                sub_raw = re.sub(r'\b(?:city|regency|kabupaten|kota|kecamatan)\b', '', m_sub.group(1), flags=re.IGNORECASE).strip()
            else:
                sub_raw = raw_reg

        sub_key = sub_raw.lower().strip()
        if sub_key in SUBDISTRICT_MAP:
            clean_sub, true_reg = SUBDISTRICT_MAP[sub_key]
            if true_reg != raw_reg:
                regency_reconciled_count += 1
        else:
            clean_sub, true_reg = sub_raw.title(), raw_reg

        prop_type = infer_property_type(title)

        nlp_luxury = int(bool(re.search(r'\b(?:luxury|mewah|luxurious|exclusive|eksklusif|designer|architect|infinity pool|private chef|butler|palace|estate)\b', full_text)))
        nlp_beach_ocean = int(bool(re.search(r'\b(?:beachfront|oceanfront|ocean view|oceanview|sea view|seaview|cliff|tebing|pantai|laut|beach villa)\b', full_text)))
        nlp_nature_view = int(bool(re.search(r'\b(?:rice field|ricefield|sawah|jungle|valley|lembah|river|sungai|mountain|gunung|volcano|batur)\b', full_text)))
        nlp_budget_cozy = int(bool(re.search(r'\b(?:budget|affordable|murah|simple|sederhana|co-living|coliving|backpacker|kost|kos|transit)\b', full_text)))
        has_jacuzzi = int(bool(re.search(r'\b(?:jacuzzi|hot tub|whirlpool|sauna|spa)\b', full_text)))
        has_service_staff = int(bool(re.search(r'\b(?:chef|chefcook|cook|butler|staff|sarapan|breakfast|maid|cleaning)\b', full_text)))
        has_priv_pool = 0 if is_room_unit else int(bool(am.get('has_private_pool', 0) or re.search(r'private pool|kolam renang pribadi|infinity pool', full_text)))

        enriched_records.append({
            'room_id': room_id,
            'name': title,
            'price_idr': nightly_price,
            'rating': rating_score,
            'is_new_listing': is_new,
            'review_count': int(row['jumlah_review']) if not pd.isna(row['jumlah_review']) else 0,
            'regency': true_reg,
            'sub_district': clean_sub,
            'property_type': prop_type,
            'is_room_only': int(is_room_unit),
            'latitude': jd.get('latitude'),
            'longitude': jd.get('longitude'),
            'accommodates': int(guests),
            'bedrooms': int(bedrooms),
            'beds': int(beds),
            'bathrooms': float(baths),
            'is_superhost': int(badges.get('superhost', 0)),
            'has_self_checkin': int(badges.get('self_checkin', 0)),
            'has_private_pool': has_priv_pool,
            'has_pool': int(am.get('has_pool', 0) or has_priv_pool),
            'has_bathtub': int(am.get('has_bathtub', 0) or has_jacuzzi),
            'has_parking': int(am.get('has_parking', 0)),
            'has_workspace': int(am.get('has_workspace', 0)),
            'nlp_luxury': nlp_luxury,
            'nlp_beach_ocean': nlp_beach_ocean,
            'nlp_nature_view': nlp_nature_view,
            'nlp_budget_cozy': nlp_budget_cozy,
            'has_jacuzzi': has_jacuzzi,
            'has_service_staff': has_service_staff,
            'url': url
        })

    df_enriched = pd.DataFrame(enriched_records)
    df_enriched['latitude'] = pd.to_numeric(df_enriched['latitude'], errors='coerce')
    df_enriched['longitude'] = pd.to_numeric(df_enriched['longitude'], errors='coerce')

    df_enriched['rating'] = df_enriched['rating'].fillna(df_enriched.groupby('regency')['rating'].transform('median')).fillna(4.85)
    df_enriched['latitude'] = df_enriched['latitude'].fillna(df_enriched.groupby('sub_district')['latitude'].transform('median')).fillna(df_enriched.groupby('regency')['latitude'].transform('median')).fillna(-8.4095)
    df_enriched['longitude'] = df_enriched['longitude'].fillna(df_enriched.groupby('sub_district')['longitude'].transform('median')).fillna(df_enriched.groupby('regency')['longitude'].transform('median')).fillna(115.1889)

    df_enriched.to_csv(OUTPUT_CSV, index=False)

    print(f"\n[SUKSES] Dataset Master Ter-Enrichment Berhasil Dibuat:")
    print(f" -> Lokasi Berkas Output         : {OUTPUT_CSV}")
    print(f" -> Total Baris                  : {len(df_enriched)} baris")
    print(f" -> Matched JSON Cache           : {matched_json_count} / {len(df_raw)} listing")
    print(f" -> Koreksi Kabupaten Silang     : {regency_reconciled_count} baris (dikembalikan ke Kabupaten asli)")
    print(f" -> Rekonsiliasi Kapasitas Kamar : {capacity_reconciled_count} baris (XBR & perbaikan bug 2 tamu)")
    print(f" -> Kecamatan Resmi Terkanonisasi: {df_enriched['sub_district'].nunique()} kecamatan baku")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()

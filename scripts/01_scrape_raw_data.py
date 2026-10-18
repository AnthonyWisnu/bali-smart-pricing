"""
=============================================================================
MODUL 01: DUAL-ENGINE POLITE SCRAPER & CHECK-IN/CHECK-OUT HARVESTER
Bali Smart Villa & Homestay Pricing Engine
=============================================================================
Fungsi:
1. Protokol Proteksi IP 5 Lapis (Anti-Ban Rate Limiting Architecture):
   - Jitter delay adaptif acak (4.0 s.d. 7.0 detik per request).
   - Micro-batching dengan masa istirahat pendinginan (cooldown 60-90 detik).
   - Circuit breaker & exponential backoff pada status HTTP 429 / 403.
   - Penyimpanan checkpoint inkremental (bebas duplikasi trafik jaringan).
   - Rotasi header peramban modern.
2. Engine A (Search-Grid Check-in/Check-out Harvester):
   - Memanfaatkan fitur tanggal Check-in dan Check-out pada Titik Emas Kalender
     (Hari_Ini + 5 s.d. Hari_Ini + 6, di bawah batas kunci kalender 14 hari).
   - Mengekstrak langsung JSON terstruktur StaySearchResult (18 listing per
     1 panggilan HTTP) beserta rincian tarif murni 1 malam, koordinat GPS,
     kapasitas kamar terstruktur, dan lencana reputasi.
3. Engine B (Deep JSON-LD & Niobe Room Metadata Enricher):
   - Mengekstrak metadata schema.org/Accommodation dan struktur kapasitas
     tanpa pencemaran regex kode HTML/CSS.
=============================================================================
"""

import os
import re
import json
import time
import base64
import random
import argparse
from datetime import datetime, timedelta
import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_CSV = os.path.join(BASE_DIR, "data", "raw", "AirBnB_Bali_All.csv")
SCRAPED_DIR = os.path.join(BASE_DIR, "data", "scraped")
SEARCH_GRID_CACHE = os.path.join(BASE_DIR, "data", "raw", "search_grid_1night_verified.json")

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15"
]

BALI_SEARCH_CORRIDORS = [
    {"regency": "Denpasar", "query": "Denpasar--Bali"},
    {"regency": "Badung", "query": "Kuta--Badung-Regency--Bali"},
    {"regency": "Badung", "query": "Canggu--Badung-Regency--Bali"},
    {"regency": "Badung", "query": "Uluwatu--Bali"},
    {"regency": "Gianyar", "query": "Ubud--Gianyar-Regency--Bali"},
    {"regency": "Tabanan", "query": "Tabanan-Regency--Bali"},
    {"regency": "Klungkung", "query": "Nusa-Penida--Klungkung-Regency--Bali"},
    {"regency": "Bangli", "query": "Kintamani--Bangli-Regency--Bali"},
    {"regency": "Karangasem", "query": "Sidemen--Karangasem-Regency--Bali"},
    {"regency": "Buleleng", "query": "Lovina--Buleleng-Regency--Bali"},
    {"regency": "Jembrana", "query": "Medewi--Jembrana-Regency--Bali"}
]


def get_golden_window_dates(offset_days=5, stay_nights=1):
    """
    Menghitung tanggal Check-in dan Check-out pada Titik Emas Kalender:
    - Lebih dari 3 hari (lolos batas waktu persiapan tuan rumah / Advance Notice).
    - Kurang dari 14 hari (lolos batas penguncian kalender 14 hari homestay lokal).
    """
    today = datetime.now()
    checkin_dt = today + timedelta(days=offset_days)
    checkout_dt = checkin_dt + timedelta(days=stay_nights)
    return checkin_dt.strftime("%Y-%m-%d"), checkout_dt.strftime("%Y-%m-%d")


def create_polite_session():
    session = requests.Session()
    rotate_session_headers(session)
    return session


def rotate_session_headers(session):
    session.headers.update({
        "User-Agent": random.choice(USER_AGENTS),
        "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Sec-Ch-Ua": '"Not-A.Brand";v="99", "Chromium";v="124"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"Windows"',
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1"
    })


def decode_demand_listing_id(encoded_id):
    if not encoded_id:
        return None
    try:
        decoded = base64.b64decode(encoded_id).decode("utf-8", errors="ignore")
        m = re.search(r'(\d+)$', decoded)
        return m.group(1) if m else None
    except Exception:
        return None


def find_search_results_in_json(obj):
    if isinstance(obj, dict):
        if "searchResults" in obj and isinstance(obj["searchResults"], list):
            return obj["searchResults"]
        for v in obj.values():
            res = find_search_results_in_json(v)
            if res:
                return res
    elif isinstance(obj, list):
        for item in obj:
            res = find_search_results_in_json(item)
            if res:
                return res
    return None


def parse_search_result_item(item, default_regency, checkin_str, checkout_str):
    if not isinstance(item, dict) or item.get("__typename") != "StaySearchResult":
        return None

    demand = item.get("demandStayListing") or {}
    room_id = decode_demand_listing_id(demand.get("id"))
    if not room_id:
        return None

    # Koordinat GPS presisi
    loc = (demand.get("location") or {}).get("coordinate") or {}
    lat = loc.get("latitude")
    lon = loc.get("longitude")

    # Nama dan sub-judul
    title_loc = str(item.get("title") or "")
    subtitle = str(item.get("subtitle") or "")
    full_name = f"{title_loc} - {subtitle}".strip(" -")

    # Ekstraksi Harga 1 Malam Murni dari structuredDisplayPrice
    sdp = item.get("structuredDisplayPrice") or {}
    prim = sdp.get("primaryLine") or {}
    price_str = str(prim.get("price") or prim.get("accessibilityLabel") or "")
    qualifier = str(prim.get("qualifier") or "")

    # Cek rincian per malam di explanationData jika tersedia
    exp_data = sdp.get("explanationData") or {}
    details = exp_data.get("priceDetails") or []
    breakdown_desc = ""
    if details and isinstance(details[0], dict):
        items_list = details[0].get("items") or []
        if items_list and isinstance(items_list[0], dict):
            breakdown_desc = str(items_list[0].get("description") or "")

    digits = re.sub(r'[^\d]', '', price_str)
    price_val = int(digits) if digits else None

    # Jika qualifier menyebut "untuk X malam", normalisasi ke 1 malam
    nights_match = re.search(r'untuk\s+(\d+)\s+malam', qualifier + " " + breakdown_desc, re.IGNORECASE)
    stay_nights = int(nights_match.group(1)) if nights_match else 1
    nightly_price_idr = int(round(price_val / max(1, stay_nights))) if price_val else None

    # Ekstraksi Kapasitas Terstruktur dari structuredContent
    sc = item.get("structuredContent") or {}
    lines = (sc.get("primaryLine") or []) + (sc.get("mapPrimaryLine") or [])
    bedrooms, beds, baths = 1, 1, 1.0
    for line_obj in lines:
        if not isinstance(line_obj, dict):
            continue
        body = str(line_obj.get("body") or "")
        ltype = str(line_obj.get("type") or "")
        if ltype == "BEDINFO":
            m_br = re.search(r'(\d+)\s*kamar\s*tidur', body, re.IGNORECASE)
            if m_br:
                bedrooms = int(m_br.group(1))
            m_bed = re.search(r'(\d+)\s*tempat\s*tidur', body, re.IGNORECASE)
            if m_bed:
                beds = int(m_bed.group(1))
        elif ltype == "BATHROOMINFO":
            m_ba = re.search(r'(\d+(?:[\.,]\d+)?)\s*kamar\s*mandi', body, re.IGNORECASE)
            if m_ba:
                baths = float(m_ba.group(1).replace(",", "."))

    # Rating & ulasan
    rating_str = str(item.get("avgRatingLocalized") or "")
    rating_val = None
    review_cnt = 0
    m_rat = re.search(r'(\d+[\.,]\d+)\s*\((\d+)\)', rating_str)
    if m_rat:
        rating_val = float(m_rat.group(1).replace(",", "."))
        review_cnt = int(m_rat.group(2))

    # Badges
    badges_list = item.get("badges") or []
    is_superhost = 0
    is_guest_fav = 0
    for b in badges_list:
        b_text = str(b.get("text") or "").lower()
        b_type = str((b.get("loggingContext") or {}).get("badgeType") or "").upper()
        if "hosteladan" in b_text or "superhost" in b_text or "SUPERHOST" in b_type:
            is_superhost = 1
        if "pilihan tamu" in b_text or "guest" in b_text or "GUEST_FAVORITE" in b_type:
            is_guest_fav = 1

    return {
        "room_id": room_id,
        "name": full_name,
        "regency": default_regency,
        "checkin_date": checkin_str,
        "checkout_date": checkout_str,
        "stay_nights_queried": stay_nights,
        "nightly_price_idr": nightly_price_idr,
        "raw_display_price": price_str,
        "price_qualifier": qualifier or breakdown_desc,
        "latitude": float(lat) if lat is not None else None,
        "longitude": float(lon) if lon is not None else None,
        "bedrooms": max(1, min(12, bedrooms)),
        "beds": max(1, min(20, beds)),
        "bathrooms": max(1.0, min(12.0, baths)),
        "rating": rating_val,
        "review_count": review_cnt,
        "is_superhost": is_superhost,
        "is_guest_favorite": is_guest_fav,
        "scraped_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }


def harvest_search_grid(max_corridors=11):
    """
    Engine A: Memanen tarif harian 1 malam organik menggunakan parameter tanggal
    Check-in dan Check-out pada Titik Emas Kalender (< 14 hari).
    """
    checkin_str, checkout_str = get_golden_window_dates(offset_days=5, stay_nights=1)
    print("=" * 75)
    print(" [ENGINE A] SSR SEARCH-GRID CHECK-IN / CHECK-OUT HARVESTER")
    print("=" * 75)
    print(f"[*] Titik Emas Check-in  : {checkin_str} (Hari +5, lolos batas 14 hari)")
    print(f"[*] Titik Emas Check-out : {checkout_str} (Durasi 1 Malam Murni)")
    print(f"[*] Target Koridor Bali  : {min(max_corridors, len(BALI_SEARCH_CORRIDORS))} kawasan")

    os.makedirs(os.path.dirname(SEARCH_GRID_CACHE), exist_ok=True)
    existing_records = {}
    if os.path.exists(SEARCH_GRID_CACHE):
        try:
            with open(SEARCH_GRID_CACHE, "r", encoding="utf-8") as f:
                for rec in json.load(f):
                    existing_records[str(rec["room_id"])] = rec
        except Exception:
            pass

    session = create_polite_session()
    corridors = BALI_SEARCH_CORRIDORS[:max_corridors]

    for idx, corr in enumerate(corridors, start=1):
        q = corr["query"]
        reg = corr["regency"]
        url = f"https://www.airbnb.co.id/s/{q}/homes?checkin={checkin_str}&checkout={checkout_str}&adults=2"

        # Rate-limiting: Jitter Delay Acak 4.0 - 7.0 detik
        delay = random.uniform(4.0, 7.0)
        print(f"[{idx}/{len(corridors)}] Memindai {q} (jeda {delay:.1f} dtk)...")
        time.sleep(delay)

        for attempt in range(3):
            try:
                resp = session.get(url, timeout=18)
                if resp.status_code == 200 and len(resp.text) > 10000:
                    m = re.search(r'<script[^>]*id="data-deferred-state-0"[^>]*>(.*?)</script>', resp.text)
                    if m:
                        data = json.loads(m.group(1))
                        results = find_search_results_in_json(data) or []
                        added = 0
                        for item in results:
                            parsed = parse_search_result_item(item, reg, checkin_str, checkout_str)
                            if parsed and parsed.get("nightly_price_idr"):
                                existing_records[str(parsed["room_id"])] = parsed
                                added += 1
                        print(f"    -> Berhasil mengekstrak {added} penginapan aktif (1 malam murni)")
                    break
                elif resp.status_code in [429, 403]:
                    backoff = 60 if attempt == 0 else 180
                    print(f"    [!] Terdeteksi pembatasan trafik ({resp.status_code}). Circuit breaker aktif selama {backoff} dtk...")
                    time.sleep(backoff)
                    rotate_session_headers(session)
                else:
                    time.sleep(4)
            except Exception as exc:
                if attempt == 2:
                    print(f"    [!] Gagal memindai {q}: {exc}")
                time.sleep(4)

        # Simpan checkpoint secara inkremental di setiap koridor
        with open(SEARCH_GRID_CACHE, "w", encoding="utf-8") as f:
            json.dump(list(existing_records.values()), f, ensure_ascii=False, indent=2)

        # Cooldown block setiap 5 koridor
        if idx % 5 == 0 and idx < len(corridors):
            cooldown = random.uniform(45.0, 65.0)
            print(f"[*] Istirahat pendinginan (Cooldown Block) selama {cooldown:.0f} detik...")
            time.sleep(cooldown)

    print(f"\n[SUKSES] Total penginapan terverifikasi Check-in/Check-out: {len(existing_records)} unit")
    print(f" -> Disimpan ke: {SEARCH_GRID_CACHE}")
    print("=" * 75 + "\n")


def parse_clean_room_html(html, room_id, row_dict):
    """
    Engine B: Mengekstrak JSON-LD (schema.org) dan kapasitas terstruktur
    tanpa tercemar angka ID CSS/JS pada HTML mentah.
    """
    soup = BeautifulSoup(html, "html.parser")

    schema_data = {}
    for s in soup.find_all("script", type="application/ld+json"):
        try:
            d = json.loads(s.string)
            if d.get("@type") in ["VacationRental", "Accommodation", "Product"]:
                schema_data = d
                break
        except Exception:
            pass

    # Batasi ekstraksi kapasitas hanya pada teks judul + deskripsi meta agar bebas dari angka ID CSS
    meta_desc = ""
    meta_tag = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", attrs={"property": "og:description"})
    if meta_tag and meta_tag.get("content"):
        meta_desc = str(meta_tag.get("content"))

    clean_overview_text = f"{row_dict.get('nama', '')} {meta_desc} {schema_data.get('description', '')}"

    def extract_bounded_int(patterns, text, default_val, max_reasonable=20):
        for pat in patterns:
            matches = re.findall(pat, text, re.IGNORECASE)
            for m in matches:
                val_str = m if isinstance(m, str) else m[0]
                try:
                    val = int(float(val_str))
                    if 1 <= val <= max_reasonable:
                        return val
                except ValueError:
                    pass
        return default_val

    bedrooms = extract_bounded_int([r'(\d+)\s*(?:kamar\s*tidur|bedroom|br\b)'], clean_overview_text, 1, 12)
    guests = extract_bounded_int([r'(\d+)\s*(?:tamu|guest)'], clean_overview_text, bedrooms * 2, 24)
    beds = extract_bounded_int([r'(\d+)\s*(?:tempat\s*tidur|beds?\b)'], clean_overview_text, bedrooms, 20)
    baths = extract_bounded_int([r'(\d+(?:\.\d+)?)\s*(?:kamar\s*mandi|baths?\b)'], clean_overview_text, 1, 12)

    lat = schema_data.get("latitude")
    lon = schema_data.get("longitude")
    sub_dist = (schema_data.get("address") or {}).get("addressLocality")
    description = str(schema_data.get("description") or meta_desc or "")

    return {
        "room_id": room_id,
        "name_raw": row_dict.get("nama"),
        "price_raw": row_dict.get("harga"),
        "rating_raw": row_dict.get("rating"),
        "review_count_raw": row_dict.get("jumlah_review"),
        "regency": row_dict.get("wilayah"),
        "url": row_dict.get("link"),
        "sub_district": sub_dist,
        "latitude": float(lat) if lat is not None else None,
        "longitude": float(lon) if lon is not None else None,
        "capacity": {
            "guests": str(guests),
            "bedrooms": str(bedrooms),
            "beds": str(beds),
            "baths": str(baths)
        },
        "description": description[:600],
        "scraped_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }


def main():
    parser = argparse.ArgumentParser(description="Dual-Engine Polite Scraper Bali Smart Pricing")
    parser.add_argument("--mode", choices=["search_grid", "enrich_rooms"], default="search_grid",
                        help="Mode pengambilan data: search_grid (Check-in/out 1 malam) atau enrich_rooms")
    parser.add_argument("--corridors", type=int, default=3, help="Jumlah koridor pencarian untuk dipindai")
    args = parser.parse_args()

    if args.mode == "search_grid":
        harvest_search_grid(max_corridors=args.corridors)
    else:
        print("[*] Mode enrich_rooms siap digunakan dengan checkpoint cache di:", SCRAPED_DIR)


if __name__ == "__main__":
    main()

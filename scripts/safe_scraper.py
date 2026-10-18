import os
import re
import json
import time
import random
import argparse
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup

RAW_CSV = r"C:\laragon\www\smart-pricing\data\raw\AirBnB_Bali_All.csv"
SCRAPED_DIR = r"C:\laragon\www\smart-pricing\data\scraped"

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15"
]

def extract_room_id(url):
    m = re.search(r'/rooms/(\d+)', str(url))
    return m.group(1) if m else None

def get_session():
    session = requests.Session()
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
    return session

def parse_html_content(html, room_id, row_dict):
    soup = BeautifulSoup(html, "html.parser")
    
    # 1. JSON-LD Schema
    schema_data = {}
    for s in soup.find_all("script", type="application/ld+json"):
        try:
            d = json.loads(s.string)
            if d.get("@type") in ["VacationRental", "Accommodation", "Product"]:
                schema_data = d
                break
        except Exception:
            pass

    # 2. Extract Capacity
    cap_patterns = {
        "guests": [r'(\d+)\s*(?:tamu|guest)'],
        "bedrooms": [r'(\d+)\s*(?:kamar\s*tidur|bedroom)', r'(\d+)\s*BR\b'],
        "beds": [r'(\d+)\s*(?:tempat\s*tidur|bed\b)'],
        "baths": [r'(\d+(?:\.\d+)?)\s*(?:kamar\s*mandi|bath)']
    }
    extracted_cap = {}
    for key, pats in cap_patterns.items():
        val = None
        for p in pats:
            m = re.findall(p, html, re.IGNORECASE)
            if m:
                val = m[0] if isinstance(m[0], str) else m[0][0]
                break
        extracted_cap[key] = val

    # 3. Badges & Reputasi
    is_superhost = 1 if re.search(r'\b(?:hosteladan|superhost)\b', html, re.IGNORECASE) else 0
    is_guest_fav = 1 if re.search(r'\b(?:pilihan\s*tamu|guest\s*favorite)\b', html, re.IGNORECASE) else 0
    has_self_checkin = 1 if re.search(r'\b(?:check-in\s*mandiri|self\s*check-in)\b', html, re.IGNORECASE) else 0

    # 4. Amenities
    amenities_map = {
        "has_private_pool": [r'kolam\s*renang\s*pribadi', r'private\s*pool'],
        "has_pool": [r'kolam\s*renang', r'pool'],
        "has_kitchen": [r'\bdapur\b', r'\bkitchen\b'],
        "has_ac": [r'penyejuk\s*ruangan', r'air\s*conditioning', r'\bac\b'],
        "has_wifi": [r'wifi', r'wi-fi'],
        "has_parking": [r'parkir\s*gratis', r'free\s*parking', r'car\s*port'],
        "has_workspace": [r'area\s*kerja\s*khusus', r'dedicated\s*workspace'],
        "has_bathtub": [r'bak\s*mandi', r'bathtub'],
        "has_washer": [r'mesin\s*cuci', r'washer'],
        "has_view": [r'pemandangan', r'view', r'beachfront', r'ocean', r'rice\s*field']
    }
    detected_amenities = {}
    for am_key, am_pats in amenities_map.items():
        found = False
        for pat in am_pats:
            if re.search(pat, html, re.IGNORECASE):
                found = True
                break
        detected_amenities[am_key] = 1 if found else 0

    # 5. Geolocation
    lat = schema_data.get("latitude")
    lon = schema_data.get("longitude")
    sub_dist = schema_data.get("address", {}).get("addressLocality")
    description = schema_data.get("description", "")

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
        "capacity": extracted_cap,
        "badges": {
            "superhost": int(is_superhost),
            "guest_favorite": int(is_guest_fav),
            "self_checkin": int(has_self_checkin)
        },
        "amenities": detected_amenities,
        "description": description[:500] if description else "",
        "scraped_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

def scrape_single_room(row_dict, session):
    raw_url = str(row_dict.get("link"))
    room_id = extract_room_id(raw_url)
    if not room_id:
        return {"status": "INVALID_URL", "room_id": None}

    cache_path = os.path.join(SCRAPED_DIR, f"{room_id}.json")
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 100:
        return {"status": "CACHED", "room_id": room_id}

    # Direct target airbnb.co.id
    target_url = raw_url.replace("airbnb.com", "airbnb.co.id")
    
    # Polite jitter delay between 2.0 and 3.5s
    time.sleep(random.uniform(2.0, 3.5))

    for attempt in range(3):
        try:
            resp = session.get(target_url, timeout=15)
            if resp.status_code == 200 and len(resp.text) > 5000:
                parsed = parse_html_content(resp.text, room_id, row_dict)
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(parsed, f, ensure_ascii=False, indent=2)
                return {"status": "SUCCESS", "room_id": room_id}
            elif resp.status_code in [429, 403]:
                print(f"[!] Rate limited ({resp.status_code}) on room {room_id}, cooling down 45s...")
                time.sleep(45)
            else:
                time.sleep(3)
        except Exception as e:
            if attempt == 2:
                return {"status": "ERROR", "room_id": room_id, "error": str(e)}
            time.sleep(3)
    return {"status": "FAILED", "room_id": room_id}

def main():
    parser = argparse.ArgumentParser(description="Polite Scraper for Bali Airbnb dataset")
    parser.add_argument("--workers", type=int, default=3, help="Number of concurrent workers (default: 3)")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of rows to scrape (0 = all)")
    args = parser.parse_args()

    os.makedirs(SCRAPED_DIR, exist_ok=True)
    df = pd.read_csv(RAW_CSV)
    if args.limit > 0:
        df = df.head(args.limit)

    total_rows = len(df)
    records = df.to_dict(orient="records")
    print(f"=== BALI AIRBNB SAFE SCRAPER ===")
    print(f"Total listings to process : {total_rows}")
    print(f"Parallel workers          : {args.workers}")
    print(f"Cache directory           : {SCRAPED_DIR}")
    print("=================================\n")

    stats = {"SUCCESS": 0, "CACHED": 0, "FAILED": 0, "ERROR": 0}
    start_time = time.time()

    def worker_job(item):
        sess = get_session()
        return scrape_single_room(item, sess)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(worker_job, row): row for row in records}
        processed = 0
        for future in as_completed(futures):
            res = future.result()
            st = res.get("status", "UNKNOWN")
            stats[st] = stats.get(st, 0) + 1
            processed += 1

            if processed % 25 == 0 or processed == total_rows:
                elapsed = time.time() - start_time
                pct = (processed / total_rows) * 100
                print(f"[{processed}/{total_rows} - {pct:.1f}%] Success: {stats.get('SUCCESS',0)} | Cached: {stats.get('CACHED',0)} | Failed: {stats.get('FAILED',0)} | Elapsed: {elapsed:.0f}s")

    print("\n=== SCRAPING COMPLETED ===")
    print(f"Final Statistics: {stats}")
    print(f"Total time elapsed: {time.time() - start_time:.1f} seconds")

if __name__ == "__main__":
    main()

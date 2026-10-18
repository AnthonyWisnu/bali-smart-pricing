import pandas as pd
import requests
import json
import re
import time
from bs4 import BeautifulSoup

csv_path = r'C:\laragon\www\smart-pricing\data\raw\AirBnB_Bali_All.csv'
df = pd.read_csv(csv_path)

# Pick 5 distinct listings from 5 different regencies
sample_indices = [
    df[df['wilayah'] == 'Kota Denpasar'].index[0],
    df[df['wilayah'] == 'Kabupaten Gianyar'].index[0],
    df[df['wilayah'] == 'Kabupaten Bangli'].index[0],
    df[df['wilayah'] == 'Kabupaten Karangasem'].index[0],
    df[df['wilayah'] == 'Kabupaten Badung'].index[0]
]

print(f"Selected {len(sample_indices)} samples across 5 regencies:")

session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    'Accept-Language': 'id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
})

results = []

for idx in sample_indices:
    row = df.iloc[idx]
    regency = str(row['wilayah'])
    nama = str(row['nama'])
    price = str(row['harga'])
    raw_url = str(row['link'])
    
    # Ensure URL targets airbnb.co.id directly to avoid GeoIP redirect form
    url = raw_url.replace('airbnb.com', 'airbnb.co.id')
    
    print(f"\n--- Fetching Sample [Index {idx}] [{regency}] ---")
    print(f"Nama: {nama}")
    print(f"URL: {url}")
    
    time.sleep(2.5) # polite delay
    try:
        resp = session.get(url, timeout=15)
        html = resp.text
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # 1. JSON-LD Schema extraction
        schema_data = {}
        for s in soup.find_all('script', type='application/ld+json'):
            try:
                d = json.loads(s.string)
                if d.get('@type') in ['VacationRental', 'Accommodation', 'Product']:
                    schema_data = d
                    break
            except Exception:
                pass
                
        # 2. Extract Capacity (regex in both id and en)
        cap_patterns = {
            'guests': [r'(\d+)\s*(?:tamu|guest)'],
            'bedrooms': [r'(\d+)\s*(?:kamar\s*tidur|bedroom)', r'(\d+)\s*BR\b'],
            'beds': [r'(\d+)\s*(?:tempat\s*tidur|bed\b)'],
            'baths': [r'(\d+(?:\.\d+)?)\s*(?:kamar\s*mandi|bath)']
        }
        
        extracted_cap = {}
        for key, pats in cap_patterns.items():
            val = None
            for p in pats:
                m = re.findall(p, html, re.IGNORECASE)
                if m:
                    res = m[0] if isinstance(m[0], str) else m[0][0]
                    val = res
                    break
            extracted_cap[key] = val
            
        # 3. Badges & Reputasi
        is_superhost = 1 if re.search(r'\b(?:hosteladan|superhost)\b', html, re.IGNORECASE) else 0
        is_guest_fav = 1 if re.search(r'\b(?:pilihan\s*tamu|guest\s*favorite)\b', html, re.IGNORECASE) else 0
        
        # 4. Amenities Check
        amenities_map = {
            'has_private_pool': [r'kolam\s*renang\s*pribadi', r'private\s*pool'],
            'has_pool': [r'kolam\s*renang', r'pool'],
            'has_kitchen': [r'\bdapur\b', r'\bkitchen\b'],
            'has_ac': [r'penyejuk\s*ruangan', r'air\s*conditioning', r'\bac\b'],
            'has_wifi': [r'wifi', r'wi-fi'],
            'has_parking': [r'parkir\s*gratis', r'free\s*parking', r'car\s*port'],
            'has_workspace': [r'area\s*kerja\s*khusus', r'dedicated\s*workspace'],
            'has_view': [r'pemandangan', r'view', r'beachfront', r'ocean', r'rice\s*field']
        }
        
        detected_amenities = {}
        for am_key, am_pats in amenities_map.items():
            found = False
            for pat in am_pats:
                if re.search(pat, html, re.IGNORECASE):
                    found = True
                    break
            detected_amenities[am_key] = 1 if found else 0
            
        # 5. Geolocation from Schema
        lat = schema_data.get('latitude')
        lon = schema_data.get('longitude')
        sub_dist = schema_data.get('address', {}).get('addressLocality')
        
        item_summary = {
            'index': int(idx),
            'regency': regency,
            'name': nama,
            'price': price,
            'sub_district': sub_dist,
            'lat': float(lat) if lat is not None else None,
            'lon': float(lon) if lon is not None else None,
            'capacity': extracted_cap,
            'badges': {'superhost': int(is_superhost), 'guest_fav': int(is_guest_fav)},
            'amenities': detected_amenities,
            'html_length': len(html),
            'has_schema': bool(schema_data)
        }
        results.append(item_summary)
        print("Extracted Summary:")
        print(json.dumps(item_summary, indent=2))
        
    except Exception as e:
        print(f"Error fetching {url}: {e}")

print("\n\n=== FINAL COMPARATIVE CONSISTENCY MATRIX ===")
summary_rows = []
for r in results:
    row_dict = {
        'Index': r['index'],
        'Regency': r['regency'],
        'Locality': r['sub_district'],
        'Coords': f"{r['lat']}, {r['lon']}" if r['lat'] else 'N/A',
        'Guests': r['capacity']['guests'],
        'Bedrooms': r['capacity']['bedrooms'],
        'Beds': r['capacity']['beds'],
        'Baths': r['capacity']['baths'],
        'PrivatePool': r['amenities']['has_private_pool'],
        'Kitchen': r['amenities']['has_kitchen'],
        'AC': r['amenities']['has_ac'],
        'Wifi': r['amenities']['has_wifi'],
        'Parking': r['amenities']['has_parking'],
        'Superhost': r['badges']['superhost']
    }
    summary_rows.append(row_dict)

comp_df = pd.DataFrame(summary_rows)
print(comp_df.to_string())

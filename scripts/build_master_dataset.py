import os
import re
import json
import glob
import pandas as pd
import numpy as np

RAW_CSV = r"C:\laragon\www\smart-pricing\data\raw\AirBnB_Bali_All.csv"
SCRAPED_DIR = r"C:\laragon\www\smart-pricing\data\scraped"
OUTPUT_CSV = r"C:\laragon\www\smart-pricing\data\processed\bali_airbnb_enriched_master.csv"

def clean_price(val):
    if pd.isna(val):
        return None
    val_str = str(val).strip()
    digits = re.sub(r'[^\d]', '', val_str)
    return int(digits) if digits else None

def clean_rating(val):
    if pd.isna(val):
        return np.nan, 1
    val_str = str(val).strip()
    if val_str.lower() in ['new', 'baru', 'none', '']:
        return np.nan, 1
    try:
        score = float(val_str)
        return score, 0
    except ValueError:
        return np.nan, 1

def infer_property_type(name):
    n = str(name).lower()
    if 'villa' in n:
        return 'Villa'
    elif any(k in n for k in ['house', 'home', 'rumah']):
        return 'Entire House'
    elif any(k in n for k in ['apartment', 'apartemen', 'condo', 'loft']):
        return 'Apartment'
    elif any(k in n for k in ['hut', 'cottage', 'bungalow', 'cabin']):
        return 'Cottage/Bungalow'
    elif any(k in n for k in ['room', 'kamar', 'suite', 'guest house', 'guesthouse', 'homestay']):
        return 'Private Room/Homestay'
    return 'Other'

def main():
    print("=== BUILDING BALI SMART PRICING MASTER DATASET ===")
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    
    df_raw = pd.read_csv(RAW_CSV)
    print(f"Loaded raw dataset: {len(df_raw)} rows")
    
    enriched_rows = []
    
    for idx, row in df_raw.iterrows():
        url = str(row['link'])
        m = re.search(r'/rooms/(\d+)', url)
        room_id = m.group(1) if m else None
        
        json_data = {}
        if room_id:
            json_file = os.path.join(SCRAPED_DIR, f"{room_id}.json")
            if os.path.exists(json_file):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                except Exception:
                    pass
                    
        # Extract features
        price_num = clean_price(row['harga'])
        rating_score, is_new = clean_rating(row['rating'])
        
        cap = json_data.get('capacity', {})
        badges = json_data.get('badges', {})
        amenities = json_data.get('amenities', {})
        
        guests = cap.get('guests')
        bedrooms = cap.get('bedrooms')
        beds = cap.get('beds')
        baths = cap.get('baths')
        
        # Regex fallback on title if scraped capacity is missing
        title = str(row['nama'])
        if not bedrooms:
            m_br = re.search(r'(\d+)\s*(?:br|bedroom|kamar)', title, re.IGNORECASE)
            bedrooms = int(m_br.group(1)) if m_br else 1
        if not guests:
            guests = int(bedrooms) * 2 if bedrooms else 2
        if not beds:
            beds = bedrooms
        if not baths:
            baths = 1.0

        prop_type = infer_property_type(title)
        
        record = {
            'room_id': room_id,
            'name': title,
            'price_idr': price_num,
            'rating': rating_score,
            'is_new_listing': is_new,
            'review_count': int(row['jumlah_review']) if not pd.isna(row['jumlah_review']) else 0,
            'regency': str(row['wilayah']).replace('Kabupaten ', '').replace('Kota ', '').strip(),
            'sub_district': json_data.get('sub_district'),
            'property_type': prop_type,
            'latitude': json_data.get('latitude'),
            'longitude': json_data.get('longitude'),
            'accommodates': int(guests) if guests is not None else 2,
            'bedrooms': int(bedrooms) if bedrooms is not None else 1,
            'beds': int(beds) if beds is not None else 1,
            'bathrooms': float(baths) if baths is not None else 1.0,
            'is_superhost': badges.get('superhost', 0),
            'is_guest_favorite': badges.get('guest_favorite', 0),
            'has_self_checkin': badges.get('self_checkin', 0),
            'has_private_pool': amenities.get('has_private_pool', 0),
            'has_pool': amenities.get('has_pool', 0),
            'has_kitchen': amenities.get('has_kitchen', 0),
            'has_ac': amenities.get('has_ac', 0),
            'has_wifi': amenities.get('has_wifi', 1),
            'has_parking': amenities.get('has_parking', 0),
            'has_workspace': amenities.get('has_workspace', 0),
            'has_bathtub': amenities.get('has_bathtub', 0),
            'has_washer': amenities.get('has_washer', 0),
            'has_view': amenities.get('has_view', 0),
            'url': url
        }
        enriched_rows.append(record)
        
    df_enriched = pd.DataFrame(enriched_rows)
    
    # Impute missing ratings by regency median
    regency_medians = df_enriched.groupby('regency')['rating'].transform('median')
    global_median = df_enriched['rating'].median()
    df_enriched['rating'] = df_enriched['rating'].fillna(regency_medians).fillna(global_median)
    
    df_enriched.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[OK] Enriched master dataset successfully saved to:")
    print(f"     -> {OUTPUT_CSV}")
    print(f"Total Rows: {len(df_enriched)}")
    print(f"Total Columns: {len(df_enriched.columns)}")
    print(f"Columns: {list(df_enriched.columns)}")

if __name__ == "__main__":
    main()

import pandas as pd
import time
import os
from serpapi import GoogleSearch

# ── CONFIG ────────────────────────────────────────────────────────────────────
SERPAPI_KEY = 'c68fcf09becc7d8f17880a65e173ce9e9b36f0b51d8c13a0041b611e6e469b62'
IN_MISSING  = 'missing_to_scrape.csv'
IN_MERGED   = 'master_990_with_reviews.csv'
OUT_MERGED  = 'master_990_with_reviews.csv'

MIN_REVIEW_WORDS = 8
DELAY_SECONDS    = 2

# ── 10 New restaurants ────────────────────────────────────────────────────────
NEW_RESTAURANTS = [
    {'name': 'DubuYo Mini @ Mayang Mall', 'municipality': 'Kuala Terengganu', 'categories': 'Korean Restaurant', 'cuisine_type': 'Korean', 'latitude': 5.3296, 'longitude': 103.137, 'address': 'Mayang Mall, Lot LG-06', 'rating': 3.7},
    {'name': 'Padi House, Mayang Mall', 'municipality': 'Kuala Terengganu', 'categories': 'Restaurant', 'cuisine_type': 'Asian', 'latitude': 5.3296, 'longitude': 103.137, 'address': 'Lot GF-12, Mayang Mall', 'rating': 4.4},
    {'name': 'Penang Chendul Mayang Mall', 'municipality': 'Kuala Terengganu', 'categories': 'Restaurant', 'cuisine_type': 'Malay', 'latitude': 5.3296, 'longitude': 103.137, 'address': 'Lot L3-35, Mayang Mall', 'rating': 4.6},
    {'name': "Me'nate Steak Hub (Kuala Terengganu)", 'municipality': 'Kuala Terengganu', 'categories': 'Steak House', 'cuisine_type': 'Western', 'latitude': 5.3296, 'longitude': 103.137, 'address': 'GF-21, Mayang Mall', 'rating': 4.8},
    {'name': 'MOMOYO @ Mayang Mall', 'municipality': 'Kuala Terengganu', 'categories': 'Cafe', 'cuisine_type': 'Cafe', 'latitude': 5.3296, 'longitude': 103.137, 'address': 'LG-12, Mayang Mall', 'rating': 4.8},
    {'name': 'Shabuyaki by Nippon Sushi @ Mayang Mall', 'municipality': 'Kuala Terengganu', 'categories': 'Japanese Restaurant', 'cuisine_type': 'Japanese', 'latitude': 5.3296, 'longitude': 103.137, 'address': 'LG-23A, Mayang Mall', 'rating': 4.5},
    {'name': 'RICHEESE FACTORY Mayang Mall', 'municipality': 'Kuala Terengganu', 'categories': 'Fast Food Restaurant', 'cuisine_type': 'Fast Food', 'latitude': 5.3296, 'longitude': 103.137, 'address': 'L3-37, Mayang Mall', 'rating': 4.1},
    {'name': 'Ayam Gepuk Ibuk Bapak KT', 'municipality': 'Kuala Terengganu', 'categories': 'Restaurant', 'cuisine_type': 'Indonesian', 'latitude': 5.3296, 'longitude': 103.137, 'address': '219, Jln Sultan Zainal Abidin', 'rating': 2.4},
    {'name': 'emart24 Kuala Ibai', 'municipality': 'Kuala Terengganu', 'categories': 'Convenience Store', 'cuisine_type': 'Fast Food', 'latitude': 5.3296, 'longitude': 103.137, 'address': 'Taman Ibai Landmark', 'rating': 4.8},
    {'name': 'Big Apple Donuts @ Air Jernih', 'municipality': 'Kuala Terengganu', 'categories': 'Dessert Shop', 'cuisine_type': 'Dessert', 'latitude': 5.3296, 'longitude': 103.137, 'address': '1A, Jalan Air Jernih', 'rating': 3.8},
]

def get_place_details(name, municipality):
    queries = [f"{name} {municipality}", f"{name} Terengganu"]
    for q in queries:
        params = {"engine": "google_maps", "q": q, "api_key": SERPAPI_KEY}
        try:
            results = GoogleSearch(params).get_dict()
            # Check local_results (list)
            if "local_results" in results and results["local_results"]:
                place = results["local_results"][0]
                return place.get("data_id"), place.get("title")
            # Check place_results (direct match)
            if "place_results" in results:
                place = results["place_results"]
                return place.get("data_id"), place.get("title")
        except Exception as e:
            print(f"   ! Error: {e}")
        time.sleep(DELAY_SECONDS)
    return None, None

def scrape_reviews(data_id):
    if not data_id: return ""
    params = {"engine": "google_maps_reviews", "data_id": data_id, "api_key": SERPAPI_KEY}
    try:
        results = GoogleSearch(params).get_dict()
        reviews = results.get("reviews", [])
        texts = [r.get("snippet", "").replace('\n', ' ') for r in reviews if len(r.get("snippet", "").split()) >= MIN_REVIEW_WORDS]
        return " | ".join(texts[:10])
    except: return ""

def run_pipeline():
    # 1. Load data
    df_master = pd.read_csv(IN_MERGED) if os.path.exists(IN_MERGED) else pd.DataFrame()
    
    # Mode A: Load missing items from CSV
    missing_list = []
    if os.path.exists(IN_MISSING):
        print(f"Loading missing restaurants from {IN_MISSING}...")
        missing_df = pd.read_csv(IN_MISSING)
        missing_list = missing_df.to_dict('records')
    
    # Combine Mode A (Missing) and Mode C (New)
    all_to_process = missing_list + NEW_RESTAURANTS
    processed_data = []

    print(f"Total to process: {len(all_to_process)} restaurants...")
    for res in all_to_process:
        name = res.get('name')
        muni = res.get('municipality', 'Terengganu')
        print(f"-> Processing: {name}")
        
        data_id, official_name = get_place_details(name, muni)
        if data_id:
            print(f"   ✓ Found. Scraping reviews...")
            res['reviews'] = scrape_reviews(data_id)
            res['data_id'] = data_id
        else:
            print(f"   X Not found. Skipping reviews.")
            res['reviews'] = res.get('reviews', "") # keep existing if any
        
        processed_data.append(res)
        time.sleep(DELAY_SECONDS)

    # Merge everything and save
    df_new = pd.DataFrame(processed_data)
    df_final = pd.concat([df_master, df_new], ignore_index=True).drop_duplicates(subset=['name'], keep='last')
    
    df_final.to_csv(OUT_MERGED, index=False)
    print(f"\nDONE! Saved to {OUT_MERGED}. Total rows: {len(df_final)}")

if __name__ == "__main__":
    run_pipeline()

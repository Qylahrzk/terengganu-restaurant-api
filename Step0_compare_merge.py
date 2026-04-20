"""
STEP 0 — Compare & Merge (Fixed with Fuzzy Matching)
=====================================================
Matches old JSON (with reviews) to new Supabase JSON (980 restaurants).

Matching strategy (in order):
  1. Exact name match (case-insensitive)
  2. Fuzzy name match (handles capitalisation, extra words like "Terengganu")
  3. Coordinate match (lat/lon)
  4. Address first-part match

Install: pip install rapidfuzz pandas

Input : terengganu_restaurants.json  (old file with reviews)
        restaurants.json             (new Supabase export)
Output: master_980_with_reviews.csv
        missing_to_scrape.csv

Run: python step0_compare_merge.py
"""

import json
import pandas as pd
from rapidfuzz import fuzz, process

# ── Input files ───────────────────────────────────────────────────────────────
OLD_JSON    = 'terengganu_restaurants.json'
NEW_JSON    = 'restaurants.json'
OUT_MERGED  = 'master_980_with_reviews.csv'
OUT_MISSING = 'missing_to_scrape.csv'

FUZZY_THRESHOLD = 75   # lower = more lenient matching

# ─────────────────────────────────────────────────────────────────────────────

def load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)

def round_coord(val, decimals=4):
    try:
        return round(float(val), decimals)
    except (TypeError, ValueError):
        return None

def extract_address_key(address):
    if not address or str(address) in ['nan', '', 'NaN']:
        return None
    parts = str(address).strip().split(',')
    return parts[0].strip().lower() if parts else None

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading files...")
old_data = load_json(OLD_JSON)
new_data = load_json(NEW_JSON)
print(f"  Old file: {len(old_data)} rows")
print(f"  New file: {len(new_data)} restaurants")

# ── Build lookup tables ───────────────────────────────────────────────────────
print("\nBuilding lookup tables...")

old_by_name = {}
for r in old_data:
    name = r['Name'].strip()
    if name not in old_by_name:
        old_by_name[name] = []
    old_by_name[name].append(r)

old_by_name_lower  = {k.lower(): k for k in old_by_name.keys()}
old_name_list      = list(old_by_name.keys())
old_name_list_lower = [n.lower() for n in old_name_list]

old_by_latlon = {}
for r in old_data:
    lat = round_coord(r.get('Latitude'))
    lon = round_coord(r.get('Longitude'))
    if lat and lon and lat != 5.3296 and lon != 103.137:
        key = (lat, lon)
        if key not in old_by_latlon:
            old_by_latlon[key] = r['Name'].strip()

old_by_address = {}
for r in old_data:
    addr_key = extract_address_key(r.get('Address', ''))
    if addr_key and len(addr_key) > 3:
        if addr_key not in old_by_address:
            old_by_address[addr_key] = r['Name'].strip()

print(f"  Unique old names   : {len(old_by_name)}")
print(f"  Lat/lon entries    : {len(old_by_latlon)}")
print(f"  Address entries    : {len(old_by_address)}")

# ── Match function ────────────────────────────────────────────────────────────

def find_old_name(new_name, lat, lon, address):
    new_lower = new_name.lower().strip()

    # 1. Exact case-insensitive
    if new_lower in old_by_name_lower:
        return old_by_name_lower[new_lower], 'exact'

    # 2. Fuzzy name (token_set_ratio handles word order + extra words)
    match = process.extractOne(
        new_lower, old_name_list_lower,
        scorer=fuzz.token_set_ratio,
    )
    if match and match[1] >= FUZZY_THRESHOLD:
        old_name = old_name_list[match[2]]
        return old_name, f'fuzzy({match[1]:.0f})'

    # 3. Coordinates
    if lat and lon and lat != 5.3296 and lon != 103.137:
        key = (round_coord(lat), round_coord(lon))
        if key in old_by_latlon:
            return old_by_latlon[key], 'coords'

    # 4. Address first part
    addr_key = extract_address_key(address)
    if addr_key and len(addr_key) > 5 and addr_key in old_by_address:
        return old_by_address[addr_key], 'address'

    return None, None

# ── Match all ─────────────────────────────────────────────────────────────────
print("\nMatching restaurants...")

merged_rows  = []
missing_list = []
counts       = {'exact': 0, 'fuzzy': 0, 'coords': 0, 'address': 0, 'none': 0}

for new_r in new_data:
    new_name = new_r['name'].strip()
    lat      = new_r.get('latitude')
    lon      = new_r.get('longitude')
    address  = new_r.get('address', '')

    old_name, match_type = find_old_name(new_name, lat, lon, address)

    if old_name and old_name in old_by_name:
        reviews      = old_by_name[old_name]
        all_reviews  = ' '.join(
            str(r.get('Review_Text', '')) for r in reviews
            if r.get('Review_Text') and str(r.get('Review_Text')) != 'nan'
        )
        cleaned_text = ' '.join(
            str(r.get('Cleaned_Text', '')) for r in reviews
            if r.get('Cleaned_Text') and str(r.get('Cleaned_Text')) != 'nan'
        )

        merged_rows.append({
            'name'             : new_name,
            'municipality'     : new_r.get('municipality', ''),
            'categories'       : new_r.get('categories', ''),
            'cuisine_type'     : new_r.get('cuisine_type', ''),
            'latitude'         : lat,
            'longitude'        : lon,
            'address'          : address,
            'rating'           : new_r.get('rating', 0),
            'coordinate_source': new_r.get('coordinate_source', ''),
            'review_text'      : all_reviews[:5000],
            'cleaned_text'     : cleaned_text[:5000],
            'has_reviews'      : True,
            'match_type'       : match_type,
            'old_name_matched' : old_name,
        })

        mtype = 'fuzzy' if 'fuzzy' in match_type else match_type
        counts[mtype] = counts.get(mtype, 0) + 1

    else:
        counts['none'] += 1
        merged_rows.append({
            'name'             : new_name,
            'municipality'     : new_r.get('municipality', ''),
            'categories'       : new_r.get('categories', ''),
            'cuisine_type'     : new_r.get('cuisine_type', ''),
            'latitude'         : lat,
            'longitude'        : lon,
            'address'          : address,
            'rating'           : new_r.get('rating', 0),
            'coordinate_source': new_r.get('coordinate_source', ''),
            'review_text'      : '',
            'cleaned_text'     : '',
            'has_reviews'      : False,
            'match_type'       : 'none',
            'old_name_matched' : '',
        })
        missing_list.append({
            'name'        : new_name,
            'municipality': new_r.get('municipality', ''),
            'address'     : address,
            'latitude'    : lat,
            'longitude'   : lon,
            'categories'  : new_r.get('categories', ''),
        })

# ── Save ──────────────────────────────────────────────────────────────────────
pd.DataFrame(merged_rows).to_csv(OUT_MERGED,  index=False, encoding='utf-8-sig')
pd.DataFrame(missing_list).to_csv(OUT_MISSING, index=False, encoding='utf-8-sig')

total_matched = len(new_data) - counts['none']

print(f"\n{'='*55}")
print(f"  STEP 0 COMPLETE")
print(f"{'='*55}")
print(f"  Total restaurants    : {len(new_data)}")
print(f"  Matched (exact)      : {counts['exact']}")
print(f"  Matched (fuzzy)      : {counts.get('fuzzy', 0)}")
print(f"  Matched (coords)     : {counts['coords']}")
print(f"  Matched (address)    : {counts['address']}")
print(f"  ─────────────────────────────────────────")
print(f"  Total with reviews   : {total_matched}")
print(f"  Still missing        : {counts['none']}")
print(f"\n  Output: {OUT_MERGED}")
print(f"  Output: {OUT_MISSING}")
print(f"{'='*55}")

if counts['none'] > 0:
    print(f"\nRestaurants still missing (need SerpApi):")
    for r in missing_list:
        print(f"  - {r['name']} ({r['municipality']})")

print(f"\nNext: Run step1_scrape_reviews.py to scrape {counts['none']} restaurants")
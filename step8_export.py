import pandas as pd
import numpy as np
import json
import os
import time
import urllib.request
import urllib.parse
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
INPUT_KBF  = 'kbf_outputs/kbf_restaurant_profiles.csv'
OUTPUT_DIR = 'export_outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Enable/disable Nominatim geocoding
# Set to True  → geocodes addresses (takes ~5 min, more accurate)
# Set to False → skips geocoding, uses district centroids only (fast)
ENABLE_GEOCODING = True

# Nominatim rate limit — must wait 1 second between requests (free API rule)
GEOCODING_DELAY  = 1.1


# ============================================================
# DISTRICT CENTROID COORDINATES
# Approximate centre GPS of each Terengganu district
# Used as fallback when no GPS or address available
# ============================================================
DISTRICT_CENTROIDS = {
    'Kuala Terengganu' : (5.3296,  103.1370),
    'Besut'            : (5.7964,  102.5615),
    'Kemaman'          : (4.2330,  103.4193),
    'Dungun'           : (4.7590,  103.4243),
    'Hulu Terengganu'  : (5.0328,  102.8985),
    'Marang'           : (5.2033,  103.2148),
    'Setiu'            : (5.6897,  102.7112),
}


# ============================================================
# MUNICIPALITY CLEANING FUNCTION
# Maps messy/raw municipality values → clean 7 district names
# ============================================================
def clean_municipality(muni):
    if not isinstance(muni, str):
        return 'Kuala Terengganu'

    muni_lower = muni.strip().lower()

    # Kuala Terengganu
    if any(k in muni_lower for k in [
        'kuala terengganu', 'gong badak', 'batu buruk', 'losong',
        'tanjung', 'cina', 'taman raya', 'kg.tok', 'jln tok',
        'bandar baru', 'gong pauh', 'banggol', 'kampung buluh',
        'kampung batu enam', 'kampung batu 24', 'kampung pak sabah',
        'batu enam', 'batu 24', 'pak sabah', 'tok jembal',
        'gong pasir', 'telaga', 'bukit tumbuh', 'wakaf',
        'pantai batu', 'persiaran coast',
    ]):
        return 'Kuala Terengganu'

    # Besut
    if any(k in muni_lower for k in [
        'besut', 'jerteh', 'permaisuri', 'kg raja', 'kampung raja',
        'bukit keluang', 'penarik', 'perhentian', 'pulau',
        'kampung air tawar', 'pasir putih',
    ]):
        return 'Besut'

    # Kemaman
    if any(k in muni_lower for k in [
        'kemaman', 'chukai', 'kijal', 'kerteh', 'geliga',
        'jakar', 'cukai', 'cherating', 'balok', 'kemasek',
    ]):
        return 'Kemaman'

    # Dungun
    if any(k in muni_lower for k in [
        'dungun', 'paka', 'sura', 'teluk lipat',
        'batu 48', 'batu 49', 'batu, 49',
    ]):
        return 'Dungun'

    # Hulu Terengganu
    if any(k in muni_lower for k in [
        'hulu terengganu', 'kuala berang', 'ajil',
    ]):
        return 'Hulu Terengganu'

    # Marang
    if any(k in muni_lower for k in [
        'marang', 'merang', 'rusila',
    ]):
        return 'Marang'

    # Setiu
    if any(k in muni_lower for k in [
        'setiu', 'chalok',
    ]):
        return 'Setiu'

    # Default fallback — Kuala Terengganu (most common district)
    return 'Kuala Terengganu'


# ============================================================
# NOMINATIM GEOCODING FUNCTION
# Converts address text → (latitude, longitude)
# Free, no API key, uses OpenStreetMap data
# ============================================================
def geocode_address(address):
    if not isinstance(address, str) or address.strip() == '':
        return None
    try:
        encoded = urllib.parse.quote(address)
        url     = (f'https://nominatim.openstreetmap.org/search'
                   f'?q={encoded}&format=json&limit=1&countrycodes=my')
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'FYP-TerengganuRestaurantApp/1.0 (academic research)'}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read())
            if data:
                return float(data[0]['lat']), float(data[0]['lon'])
    except Exception:
        pass
    return None


# ============================================================
# STEP 1: Load Data
# ============================================================
print("=" * 60)
print("  STEP 8 — EXPORT JSON FOR FLUTTER & SUPABASE")
print("=" * 60)

df = pd.read_csv(INPUT_KBF)
print(f"\n✅ Loaded: {len(df):,} restaurant profiles")

# Clean municipality names
df['Municipality'] = df['Municipality'].apply(clean_municipality)

print(f"\n   District distribution after cleaning:")
for district, count in df['Municipality'].value_counts().items():
    print(f"   {district:<25} : {count:,}")

# Coordinate analysis
has_coords  = df['Latitude'].notna() & df['Longitude'].notna()
has_address = df['Address'].notna() & (df['Address'].astype(str).str.strip() != '')
print(f"\n   Coordinate availability:")
print(f"   Already have coordinates      : {has_coords.sum():,}")
print(f"   Missing + have address        : {(~has_coords & has_address).sum():,}")
print(f"   Missing + no address          : {(~has_coords & ~has_address).sum():,}")


# ============================================================
# STEP 2: Resolve Coordinates for All Restaurants
# ============================================================
print(f"\n⏳ Resolving coordinates...")
if ENABLE_GEOCODING:
    missing_with_addr = (~has_coords & has_address).sum()
    if missing_with_addr > 0:
        print(f"   Geocoding {missing_with_addr} addresses")
        print(f"   Estimated time: ~{int(missing_with_addr * GEOCODING_DELAY)} seconds\n")

latitudes     = []
longitudes    = []
coord_sources = []
original_count = geocoded_count = centroid_count = failed_count = 0

for idx, row in df.iterrows():
    lat  = row['Latitude']
    lon  = row['Longitude']
    muni = str(row['Municipality'])

    # TIER 1 — Already has GPS coordinates (exact)
    if pd.notna(lat) and pd.notna(lon):
        latitudes.append(float(lat))
        longitudes.append(float(lon))
        coord_sources.append('original')
        original_count += 1

    # TIER 2 — No GPS but has address → geocode via Nominatim
    elif ENABLE_GEOCODING and pd.notna(row.get('Address')) and str(row['Address']).strip() != '':
        result = geocode_address(str(row['Address']))
        time.sleep(GEOCODING_DELAY)  # respect rate limit

        if result:
            latitudes.append(result[0])
            longitudes.append(result[1])
            coord_sources.append('geocoded')
            geocoded_count += 1
            print(f"   ✅ Geocoded: {str(row['Name'])[:45]}")
        else:
            # Geocoding failed → fall back to district centroid
            centroid = DISTRICT_CENTROIDS.get(muni, (5.3296, 103.1370))
            latitudes.append(centroid[0])
            longitudes.append(centroid[1])
            coord_sources.append('district_centroid')
            centroid_count += 1
            failed_count += 1

    # TIER 3 — No GPS, no address → district centroid (approximate)
    else:
        centroid = DISTRICT_CENTROIDS.get(muni, (5.3296, 103.1370))
        latitudes.append(centroid[0])
        longitudes.append(centroid[1])
        coord_sources.append('district_centroid')
        centroid_count += 1

df['Latitude']          = latitudes
df['Longitude']         = longitudes
df['Coordinate_Source'] = coord_sources

print(f"\n✅ Coordinate resolution complete:")
print(f"   Original GPS       : {original_count:,}  (exact)")
print(f"   Geocoded           : {geocoded_count:,}  (good estimate)")
print(f"   District centroid  : {centroid_count:,}  (approximate)")
if failed_count > 0:
    print(f"   Geocoding failed   : {failed_count:,} (fell back to centroid)")


# ============================================================
# STEP 3: Clean & Prepare Final Data
# ============================================================
print(f"\n⏳ Cleaning data...")

# Convert Yes/No → proper Python booleans
bool_cols = [
    'Is_Halal', 'Is_Vegetarian', 'Is_Vegan',
    'Has_Parking', 'Is_Family_Friendly', 'Is_Romantic',
    'Has_Scenic_View', 'Has_Outdoor', 'Has_Wifi'
]
for col in bool_cols:
    df[col] = df[col].apply(
        lambda x: True if str(x).strip().lower() == 'yes' else False
    )

# Fill missing values
df['Topic_1_Pct']    = df['Topic_1_Pct'].fillna(0.0).round(2)
df['Topic_2_Pct']    = df['Topic_2_Pct'].fillna(0.0).round(2)
df['Topic_3_Pct']    = df['Topic_3_Pct'].fillna(0.0).round(2)
df['Dominant_Topic'] = df['Dominant_Topic'].fillna(0).astype(int)
df['Topic_Label']    = df['Topic_Label'].fillna('No Reviews')
df['Rating']         = df['Rating'].fillna(df['Rating'].median()).round(1)
df['Address']        = df['Address'].fillna('')
df['Cuisine_Type']   = df['Cuisine_Type'].fillna('Other')
df['Categories']     = df['Categories'].fillna('')
df['Rating_Band']    = df['Rating_Band'].fillna('')

print(f"✅ Data cleaned")


# ============================================================
# STEP 4: Build JSON Structure
# ============================================================
print(f"\n⏳ Building JSON...")

restaurants_json = []

for idx, row in df.iterrows():
    restaurant = {
        # Identity
        "id"                 : idx + 1,
        "name"               : str(row['Name']) if pd.notna(row['Name']) else '',
        "address"            : str(row['Address']),
        "municipality"       : str(row['Municipality']),

        # Cuisine & Category
        "categories"         : str(row['Categories']),
        "cuisine_type"       : str(row['Cuisine_Type']),

        # Rating
        "rating"             : float(row['Rating']),
        "rating_band"        : str(row['Rating_Band']),

        # Location
        # coordinate_source:
        #   "original"          → exact GPS (most accurate, use precise distance)
        #   "geocoded"          → from address (good, show "~X km away")
        #   "district_centroid" → approximate (show "In X district area")
        "latitude"           : float(row['Latitude']),
        "longitude"          : float(row['Longitude']),
        "coordinate_source"  : str(row['Coordinate_Source']),

        # KBF Filter Attributes (for Flutter filtering)
        "is_halal"           : bool(row['Is_Halal']),
        "is_vegetarian"      : bool(row['Is_Vegetarian']),
        "is_vegan"           : bool(row['Is_Vegan']),
        "has_parking"        : bool(row['Has_Parking']),
        "is_family_friendly" : bool(row['Is_Family_Friendly']),
        "is_romantic"        : bool(row['Is_Romantic']),
        "has_scenic_view"    : bool(row['Has_Scenic_View']),
        "has_outdoor"        : bool(row['Has_Outdoor']),
        "has_wifi"           : bool(row['Has_Wifi']),

        # LDA Topic Attributes (for Flutter scoring)
        "dominant_topic"     : int(row['Dominant_Topic']),
        "topic_label"        : str(row['Topic_Label']),
        "topic_1_pct"        : float(row['Topic_1_Pct']),
        "topic_2_pct"        : float(row['Topic_2_Pct']),
        "topic_3_pct"        : float(row['Topic_3_Pct']),
    }
    restaurants_json.append(restaurant)

final_json = {
    "metadata": {
        "total_restaurants"  : len(restaurants_json),
        "generated_at"       : pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        "version"            : "1.0",
        "districts"          : sorted(df['Municipality'].unique().tolist()),
        "cuisine_types"      : sorted(df['Cuisine_Type'].unique().tolist()),
        "topic_labels"       : sorted(df['Topic_Label'].unique().tolist()),
        "coordinate_sources" : {
            "original"          : original_count,
            "geocoded"          : geocoded_count,
            "district_centroid" : centroid_count,
        }
    },
    "restaurants": restaurants_json
}

print(f"✅ JSON built: {len(restaurants_json):,} restaurants")


# ============================================================
# STEP 5: Save Output Files
# ============================================================

# 1. Flutter assets JSON
flutter_path = f'{OUTPUT_DIR}/restaurants.json'
with open(flutter_path, 'w', encoding='utf-8') as f:
    json.dump(final_json, f, ensure_ascii=False, indent=2)
print(f"\n✅ Saved: restaurants.json  ({os.path.getsize(flutter_path)/1024:.1f} KB)")
print(f"   → Copy to Flutter assets/ folder")

# 2. Supabase JSON (same content, separate file)
supabase_json_path = f'{OUTPUT_DIR}/restaurants_supabase.json'
with open(supabase_json_path, 'w', encoding='utf-8') as f:
    json.dump(final_json, f, ensure_ascii=False, indent=2)
print(f"✅ Saved: restaurants_supabase.json  ({os.path.getsize(supabase_json_path)/1024:.1f} KB)")
print(f"   → Upload via Supabase dashboard if needed")

# 3. Updated CSV with clean districts + resolved coordinates
csv_path = f'{OUTPUT_DIR}/kbf_restaurant_profiles_final.csv'
df.to_csv(csv_path, index=False)
print(f"✅ Saved: kbf_restaurant_profiles_final.csv")
print(f"   → Updated dataset with clean municipality + coordinates")


# ============================================================
# STEP 6: Generate Supabase SQL Insert Script
# ============================================================
print(f"\n⏳ Generating Supabase SQL script...")

def esc(v):
    """Safely escape values for SQL INSERT statements."""
    if v is None:                   return 'NULL'
    if isinstance(v, bool):         return 'TRUE' if v else 'FALSE'
    if isinstance(v, (int, float)): return str(v)
    return "'" + str(v).replace("'", "''") + "'"

sql_lines = []
sql_lines.append("-- ============================================================")
sql_lines.append("-- AUTO-GENERATED SQL")
sql_lines.append("-- HOW TO USE:")
sql_lines.append("-- 1. Go to your Supabase project dashboard")
sql_lines.append("-- 2. Click SQL Editor in the left menu")
sql_lines.append("-- 3. Paste this entire file and click Run")
sql_lines.append(f"-- Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
sql_lines.append(f"-- Total restaurants: {len(restaurants_json)}")
sql_lines.append("-- ============================================================")
sql_lines.append("")
sql_lines.append("-- STEP 1: Remove old tables and views")
sql_lines.append("DROP VIEW IF EXISTS restaurant_knowledge_base;")
sql_lines.append("DROP VIEW IF EXISTS unique_restaurants;")
sql_lines.append("DROP TABLE IF EXISTS restaurant_profiles CASCADE;")
sql_lines.append("DROP TABLE IF EXISTS restaurants CASCADE;")
sql_lines.append("")
sql_lines.append("-- STEP 2: Create new restaurant_profiles table")
sql_lines.append("""CREATE TABLE restaurant_profiles (
  id                   SERIAL PRIMARY KEY,
  name                 TEXT,
  address              TEXT,
  municipality         TEXT,
  categories           TEXT,
  cuisine_type         TEXT,
  rating               FLOAT4,
  rating_band          TEXT,
  latitude             FLOAT8,
  longitude            FLOAT8,
  coordinate_source    TEXT,
  is_halal             BOOLEAN DEFAULT FALSE,
  is_vegetarian        BOOLEAN DEFAULT FALSE,
  is_vegan             BOOLEAN DEFAULT FALSE,
  has_parking          BOOLEAN DEFAULT FALSE,
  is_family_friendly   BOOLEAN DEFAULT FALSE,
  is_romantic          BOOLEAN DEFAULT FALSE,
  has_scenic_view      BOOLEAN DEFAULT FALSE,
  has_outdoor          BOOLEAN DEFAULT FALSE,
  has_wifi             BOOLEAN DEFAULT FALSE,
  dominant_topic       INT,
  topic_label          TEXT,
  topic_1_pct          FLOAT8,
  topic_2_pct          FLOAT8,
  topic_3_pct          FLOAT8,
  created_at           TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
""")
sql_lines.append("-- STEP 3: Insert restaurant data (in batches of 50)")

insert_cols = (
    "(name, address, municipality, categories, cuisine_type, "
    "rating, rating_band, latitude, longitude, coordinate_source, "
    "is_halal, is_vegetarian, is_vegan, has_parking, is_family_friendly, "
    "is_romantic, has_scenic_view, has_outdoor, has_wifi, "
    "dominant_topic, topic_label, topic_1_pct, topic_2_pct, topic_3_pct)"
)

for batch_start in range(0, len(restaurants_json), 50):
    batch  = restaurants_json[batch_start:batch_start + 50]
    values = []
    for r in batch:
        val = (
            f"({esc(r['name'])}, {esc(r['address'])}, {esc(r['municipality'])}, "
            f"{esc(r['categories'])}, {esc(r['cuisine_type'])}, "
            f"{esc(r['rating'])}, {esc(r['rating_band'])}, "
            f"{esc(r['latitude'])}, {esc(r['longitude'])}, {esc(r['coordinate_source'])}, "
            f"{esc(r['is_halal'])}, {esc(r['is_vegetarian'])}, {esc(r['is_vegan'])}, "
            f"{esc(r['has_parking'])}, {esc(r['is_family_friendly'])}, "
            f"{esc(r['is_romantic'])}, {esc(r['has_scenic_view'])}, "
            f"{esc(r['has_outdoor'])}, {esc(r['has_wifi'])}, "
            f"{esc(r['dominant_topic'])}, {esc(r['topic_label'])}, "
            f"{esc(r['topic_1_pct'])}, {esc(r['topic_2_pct'])}, {esc(r['topic_3_pct'])})"
        )
        values.append(val)
    sql_lines.append(
        f"INSERT INTO restaurant_profiles {insert_cols} VALUES\n"
        + ",\n".join(values) + ";"
    )
    sql_lines.append("")

sql_lines.append("-- STEP 4: Grant read permissions to Flutter app")
sql_lines.append("GRANT SELECT ON restaurant_profiles TO anon;")
sql_lines.append("GRANT SELECT ON restaurant_profiles TO authenticated;")
sql_lines.append("")
sql_lines.append("-- STEP 5: Verify — should return 1051")
sql_lines.append("SELECT COUNT(*) FROM restaurant_profiles;")

sql_path = f'{OUTPUT_DIR}/supabase_insert.sql'
with open(sql_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(sql_lines))

print(f"✅ Saved: supabase_insert.sql  ({os.path.getsize(sql_path)/1024:.1f} KB)")
print(f"   → Paste into Supabase SQL Editor and run")


# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("  EXPORT SUMMARY")
print("=" * 60)
print(f"  Total restaurants exported   : {len(restaurants_json):,}")
print(f"\n  Coordinate sources:")
print(f"    Original GPS               : {original_count:,}  (exact distance)")
print(f"    Geocoded from address      : {geocoded_count:,}  (~estimated distance)")
print(f"    District centroid          : {centroid_count:,}  (district area only)")
print(f"\n  Districts  : {final_json['metadata']['districts']}")
print(f"\n  Cuisines   : {final_json['metadata']['cuisine_types']}")
print(f"\n  Topics     : {final_json['metadata']['topic_labels']}")
print(f"\n  KBF filter attribute counts:")
for col, key in [
    ('Is_Halal','is_halal'), ('Is_Vegetarian','is_vegetarian'),
    ('Is_Vegan','is_vegan'), ('Has_Parking','has_parking'),
    ('Is_Family_Friendly','is_family_friendly'),
    ('Is_Romantic','is_romantic'), ('Has_Scenic_View','has_scenic_view'),
    ('Has_Outdoor','has_outdoor'), ('Has_Wifi','has_wifi'),
]:
    count = sum(1 for r in restaurants_json if r[key])
    print(f"    {col:<25} : {count:,} restaurants")
print("=" * 60)
print("\n  Output files → /export_outputs/")
print("  ┌──────────────────────────────────────────────────┐")
print("  │ restaurants.json           → Flutter assets/     │")
print("  │ restaurants_supabase.json  → Supabase upload     │")
print("  │ supabase_insert.sql        → Supabase SQL Editor │")
print("  │ kbf_restaurant_profiles_   → Updated CSV file    │")
print("  │ final.csv                                        │")
print("  └──────────────────────────────────────────────────┘")
print("\n  WHAT TO DO NEXT:")
print("  1. Run supabase_insert.sql in Supabase SQL Editor")
print("  2. Copy restaurants.json → Flutter assets/ folder")
print("  3. Add in pubspec.yaml:")
print("       assets:")
print("         - assets/restaurants.json")
print("  4. Build step8_api.py (Flask API)")
print("=" * 60)
print("\n  ➡  Next: step8_api.py\n")
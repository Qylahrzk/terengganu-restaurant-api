"""
STEP 7 — Export & Update Supabase
===================================
Builds final restaurant profiles from KBF output.
Generates:
  - SQL UPDATE for existing restaurants (update LDA + KBF fields)
  - SQL INSERT for new restaurants not yet in Supabase

Fixes vs previous version:
  1. else: indentation bug FIXED — was FOR-else (only ran once), now IF-else
     (runs per-row) → all unmatched restaurants now get an INSERT
  2. clean_str() strips Python list values like ["Malay"] → Malay
  3. 'nan' coordinate_source handled cleanly
  4. All 6 new KBF columns included
  5. WHERE clause indentation fixed in UPDATE SQL
  6. ALTER TABLE statements auto-included at top of SQL file

Input : master_990_kbf.csv
        restaurants.json  (original Supabase export — preserves id + price_level)
Output: final_990_profiles.json
        supabase_update_990.sql

Run: python s7_export.py
"""

import pandas as pd
import json
import math
import ast

IN_KBF      = 'master_990_kbf.csv'
IN_SUPABASE = 'restaurants.json'
OUT_JSON    = 'final_990_profiles.json'
OUT_SQL     = 'supabase_update_990.sql'


# ── Helpers ───────────────────────────────────────────────────────────────────

def clean_str(v):
    """
    Converts a value to a clean plain string.
    - Strips Python list formatting e.g. ["Malay"] → Malay
    - Handles NaN, None, 'nan', 'none' safely → returns ''
    """
    if v is None:
        return ''
    if isinstance(v, float) and math.isnan(v):
        return ''
    s = str(v).strip()
    if s.lower() in ('nan', 'none', ''):
        return ''
    # Strip Python list format e.g. ["Malay"] or ['Western', 'Fusion']
    if s.startswith('[') and s.endswith(']'):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return str(parsed[0]).strip() if parsed else ''
        except Exception:
            return s.strip("[]\"' ")
    return s


def rating_band(r):
    try:
        r = float(r)
        if r >= 4.5: return 'Excellent (≥4.5 ★★★★★)'
        if r >= 4.0: return 'Very Good (≥4.0 ★★★★)'
        if r >= 3.5: return 'Good (≥3.5 ★★★)'
        if r >= 3.0: return 'Average (≥3.0 ★★)'
        return 'Below Average (<3.0 ★)'
    except (ValueError, TypeError):
        return 'Invalid Input'


def safe_float(v):
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except Exception:
        return None


def sql_str(v):
    """SQL string literal — handles NULL, empty, and list-formatted values."""
    if v is None:
        return 'NULL'
    s = clean_str(v)
    if s == '':
        return "''"
    return "'" + s.replace("'", "''") + "'"


def sql_bool(v):
    return 'TRUE' if v else 'FALSE'


def sql_num(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return 'NULL'
    return str(v)


# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading files...")
df = pd.read_csv(IN_KBF)
with open(IN_SUPABASE, encoding='utf-8') as f:
    supabase_data = json.load(f)

supabase_by_name = {r['name']: r for r in supabase_data}
print(f"  KBF rows      : {len(df)}")
print(f"  Supabase rows : {len(supabase_data)}")


# ── Build profiles + SQL ──────────────────────────────────────────────────────
print("Building final profiles...")
profiles    = []
sql_updates = []
sql_inserts = []

for _, row in df.iterrows():
    name      = clean_str(row.get('name', ''))
    sb        = supabase_by_name.get(name, {})
    record_id = sb.get('id')
    price_lvl = sb.get('price_level')

    p = {
        'id'                : record_id,
        'name'              : name,
        'address'           : clean_str(row.get('address', '')),
        'municipality'      : clean_str(row.get('municipality', '')),
        'categories'        : clean_str(row.get('categories', '')),
        'cuisine_type'      : clean_str(row.get('cuisine_type', '')),
        'rating'            : float(row.get('rating', 0) or 0),
        'rating_band'       : rating_band(row.get('rating', 0)),
        'latitude'          : safe_float(row.get('latitude')),
        'longitude'         : safe_float(row.get('longitude')),
        'coordinate_source' : clean_str(row.get('coordinate_source', '')),
        'price_level'       : price_lvl,
        # Dietary
        'is_halal'          : bool(row.get('is_halal', False)),
        'is_vegetarian'     : bool(row.get('is_vegetarian', False)),
        'is_vegan'          : bool(row.get('is_vegan', False)),
        # Facilities & Accessibility
        'has_parking'       : bool(row.get('has_parking', False)),
        'is_accessible'     : bool(row.get('is_accessible', False)),
        'has_ac'            : bool(row.get('has_ac', False)),
        'has_wifi'          : bool(row.get('has_wifi', False)),
        'has_outdoor'       : bool(row.get('has_outdoor', False)),
        # Vibes & Occasions
        'is_family_friendly': bool(row.get('is_family_friendly', False)),
        'is_group_friendly' : bool(row.get('is_group_friendly', False)),
        'is_casual'         : bool(row.get('is_casual', False)),
        'is_romantic'       : bool(row.get('is_romantic', False)),
        'has_scenic_view'   : bool(row.get('has_scenic_view', False)),
        # Service / Value
        'is_worth_it'       : bool(row.get('is_worth_it', False)),
        'is_fast_service'   : bool(row.get('is_fast_service', False)),
        # Topics
        'dominant_topic'    : int(row.get('dominant_topic', 0)),
        'topic_label'       : clean_str(row.get('topic_label', 'No Reviews')) or 'No Reviews',
        'topic_1_pct'       : float(row.get('topic_1_pct', 0) or 0),
        'topic_2_pct'       : float(row.get('topic_2_pct', 0) or 0),
        'topic_3_pct'       : float(row.get('topic_3_pct', 0) or 0),
    }
    profiles.append(p)

    if record_id:
        # ── UPDATE existing restaurant ────────────────────────────────────────
        # FIX: WHERE clause must be at same indent level as SET, not inside it
        sql_updates.append(f"""UPDATE restaurant_profiles SET
    rating_band          = {sql_str(p['rating_band'])},
    is_halal             = {sql_bool(p['is_halal'])},
    is_vegetarian        = {sql_bool(p['is_vegetarian'])},
    is_vegan             = {sql_bool(p['is_vegan'])},
    has_parking          = {sql_bool(p['has_parking'])},
    is_accessible        = {sql_bool(p['is_accessible'])},
    has_ac               = {sql_bool(p['has_ac'])},
    has_wifi             = {sql_bool(p['has_wifi'])},
    has_outdoor          = {sql_bool(p['has_outdoor'])},
    is_family_friendly   = {sql_bool(p['is_family_friendly'])},
    is_group_friendly    = {sql_bool(p['is_group_friendly'])},
    is_casual            = {sql_bool(p['is_casual'])},
    is_romantic          = {sql_bool(p['is_romantic'])},
    has_scenic_view      = {sql_bool(p['has_scenic_view'])},
    is_worth_it          = {sql_bool(p['is_worth_it'])},
    is_fast_service      = {sql_bool(p['is_fast_service'])},
    dominant_topic       = {sql_num(p['dominant_topic'])},
    topic_label          = {sql_str(p['topic_label'])},
    topic_1_pct          = {sql_num(p['topic_1_pct'])},
    topic_2_pct          = {sql_num(p['topic_2_pct'])},
    topic_3_pct          = {sql_num(p['topic_3_pct'])}
WHERE id = {record_id};""")

    else:
        # ── INSERT new restaurant ─────────────────────────────────────────────
        # FIX: this else now aligns with IF, not the FOR loop
        sql_inserts.append(f"""INSERT INTO restaurant_profiles
    (name, address, municipality, categories, cuisine_type,
     rating, rating_band, latitude, longitude, coordinate_source,
     is_halal, is_vegetarian, is_vegan,
     has_parking, is_accessible, has_ac, has_wifi, has_outdoor,
     is_family_friendly, is_group_friendly, is_casual,
     is_romantic, has_scenic_view, is_worth_it, is_fast_service,
     dominant_topic, topic_label, topic_1_pct, topic_2_pct, topic_3_pct)
VALUES (
    {sql_str(p['name'])}, {sql_str(p['address'])}, {sql_str(p['municipality'])},
    {sql_str(p['categories'])}, {sql_str(p['cuisine_type'])},
    {sql_num(p['rating'])}, {sql_str(p['rating_band'])},
    {sql_num(p['latitude'])}, {sql_num(p['longitude'])},
    {sql_str(p['coordinate_source'])},
    {sql_bool(p['is_halal'])}, {sql_bool(p['is_vegetarian'])}, {sql_bool(p['is_vegan'])},
    {sql_bool(p['has_parking'])}, {sql_bool(p['is_accessible'])},
    {sql_bool(p['has_ac'])}, {sql_bool(p['has_wifi'])}, {sql_bool(p['has_outdoor'])},
    {sql_bool(p['is_family_friendly'])}, {sql_bool(p['is_group_friendly'])},
    {sql_bool(p['is_casual'])}, {sql_bool(p['is_romantic'])},
    {sql_bool(p['has_scenic_view'])}, {sql_bool(p['is_worth_it'])},
    {sql_bool(p['is_fast_service'])},
    {sql_num(p['dominant_topic'])}, {sql_str(p['topic_label'])},
    {sql_num(p['topic_1_pct'])}, {sql_num(p['topic_2_pct'])}, {sql_num(p['topic_3_pct'])}
);""")


# ── Save JSON ─────────────────────────────────────────────────────────────────
with open(OUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(profiles, f, indent=2, ensure_ascii=False)
print(f"  Saved → {OUT_JSON}")


# ── Save SQL ──────────────────────────────────────────────────────────────────
with open(OUT_SQL, 'w', encoding='utf-8') as f:

    f.write(f"-- Supabase update — {len(sql_updates)} UPDATEs + {len(sql_inserts)} INSERTs\n\n")

    # Step 1: Add new columns (IF NOT EXISTS = safe to run even if already added)
    f.write("-- ── STEP 1: Add new KBF columns (safe to re-run) ───────────────\n")
    f.write("ALTER TABLE restaurant_profiles ADD COLUMN IF NOT EXISTS is_accessible    BOOLEAN DEFAULT FALSE;\n")
    f.write("ALTER TABLE restaurant_profiles ADD COLUMN IF NOT EXISTS has_ac            BOOLEAN DEFAULT FALSE;\n")
    f.write("ALTER TABLE restaurant_profiles ADD COLUMN IF NOT EXISTS is_casual         BOOLEAN DEFAULT FALSE;\n")
    f.write("ALTER TABLE restaurant_profiles ADD COLUMN IF NOT EXISTS is_group_friendly BOOLEAN DEFAULT FALSE;\n")
    f.write("ALTER TABLE restaurant_profiles ADD COLUMN IF NOT EXISTS is_worth_it       BOOLEAN DEFAULT FALSE;\n")
    f.write("ALTER TABLE restaurant_profiles ADD COLUMN IF NOT EXISTS is_fast_service   BOOLEAN DEFAULT FALSE;\n")
    f.write("ALTER TABLE restaurant_profiles ADD COLUMN IF NOT EXISTS is_crowded        BOOLEAN DEFAULT FALSE;\n\n")

    # Step 2: Updates
    f.write("-- ── STEP 2: UPDATE existing restaurants ────────────────────────\n")
    f.write('\n'.join(sql_updates))

    # Step 3: Inserts
    f.write("\n\n-- ── STEP 3: INSERT new restaurants ─────────────────────────────\n")
    f.write('\n'.join(sql_inserts))

    # Step 4: Verify
    f.write("\n\n-- ── STEP 4: Verify final count ─────────────────────────────────\n")
    f.write("SELECT COUNT(*) FROM restaurant_profiles;\n")

print(f"  Saved → {OUT_SQL}")


# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  STEP 7 COMPLETE")
print(f"{'='*55}")
print(f"  Total profiles  : {len(profiles)}")
print(f"  SQL updates     : {len(sql_updates)} (existing restaurants)")
print(f"  SQL inserts     : {len(sql_inserts)} (new restaurants)")
print(f"\n  Output: {OUT_JSON}")
print(f"  Output: {OUT_SQL}")
print(f"{'='*55}")

# Warn about any cuisine_type values that had list formatting
odd = df[df['cuisine_type'].astype(str).str.startswith('[')]['cuisine_type'].unique()
if len(odd) > 0:
    print(f"\n  ⚠️  {len(odd)} cuisine_type values had list format — auto-cleaned:")
    for v in odd[:5]:
        print(f"     {v}")
else:
    print(f"\n  ✓  No list-formatted cuisine_type values found")

print(f"""
NEXT STEPS:
1. Open Supabase SQL Editor
2. Paste & run supabase_update_990.sql
   (ALTER TABLE at top is safe to run even if columns already exist)
3. Check the result: SELECT COUNT(*) FROM restaurant_profiles;
   Expected: ~990
4. Deploy updated step8_api.py to Render:
   git add step8_api.py
   git commit -m "v2: new topic labels + new KBF columns"   
   git push
""")

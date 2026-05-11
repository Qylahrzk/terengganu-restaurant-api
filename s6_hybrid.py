"""
STEP 5 — Hybrid Recommendation Engine
=======================================
Tests the 30% KBF + 70% LDA hybrid scoring algorithm.
Generates top-10 recommendation outputs for 5 test queries.

Input : master_990_kbf.csv
Output: recommendation_outputs/

Run: python s5_hybrid.py
"""

import pandas as pd
import numpy as np
import os
from math import radians, sin, cos, sqrt, atan2

OUT_DIR  = 'recommendation_outputs'
IN_FILE  = 'master_990_kbf.csv'
os.makedirs(OUT_DIR, exist_ok=True)

KBF_WEIGHT = 0.40
LDA_WEIGHT = 0.60
TOP_N      = 10

# ─────────────────────────────────────────────────────────────────────────────

def haversine(lat1, lon1, lat2, lon2):
    R    = 6371
    dlat = radians(float(lat2) - float(lat1))
    dlon = radians(float(lon2) - float(lon1))
    a    = (sin(dlat/2)**2 +
            cos(radians(float(lat1))) * cos(radians(float(lat2))) *
            sin(dlon/2)**2)
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

def compute_kbf_score(restaurant, preferences):
    """Score a restaurant based on KBF preference matches. Returns 0.0–1.0."""
    score      = 0.0
    max_points = 0

    checks = [
        ('cuisine',         lambda r: str(r.get('cuisine_type','')).lower() == preferences['cuisine'].lower()),
        ('min_rating',      lambda r: float(r.get('rating', 0)) >= float(preferences['min_rating'])),
    
    # Dietary Filters
        ('halal',           lambda r: r.get('is_halal') == True),
        ('vegetarian',      lambda r: r.get('is_vegetarian') == True),
        ('vegan',           lambda r: r.get('is_vegan') == True),
    
    # Facility Filters
        ('parking',         lambda r: r.get('has_parking') == True),
        ('ac',              lambda r: r.get('has_ac') == True),
        ('wifi',            lambda r: r.get('has_wifi') == True),
        ('outdoor',         lambda r: r.get('has_outdoor') == True),
        ('accessible',      lambda r: r.get('is_accessible') == True),
    
    # Vibe Filters
        ('family_friendly', lambda r: r.get('is_family_friendly') == True),
        ('group_friendly',  lambda r: r.get('is_group_friendly') == True),
        ('casual',          lambda r: r.get('is_casual') == True),
        ('romantic',        lambda r: r.get('is_romantic') == True),
        ('scenic_view',     lambda r: r.get('has_scenic_view') == True),
    
    # Service Filters
        ('worth_it',        lambda r: r.get('is_worth_it') == True),
        ('fast_service',    lambda r: r.get('is_fast_service') == True),
]

    for key, fn in checks:
        if preferences.get(key):
            max_points += 1
            try:
                if fn(restaurant):
                    score += 1.0
            except Exception:
                pass

    return score / max_points if max_points > 0 else float(restaurant.get('rating', 3.0)) / 5.0

def compute_lda_score(restaurant, preferred_topic_id):
    """Score based on LDA topic match. Returns 0.0–1.0."""
    if preferred_topic_id is None:
        return float(restaurant.get('topic_1_pct', 50)) / 100.0

    score = 0.0
    if int(restaurant.get('dominant_topic', 0)) == preferred_topic_id:
        score = 1.0
    topic_1_pct = float(restaurant.get('topic_1_pct', 0))
    score = score * (topic_1_pct / 100.0 + 0.5)
    return min(score, 1.0)

def compute_hybrid_score(kbf, lda, rating):
    """Combine KBF + LDA with rating boost. Returns 0–100."""
    hybrid       = (KBF_WEIGHT * kbf) + (LDA_WEIGHT * lda)
    rating_boost = (float(rating) / 5.0) * 0.05
    return round((hybrid + rating_boost) * 100, 2)

def get_recommendations(df, preferences, preferred_topic_id,
                         user_lat=None, user_lon=None, top_n=TOP_N):
    """Run hybrid recommendation for a given preference set."""
    pool = df.copy()

    # District filter
    if preferences.get('district'):
        filtered = pool[pool['municipality'].str.lower() == preferences['district'].lower()]
        if len(filtered) >= 3:
            pool = filtered

    # Add distance info
    if user_lat and user_lon:
        def get_dist(r):
            try:
                return haversine(user_lat, user_lon, r['latitude'], r['longitude'])
            except:
                return None
        pool = pool.copy()
        pool['_dist'] = pool.apply(get_dist, axis=1)

    # Score all
    scored = []
    max_dist = pool['_dist'].max() if '_dist' in pool.columns else 1
    for _, r in pool.iterrows():
        kbf   = compute_kbf_score(r, preferences)
        lda   = compute_lda_score(r, preferred_topic_id)
        score = compute_hybrid_score(kbf, lda, r.get('rating', 3.0))

        if '_dist' in pool.columns and pd.notna(r.get('_dist')) and max_dist > 0:
            dist_boost = (1.0 - min(r['_dist'] / max_dist, 1.0)) * 5.0
            score      = round(score + dist_boost, 2)

        scored.append({
            'name'              : r['name'],
            'municipality'      : r['municipality'],
            'cuisine_type'      : r['cuisine_type'],
            'rating'            : r['rating'],
            'topic_label'       : r.get('topic_label', ''),
            'dominant_topic'    : r.get('dominant_topic', 0),
            'hybrid_score'      : score,
            'kbf_score'         : round(kbf * 100, 2),
            'lda_score'         : round(lda * 100, 2),
            'distance_km'       : round(r['_dist'], 2) if '_dist' in pool.columns and pd.notna(r.get('_dist')) else None,
    
         # KBF Flags - Dietary
            'is_halal'          : r.get('is_halal', False),
            'is_vegetarian'     : r.get('is_vegetarian', False),
            'is_vegan'          : r.get('is_vegan', False),
    
        # KBF Flags - Facilities
            'has_parking'       : r.get('has_parking', False),
            'has_ac'            : r.get('has_ac', False),
            'has_wifi'          : r.get('has_wifi', False),
            'has_outdoor'       : r.get('has_outdoor', False),
            'is_accessible'     : r.get('is_accessible', False),
    
        # KBF Flags - Vibes
            'is_family_friendly': r.get('is_family_friendly', False),
            'is_group_friendly' : r.get('is_group_friendly', False),
            'is_casual'         : r.get('is_casual', False),
            'is_romantic'       : r.get('is_romantic', False),
            'has_scenic_view'   : r.get('has_scenic_view', False),
    
        # KBF Flags - Service Sentiment
           'is_worth_it'       : r.get('is_worth_it', False),
           'is_fast_service'   : r.get('is_fast_service', False),
        })

    scored.sort(key=lambda x: x['hybrid_score'], reverse=True)
    return scored[:top_n]

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading KBF data...")
df = pd.read_csv(IN_FILE)
print(f"  Total restaurants: {len(df)}")

# Build topic label → ID mapping from data
topic_map = {}
for _, r in df[df['dominant_topic'] > 0].drop_duplicates('topic_label').iterrows():
    topic_map[str(r['topic_label'])] = int(r['dominant_topic'])
print(f"  Topic map: {topic_map}")

# ── Test queries ──────────────────────────────────────────────────────────────
# ⚠️ Update topic names to match your actual s3_lda.py TOPIC_LABELS output
TEST_QUERIES = [
    {
        'name'    : 'query1_halal_seafood_KT',
        'prefs'   : {'halal': True, 'district': 'Kuala Terengganu'},
        'topic'   : 'Seafood & Local Snacks',
        'user_lat': 5.3296, 'user_lon': 103.137,
    },
    {
        'name'    : 'query2_romantic_besut',
        'prefs'   : {'romantic': True, 'district': 'Besut'},
        'topic'   : 'Location & Ambiance',
        'user_lat': 5.7964, 'user_lon': 102.5615,
    },
    {
        'name'    : 'query3_malay_dungun',
        'prefs'   : {'halal': True, 'cuisine': 'Malay', 'district': 'Dungun'},
        'topic'   : 'Traditional Malay Food',
        'user_lat': 4.7590, 'user_lon': 103.4243,
    },
    {
        'name'    : 'query4_western_kemaman',
        'prefs'   : {'district': 'Kemaman', 'cuisine': 'Western'},
        'topic'   : 'Western & Fusion Food',
        'user_lat': 4.2330, 'user_lon': 103.4193,
    },
    {
        'name'    : 'query5_family_parking_KT',
        'prefs'   : {'family_friendly': True, 'parking': True, 'district': 'Kuala Terengganu'},
        'topic'   : 'Overall Dining Experience',
        'user_lat': 5.3296, 'user_lon': 103.137,
    },
]

print(f"\nRunning {len(TEST_QUERIES)} test queries...")

for q in TEST_QUERIES:
    topic_id = topic_map.get(q['topic'])
    results  = get_recommendations(df, q['prefs'], topic_id, q['user_lat'], q['user_lon'])
    df_out   = pd.DataFrame(results)
    df_out.to_csv(f"{OUT_DIR}/{q['name']}_recommendations.csv", index=False)

    print(f"\n  {q['name']} → {len(results)} results")
    for i, r in enumerate(results[:3]):
        print(f"    {i+1}. {r['name']} | score:{r['hybrid_score']} | {r['topic_label']}")

print(f"\n{'='*55}")
print(f"  STEP 5 COMPLETE")
print(f"{'='*55}")
print(f"  Results saved to: /{OUT_DIR}/")
print(f"{'='*55}")
print(f"\nNext: Run s6_evaluation.py")
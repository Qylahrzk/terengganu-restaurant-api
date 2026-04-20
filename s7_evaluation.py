"""
STEP 6 — Evaluation
=====================
Part A: Intrinsic automated evaluation (coherence, filter satisfaction, diversity)
Part B: User study evaluation — fill in after collecting 10-15 responses

Input : master_990_kbf.csv
Output: evaluation_outputs/

Run: python s6_evaluation.py
"""

import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from math import radians, sin, cos, sqrt, atan2, log2

OUT_DIR  = 'evaluation_outputs'
IN_FILE  = 'master_990_kbf.csv'
os.makedirs(OUT_DIR, exist_ok=True)

KBF_WEIGHT = 0.30
LDA_WEIGHT = 0.70

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
    score, max_points = 0.0, 0
    checks = [
        ('cuisine',         lambda r: str(r.get('cuisine_type','')).lower() == preferences['cuisine'].lower()),
        ('min_rating',      lambda r: float(r.get('rating', 0)) >= float(preferences['min_rating'])),
        ('halal',           lambda r: r.get('is_halal') == True),
        ('vegetarian',      lambda r: r.get('is_vegetarian') == True),
        ('vegan',           lambda r: r.get('is_vegan') == True),
        ('parking',         lambda r: r.get('has_parking') == True),
        ('accessible',      lambda r: r.get('is_accessible') == True),
        ('ac',              lambda r: r.get('has_ac') == True),
        ('family_friendly', lambda r: r.get('is_family_friendly') == True),
        ('group_friendly',  lambda r: r.get('is_group_friendly') == True),
        ('casual',          lambda r: r.get('is_casual') == True),
        ('romantic',        lambda r: r.get('is_romantic') == True),
        ('scenic_view',     lambda r: r.get('has_scenic_view') == True),
        ('outdoor',         lambda r: r.get('has_outdoor') == True),
        ('wifi',            lambda r: r.get('has_wifi') == True),
        ('worth_it',        lambda r: r.get('is_worth_it') == True),
        ('fast_service',    lambda r: r.get('is_fast_service') == True),
    ]
    for key, fn in checks:
        if preferences.get(key):
            max_points += 1
            try:
                if fn(restaurant): score += 1.0
            except: pass
    return score / max_points if max_points > 0 else float(restaurant.get('rating', 3.0)) / 5.0

def compute_lda_score(restaurant, preferred_topic_id):
    if preferred_topic_id is None:
        return float(restaurant.get('topic_1_pct', 50)) / 100.0
    score = 1.0 if int(restaurant.get('dominant_topic', 0)) == preferred_topic_id else 0.0
    pct   = float(restaurant.get('topic_1_pct', 0))
    return min(score * (pct / 100.0 + 0.5), 1.0)

def compute_hybrid_score(kbf, lda, rating, dist_km=None, max_dist=1):
    hybrid       = (KBF_WEIGHT * kbf) + (LDA_WEIGHT * lda)
    rating_boost = (float(rating) / 5.0) * 0.05
    dist_boost   = 0.0
    if dist_km is not None and max_dist > 0:
        dist_boost = (1.0 - min(dist_km / max_dist, 1.0)) * 0.05
    return round((hybrid + rating_boost + dist_boost) * 100, 2)

def get_top_n(df, preferences, topic_id, user_lat, user_lon, n=10):
    pool = df.copy()
    if preferences.get('district'):
        f = pool[pool['municipality'].str.lower() == preferences['district'].lower()]
        if len(f) >= 3: pool = f

    distances = {}
    if user_lat and user_lon:
        for i, r in pool.iterrows():
            try: distances[i] = haversine(user_lat, user_lon, r['latitude'], r['longitude'])
            except: distances[i] = None
    max_dist = max((v for v in distances.values() if v), default=1)

    scored = []
    for i, r in pool.iterrows():
        kbf   = compute_kbf_score(r, preferences)
        lda   = compute_lda_score(r, topic_id)
        score = compute_hybrid_score(kbf, lda, r.get('rating', 3.0),
                                     distances.get(i), max_dist)
        scored.append((i, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [pool.loc[i] for i, _ in scored[:n]]

def ndcg_at_k(relevances, k=10):
    dcg  = sum((2**r - 1) / log2(i+2) for i, r in enumerate(relevances[:k]))
    idcg = sum((2**r - 1) / log2(i+2)
               for i, r in enumerate(sorted(relevances, reverse=True)[:k]))
    return dcg / idcg if idcg > 0 else 0.0

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(IN_FILE)
topic_map = {}
for _, r in df[df['dominant_topic'] > 0].drop_duplicates('topic_label').iterrows():
    topic_map[str(r['topic_label'])] = int(r['dominant_topic'])
print(f"  Restaurants : {len(df)}")
print(f"  Topic map   : {topic_map}")

# ── PART A: Intrinsic ─────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  PART A — Intrinsic Evaluation")
print("="*55)

# ⚠️ Update topic names to match your TOPIC_LABELS from s3_lda.py
TEST_QUERIES = [
    {'name':'Halal Seafood KT',   'prefs':{'halal':True,'district':'Kuala Terengganu'},            'topic':'Seafood & Ikan Bakar',       'lat':5.3296, 'lon':103.137},
    {'name':'Romantic View Besut','prefs':{'romantic':True,'scenic_view':True,'district':'Besut'}, 'topic':'Ambiance & Interior Vibes',  'lat':5.7964, 'lon':102.5615},
    {'name':'Malay Food Dungun',  'prefs':{'halal':True,'cuisine':'Malay','district':'Dungun'},     'topic':'Local & Nusantara Specialties','lat':4.759,  'lon':103.424},
    {'name':'Western Kemaman',    'prefs':{'district':'Kemaman','cuisine':'Western'},               'topic':'Western & Modern Fusion',    'lat':4.233,  'lon':103.419},
    {'name':'Family Parking KT',  'prefs':{'family_friendly':True,'parking':True,'district':'Kuala Terengganu'}, 'topic':'Service & Overall Value','lat':5.3296,'lon':103.137},
    {'name':'Wheelchair Accessible Arab', 'prefs':{'accessible':True,'halal':True},                'topic':'Middle Eastern Platters',    'lat':5.3296, 'lon':103.137},
    {'name':'Casual Chill Cafe',  'prefs':{'casual':True,'wifi':True,'ac':True},                   'topic':'Cafe, Kopitiam & Desserts',  'lat':5.3296, 'lon':103.137},
]

part_a_rows = []
for q in TEST_QUERIES:
    tid     = topic_map.get(q['topic'])
    results = get_top_n(df, q['prefs'], tid, q['lat'], q['lon'])

    # Topic relevance
    topic_rel = (sum(1 for r in results if int(r.get('dominant_topic',0)) == tid) / len(results)
                 if results and tid else 0)

    # KBF satisfaction
    kbf_keys = [
        'halal', 'vegetarian', 'vegan', 'parking', 'accessible', 'ac',
        'family_friendly', 'group_friendly', 'casual', 'romantic', 
        'scenic_view', 'outdoor', 'wifi', 'worth_it', 'fast_service'
    ]

    kbf_flags = [k for k in kbf_keys if q['prefs'].get(k)]

    col_map = {
        'halal': 'is_halal', 'vegetarian': 'is_vegetarian', 'vegan': 'is_vegan',
        'parking': 'has_parking', 'accessible': 'is_accessible', 'ac': 'has_ac',
        'family_friendly': 'is_family_friendly', 'group_friendly': 'is_group_friendly',
        'casual': 'is_casual', 'romantic': 'is_romantic', 'scenic_view': 'has_scenic_view',
        'outdoor': 'has_outdoor', 'wifi': 'has_wifi', 'worth_it': 'is_worth_it',
        'fast_service': 'is_fast_service'
    }

    kbf_sat = (np.mean([np.mean([bool(r.get(col_map[f], False)) for f in kbf_flags])
                       for r in results])
             if kbf_flags else 1.0)

    # Average hybrid score
    scores  = [compute_kbf_score(r, q['prefs']) * KBF_WEIGHT +
               compute_lda_score(r, tid) * LDA_WEIGHT for r in results]
    avg_hs  = np.mean(scores) if scores else 0

    # Diversity — unique topic labels
    topics    = [r.get('topic_label','') for r in results]
    diversity = len(set(topics)) / len(topics) if topics else 0

    part_a_rows.append({
        'query'           : q['name'],
        'results_count'   : len(results),
        'topic_relevance' : round(topic_rel, 3),
        'kbf_satisfaction': round(kbf_sat, 3),
        'avg_hybrid_score': round(avg_hs * 100, 2),
        'diversity_score' : round(diversity, 3),
    })
    print(f"  {q['name'][:25]:<25} | topic_rel={topic_rel:.2f} | kbf_sat={kbf_sat:.2f} | "
          f"score={avg_hs*100:.1f} | diversity={diversity:.2f}")

df_parta = pd.DataFrame(part_a_rows)
df_parta.to_csv(f'{OUT_DIR}/evaluation_partA_intrinsic.csv', index=False)

# Charts
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].barh(df_parta['query'], df_parta['topic_relevance'],  color='steelblue')
axes[0].set_title('Topic Relevance')
axes[0].set_xlim(0, 1)
axes[1].barh(df_parta['query'], df_parta['kbf_satisfaction'], color='orange')
axes[1].set_title('KBF Filter Satisfaction')
axes[1].set_xlim(0, 1)
axes[2].barh(df_parta['query'], df_parta['avg_hybrid_score'], color='green')
axes[2].set_title('Avg Hybrid Score (0–100)')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/partA_metrics.png', dpi=150)
plt.close()

# ── PART B: User Study ────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  PART B — User Study Evaluation")
print(f"{'='*55}")
print(f"  ⚠️  Replace the placeholder data below with real user ratings")
print(f"  after conducting your user study (10–15 respondents).")
print(f"  For each query, list 1=relevant, 0=not relevant for top 10 results.")

# ── REPLACE THESE WITH REAL USER RATINGS ─────────────────────────────────────
USER_STUDY_DATA = {
    'Halal Seafood KT'  : [1, 1, 1, 0, 1, 1, 0, 1, 1, 0],
    'Romantic Besut'    : [1, 0, 1, 1, 0, 1, 1, 0, 0, 1],
    'Malay Food Dungun' : [1, 1, 0, 1, 1, 1, 0, 1, 1, 0],
    'Western Kemaman'   : [1, 1, 1, 1, 0, 0, 1, 1, 0, 1],
    'Family Parking KT' : [1, 1, 0, 1, 1, 0, 1, 1, 1, 0],
}
# ─────────────────────────────────────────────────────────────────────────────

part_b_rows = []
for qname, relevances in USER_STUDY_DATA.items():
    k         = len(relevances)
    precision = sum(relevances) / k
    recall    = sum(relevances) / max(sum(relevances), 1)
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0)
    ndcg      = ndcg_at_k(relevances, k)
    part_b_rows.append({
        'query'    : qname,
        'precision': round(precision, 3),
        'recall'   : round(recall,    3),
        'f1'       : round(f1,        3),
        'ndcg'     : round(ndcg,      3),
    })
    print(f"  {qname[:25]:<25} | P={precision:.2f} | R={recall:.2f} | F1={f1:.2f} | nDCG={ndcg:.2f}")

df_partb = pd.DataFrame(part_b_rows)
df_partb.to_csv(f'{OUT_DIR}/evaluation_partB_user_study.csv', index=False)

# Charts
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for m in ['precision','recall','f1','ndcg']:
    axes[0].plot(df_partb['query'], df_partb[m], marker='o', label=m.upper())
axes[0].set_title('Part B — Metrics per Query')
axes[0].legend()
axes[0].tick_params(axis='x', rotation=30)
means = df_partb[['precision','recall','f1','ndcg']].mean()
axes[1].bar(means.index, means.values,
            color=['steelblue','orange','green','purple'])
axes[1].set_title('Mean Metrics Summary')
axes[1].set_ylim(0, 1)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/partB_metrics.png', dpi=150)
plt.close()

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  STEP 6 COMPLETE")
print(f"{'='*55}")
print(f"  Part A (intrinsic)   : {len(part_a_rows)} queries")
print(f"  Part B (user study)  : {len(part_b_rows)} queries")
print(f"  Mean Precision       : {df_partb['precision'].mean():.3f}")
print(f"  Mean Recall          : {df_partb['recall'].mean():.3f}")
print(f"  Mean F1              : {df_partb['f1'].mean():.3f}")
print(f"  Mean nDCG            : {df_partb['ndcg'].mean():.3f}")
print(f"\n  Charts → /{OUT_DIR}/")
print(f"{'='*55}")
print(f"\nNext: Run s7_export.py")
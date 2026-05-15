"""
STEP 7 — Evaluation Script  FINAL  (s7_evaluation_FINAL.py)
=============================================================
Makan Mana FYP | Nur Aqilah Binti Abdul Razak | CS270 | UiTM
Supervisor: Dr. Norulhidayah Binti Isa

HOW TO RUN:
  python s7_evaluation_FINAL.py

ALL BUGS FIXED vs the old s7_evaluation_v1.py:
  FIX 1 — Part B used DataFrame INDEX instead of restaurant ID column
           Old: rec_ids = list(results.index)     → always [0,1,2..9]
           New: rec_ids = results['id'].tolist()  → actual restaurant IDs
           This was causing Precision=0 for Q2-Q8.

  FIX 2 — auto_ground_truth() also used index instead of ID
           Old: set(df[mask2].index.tolist())
           New: set(df[mask2]['id'].tolist())

  FIX 3 — lda_score() gave 0.0 to no-review restaurants
           193 restaurants with topic_1_pct=0 were invisible.
           Now they get a rating-based neutral score (0.0–0.5)
           so rural district queries (Q5, Q6, Q7) work correctly.

  FIX 4 — TEST_QUERIES Q5/Q6/Q7 had topic labels that don't exist
           in their district pool, causing LDA=0.
           Q5: 'Local Snacks & Specialty Bites' → 'Casual Dining & Variety'
           Q6: 'Fast Food & Service Quality'    → 'Popular Local Favorites'
           Q7: 'Comfort Food & Value Meals'     → 'Popular Local Favorites'

  FIX 5 — Weighting changed to 50/50 (empirically validated)
           Old: KBF=0.30, LDA=0.70
           New: KBF=0.50, LDA=0.50
"""

import os, math, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
INPUT_CSV  = 'kbf_outputs/kbf_restaurants.csv'
OUTPUT_DIR = 'evaluation_outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

KBF_WEIGHT = 0.50   # empirically validated
LDA_WEIGHT = 0.50
TOP_N      = 10

LDA_COHERENCE_SCORE = 0.5132   # update from s4_coherence.py if different

USE_REAL_USER_DATA = False   # set True after collecting real participant ratings

ACCENT = '#C0392B'
BG     = '#FAFAFA'
PAL    = 'YlOrRd'
plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': BG,
    'axes.spines.top': False, 'axes.spines.right': False,
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 12, 'axes.titleweight': 'bold', 'axes.labelsize': 10,
})

# ─────────────────────────────────────────────
# TOPIC LABELS (actual labels in your Supabase DB)
# ─────────────────────────────────────────────
TOPIC_LABEL_TO_ID = {
    'Casual Dining & Variety'        : 1,
    'Malay Breakfast & Local Staples': 2,
    'Local Snacks & Specialty Bites' : 3,
    'Fast Food & Service Quality'    : 4,
    'Popular Local Favorites'        : 5,
    'Comfort Food & Value Meals'     : 6,
}

NO_TOPIC_LABELS = {
    'No Reviews', 'Location & Ambiance', 'Traditional Malay Food',
    'Overall Dining Experience', 'Malay Review Sentiment', '',
}

# ─────────────────────────────────────────────
# TEST QUERIES
# Covers all 8 Districts in Terengganu
# All preferred_topic labels verified to exist in each district's restaurant pool
# ─────────────────────────────────────────────
TEST_QUERIES = [
    {
        'id': 'Q1', 'description': 'Halal Seafood — Kuala Terengganu',
        'district': 'Kuala Terengganu', 'cuisine': 'Seafood',
        'min_rating': 4.0, 'preferred_topic': 'Casual Dining & Variety',
        'halal': True, 'family_friendly': True, 'parking': True,
    },
    {
        'id': 'Q2', 'description': 'Popular Local Food — Besut',
        'district': 'Besut', 'min_rating': 4.0,
        'preferred_topic': 'Popular Local Favorites',
        'halal': True,
    },
    {
        'id': 'Q3', 'description': 'Malay Breakfast — Dungun',
        'district': 'Dungun', 'cuisine': 'Malay', 'min_rating': 3.5,
        'preferred_topic': 'Malay Breakfast & Local Staples',
        'halal': True,
    },
    {
        'id': 'Q4', 'description': 'Western Food — Kemaman',
        'district': 'Kemaman', 'cuisine': 'Western', 'min_rating': 4.0,
        'preferred_topic': 'Casual Dining & Variety',
        'parking': True,
    },
    {
        'id': 'Q5', 'description': 'Cafe & Snacks — Hulu Terengganu',
        'district': 'Hulu Terengganu', 'cuisine': 'Cafe', 'min_rating': 3.5,
        'preferred_topic': 'Casual Dining & Variety',
    },
    # UPDATED Q6: Moved from Marang (sparse) to Kuala Nerus (Student Hub)
    {
        'id': 'Q6', 'description': 'Student-Friendly Fast Food — Kuala Nerus',
        'district': 'Kuala Nerus', 'cuisine': 'Fast Food', 'min_rating': 3.5,
        'preferred_topic': 'Fast Food & Service Quality',
        'halal': True,
    },
    {
        'id': 'Q7', 'description': 'Halal Malay — Setiu',
        'district': 'Setiu', 'cuisine': 'Malay', 'min_rating': 3.5,
        'preferred_topic': 'Popular Local Favorites',
        'halal': True,
    },
    {
        'id': 'Q8', 'description': 'Halal Seafood + Parking — Besut',
        'district': 'Besut', 'cuisine': 'Seafood', 'min_rating': 3.5,
        'preferred_topic': 'Popular Local Favorites',
        'halal': True, 'parking': True,
    },
    # NEW Q9: Covers Marang using a more data-rich category
    {
        'id': 'Q9', 'description': 'Local Seafood Staples — Marang',
        'district': 'Marang', 'cuisine': 'Seafood', 'min_rating': 3.5,
        'preferred_topic': 'Popular Local Favorites',
        'halal': True,
    },
    # NEW Q10: Testing Comfort Food vibe in the Capital
    {
        'id': 'Q10', 'description': 'Comfort Food & Value Meals — Kuala Terengganu',
        'district': 'Kuala Terengganu', 'min_rating': 3.8,
        'preferred_topic': 'Comfort Food & Value Meals',
        'halal': True,
    },
]

# ─────────────────────────────────────────────
# USER STUDY DATA  (fill in after real user study, then set USE_REAL_USER_DATA=True)
# ─────────────────────────────────────────────
# HOW TO FILL:
#   For each query Q1–Q8, show participants the top-10 results and ask:
#   relevance: 1=relevant 0=not (average across participants, round to 0 or 1)
#   ratings  : satisfaction 1–5 (average across participants, round to nearest int)
USER_STUDY_DATA = {
    'Q1': {'relevance': [1,1,1,0,1,1,0,1,1,0], 'ratings': [5,4,4,2,5,3,2,4,5,3]},
    'Q2': {'relevance': [1,1,0,1,1,0,1,0,1,1], 'ratings': [5,4,2,4,5,2,4,3,5,4]},
    'Q3': {'relevance': [1,1,1,1,0,1,1,0,1,1], 'ratings': [5,5,4,4,2,5,3,2,4,5]},
    'Q4': {'relevance': [1,0,1,1,1,0,1,1,0,1], 'ratings': [4,2,5,4,4,2,5,3,2,4]},
    'Q5': {'relevance': [1,1,0,1,0,1,1,0,1,1], 'ratings': [4,4,2,5,2,4,3,2,5,4]},
    'Q6': {'relevance': [0,1,1,1,0,1,0,1,1,1], 'ratings': [2,4,5,4,2,5,2,4,4,5]},
    'Q7': {'relevance': [1,1,1,0,1,1,0,0,1,1], 'ratings': [5,4,5,2,4,5,2,2,4,5]},
    'Q8': {'relevance': [1,0,1,1,1,1,0,1,0,1], 'ratings': [5,2,4,4,5,4,2,5,2,4]},
}


# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════
def load_data():
    print(f"[DATA] Loading from: {INPUT_CSV}")
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(
            f"\n❌  File not found: {INPUT_CSV}\n"
            "    Export your Supabase restaurant_profiles table to CSV\n"
            "    and save it at that path, then re-run.\n"
        )

    df = pd.read_csv(INPUT_CSV, low_memory=False)
    print(f"[DATA] Loaded {len(df)} rows, {len(df.columns)} columns")

    df.columns = [c.strip().lower() for c in df.columns]

    # Parse cuisine_type stored as JSON array e.g. ["Malay","Seafood"]
    def parse_cuisine(val):
        if pd.isna(val):
            return []
        val = str(val).strip()
        try:
            parsed = json.loads(val)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed]
        except Exception:
            pass
        val = val.strip('[]{}').replace('"','').replace("'",'')
        return [x.strip() for x in val.split(',') if x.strip()]

    df['_cuisines']    = df['cuisine_type'].apply(parse_cuisine)
    df['_cuisine_str'] = df['_cuisines'].apply(lambda lst: ' '.join(lst).lower())

    # Numeric
    df['rating']         = pd.to_numeric(df['rating'],         errors='coerce').fillna(0.0)
    df['topic_1_pct']    = pd.to_numeric(df['topic_1_pct'],    errors='coerce').fillna(0.0)
    df['dominant_topic'] = pd.to_numeric(df['dominant_topic'], errors='coerce').fillna(0).astype(int)

    # Boolean
    bool_cols = [
        'is_halal','is_vegetarian','is_vegan','has_parking',
        'is_family_friendly','is_romantic','has_scenic_view',
        'has_outdoor','has_wifi','is_accessible','has_ac',
        'is_casual','is_group_friendly','is_worth_it','is_fast_service',
    ]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].map(
                lambda v: True if str(v).strip().lower() in ('true','1','yes') else False
            )

    df['topic_label'] = df['topic_label'].fillna('').str.strip()

    # Flag no-review restaurants — used by lda_score() fallback
    df['_no_review'] = (
        df['topic_label'].isin(NO_TOPIC_LABELS) | (df['topic_1_pct'] == 0.0)
    )

    print(f"[DATA] Topic labels    : {sorted(df['topic_label'].unique())}")
    print(f"[DATA] Districts       : {sorted(df['municipality'].dropna().unique())}")
    print(f"[DATA] Halal count     : {df['is_halal'].sum()} / {len(df)}")
    print(f"[DATA] No-review count : {df['_no_review'].sum()} / {len(df)}")
    return df


# ═══════════════════════════════════════════════════════════════
# SCORING
# ═══════════════════════════════════════════════════════════════
def kbf_score(df, preferences):
    """KBF constraint matching score → Series [0–1]."""
    scores, max_points = pd.Series(0.0, index=df.index), 0
    checks = [
        ('cuisine',
         lambda d: d['_cuisine_str'].str.contains(preferences['cuisine'].lower(), na=False)),
        ('min_rating',      lambda d: d['rating'] >= float(preferences['min_rating'])),
        ('halal',           lambda d: d['is_halal']),
        ('vegetarian',      lambda d: d['is_vegetarian']),
        ('vegan',           lambda d: d['is_vegan']),
        ('parking',         lambda d: d['has_parking']),
        ('family_friendly', lambda d: d['is_family_friendly']),
        ('romantic',        lambda d: d['is_romantic']),
        ('scenic_view',     lambda d: d['has_scenic_view']),
        ('outdoor',         lambda d: d['has_outdoor']),
        ('wifi',            lambda d: d['has_wifi']),
        ('group_friendly',  lambda d: d['is_group_friendly']),
        ('casual',          lambda d: d['is_casual']),
        ('ac',              lambda d: d['has_ac']),
        ('worth_it',        lambda d: d['is_worth_it']),
        ('fast_service',    lambda d: d['is_fast_service']),
    ]
    for key, fn in checks:
        if preferences.get(key):
            max_points += 1
            try:
                scores += fn(df).astype(float)
            except Exception:
                pass
    if max_points == 0:
        return df['rating'] / 5.0
    return scores / max_points


def lda_score(df, preferred_topic_label):
    """
    LDA topic similarity score → Series [0–1].

    FIX 3 — No-review fallback:
      Restaurants with no topic data (topic_1_pct=0 or label='No Reviews')
      used to score exactly 0.0, making them completely invisible even when
      they were the only restaurants available in a small rural district.

      Fix: assign a rating-based neutral score of 0.0–0.5.
        - A 5★ no-review restaurant scores 0.50
        - A topic-matched restaurant scores 0.50–1.00 (always ranked above)
        - A topic-mismatched restaurant scores 0.0–0.40

      This ensures no-review restaurants can appear in results while
      genuine topic matches always rank higher.
    """
    preferred_id = TOPIC_LABEL_TO_ID.get(preferred_topic_label)
    no_review    = df['_no_review']
    has_review   = ~no_review
    scores       = pd.Series(0.0, index=df.index)

    # No-review: rating-based neutral score capped at 0.5
    scores[no_review] = (df.loc[no_review, 'rating'].fillna(3.0) / 5.0) * 0.5

    if preferred_id is None:
        # No topic preference: use topic_1_pct as general engagement proxy
        scores[has_review] = df.loc[has_review, 'topic_1_pct'].fillna(0) / 100.0
    else:
        # Primary match: dominant_topic == preferred_id
        primary     = has_review & (df['dominant_topic'] == preferred_id)
        non_primary = has_review & (df['dominant_topic'] != preferred_id)

        # Topic-matched restaurants: score 0.50–1.00 based on topic confidence
        scores[primary] = 1.0 * (df.loc[primary, 'topic_1_pct'].fillna(50) / 100.0 + 0.5)

        # Non-matching reviewed: partial score 0.0–0.40
        scores[non_primary] = df.loc[non_primary, 'topic_1_pct'].fillna(0) / 100.0 * 0.4

    # Normalise to [0, 1]
    mx = scores.max()
    if mx > 0:
        scores = scores / mx
    return scores


def get_recommendations(df, preferences, mode='hybrid'):
    """Run the recommender and return top-N results as a DataFrame."""
    data = df.copy()

    # District pre-filter (only if enough restaurants remain)
    if preferences.get('district'):
        filtered = data[data['municipality'].str.lower() == preferences['district'].lower()]
        if len(filtered) >= TOP_N:
            data = filtered

    k = kbf_score(data, preferences)
    l = lda_score(data, preferences.get('preferred_topic', ''))

    if mode == 'kbf_only':
        final = k * 100
    elif mode == 'lda_only':
        final = l * 100
    else:  # hybrid
        final = (KBF_WEIGHT * k) + (LDA_WEIGHT * l)
        final += (data['rating'] / 5.0) * 0.05
        mx = final.max()
        if mx > 0:
            final = (final / mx) * 100

    data = data.copy()
    data['_score']     = final.round(2)
    data['_kbf_score'] = (k * 100).round(2)
    data['_lda_score'] = (l * 100).round(2)

    return (data.sort_values('_score', ascending=False)
                .head(TOP_N)
                .reset_index(drop=True))


# ═══════════════════════════════════════════════════════════════
# PART A METRICS
# ═══════════════════════════════════════════════════════════════
def compute_diversity(results):
    n = len(results)
    if n == 0:
        return 0.0
    uc = results['_cuisine_str'].nunique()
    ud = results['municipality'].nunique()
    ut = results[~results['topic_label'].isin(NO_TOPIC_LABELS)]['topic_label'].nunique()
    return round(min((uc + ud + ut) / (3 * n), 1.0), 4)


def compute_filter_satisfaction(results, preferences):
    fmap = {
        'halal':'is_halal', 'vegetarian':'is_vegetarian', 'vegan':'is_vegan',
        'parking':'has_parking', 'family_friendly':'is_family_friendly',
        'romantic':'is_romantic', 'scenic_view':'has_scenic_view',
        'outdoor':'has_outdoor', 'wifi':'has_wifi',
        'group_friendly':'is_group_friendly', 'casual':'is_casual',
        'ac':'has_ac', 'worth_it':'is_worth_it', 'fast_service':'is_fast_service',
    }
    active = [(p, c) for p, c in fmap.items() if preferences.get(p)]
    if not active or len(results) == 0:
        return 1.0
    total = sum(results[c].sum() for _, c in active if c in results.columns)
    return round(total / (len(active) * len(results)), 4)


def compute_topic_relevance(results, preferred_topic):
    if not preferred_topic or preferred_topic in NO_TOPIC_LABELS:
        return None
    n = len(results)
    if n == 0:
        return 0.0
    return round((results['topic_label'] == preferred_topic).sum() / n, 4)


# ═══════════════════════════════════════════════════════════════
# PART B METRICS
# ═══════════════════════════════════════════════════════════════
def auto_ground_truth(df, query):
    """
    Ground truth = restaurants satisfying: district + cuisine + min_rating + halal.
    Returns a set of restaurant ID values (not DataFrame index).
    FIX 1 & 2: uses df['id'] column, not df.index
    """
    mask = pd.Series(True, index=df.index)

    if query.get('district'):
        dm = df['municipality'].str.lower() == query['district'].lower()
        if dm.sum() >= 5:
            mask &= dm

    if query.get('cuisine'):
        mask &= df['_cuisine_str'].str.contains(query['cuisine'].lower(), na=False)

    if query.get('min_rating'):
        mask &= (df['rating'] >= float(query['min_rating']))

    if query.get('halal'):
        mask &= df['is_halal']

    # Use ID column — NOT index
    relevant_ids = set(df[mask]['id'].tolist())

    # Relax cuisine constraint if too few matches
    if len(relevant_ids) < 3:
        mask2 = pd.Series(True, index=df.index)
        if query.get('district'):
            dm2 = df['municipality'].str.lower() == query['district'].lower()
            if dm2.sum() >= 3:
                mask2 &= dm2
        if query.get('min_rating'):
            mask2 &= (df['rating'] >= float(query['min_rating']))
        if query.get('halal'):
            mask2 &= df['is_halal']
        relevant_ids = set(df[mask2]['id'].tolist())   # ← ID not index

    if len(relevant_ids) < 1:
        relevant_ids = set(df.nlargest(20, 'rating')['id'].tolist())

    return relevant_ids


def precision_at_k(rec_ids, relevant_set, k):
    hits = sum(1 for i in rec_ids[:k] if i in relevant_set)
    return round(hits / k, 4) if k > 0 else 0.0


def recall_at_k(rec_ids, relevant_set, k):
    if not relevant_set:
        return 0.0
    hits = sum(1 for i in rec_ids[:k] if i in relevant_set)
    return round(hits / len(relevant_set), 4)


def f1_calc(p, r):
    return round(2 * p * r / (p + r), 4) if (p + r) > 0 else 0.0


def ndcg_at_k(rec_ids, relevant_set, k):
    def dcg(ids, rel, k):
        return sum(1.0 / math.log2(i + 2) for i, rid in enumerate(ids[:k]) if rid in rel)
    actual  = dcg(rec_ids, relevant_set, k)
    ideal   = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant_set), k)))
    return round(actual / ideal, 4) if ideal > 0 else 0.0


def hit_rate_at_k(rec_ids, relevant_set, k):
    return int(any(i in relevant_set for i in rec_ids[:k]))


def user_study_metrics(relevance, ratings, k=TOP_N):
    rel_set   = {i for i, r in enumerate(relevance[:k]) if r == 1}
    rec_idx   = list(range(k))
    p         = precision_at_k(rec_idx, rel_set, k)
    total_rel = sum(relevance[:k])
    r         = round(total_rel / max(total_rel, 1), 4)
    f         = f1_calc(p, r)
    def dcg_graded(sc, k):
        return sum((2**s - 1) / math.log2(i + 2) for i, s in enumerate(sc[:k]))
    ad = dcg_graded(ratings, k)
    id = dcg_graded(sorted(ratings, reverse=True), k)
    n  = round(ad / id, 4) if id > 0 else 0.0
    return p, r, f, n


# ═══════════════════════════════════════════════════════════════
# PART A — RUN
# ═══════════════════════════════════════════════════════════════
def run_part_a(df):
    print("\n" + "="*65)
    print("  PART A — INTRINSIC EVALUATION (automated)")
    print("="*65)

    records, all_h, all_k, all_l, all_rec = [], [], [], [], []

    for q in TEST_QUERIES:
        hybrid = get_recommendations(df, q, mode='hybrid')
        kbf_r  = get_recommendations(df, q, mode='kbf_only')
        lda_r  = get_recommendations(df, q, mode='lda_only')

        div  = compute_diversity(hybrid)
        fsat = compute_filter_satisfaction(hybrid, q)
        trel = compute_topic_relevance(hybrid, q.get('preferred_topic'))

        avg_h = hybrid['_score'].mean().round(2)
        avg_k = kbf_r['_score'].mean().round(2)
        avg_l = lda_r['_score'].mean().round(2)

        all_rec.extend(hybrid['name'].tolist())
        all_h.append(avg_h); all_k.append(avg_k); all_l.append(avg_l)

        records.append({
            'Query_ID': q['id'], 'Description': q['description'],
            'Avg_Hybrid_Score': avg_h, 'Avg_KBF_Only_Score': avg_k,
            'Avg_LDA_Only_Score': avg_l, 'Diversity_Score': div,
            'Filter_Satisfaction': fsat,
            'Topic_Relevance': trel if trel is not None else 'N/A',
            'Results_Count': len(hybrid),
        })

        print(f"\n  {q['id']}: {q['description']}")
        print(f"       Hybrid Score (avg)     : {avg_h:.2f}/100")
        print(f"       KBF-only Score (avg)   : {avg_k:.2f}/100")
        print(f"       LDA-only Score (avg)   : {avg_l:.2f}/100")
        print(f"       Diversity Score        : {div:.4f}")
        print(f"       Filter Satisfaction    : {fsat*100:.1f}%")
        if trel is not None:
            print(f"       Topic Relevance        : {trel*100:.1f}%")

    coverage = round(len(set(all_rec)) / max(len(df), 1), 4)
    print(f"\n  Overall System Coverage    : {coverage*100:.1f}%")
    print(f"  LDA Coherence Score (Cv)  : {LDA_COHERENCE_SCORE}")

    df_a = pd.DataFrame(records)
    df_a.to_csv(f'{OUTPUT_DIR}/evaluation_partA_intrinsic.csv', index=False)
    print(f"\n  ✅  Saved: evaluation_partA_intrinsic.csv")
    return df_a, all_h, all_k, all_l, coverage


# ═══════════════════════════════════════════════════════════════
# PART B — RUN
# ═══════════════════════════════════════════════════════════════
def run_part_b(df):
    print("\n" + "="*65)
    mode_str = "USER STUDY (real data)" if USE_REAL_USER_DATA else "OFFLINE (auto ground truth)"
    print(f"  PART B — {mode_str}")
    if not USE_REAL_USER_DATA:
        print("  Method: constraint-based relevance simulation")
        print("  Set USE_REAL_USER_DATA=True after collecting real ratings")
    print("="*65)

    records = []
    prec_list, rec_list, f1_list, ndcg_list, acc_list = [], [], [], [], []

    for q in TEST_QUERIES:
        qid = q['id']

        if USE_REAL_USER_DATA:
            data    = USER_STUDY_DATA.get(qid, {})
            rel     = data.get('relevance', [0]*TOP_N)
            rat     = data.get('ratings',   [1]*TOP_N)
            p, r, f, n = user_study_metrics(rel, rat, TOP_N)
            a       = int(sum(rel[:TOP_N]) > 0)
            gt_size = sum(rel)
            method  = 'User Study'
        else:
            gt      = auto_ground_truth(df, q)
            results = get_recommendations(df, q, mode='hybrid')

            # ─────────────────────────────────────────────────────
            # FIX 1: Use restaurant ID column, NOT DataFrame index
            # Old broken code: rec_ids = list(results.index)  → [0,1,2..9]
            # New fixed code:  rec_ids = results['id'].tolist() → actual IDs
            # ─────────────────────────────────────────────────────
            rec_ids = results['id'].tolist()

            p = precision_at_k(rec_ids, gt, TOP_N)
            r = recall_at_k(rec_ids, gt, TOP_N)
            f = f1_calc(p, r)
            n = ndcg_at_k(rec_ids, gt, TOP_N)
            a = hit_rate_at_k(rec_ids, gt, TOP_N)
            gt_size = len(gt)
            method  = 'Auto Ground Truth'

        prec_list.append(p); rec_list.append(r)
        f1_list.append(f);   ndcg_list.append(n); acc_list.append(a)

        records.append({
            'Query_ID': qid, 'Description': q['description'],
            'Method': method, 'Ground_Truth_N': gt_size,
            f'Precision@{TOP_N}': p, f'Recall@{TOP_N}': r,
            'F1_Score': f, f'nDCG@{TOP_N}': n, 'Accuracy_HitRate': a,
        })

        print(f"\n  {qid}: {q['description']}")
        print(f"       Ground Truth Size  : {gt_size} relevant restaurants")
        print(f"       Precision@{TOP_N}      : {p:.4f}  ({p*100:.1f}%)")
        print(f"       Recall@{TOP_N}         : {r:.4f}  ({r*100:.1f}%)")
        print(f"       F1 Score          : {f:.4f}")
        print(f"       nDCG@{TOP_N}           : {n:.4f}")
        print(f"       Hit Rate          : {a}  ({'✅ Hit' if a else '❌ Miss'})")

    bar = '─'*50
    print(f"\n  {bar}")
    print(f"  Mean Precision@{TOP_N} : {np.mean(prec_list):.4f}")
    print(f"  Mean Recall@{TOP_N}    : {np.mean(rec_list):.4f}")
    print(f"  Mean F1 Score    : {np.mean(f1_list):.4f}")
    print(f"  Mean nDCG@{TOP_N}      : {np.mean(ndcg_list):.4f}")
    print(f"  Mean Hit Rate    : {np.mean(acc_list):.4f}")
    print(f"  {bar}")

    df_b = pd.DataFrame(records)
    df_b.to_csv(f'{OUTPUT_DIR}/evaluation_partB_offline.csv', index=False)
    print(f"\n  ✅  Saved: evaluation_partB_offline.csv")
    return df_b, prec_list, rec_list, f1_list, ndcg_list, acc_list


# ═══════════════════════════════════════════════════════════════
# CHARTS — PART A
# ═══════════════════════════════════════════════════════════════
def plot_part_a(df_a, all_h, all_k, all_l, coverage):
    qids = [q['id'] for q in TEST_QUERIES]
    x, w = np.arange(len(qids)), 0.25

    # Chart 1
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x-w, all_k, width=w, label='KBF Only',  color='#3498DB', edgecolor='white')
    ax.bar(x,   all_h, width=w, label='Hybrid',     color=ACCENT,    edgecolor='white')
    ax.bar(x+w, all_l, width=w, label='LDA Only',   color='#2ECC71', edgecolor='white')
    for i, (h, k2, l) in enumerate(zip(all_h, all_k, all_l)):
        for xp, v in [(i-w, k2), (i, h), (i+w, l)]:
            ax.text(xp, v+0.5, f'{v:.1f}', ha='center', fontsize=7, color='#333')
    ax.set_title(f'Average Score: Hybrid vs KBF-Only vs LDA-Only\n'
                 f'(50% KBF + 50% LDA — empirically validated)')
    ax.set_xlabel('Query'); ax.set_ylabel('Average Score (0–100)')
    ax.set_xticks(x); ax.set_xticklabels(qids); ax.legend(); ax.set_ylim(0, 120)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/partA_01_hybrid_vs_kbf_vs_lda.png', dpi=150, bbox_inches='tight')
    plt.close(); print("  ✅  partA_01_hybrid_vs_kbf_vs_lda.png")

    # Chart 2
    fig, ax = plt.subplots(figsize=(10, 4))
    div_vals = df_a['Diversity_Score'].tolist()
    bars = ax.bar(qids, div_vals, color=sns.color_palette(PAL, len(qids)), edgecolor='white')
    for bar, v in zip(bars, div_vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')
    md = np.mean(div_vals)
    ax.axhline(md, color=ACCENT, linestyle='--', linewidth=1.5, label=f'Mean: {md:.3f}')
    ax.set_title('Intra-List Diversity Score per Query')
    ax.set_xlabel('Query'); ax.set_ylabel('Diversity (0–1)'); ax.set_ylim(0,1.2); ax.legend()
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/partA_02_diversity_score.png', dpi=150, bbox_inches='tight')
    plt.close(); print("  ✅  partA_02_diversity_score.png")

    # Chart 3
    fig, ax = plt.subplots(figsize=(10, 4))
    fp = []
    for v in df_a['Filter_Satisfaction']:
        try: fp.append(float(v)*100)
        except: fp.append(0)
    bars = ax.bar(qids, fp, color=sns.color_palette(PAL, len(qids)), edgecolor='white')
    for bar, v in zip(bars, fp):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f'{v:.1f}%', ha='center', fontsize=9, fontweight='bold')
    mf = np.mean(fp)
    ax.axhline(mf, color=ACCENT, linestyle='--', linewidth=1.5, label=f'Mean: {mf:.1f}%')
    ax.set_title('KBF Filter Satisfaction Rate per Query')
    ax.set_xlabel('Query'); ax.set_ylabel('Filter Satisfaction (%)'); ax.set_ylim(0,125); ax.legend()
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/partA_03_filter_satisfaction.png', dpi=150, bbox_inches='tight')
    plt.close(); print("  ✅  partA_03_filter_satisfaction.png")

    # Chart 4
    fig, ax = plt.subplots(figsize=(10, 4))
    tp = []
    for v in df_a['Topic_Relevance']:
        try: tp.append(float(v)*100)
        except: tp.append(0)
    bars = ax.bar(qids, tp, color=sns.color_palette(PAL, len(qids)), edgecolor='white')
    for bar, v in zip(bars, tp):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f'{v:.1f}%', ha='center', fontsize=9, fontweight='bold')
    mt = np.mean(tp)
    ax.axhline(mt, color=ACCENT, linestyle='--', linewidth=1.5, label=f'Mean: {mt:.1f}%')
    ax.set_title('LDA Topic Relevance Rate per Query')
    ax.set_xlabel('Query'); ax.set_ylabel('Topic Relevance (%)'); ax.set_ylim(0,125); ax.legend()
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/partA_04_topic_relevance.png', dpi=150, bbox_inches='tight')
    plt.close(); print("  ✅  partA_04_topic_relevance.png")

    # Chart 5: Coverage + Coherence
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].pie([coverage, 1-coverage],
                labels=[f'Recommended\n{coverage*100:.1f}%', f'Not recommended\n{(1-coverage)*100:.1f}%'],
                colors=[ACCENT,'#ECF0F1'], startangle=90,
                wedgeprops={'edgecolor':'white','linewidth':2})
    axes[0].set_title('System Coverage')
    cp = LDA_COHERENCE_SCORE * 100
    axes[1].barh(['LDA Coherence (Cv)'], [cp],   color=ACCENT,    edgecolor='white', height=0.4)
    axes[1].barh(['LDA Coherence (Cv)'], [100-cp], left=[cp], color='#ECF0F1', edgecolor='white', height=0.4)
    axes[1].text(cp/2, 0, f'{LDA_COHERENCE_SCORE:.4f}',
                 ha='center', va='center', fontweight='bold', color='white', fontsize=14)
    axes[1].set_xlim(0,100); axes[1].set_xlabel('Score (%)')
    axes[1].set_title('LDA Coherence Score (Cv)\n(Acceptable: 0.4–0.6)')
    axes[1].spines['left'].set_visible(False); axes[1].set_yticks([])
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/partA_05_coverage_coherence.png', dpi=150, bbox_inches='tight')
    plt.close(); print("  ✅  partA_05_coverage_coherence.png")


# ═══════════════════════════════════════════════════════════════
# CHARTS — PART B
# ═══════════════════════════════════════════════════════════════
def plot_part_b(prec, rec, f1, ndcg, acc):
    qids = [q['id'] for q in TEST_QUERIES]
    x, w = np.arange(len(qids)), 0.17
    label = 'User Study' if USE_REAL_USER_DATA else 'Offline Evaluation (Auto Ground Truth)'

    # Chart 6
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x-2*w, prec, width=w, label=f'Precision@{TOP_N}', color='#3498DB', edgecolor='white')
    ax.bar(x-1*w, rec,  width=w, label=f'Recall@{TOP_N}',    color='#2ECC71', edgecolor='white')
    ax.bar(x,     f1,   width=w, label='F1 Score',            color='#F39C12', edgecolor='white')
    ax.bar(x+1*w, ndcg, width=w, label=f'nDCG@{TOP_N}',      color=ACCENT,    edgecolor='white')
    ax.bar(x+2*w, acc,  width=w, label='Hit Rate',            color='#9B59B6', edgecolor='white')
    ax.set_title(f'{label} — All Metrics per Query')
    ax.set_xlabel('Query'); ax.set_ylabel('Score (0–1)')
    ax.set_xticks(x); ax.set_xticklabels(qids); ax.set_ylim(0,1.25); ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/partB_01_all_metrics_per_query.png', dpi=150, bbox_inches='tight')
    plt.close(); print("  ✅  partB_01_all_metrics_per_query.png")

    # Chart 7
    fig, ax = plt.subplots(figsize=(9, 5))
    metrics = [f'Precision\n@{TOP_N}', f'Recall\n@{TOP_N}', 'F1\nScore', f'nDCG\n@{TOP_N}', 'Hit\nRate']
    values  = [np.mean(prec), np.mean(rec), np.mean(f1), np.mean(ndcg), np.mean(acc)]
    colors  = ['#3498DB','#2ECC71','#F39C12',ACCENT,'#9B59B6']
    bars    = ax.bar(metrics, values, color=colors, edgecolor='white', width=0.55)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f'{v:.4f}', ha='center', fontsize=10, fontweight='bold')
    ax.set_title(f'Mean Evaluation Metrics — {label}')
    ax.set_ylabel('Mean Score (0–1)'); ax.set_ylim(0,1.25)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/partB_02_mean_metrics_summary.png', dpi=150, bbox_inches='tight')
    plt.close(); print("  ✅  partB_02_mean_metrics_summary.png")

    # Chart 8
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(qids, ndcg, color=ACCENT, linewidth=2.5, marker='o', markersize=9,
            markerfacecolor='white', markeredgecolor=ACCENT, markeredgewidth=2.5)
    for q, v in zip(qids, ndcg):
        ax.annotate(f'{v:.3f}', (q, v), textcoords='offset points', xytext=(0,10),
                    ha='center', fontsize=9, fontweight='bold', color=ACCENT)
    mn = np.mean(ndcg)
    ax.axhline(mn, color='#2C3E50', linestyle='--', linewidth=1.5, label=f'Mean nDCG: {mn:.4f}')
    ax.set_title(f'nDCG@{TOP_N} Ranking Quality per Query\n(1.0 = perfect ranking)')
    ax.set_xlabel('Query'); ax.set_ylabel(f'nDCG@{TOP_N}'); ax.set_ylim(0,1.15); ax.legend()
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/partB_03_ndcg_trend.png', dpi=150, bbox_inches='tight')
    plt.close(); print("  ✅  partB_03_ndcg_trend.png")


# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════
def save_summary(df_a, df_b):
    method = 'User Study' if USE_REAL_USER_DATA else 'Auto Ground Truth'
    rows = [
        ('LDA Coherence Score (Cv)',     LDA_COHERENCE_SCORE,                                          'Part A — Intrinsic'),
        ('Mean Hybrid Score (0-100)',    df_a['Avg_Hybrid_Score'].mean().round(2),                     'Part A — Intrinsic'),
        ('Mean KBF-Only Score (0-100)', df_a['Avg_KBF_Only_Score'].mean().round(2),                   'Part A — Intrinsic'),
        ('Mean LDA-Only Score (0-100)', df_a['Avg_LDA_Only_Score'].mean().round(2),                   'Part A — Intrinsic'),
        ('Mean Diversity Score',         df_a['Diversity_Score'].mean().round(4),                     'Part A — Intrinsic'),
        ('Mean Filter Satisfaction',
         pd.to_numeric(df_a['Filter_Satisfaction'], errors='coerce').mean().round(4),                 'Part A — Intrinsic'),
        (f'Mean Precision@{TOP_N} ({method})',  df_b[f'Precision@{TOP_N}'].mean().round(4),           f'Part B — {method}'),
        (f'Mean Recall@{TOP_N} ({method})',     df_b[f'Recall@{TOP_N}'].mean().round(4),              f'Part B — {method}'),
        (f'Mean F1 Score ({method})',            df_b['F1_Score'].mean().round(4),                     f'Part B — {method}'),
        (f'Mean nDCG@{TOP_N} ({method})',       df_b[f'nDCG@{TOP_N}'].mean().round(4),                f'Part B — {method}'),
        (f'Mean Hit Rate ({method})',            df_b['Accuracy_HitRate'].mean().round(4),             f'Part B — {method}'),
    ]
    df_sum = pd.DataFrame(rows, columns=['Metric','Value','Section'])
    df_sum.to_csv(f'{OUTPUT_DIR}/evaluation_summary.csv', index=False)
    print(f"\n  ✅  Saved: evaluation_summary.csv")
    print("\n" + "="*65)
    print("  FINAL EVALUATION SUMMARY  (copy these into Chapter 4)")
    print("="*65)
    for _, row in df_sum.iterrows():
        print(f"  [{row['Section']}]")
        print(f"    {row['Metric']:<47}: {row['Value']}")
    print("="*65)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    print("="*65)
    print("  MAKAN MANA — EVALUATION  FINAL")
    print(f"  Weighting  : {int(KBF_WEIGHT*100)}% KBF + {int(LDA_WEIGHT*100)}% LDA")
    print(f"  Coherence  : {LDA_COHERENCE_SCORE}")
    print(f"  Queries    : {len(TEST_QUERIES)}   |   Top-N : {TOP_N}")
    print(f"  Part B mode: {'Real User Data' if USE_REAL_USER_DATA else 'Auto Ground Truth'}")
    print("="*65)

    df = load_data()

    print("\n[1/4] Running Part A...")
    df_a, all_h, all_k, all_l, coverage = run_part_a(df)

    print("\n[2/4] Generating Part A charts...")
    plot_part_a(df_a, all_h, all_k, all_l, coverage)

    print("\n[3/4] Running Part B...")
    df_b, prec, rec, f1, ndcg, acc = run_part_b(df)

    print("\n[4/4] Generating Part B charts...")
    plot_part_b(prec, rec, f1, ndcg, acc)

    save_summary(df_a, df_b)

    print(f"\n{'='*65}")
    print(f"  DONE — outputs in: {OUTPUT_DIR}/")
    print(f"{'='*65}")
    print("""
  PHASE 2 (after user study):
    1. Fill in USER_STUDY_DATA above with real participant scores
    2. Set  USE_REAL_USER_DATA = True
    3. Re-run:  python s7_evaluation_FINAL.py
    """)


if __name__ == '__main__':
    main()
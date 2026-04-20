"""
STEP 4 — KBF Filtering
=======================
Extracts boolean Knowledge-Based Filter attributes from review text.
Attributes: is_halal, is_vegetarian, is_vegan, has_parking,
            is_family_friendly, is_romantic, has_scenic_view,
            has_outdoor, has_wifi

Input : master_990_lda.csv
Output: master_990_kbf.csv

Run: python s4_kbf.py
"""

import pandas as pd
import re

IN_FILE  = 'master_990_lda.csv'
OUT_FILE = 'master_990_kbf.csv'

# Knowledge-Based Filtering (KBF) Patterns for Restaurant Recommendations
# Optimized for Malaysian context (English, Manglish, and Malay terms)
KBF_PATTERNS = {
    # --- DIETARY & RELIGIOUS ---
    'is_halal': [
        r'\bhalal\b', r'\bislamic\b', r'\bmuslim friendly\b',
        r'\bno pork\b', r'\bno alcohol\b', r'\bhal[aâ]l\b', r'\bsijil halal\b'
    ],
    'is_vegetarian': [
        r'\bvegetarian\b', r'\bvege\b', r'\bno meat\b',
        r'\bplant.based\b', r'\bveggies\b', r'\bsayur-sayuran\b'
    ],
    'is_vegan': [
        r'\bvegan\b', r'\bno animal\b', r'\bdairy.free\b', r'\beggless\b'
    ],

    # --- ACCESSIBILITY & INCLUSION ---
    'is_accessible': [
        r'\bwheelchair\b', r'\bdisable(d)? friendly\b', r'\bramp\b',
        r'\blift\b', r'\belevator\b', r'\bokut\b', r'\bhandicap\b',
        r'\beasy access\b', r'\bstair-free\b', r'\bground floor\b'
    ],

    # --- FACILITIES & CONVENIENCE ---
    'has_parking': [
        r'\bparking\b', r'\bcar park\b', r'\bparked\b', r'\bvalet\b',
        r'\bample parking\b', r'\beasy to park\b', r'\bparking lot\b'
    ],
    'has_wifi': [
        r'\bwifi\b', r'\bwi-fi\b', r'\bwireless\b', r'\binternet\b',
        r'\bfree wifi\b', r'\bgood wifi\b'
    ],
    'has_ac': [
        r'\baircon\b', r'\bair conditioning\b', r'\bac\b', r'\bcold\b',
        r'\bcool\b', r'\bchilly\b', r'\bsejuk\b'
    ],
    'has_outdoor': [
        r'\boutdoor\b', r'\bal fresco\b', r'\bopen air\b',
        r'\boutside seating\b', r'\bterrace\b', r'\bgarden seating\b'
    ],

    # --- VIBES & OCCASIONS ---
    'is_casual': [
        r'\bcasual\b', r'\brelaxed\b', r'\bchill\b', r'\bsantai\b', 
        r'\bno dress code\b', r'\beasygoing\b', r'\bhumble\b', 
        r'\bsimple\b', r'\bcozy\b', r'\bcosy\b', r'\blepak\b'
    ],
    'is_group_friendly': [
        r'\bgroup\b', r'\bgathering\b', r'\bbig group\b', r'\bfriends\b',
        r'\blarge party\b', r'\bcelebration\b', r'\bcolleagues\b',
        r'\bcompany dinner\b', r'\bbig table\b', r'\bmeja besar\b', 
        r'\bberamai-ramai\b'
    ],
    'is_family_friendly': [
        r'\bfamily\b', r'\bkids\b', r'\bchildren\b', r'\bbaby chair\b', 
        r'\bhigh chair\b', r'\bchild friendly\b', r'\bchild chair\b',
        r'\bbring kids\b'
    ],
    'is_romantic': [
        r'\bromantic\b', r'\bdate night\b', r'\bcouple\b',
        r'\bintimate\b', r'\bcandle\b', r'\bdating\b', r'\bdinner for two\b'
    ],
    'has_scenic_view': [
        r'\bscenic\b', r'\bbeautiful view\b', r'\bsea view\b',
        r'\bocean view\b', r'\bskyline\b', r'\bcityscape\b', r'\bsunset\b'
    ],

    # --- SERVICE & VALUE ---
    'is_worth_it': [
        r'\bworth it\b', r'\bvalue for money\b', r'\baffordable\b',
        r'\bcheap\b', r'\breasonable price\b', r'\bbudget\b', r'\bmurah\b'
    ],
    'is_fast_service': [
        r'\bfast\b', r'\bquick\b', r'\bno wait\b', r'\befficient\b',
        r'\bpantas\b', r'\bexpress\b', r'\bspeedy\b'
    ],
    'is_crowded': [
        r'\bqueue\b', r'\bcrowded\b', r'\bfull house\b', r'\blong wait\b',
        r'\bsesak\b', r'\bpacked\b', r'\breservation needed\b'
    ]
}

def extract_kbf(text):
    """Extract KBF boolean attributes from review text."""
    result = {attr: False for attr in KBF_PATTERNS}
    if not text or pd.isna(text):
        return result
    text_lower = str(text).lower()
    for attr, patterns in KBF_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                result[attr] = True
                break
    return result

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading LDA output...")
df = pd.read_csv(IN_FILE)
print(f"  Total restaurants: {len(df)}")

# ── Extract KBF ───────────────────────────────────────────────────────────────
print("\nExtracting KBF attributes from reviews...")
kbf_results = df['review_text'].apply(extract_kbf)
kbf_df      = pd.DataFrame(kbf_results.tolist())
for col in kbf_df.columns:
    df[col] = kbf_df[col]

# ── Save ──────────────────────────────────────────────────────────────────────
df.to_csv(OUT_FILE, index=False, encoding='utf-8-sig')

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  STEP 4 COMPLETE — KBF Attribute Counts")
print(f"{'='*55}")
for attr in KBF_PATTERNS:
    count = df[attr].sum()
    pct   = round(count / len(df) * 100, 1)
    print(f"  {attr:<22}: {count:>4} ({pct}%)")
print(f"\n  Output: {OUT_FILE}")
print(f"{'='*55}")
print(f"\nNext: Run s5_hybrid.py")
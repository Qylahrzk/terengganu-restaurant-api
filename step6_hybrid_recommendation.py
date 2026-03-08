import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
INPUT_KBF    = 'kbf_outputs/kbf_restaurant_profiles.csv'
INPUT_LABELS = 'lda_outputs/lda_topic_labels.csv'
OUTPUT_DIR   = 'recommendation_outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hybrid weighting — based on FYP proposal:
# "Theme/vibe is more dominant than food type"
KBF_WEIGHT = 0.30   # 30% — practical filters (district, cuisine, rating)
LDA_WEIGHT = 0.70   # 70% — thematic/social context from reviews

TOP_N = 10          # number of recommendations to return


# ============================================================
# STEP 1: Load Data
# ============================================================
print("=" * 60)
print("  HYBRID RESTAURANT RECOMMENDER — TERENGGANU")
print(f"  Weighting: {int(KBF_WEIGHT*100)}% KBF + {int(LDA_WEIGHT*100)}% LDA")
print("=" * 60)

df = pd.read_csv(INPUT_KBF)
df_labels = pd.read_csv(INPUT_LABELS)

# Build topic label → topic ID mapping
topic_label_to_id = dict(zip(df_labels['Label'], df_labels['Topic_ID'] + 1))
topic_id_to_label = dict(zip(df_labels['Topic_ID'] + 1, df_labels['Label']))

print(f"\n✅ Loaded {len(df):,} restaurant profiles")
print(f"✅ Available topic labels: {list(topic_label_to_id.keys())}\n")


# ============================================================
# STEP 2: KBF SCORING FUNCTION
# Scores each restaurant based on how well it matches
# the user's explicit filter preferences
# ============================================================
def compute_kbf_score(df, preferences):
    """
    Score each restaurant based on KBF filter matches.
    Each matched filter contributes equally to the KBF score.
    Returns a Series of scores between 0.0 and 1.0
    """
    scores     = pd.Series(0.0, index=df.index)
    max_points = 0

    # --- District filter ---
    if preferences.get('district'):
        max_points += 1
        match = df['Municipality'].str.lower() == preferences['district'].lower()
        scores += match.astype(float)

    # --- Cuisine filter ---
    if preferences.get('cuisine'):
        max_points += 1
        match = df['Cuisine_Type'].str.lower() == preferences['cuisine'].lower()
        scores += match.astype(float)

    # --- Minimum rating filter ---
    if preferences.get('min_rating'):
        max_points += 1
        match = df['Rating'] >= float(preferences['min_rating'])
        scores += match.astype(float)

    # --- Halal filter ---
    if preferences.get('halal'):
        max_points += 1
        match = df['Is_Halal'] == 'Yes'
        scores += match.astype(float)

    # --- Vegetarian filter ---
    if preferences.get('vegetarian'):
        max_points += 1
        match = df['Is_Vegetarian'] == 'Yes'
        scores += match.astype(float)

    # --- Vegan filter ---
    if preferences.get('vegan'):
        max_points += 1
        match = df['Is_Vegan'] == 'Yes'
        scores += match.astype(float)

    # --- Parking filter ---
    if preferences.get('parking'):
        max_points += 1
        match = df['Has_Parking'] == 'Yes'
        scores += match.astype(float)

    # --- Family-friendly filter ---
    if preferences.get('family_friendly'):
        max_points += 1
        match = df['Is_Family_Friendly'] == 'Yes'
        scores += match.astype(float)

    # --- Romantic filter ---
    if preferences.get('romantic'):
        max_points += 1
        match = df['Is_Romantic'] == 'Yes'
        scores += match.astype(float)

    # --- Scenic view filter ---
    if preferences.get('scenic_view'):
        max_points += 1
        match = df['Has_Scenic_View'] == 'Yes'
        scores += match.astype(float)

    # --- Outdoor seating filter ---
    if preferences.get('outdoor'):
        max_points += 1
        match = df['Has_Outdoor'] == 'Yes'
        scores += match.astype(float)

    # --- WiFi filter ---
    if preferences.get('wifi'):
        max_points += 1
        match = df['Has_Wifi'] == 'Yes'
        scores += match.astype(float)

    # Normalize to 0.0 – 1.0
    if max_points > 0:
        scores = scores / max_points
    else:
        # No filters set — use normalized rating as KBF score
        scores = df['Rating'] / 5.0

    return scores


# ============================================================
# STEP 3: LDA SCORING FUNCTION
# Scores each restaurant based on how well its review
# topics match the user's preferred theme/vibe
# ============================================================
def compute_lda_score(df, preferred_topic_id):
    """
    Score each restaurant based on topic relevance.
    Uses Topic_1_Pct (dominant topic percentage) as the signal.
    Returns a Series of scores between 0.0 and 1.0
    """
    if preferred_topic_id is None:
        # No topic preference — use dominant topic percentage as neutral signal
        return (df['Topic_1_Pct'].fillna(0) / 100.0)

    scores = pd.Series(0.0, index=df.index)

    # Primary match — dominant topic matches preferred topic
    primary_match = df['Dominant_Topic'] == preferred_topic_id
    scores += primary_match.astype(float) * 1.0

    # Secondary match — preferred topic appears as Topic 2 or 3
    col_map = {
        'Topic_2_ID': 'Topic_2_Pct',
        'Topic_3_ID': 'Topic_3_Pct'
    }
    for id_col, pct_col in col_map.items():
        if id_col in df.columns and pct_col in df.columns:
            secondary = df[id_col] == preferred_topic_id
            # Weight secondary matches by their percentage contribution
            scores += secondary.astype(float) * (df[pct_col].fillna(0) / 100.0) * 0.5

    # Boost by the actual topic percentage for dominant topic matches
    scores = scores * (df['Topic_1_Pct'].fillna(50) / 100.0 + 0.5)

    # Normalize to 0.0 – 1.0
    max_score = scores.max()
    if max_score > 0:
        scores = scores / max_score

    return scores


# ============================================================
# STEP 4: HYBRID SCORING FUNCTION
# Combines KBF and LDA scores using configured weights
# ============================================================
def compute_hybrid_score(kbf_scores, lda_scores, rating_series):
    """
    Final hybrid score = KBF_WEIGHT * KBF + LDA_WEIGHT * LDA
    Rating is used as a tiebreaker boost (±5%)
    """
    hybrid = (KBF_WEIGHT * kbf_scores) + (LDA_WEIGHT * lda_scores)

    # Add small rating boost as tiebreaker (max 5% boost)
    rating_boost = (rating_series.fillna(3.0) / 5.0) * 0.05
    hybrid = hybrid + rating_boost

    # Normalize final score to 0–100 for readability
    max_score = hybrid.max()
    if max_score > 0:
        hybrid = (hybrid / max_score) * 100

    return hybrid.round(2)


# ============================================================
# STEP 5: FALLBACK — Relax Filters
# If no exact matches found, progressively relax filters
# ============================================================
def relax_and_retry(df, preferences, min_results=5):
    """
    Progressively relax filters until at least min_results restaurants found.
    Relaxation order: optional filters first, then cuisine, then district last.
    """
    relaxation_order = [
        'wifi', 'vegan', 'outdoor', 'scenic_view',
        'romantic', 'parking', 'family_friendly',
        'vegetarian', 'halal', 'cuisine', 'min_rating'
    ]

    relaxed_prefs  = preferences.copy()
    relaxed_flags  = []
    filtered       = df.copy()

    for key in relaxation_order:
        # Apply current preferences as hard filter
        mask = pd.Series(True, index=df.index)

        if relaxed_prefs.get('district'):
            mask &= df['Municipality'].str.lower() == relaxed_prefs['district'].lower()
        if relaxed_prefs.get('cuisine'):
            mask &= df['Cuisine_Type'].str.lower() == relaxed_prefs['cuisine'].lower()
        if relaxed_prefs.get('min_rating'):
            mask &= df['Rating'] >= float(relaxed_prefs['min_rating'])
        if relaxed_prefs.get('halal'):
            mask &= df['Is_Halal'] == 'Yes'
        if relaxed_prefs.get('vegetarian'):
            mask &= df['Is_Vegetarian'] == 'Yes'
        if relaxed_prefs.get('vegan'):
            mask &= df['Is_Vegan'] == 'Yes'
        if relaxed_prefs.get('parking'):
            mask &= df['Has_Parking'] == 'Yes'
        if relaxed_prefs.get('family_friendly'):
            mask &= df['Is_Family_Friendly'] == 'Yes'
        if relaxed_prefs.get('romantic'):
            mask &= df['Is_Romantic'] == 'Yes'
        if relaxed_prefs.get('scenic_view'):
            mask &= df['Has_Scenic_View'] == 'Yes'
        if relaxed_prefs.get('outdoor'):
            mask &= df['Has_Outdoor'] == 'Yes'
        if relaxed_prefs.get('wifi'):
            mask &= df['Has_Wifi'] == 'Yes'

        filtered = df[mask]

        if len(filtered) >= min_results:
            break

        # Relax the next filter in order
        if relaxed_prefs.get(key):
            relaxed_flags.append(key)
            relaxed_prefs.pop(key)

    return filtered, relaxed_prefs, relaxed_flags


# ============================================================
# STEP 6: MAIN RECOMMEND FUNCTION
# ============================================================
def recommend(preferences, verbose=True):
    """
    Main hybrid recommendation function.

    Parameters:
    -----------
    preferences : dict with any of these keys:
        district        : str  e.g. 'Kuala Terengganu'
        cuisine         : str  e.g. 'Seafood'
        min_rating      : float e.g. 4.0
        preferred_topic : str  e.g. 'Traditional Malay Food'
        halal           : bool
        vegetarian      : bool
        vegan           : bool
        parking         : bool
        family_friendly : bool
        romantic        : bool
        scenic_view     : bool
        outdoor         : bool
        wifi            : bool

    Returns:
    --------
    DataFrame of top N recommended restaurants with scores
    """

    if verbose:
        print("\n" + "=" * 60)
        print("  USER PREFERENCES")
        print("=" * 60)
        for k, v in preferences.items():
            print(f"  {k:<20} : {v}")
        print()

    # --- Resolve preferred topic ---
    preferred_topic_id = None
    if preferences.get('preferred_topic'):
        preferred_topic_id = topic_label_to_id.get(preferences['preferred_topic'])
        if preferred_topic_id is None:
            if verbose:
                print(f"  ⚠️  Topic '{preferences['preferred_topic']}' not found.")
                print(f"  Available: {list(topic_label_to_id.keys())}\n")

    # --- Apply hard filters (district is always kept) ---
    df_filtered = df.copy()
    relaxed_flags = []

    # Apply district as strict filter always
    if preferences.get('district'):
        df_district = df_filtered[
            df_filtered['Municipality'].str.lower() == preferences['district'].lower()
        ]
        # If district returns too few, keep all
        if len(df_district) >= 3:
            df_filtered = df_district
        else:
            if verbose:
                print(f"  ⚠️  District '{preferences['district']}' has fewer than 3 restaurants.")
                print(f"      Searching across all districts instead.\n")

    # Apply remaining filters
    prefs_no_district = {k: v for k, v in preferences.items()
                         if k not in ['district', 'preferred_topic']}

    # Check how many restaurants match all filters
    test_mask = pd.Series(True, index=df_filtered.index)
    if prefs_no_district.get('cuisine'):
        test_mask &= df_filtered['Cuisine_Type'].str.lower() == prefs_no_district['cuisine'].lower()
    if prefs_no_district.get('min_rating'):
        test_mask &= df_filtered['Rating'] >= float(prefs_no_district['min_rating'])
    if prefs_no_district.get('halal'):
        test_mask &= df_filtered['Is_Halal'] == 'Yes'
    if prefs_no_district.get('vegetarian'):
        test_mask &= df_filtered['Is_Vegetarian'] == 'Yes'
    if prefs_no_district.get('vegan'):
        test_mask &= df_filtered['Is_Vegan'] == 'Yes'
    if prefs_no_district.get('parking'):
        test_mask &= df_filtered['Has_Parking'] == 'Yes'
    if prefs_no_district.get('family_friendly'):
        test_mask &= df_filtered['Is_Family_Friendly'] == 'Yes'
    if prefs_no_district.get('romantic'):
        test_mask &= df_filtered['Is_Romantic'] == 'Yes'
    if prefs_no_district.get('scenic_view'):
        test_mask &= df_filtered['Has_Scenic_View'] == 'Yes'
    if prefs_no_district.get('outdoor'):
        test_mask &= df_filtered['Has_Outdoor'] == 'Yes'
    if prefs_no_district.get('wifi'):
        test_mask &= df_filtered['Has_Wifi'] == 'Yes'

    exact_matches = df_filtered[test_mask]

    # If too few exact matches — relax filters
    if len(exact_matches) < TOP_N:
        if verbose:
            print(f"  ℹ️  Only {len(exact_matches)} exact matches found.")
            print(f"      Relaxing filters to find closest matches...\n")
        df_filtered, relaxed_prefs, relaxed_flags = relax_and_retry(
            df_filtered, prefs_no_district, min_results=TOP_N
        )
        if relaxed_flags and verbose:
            print(f"  🔄 Relaxed filters: {relaxed_flags}\n")
    else:
        df_filtered = exact_matches

    if len(df_filtered) == 0:
        if verbose:
            print("  ❌ No restaurants found even after relaxing filters.")
            print("     Returning top-rated restaurants overall.\n")
        df_filtered = df.nlargest(TOP_N, 'Rating')

    # --- Compute scores ---
    kbf_scores    = compute_kbf_score(df_filtered, preferences)
    lda_scores    = compute_lda_score(df_filtered, preferred_topic_id)
    hybrid_scores = compute_hybrid_score(kbf_scores, lda_scores, df_filtered['Rating'])

    df_filtered = df_filtered.copy()
    df_filtered['KBF_Score']    = (kbf_scores * 100).round(2)
    df_filtered['LDA_Score']    = (lda_scores * 100).round(2)
    df_filtered['Hybrid_Score'] = hybrid_scores

    # --- Rank and return top N ---
    results = (df_filtered
               .sort_values('Hybrid_Score', ascending=False)
               .head(TOP_N)
               .reset_index(drop=True))
    results.index += 1   # rank starts at 1

    # --- Display results ---
    if verbose:
        print("=" * 60)
        print(f"  TOP {TOP_N} RECOMMENDED RESTAURANTS")
        if relaxed_flags:
            print(f"  (Filters relaxed: {', '.join(relaxed_flags)})")
        print("=" * 60)

        for rank, row in results.iterrows():
            print(f"\n  #{rank}  {row['Name']}")
            print(f"       📍 {row['Municipality']}  |  🍽️  {row['Cuisine_Type']}  |  ⭐ {row['Rating']}★")
            print(f"       🏷️  Topic: {row['Topic_Label']}")
            print(f"       📊 Hybrid: {row['Hybrid_Score']:.1f}  |  KBF: {row['KBF_Score']:.1f}  |  LDA: {row['LDA_Score']:.1f}")

            # Show matched filters
            flags = []
            if row.get('Is_Halal')         == 'Yes': flags.append('✅ Halal')
            if row.get('Is_Family_Friendly')== 'Yes': flags.append('👨‍👩‍👧 Family')
            if row.get('Has_Parking')       == 'Yes': flags.append('🅿️ Parking')
            if row.get('Is_Romantic')       == 'Yes': flags.append('💑 Romantic')
            if row.get('Has_Scenic_View')   == 'Yes': flags.append('🌊 Scenic')
            if row.get('Has_Outdoor')       == 'Yes': flags.append('🌿 Outdoor')
            if row.get('Has_Wifi')          == 'Yes': flags.append('📶 WiFi')
            if row.get('Is_Vegetarian')     == 'Yes': flags.append('🥗 Vegetarian')
            if flags:
                print(f"       {' | '.join(flags)}")

        print("\n" + "=" * 60)

    return results


# ============================================================
# STEP 7: SAVE RESULTS FUNCTION
# ============================================================
def save_results(results, query_name='query'):
    """Save recommendation results to CSV."""
    save_cols = [
        'Name', 'Municipality', 'Cuisine_Type', 'Rating',
        'Topic_Label', 'Hybrid_Score', 'KBF_Score', 'LDA_Score',
        'Is_Halal', 'Is_Family_Friendly', 'Has_Parking',
        'Is_Romantic', 'Has_Scenic_View', 'Has_Outdoor', 'Has_Wifi',
        'Is_Vegetarian', 'Is_Vegan', 'Address'
    ]
    filename = f'{OUTPUT_DIR}/{query_name}_recommendations.csv'
    results[save_cols].to_csv(filename, index_label='Rank')
    print(f"\n✅ Results saved: {filename}\n")


# ============================================================
# STEP 8: RUN SAMPLE QUERIES
# ============================================================
if __name__ == '__main__':

    print("\n" + "🔎 " * 20)
    print("  RUNNING SAMPLE RECOMMENDATION QUERIES")
    print("🔎 " * 20)

    # ----------------------------------------------------------
    # Query 1: Family outing in Kuala Terengganu
    # ----------------------------------------------------------
    query1 = {
        'district'        : 'Kuala Terengganu',
        'cuisine'         : 'Seafood',
        'min_rating'      : 4.0,
        'preferred_topic' : 'Seafood & Local Snacks',
        'halal'           : True,
        'family_friendly' : True,
        'parking'         : True,
    }
    results1 = recommend(query1)
    save_results(results1, 'query1_family_seafood_KT')

    # ----------------------------------------------------------
    # Query 2: Romantic dinner in Besut with scenic view
    # ----------------------------------------------------------
    query2 = {
        'district'        : 'Besut',
        'min_rating'      : 4.0,
        'preferred_topic' : 'Location & Ambiance',
        'romantic'        : True,
        'scenic_view'     : True,
        'halal'           : True,
    }
    results2 = recommend(query2)
    save_results(results2, 'query2_romantic_besut')

    # ----------------------------------------------------------
    # Query 3: Traditional Malay food in Dungun
    # ----------------------------------------------------------
    query3 = {
        'district'        : 'Dungun',
        'cuisine'         : 'Malay',
        'min_rating'      : 3.5,
        'preferred_topic' : 'Traditional Malay Food',
        'halal'           : True,
    }
    results3 = recommend(query3)
    save_results(results3, 'query3_malay_dungun')

    # ----------------------------------------------------------
    # Query 4: Western food cafe in Kemaman
    # ----------------------------------------------------------
    query4 = {
        'district'        : 'Kemaman',
        'cuisine'         : 'Western',
        'min_rating'      : 4.0,
        'preferred_topic' : 'Western & Fusion Food',
        'parking'         : True,
    }
    results4 = recommend(query4)
    save_results(results4, 'query4_western_kemaman')

    print("=" * 60)
    print("  ALL QUERIES COMPLETE")
    print(f"  Results saved in /{OUTPUT_DIR}/")
    print("=" * 60)
    print("\n  ➡  Next step: step7_evaluation.py\n")


# ============================================================
# HOW TO USE IN YOUR OWN CODE
# ============================================================
# from step6_hybrid_recommendation import recommend, save_results
#
# my_preferences = {
#     'district'        : 'Kuala Terengganu',
#     'cuisine'         : 'Malay',
#     'min_rating'      : 4.0,
#     'preferred_topic' : 'Traditional Malay Food',
#     'halal'           : True,
#     'family_friendly' : True,
# }
#
# results = recommend(my_preferences)
# save_results(results, 'my_query')
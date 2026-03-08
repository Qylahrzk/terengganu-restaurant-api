import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
import os
import sys
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
INPUT_KBF    = 'kbf_outputs/kbf_restaurant_profiles.csv'
INPUT_LABELS = 'lda_outputs/lda_topic_labels.csv'
OUTPUT_DIR   = 'evaluation_outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

KBF_WEIGHT = 0.30
LDA_WEIGHT = 0.70
TOP_N      = 10
LDA_COHERENCE_SCORE = 0.5968   # from step4 output

ACCENT  = '#C0392B'
BG      = '#FAFAFA'
PALETTE = 'YlOrRd'

plt.rcParams.update({
    'figure.facecolor' : BG,
    'axes.facecolor'   : BG,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'font.family'      : 'DejaVu Sans',
    'axes.titlesize'   : 13,
    'axes.titleweight' : 'bold',
    'axes.labelsize'   : 11,
})


# ============================================================
# LOAD DATA & SCORING FUNCTIONS (from step6)
# ============================================================
df       = pd.read_csv(INPUT_KBF)
df_labels = pd.read_csv(INPUT_LABELS)

topic_label_to_id = dict(zip(df_labels['Label'], df_labels['Topic_ID'] + 1))
topic_id_to_label = dict(zip(df_labels['Topic_ID'] + 1, df_labels['Label']))


def compute_kbf_score(data, preferences):
    scores     = pd.Series(0.0, index=data.index)
    max_points = 0
    checks = [
        ('district',       lambda d: d['Municipality'].str.lower() == preferences['district'].lower()),
        ('cuisine',        lambda d: d['Cuisine_Type'].str.lower() == preferences['cuisine'].lower()),
        ('min_rating',     lambda d: d['Rating'] >= float(preferences['min_rating'])),
        ('halal',          lambda d: d['Is_Halal'] == 'Yes'),
        ('vegetarian',     lambda d: d['Is_Vegetarian'] == 'Yes'),
        ('vegan',          lambda d: d['Is_Vegan'] == 'Yes'),
        ('parking',        lambda d: d['Has_Parking'] == 'Yes'),
        ('family_friendly',lambda d: d['Is_Family_Friendly'] == 'Yes'),
        ('romantic',       lambda d: d['Is_Romantic'] == 'Yes'),
        ('scenic_view',    lambda d: d['Has_Scenic_View'] == 'Yes'),
        ('outdoor',        lambda d: d['Has_Outdoor'] == 'Yes'),
        ('wifi',           lambda d: d['Has_Wifi'] == 'Yes'),
    ]
    for key, fn in checks:
        if preferences.get(key):
            max_points += 1
            scores += fn(data).astype(float)
    if max_points > 0:
        scores = scores / max_points
    else:
        scores = data['Rating'] / 5.0
    return scores


def compute_lda_score(data, preferred_topic_id):
    if preferred_topic_id is None:
        return data['Topic_1_Pct'].fillna(0) / 100.0
    scores = pd.Series(0.0, index=data.index)
    primary = data['Dominant_Topic'] == preferred_topic_id
    scores += primary.astype(float) * 1.0
    for id_col, pct_col in [('Topic_2_ID','Topic_2_Pct'),('Topic_3_ID','Topic_3_Pct')]:
        if id_col in data.columns:
            sec = data[id_col] == preferred_topic_id
            scores += sec.astype(float) * (data[pct_col].fillna(0)/100.0) * 0.5
    scores = scores * (data['Topic_1_Pct'].fillna(50)/100.0 + 0.5)
    mx = scores.max()
    if mx > 0:
        scores = scores / mx
    return scores


def get_recommendations(preferences, mode='hybrid'):
    preferred_topic_id = None
    if preferences.get('preferred_topic'):
        preferred_topic_id = topic_label_to_id.get(preferences['preferred_topic'])

    data = df.copy()

    # Apply district filter
    if preferences.get('district'):
        d = data[data['Municipality'].str.lower() == preferences['district'].lower()]
        if len(d) >= 3:
            data = d

    kbf_s = compute_kbf_score(data, preferences)
    lda_s = compute_lda_score(data, preferred_topic_id)

    if mode == 'kbf_only':
        final = kbf_s * 100
    elif mode == 'lda_only':
        final = lda_s * 100
    else:
        final = ((KBF_WEIGHT * kbf_s) + (LDA_WEIGHT * lda_s))
        rating_boost = (data['Rating'].fillna(3.0) / 5.0) * 0.05
        final = final + rating_boost
        mx = final.max()
        if mx > 0:
            final = (final / mx) * 100

    data = data.copy()
    data['Score']     = final.round(2)
    data['KBF_Score'] = (kbf_s * 100).round(2)
    data['LDA_Score'] = (lda_s * 100).round(2)

    return data.sort_values('Score', ascending=False).head(TOP_N).reset_index(drop=True)


# ============================================================
# TEST QUERIES — 8 diverse scenarios
# ============================================================
TEST_QUERIES = [
    {
        'id'             : 'Q1',
        'description'    : 'Family Seafood Outing — Kuala Terengganu',
        'district'       : 'Kuala Terengganu',
        'cuisine'        : 'Seafood',
        'min_rating'     : 4.0,
        'preferred_topic': 'Seafood & Local Snacks',
        'halal'          : True,
        'family_friendly': True,
        'parking'        : True,
    },
    {
        'id'             : 'Q2',
        'description'    : 'Romantic Dinner — Besut',
        'district'       : 'Besut',
        'min_rating'     : 4.0,
        'preferred_topic': 'Location & Ambiance',
        'romantic'       : True,
        'scenic_view'    : True,
        'halal'          : True,
    },
    {
        'id'             : 'Q3',
        'description'    : 'Traditional Malay Food — Dungun',
        'district'       : 'Dungun',
        'cuisine'        : 'Malay',
        'min_rating'     : 3.5,
        'preferred_topic': 'Traditional Malay Food',
        'halal'          : True,
    },
    {
        'id'             : 'Q4',
        'description'    : 'Western Food — Kemaman',
        'district'       : 'Kemaman',
        'cuisine'        : 'Western',
        'min_rating'     : 4.0,
        'preferred_topic': 'Western & Fusion Food',
        'parking'        : True,
    },
    {
        'id'             : 'Q5',
        'description'    : 'Budget Cafe — Hulu Terengganu',
        'district'       : 'Hulu Terengganu',
        'cuisine'        : 'Cafe',
        'min_rating'     : 3.0,
        'preferred_topic': 'Overall Dining Experience',
    },
    {
        'id'             : 'Q6',
        'description'    : 'Vegetarian Friendly — Kuala Terengganu',
        'district'       : 'Kuala Terengganu',
        'min_rating'     : 3.5,
        'preferred_topic': 'Overall Dining Experience',
        'vegetarian'     : True,
    },
    {
        'id'             : 'Q7',
        'description'    : 'Scenic Outdoor Dining — Marang',
        'district'       : 'Marang',
        'min_rating'     : 3.5,
        'preferred_topic': 'Location & Ambiance',
        'scenic_view'    : True,
        'outdoor'        : True,
    },
    {
        'id'             : 'Q8',
        'description'    : 'Fast Food Family — Besut',
        'district'       : 'Besut',
        'cuisine'        : 'Fast Food',
        'min_rating'     : 3.5,
        'preferred_topic': 'Overall Dining Experience',
        'halal'          : True,
        'family_friendly': True,
    },
]


# ============================================================
# PART A — INTRINSIC EVALUATION METRICS
# ============================================================
def compute_diversity(results):
    """Intra-list diversity: unique cuisines + districts + topics / possible max."""
    unique_cuisines  = results['Cuisine_Type'].nunique()
    unique_districts = results['Municipality'].nunique()
    unique_topics    = results['Topic_Label'].nunique()
    # Normalize by TOP_N
    diversity = (unique_cuisines + unique_districts + unique_topics) / (3 * min(TOP_N, len(results)))
    return round(min(diversity, 1.0), 4)


def compute_filter_satisfaction(results, preferences):
    """% of requested KBF filters satisfied on average across top 10."""
    filter_cols = {
        'halal'          : 'Is_Halal',
        'vegetarian'     : 'Is_Vegetarian',
        'vegan'          : 'Is_Vegan',
        'parking'        : 'Has_Parking',
        'family_friendly': 'Is_Family_Friendly',
        'romantic'       : 'Is_Romantic',
        'scenic_view'    : 'Has_Scenic_View',
        'outdoor'        : 'Has_Outdoor',
        'wifi'           : 'Has_Wifi',
    }
    active_filters = [(pref, col) for pref, col in filter_cols.items()
                      if preferences.get(pref)]
    if not active_filters:
        return 1.0   # no filters = 100% satisfied

    total_satisfied = 0
    for _, col in active_filters:
        if col in results.columns:
            total_satisfied += (results[col] == 'Yes').sum()

    max_possible = len(active_filters) * len(results)
    return round(total_satisfied / max_possible, 4) if max_possible > 0 else 0.0


def compute_topic_relevance(results, preferred_topic):
    """% of top 10 results whose dominant topic matches preferred topic."""
    if not preferred_topic:
        return None
    matched = (results['Topic_Label'] == preferred_topic).sum()
    return round(matched / len(results), 4)


def compute_coverage(all_results_names):
    """% of total restaurant pool covered by recommendations."""
    unique_recommended = len(set(all_results_names))
    return round(unique_recommended / len(df), 4)


def run_part_a():
    print("\n" + "=" * 60)
    print("  PART A — INTRINSIC EVALUATION")
    print("=" * 60)

    records          = []
    all_hybrid_scores = []
    all_kbf_scores   = []
    all_lda_scores   = []
    all_recommended  = []

    for q in TEST_QUERIES:
        qid  = q['id']
        desc = q['description']

        # Get results for all 3 modes
        hybrid_results = get_recommendations(q, mode='hybrid')
        kbf_results    = get_recommendations(q, mode='kbf_only')
        lda_results    = get_recommendations(q, mode='lda_only')

        # Compute metrics
        diversity    = compute_diversity(hybrid_results)
        filter_sat   = compute_filter_satisfaction(hybrid_results, q)
        topic_rel    = compute_topic_relevance(hybrid_results, q.get('preferred_topic'))
        avg_hybrid   = hybrid_results['Score'].mean().round(2)
        avg_kbf      = kbf_results['Score'].mean().round(2)
        avg_lda      = lda_results['Score'].mean().round(2)

        all_recommended.extend(hybrid_results['Name'].tolist())
        all_hybrid_scores.append(avg_hybrid)
        all_kbf_scores.append(avg_kbf)
        all_lda_scores.append(avg_lda)

        record = {
            'Query_ID'            : qid,
            'Description'         : desc,
            'Avg_Hybrid_Score'    : avg_hybrid,
            'Avg_KBF_Only_Score'  : avg_kbf,
            'Avg_LDA_Only_Score'  : avg_lda,
            'Diversity_Score'     : diversity,
            'Filter_Satisfaction' : filter_sat,
            'Topic_Relevance'     : topic_rel if topic_rel is not None else 'N/A',
            'Results_Count'       : len(hybrid_results),
        }
        records.append(record)

        print(f"\n  {qid}: {desc}")
        print(f"       Hybrid Score (avg)   : {avg_hybrid:.2f}")
        print(f"       KBF-only Score (avg) : {avg_kbf:.2f}")
        print(f"       LDA-only Score (avg) : {avg_lda:.2f}")
        print(f"       Diversity Score      : {diversity:.4f}")
        print(f"       Filter Satisfaction  : {filter_sat*100:.1f}%")
        if topic_rel is not None:
            print(f"       Topic Relevance      : {topic_rel*100:.1f}%")

    # Overall coverage
    coverage = compute_coverage(all_recommended)

    print(f"\n  {'='*40}")
    print(f"  Overall System Coverage    : {coverage*100:.1f}%")
    print(f"  LDA Coherence Score (c_v)  : {LDA_COHERENCE_SCORE}")
    print(f"  {'='*40}")

    df_results = pd.DataFrame(records)
    df_results.to_csv(f'{OUTPUT_DIR}/evaluation_partA_intrinsic.csv', index=False)
    print(f"\n✅ Saved: evaluation_partA_intrinsic.csv")

    return df_results, all_hybrid_scores, all_kbf_scores, all_lda_scores, coverage


# ============================================================
# PART A — CHARTS
# ============================================================
def plot_part_a(df_results, all_hybrid, all_kbf, all_lda, coverage):
    query_ids = [q['id'] for q in TEST_QUERIES]
    x = np.arange(len(query_ids))
    w = 0.25

    # --- Chart 1: Hybrid vs KBF-only vs LDA-only ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - w, all_kbf,    width=w, label='KBF Only',   color='#3498DB', edgecolor='white')
    ax.bar(x,     all_hybrid, width=w, label='Hybrid',      color=ACCENT,    edgecolor='white')
    ax.bar(x + w, all_lda,    width=w, label='LDA Only',   color='#2ECC71', edgecolor='white')
    ax.set_title('Average Score: Hybrid vs KBF-Only vs LDA-Only')
    ax.set_xlabel('Query')
    ax.set_ylabel('Average Score (0–100)')
    ax.set_xticks(x)
    ax.set_xticklabels(query_ids)
    ax.legend()
    ax.set_ylim(0, 110)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/partA_01_hybrid_vs_kbf_vs_lda.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Chart saved: partA_01_hybrid_vs_kbf_vs_lda.png")

    # --- Chart 2: Diversity Score per Query ---
    fig, ax = plt.subplots(figsize=(10, 5))
    diversity_vals = df_results['Diversity_Score'].tolist()
    colors = sns.color_palette(PALETTE, len(query_ids))
    bars = ax.bar(query_ids, diversity_vals, color=colors, edgecolor='white')
    for bar, val in zip(bars, diversity_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')
    ax.set_title('Intra-List Diversity Score per Query')
    ax.set_xlabel('Query')
    ax.set_ylabel('Diversity Score (0–1)')
    ax.set_ylim(0, 1.2)
    ax.axhline(np.mean(diversity_vals), color=ACCENT, linestyle='--',
               linewidth=1.5, label=f'Mean: {np.mean(diversity_vals):.3f}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/partA_02_diversity_score.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Chart saved: partA_02_diversity_score.png")

    # --- Chart 3: Filter Satisfaction Rate ---
    fig, ax = plt.subplots(figsize=(10, 5))
    filter_vals = [float(v) * 100 if v != 'N/A' else 0
                   for v in df_results['Filter_Satisfaction'].tolist()]
    colors = sns.color_palette(PALETTE, len(query_ids))
    bars = ax.bar(query_ids, filter_vals, color=colors, edgecolor='white')
    for bar, val in zip(bars, filter_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold')
    ax.set_title('Filter Satisfaction Rate per Query (%)')
    ax.set_xlabel('Query')
    ax.set_ylabel('Filter Satisfaction (%)')
    ax.set_ylim(0, 120)
    ax.axhline(np.mean(filter_vals), color=ACCENT, linestyle='--',
               linewidth=1.5, label=f'Mean: {np.mean(filter_vals):.1f}%')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/partA_03_filter_satisfaction.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Chart saved: partA_03_filter_satisfaction.png")

    # --- Chart 4: Topic Relevance Rate ---
    fig, ax = plt.subplots(figsize=(10, 5))
    topic_vals = []
    for v in df_results['Topic_Relevance'].tolist():
        try:
            topic_vals.append(float(v) * 100)
        except:
            topic_vals.append(0)
    colors = sns.color_palette(PALETTE, len(query_ids))
    bars = ax.bar(query_ids, topic_vals, color=colors, edgecolor='white')
    for bar, val in zip(bars, topic_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold')
    ax.set_title('Topic Relevance Rate per Query (%)')
    ax.set_xlabel('Query')
    ax.set_ylabel('Topic Relevance (%)')
    ax.set_ylim(0, 120)
    ax.axhline(np.mean(topic_vals), color=ACCENT, linestyle='--',
               linewidth=1.5, label=f'Mean: {np.mean(topic_vals):.1f}%')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/partA_04_topic_relevance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Chart saved: partA_04_topic_relevance.png")

    # --- Chart 5: System Coverage + Coherence Summary ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Coverage pie
    axes[0].pie([coverage, 1-coverage],
                labels=[f'Covered\n{coverage*100:.1f}%', f'Not covered\n{(1-coverage)*100:.1f}%'],
                colors=[ACCENT, '#ECF0F1'], startangle=90,
                wedgeprops={'edgecolor':'white', 'linewidth':2})
    axes[0].set_title('System Coverage\n(% of restaurants recommended)')

    # Coherence score gauge
    coherence_pct = LDA_COHERENCE_SCORE * 100
    axes[1].barh(['LDA Coherence\n(c_v)'], [coherence_pct],
                 color=ACCENT, edgecolor='white', height=0.4)
    axes[1].barh(['LDA Coherence\n(c_v)'], [100 - coherence_pct],
                 left=[coherence_pct], color='#ECF0F1', edgecolor='white', height=0.4)
    axes[1].text(coherence_pct/2, 0, f'{LDA_COHERENCE_SCORE:.4f}',
                 ha='center', va='center', fontweight='bold', color='white', fontsize=14)
    axes[1].set_xlim(0, 100)
    axes[1].set_xlabel('Score (%)')
    axes[1].set_title('LDA Topic Coherence Score\n(c_v, higher = better)')
    axes[1].spines['left'].set_visible(False)
    axes[1].set_yticks([])

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/partA_05_coverage_coherence.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Chart saved: partA_05_coverage_coherence.png")


# ============================================================
# PART B — USER STUDY EVALUATION
# ============================================================
# HOW TO USE:
# For each query, show top 10 results to real users.
# Ask them: "Is this restaurant relevant to your query? (1=Yes, 0=No)"
# Also ask: "Rate this restaurant for this query (1-5)"
# Enter their responses in USER_STUDY_DATA below.
# Each entry: list of 10 values (one per recommended restaurant)
# ============================================================

# PLACEHOLDER DATA — replace with real user ratings
# Format: query_id → list of 10 relevance judgements (1=relevant, 0=not relevant)
# and list of 10 graded ratings (1-5 scale)
# Below uses simulated placeholder values — REPLACE with real user data

USER_STUDY_DATA = {
    'Q1': {
        'relevance': [1, 1, 1, 0, 1, 1, 0, 1, 1, 0],   # 1=relevant, 0=not relevant
        'ratings'  : [5, 4, 4, 2, 5, 3, 2, 4, 5, 3],   # 1-5 scale
    },
    'Q2': {
        'relevance': [1, 1, 0, 1, 1, 0, 1, 0, 1, 1],
        'ratings'  : [5, 4, 2, 4, 5, 2, 4, 3, 5, 4],
    },
    'Q3': {
        'relevance': [1, 1, 1, 1, 0, 1, 1, 0, 1, 1],
        'ratings'  : [5, 5, 4, 4, 2, 5, 3, 2, 4, 5],
    },
    'Q4': {
        'relevance': [1, 0, 1, 1, 1, 0, 1, 1, 0, 1],
        'ratings'  : [4, 2, 5, 4, 4, 2, 5, 3, 2, 4],
    },
    'Q5': {
        'relevance': [1, 1, 0, 1, 0, 1, 1, 0, 1, 1],
        'ratings'  : [4, 4, 2, 5, 2, 4, 3, 2, 5, 4],
    },
    'Q6': {
        'relevance': [0, 1, 1, 1, 0, 1, 0, 1, 1, 1],
        'ratings'  : [2, 4, 5, 4, 2, 5, 2, 4, 4, 5],
    },
    'Q7': {
        'relevance': [1, 1, 1, 0, 1, 1, 0, 0, 1, 1],
        'ratings'  : [5, 4, 5, 2, 4, 5, 2, 2, 4, 5],
    },
    'Q8': {
        'relevance': [1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
        'ratings'  : [5, 2, 4, 4, 5, 4, 2, 5, 2, 4],
    },
}


def compute_precision_at_k(relevance, k=10):
    """Precision@K = relevant results in top K / K"""
    return round(sum(relevance[:k]) / k, 4)


def compute_recall_at_k(relevance, total_relevant, k=10):
    """Recall@K = relevant results in top K / total relevant in pool"""
    if total_relevant == 0:
        return 0.0
    return round(sum(relevance[:k]) / total_relevant, 4)


def compute_f1(precision, recall):
    """F1 = 2 * P * R / (P + R)"""
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


def compute_ndcg_at_k(ratings, k=10):
    """
    nDCG@K — Normalized Discounted Cumulative Gain
    ratings: list of relevance scores (1-5 scale)
    """
    def dcg(scores):
        return sum([(2**r - 1) / np.log2(i + 2)
                    for i, r in enumerate(scores[:k])])
    actual_dcg  = dcg(ratings)
    ideal_dcg   = dcg(sorted(ratings, reverse=True))
    if ideal_dcg == 0:
        return 0.0
    return round(actual_dcg / ideal_dcg, 4)


def run_part_b():
    print("\n" + "=" * 60)
    print("  PART B — USER STUDY EVALUATION")
    print("  ⚠️  Currently using PLACEHOLDER data.")
    print("  Replace USER_STUDY_DATA with real user ratings.")
    print("=" * 60)

    records = []

    precision_list = []
    recall_list    = []
    f1_list        = []
    ndcg_list      = []

    for q in TEST_QUERIES:
        qid  = q['id']
        desc = q['description']
        data = USER_STUDY_DATA.get(qid, {})

        relevance      = data.get('relevance', [0]*TOP_N)
        ratings        = data.get('ratings',   [1]*TOP_N)
        total_relevant = sum(relevance)

        precision = compute_precision_at_k(relevance, TOP_N)
        recall    = compute_recall_at_k(relevance, total_relevant, TOP_N)
        f1        = compute_f1(precision, recall)
        ndcg      = compute_ndcg_at_k(ratings, TOP_N)

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        ndcg_list.append(ndcg)

        records.append({
            'Query_ID'      : qid,
            'Description'   : desc,
            'Precision@10'  : precision,
            'Recall@10'     : recall,
            'F1_Score'      : f1,
            'nDCG@10'       : ndcg,
            'Total_Relevant': total_relevant,
        })

        print(f"\n  {qid}: {desc}")
        print(f"       Precision@10  : {precision:.4f}  ({precision*100:.1f}%)")
        print(f"       Recall@10     : {recall:.4f}  ({recall*100:.1f}%)")
        print(f"       F1 Score      : {f1:.4f}")
        print(f"       nDCG@10       : {ndcg:.4f}")

    print(f"\n  {'='*40}")
    print(f"  Mean Precision@10 : {np.mean(precision_list):.4f}")
    print(f"  Mean Recall@10    : {np.mean(recall_list):.4f}")
    print(f"  Mean F1 Score     : {np.mean(f1_list):.4f}")
    print(f"  Mean nDCG@10      : {np.mean(ndcg_list):.4f}")
    print(f"  {'='*40}")

    df_results = pd.DataFrame(records)
    df_results.to_csv(f'{OUTPUT_DIR}/evaluation_partB_user_study.csv', index=False)
    print(f"\n✅ Saved: evaluation_partB_user_study.csv")

    return df_results, precision_list, recall_list, f1_list, ndcg_list


# ============================================================
# PART B — CHARTS
# ============================================================
def plot_part_b(precision_list, recall_list, f1_list, ndcg_list):
    query_ids = [q['id'] for q in TEST_QUERIES]
    x = np.arange(len(query_ids))
    w = 0.2

    # --- Chart 6: All metrics per query ---
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - 1.5*w, precision_list, width=w, label='Precision@10', color='#3498DB', edgecolor='white')
    ax.bar(x - 0.5*w, recall_list,    width=w, label='Recall@10',    color='#2ECC71', edgecolor='white')
    ax.bar(x + 0.5*w, f1_list,        width=w, label='F1 Score',     color='#F39C12', edgecolor='white')
    ax.bar(x + 1.5*w, ndcg_list,      width=w, label='nDCG@10',      color=ACCENT,    edgecolor='white')
    ax.set_title('Evaluation Metrics per Query (User Study)')
    ax.set_xlabel('Query')
    ax.set_ylabel('Score (0–1)')
    ax.set_xticks(x)
    ax.set_xticklabels(query_ids)
    ax.set_ylim(0, 1.2)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/partB_01_all_metrics_per_query.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Chart saved: partB_01_all_metrics_per_query.png")

    # --- Chart 7: Mean metrics summary bar ---
    fig, ax = plt.subplots(figsize=(8, 5))
    metrics = ['Precision@10', 'Recall@10', 'F1 Score', 'nDCG@10']
    values  = [np.mean(precision_list), np.mean(recall_list),
               np.mean(f1_list),        np.mean(ndcg_list)]
    colors  = ['#3498DB', '#2ECC71', '#F39C12', ACCENT]
    bars = ax.bar(metrics, values, color=colors, edgecolor='white', width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', fontsize=11, fontweight='bold')
    ax.set_title('Overall Mean Evaluation Metrics (User Study)')
    ax.set_ylabel('Mean Score (0–1)')
    ax.set_ylim(0, 1.2)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/partB_02_mean_metrics_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Chart saved: partB_02_mean_metrics_summary.png")

    # --- Chart 8: nDCG trend across queries ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(query_ids, ndcg_list, color=ACCENT, linewidth=2.5,
            marker='o', markersize=8, markerfacecolor='white',
            markeredgecolor=ACCENT, markeredgewidth=2)
    ax.axhline(np.mean(ndcg_list), color='#2C3E50', linestyle='--',
               linewidth=1.5, label=f'Mean nDCG: {np.mean(ndcg_list):.4f}')
    ax.set_title('nDCG@10 Score per Query')
    ax.set_xlabel('Query')
    ax.set_ylabel('nDCG@10')
    ax.set_ylim(0, 1.1)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/partB_03_ndcg_trend.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Chart saved: partB_03_ndcg_trend.png")


# ============================================================
# FINAL SUMMARY REPORT
# ============================================================
def save_summary(df_partA, df_partB):
    summary = {
        'Metric'  : [],
        'Value'   : [],
        'Section' : [],
    }

    # Part A metrics
    summary['Metric'].append('LDA Coherence Score (c_v)')
    summary['Value'].append(LDA_COHERENCE_SCORE)
    summary['Section'].append('Part A — Intrinsic')

    summary['Metric'].append('Mean Hybrid Score')
    summary['Value'].append(df_partA['Avg_Hybrid_Score'].mean().round(2))
    summary['Section'].append('Part A — Intrinsic')

    summary['Metric'].append('Mean KBF-Only Score')
    summary['Value'].append(df_partA['Avg_KBF_Only_Score'].mean().round(2))
    summary['Section'].append('Part A — Intrinsic')

    summary['Metric'].append('Mean LDA-Only Score')
    summary['Value'].append(df_partA['Avg_LDA_Only_Score'].mean().round(2))
    summary['Section'].append('Part A — Intrinsic')

    summary['Metric'].append('Mean Diversity Score')
    summary['Value'].append(df_partA['Diversity_Score'].mean().round(4))
    summary['Section'].append('Part A — Intrinsic')

    summary['Metric'].append('Mean Filter Satisfaction')
    fs_vals = pd.to_numeric(df_partA['Filter_Satisfaction'], errors='coerce')
    summary['Value'].append(fs_vals.mean().round(4))
    summary['Section'].append('Part A — Intrinsic')

    # Part B metrics
    summary['Metric'].append('Mean Precision@10')
    summary['Value'].append(df_partB['Precision@10'].mean().round(4))
    summary['Section'].append('Part B — User Study')

    summary['Metric'].append('Mean Recall@10')
    summary['Value'].append(df_partB['Recall@10'].mean().round(4))
    summary['Section'].append('Part B — User Study')

    summary['Metric'].append('Mean F1 Score')
    summary['Value'].append(df_partB['F1_Score'].mean().round(4))
    summary['Section'].append('Part B — User Study')

    summary['Metric'].append('Mean nDCG@10')
    summary['Value'].append(df_partB['nDCG@10'].mean().round(4))
    summary['Section'].append('Part B — User Study')

    df_summary = pd.DataFrame(summary)
    df_summary.to_csv(f'{OUTPUT_DIR}/evaluation_summary.csv', index=False)
    print(f"\n✅ Saved: evaluation_summary.csv")

    print("\n" + "=" * 60)
    print("  FINAL EVALUATION SUMMARY")
    print("=" * 60)
    for _, row in df_summary.iterrows():
        print(f"  [{row['Section']}]")
        print(f"  {row['Metric']:<35} : {row['Value']}")
    print("=" * 60)


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':

    print("=" * 60)
    print("  STEP 7 — EVALUATION")
    print(f"  Hybrid Weighting: {int(KBF_WEIGHT*100)}% KBF + {int(LDA_WEIGHT*100)}% LDA")
    print(f"  Test Queries: {len(TEST_QUERIES)}")
    print(f"  Top N: {TOP_N}")
    print("=" * 60)

    # Run Part A
    df_partA, hybrid_s, kbf_s, lda_s, coverage = run_part_a()
    plot_part_a(df_partA, hybrid_s, kbf_s, lda_s, coverage)

    # Run Part B
    df_partB, prec, rec, f1, ndcg = run_part_b()
    plot_part_b(prec, rec, f1, ndcg)

    # Save summary
    save_summary(df_partA, df_partB)

    print("\n" + "=" * 60)
    print("  EVALUATION COMPLETE")
    print(f"  All outputs saved in /{OUTPUT_DIR}/")
    print("=" * 60)
    print("\n  Part A Charts:")
    print("  - partA_01_hybrid_vs_kbf_vs_lda.png")
    print("  - partA_02_diversity_score.png")
    print("  - partA_03_filter_satisfaction.png")
    print("  - partA_04_topic_relevance.png")
    print("  - partA_05_coverage_coherence.png")
    print("\n  Part B Charts:")
    print("  - partB_01_all_metrics_per_query.png")
    print("  - partB_02_mean_metrics_summary.png")
    print("  - partB_03_ndcg_trend.png")
    print("\n  CSV Reports:")
    print("  - evaluation_partA_intrinsic.csv")
    print("  - evaluation_partB_user_study.csv")
    print("  - evaluation_summary.csv")
    print("=" * 60)
    print("\n  ⚠️  IMPORTANT: Replace USER_STUDY_DATA in this script")
    print("     with real ratings collected from your user study.")
    print("     Then re-run to get accurate Part B metrics.\n")
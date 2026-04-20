"""
STEP 3 — LDA Modelling
=======================
Trains LDA topic model on cleaned reviews.
Automatically finds optimal number of topics (k) using coherence score.
Assigns dominant topic to each restaurant.

Input : master_990_terengganu_preprocessed.csv
Output: master_990_lda.csv
        lda_outputs/ (model files, charts, topic labels)

Run: python s3_lda.py
"""

import pandas as pd
import numpy as np
import os
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

import gensim
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel

IN_FILE     = 'master_990_terengganu_preprocessed.csv'
OUT_FILE    = 'master_990_lda.csv'
OUT_DIR     = 'lda_outputs'
os.makedirs(OUT_DIR, exist_ok=True)

# ── LDA Config ────────────────────────────────────────────────────────────────
K_MIN       = 4    # minimum topics to try
K_MAX       = 12   # maximum topics to try
K_STEP      = 1
LDA_PASSES  = 15
LDA_ITER    = 100
RANDOM_SEED = 42

# ─────────────────────────────────────────────────────────────────────────────

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading preprocessed data...")
df = pd.read_csv(IN_FILE)

# Only use restaurants with sufficient text for LDA training
df_lda = df[df['has_sufficient_text'] == True].copy()
print(f"  Total restaurants       : {len(df)}")
print(f"  Used for LDA training   : {len(df_lda)}")
print(f"  Skipped (no reviews)    : {len(df) - len(df_lda)}")

# ── Build corpus ──────────────────────────────────────────────────────────────
print("\nBuilding corpus...")
texts = [str(t).split() for t in df_lda['cleaned_text']]
dictionary = corpora.Dictionary(texts)

# Filter extremes — remove very rare and very common words
dictionary.filter_extremes(no_below=3, no_above=0.85)
corpus = [dictionary.doc2bow(t) for t in texts]

print(f"  Dictionary size : {len(dictionary)} unique tokens")
print(f"  Corpus size     : {len(corpus)} documents")

# ── Find optimal k ────────────────────────────────────────────────────────────
print(f"\nFinding optimal k ({K_MIN}–{K_MAX} topics)...")

k_values   = list(range(K_MIN, K_MAX + 1, K_STEP))
coherences = []

for k in k_values:
    print(f"  Testing k={k}...", end=' ', flush=True)
    model = LdaModel(
        corpus           = corpus,
        id2word          = dictionary,
        num_topics       = k,
        passes           = LDA_PASSES,
        iterations       = LDA_ITER,
        random_state     = RANDOM_SEED,
        per_word_topics  = True,
        minimum_probability = 0.0,
    )
    cm = CoherenceModel(
        model      = model,
        texts      = texts,
        dictionary = dictionary,
        coherence  = 'c_v',
        processes  = 1,
    )
    score = cm.get_coherence()
    coherences.append(score)
    print(f"coherence = {score:.4f}")

# Pick best k
best_idx = int(np.argmax(coherences))
best_k   = k_values[best_idx]
best_coh = coherences[best_idx]

print(f"\n  ✓ Best k = {best_k} (coherence = {best_coh:.4f})")

# ── Plot coherence scores ─────────────────────────────────────────────────────
plt.figure(figsize=(10, 5))
plt.plot(k_values, coherences, 'bo-', linewidth=2, markersize=8)
plt.axvline(x=best_k, color='red', linestyle='--', label=f'Best k={best_k}')
plt.xlabel('Number of Topics (k)', fontsize=12)
plt.ylabel('Coherence Score (c_v)', fontsize=12)
plt.title('LDA Topic Coherence Score vs Number of Topics', fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/coherence_score_chart.png', dpi=150)
plt.close()
print(f"  Saved coherence chart → {OUT_DIR}/coherence_score_chart.png")

# ── Train final model with best k ─────────────────────────────────────────────
print(f"\nTraining final LDA model (k={best_k})...")
final_model = LdaModel(
    corpus           = corpus,
    id2word          = dictionary,
    num_topics       = best_k,
    passes           = LDA_PASSES,
    iterations       = LDA_ITER,
    random_state     = RANDOM_SEED,
    per_word_topics  = True,
    minimum_probability = 0.0,
)

# Save model
model_path = f'{OUT_DIR}/lda_final_model'
final_model.save(model_path)
print(f"  Saved model → {model_path}")

# ── Print topics ──────────────────────────────────────────────────────────────
print(f"\nTop words per topic:")
topic_keywords = {}
for i in range(best_k):
    words = [w for w, _ in final_model.show_topic(i, topn=10)]
    topic_keywords[i] = words
    print(f"  Topic {i+1}: {', '.join(words)}")

# ── Assign topics to restaurants ─────────────────────────────────────────────
print("\nAssigning topics to restaurants...")

topic_assignments = []
for bow in corpus:
    topic_dist        = final_model.get_document_topics(bow, minimum_probability=0.0)
    topic_dist_sorted = sorted(topic_dist, key=lambda x: x[1], reverse=True)

    dominant = topic_dist_sorted[0][0] + 1   # 1-indexed
    top1_pct = round(topic_dist_sorted[0][1] * 100, 2)
    top2_pct = round(topic_dist_sorted[1][1] * 100, 2) if len(topic_dist_sorted) > 1 else 0.0
    top3_pct = round(topic_dist_sorted[2][1] * 100, 2) if len(topic_dist_sorted) > 2 else 0.0

    topic_assignments.append({
        'dominant_topic': dominant,
        'topic_1_pct'   : top1_pct,
        'topic_2_pct'   : top2_pct,
        'topic_3_pct'   : top3_pct,
    })

df_lda = df_lda.copy()
df_lda['dominant_topic'] = [t['dominant_topic'] for t in topic_assignments]
df_lda['topic_1_pct']    = [t['topic_1_pct']    for t in topic_assignments]
df_lda['topic_2_pct']    = [t['topic_2_pct']    for t in topic_assignments]
df_lda['topic_3_pct']    = [t['topic_3_pct']    for t in topic_assignments]

# Save LDA results for restaurants with reviews
df_lda[['name', 'dominant_topic', 'topic_1_pct', 'topic_2_pct', 'topic_3_pct']].to_csv(
    f'{OUT_DIR}/lda_restaurant_topics.csv', index=False
)

# ── IMPORTANT: Manual topic labelling required ────────────────────────────────
print(f"\n{'='*55}")
print(f"  ⚠️  MANUAL ACTION REQUIRED")
print(f"{'='*55}")
print(f"  Look at the topic keywords printed above.")
print(f"  Edit the TOPIC_LABELS dict below in this script,")
print(f"  then re-run to apply labels.")
print(f"{'='*55}")

# ── EDIT THIS based on your topic keywords above ──────────────────────────────
# Run once → see keywords → update these labels → re-run
TOPIC_LABELS = {
    1: "Casual Dining & Variety",
    2: "Malay Breakfast & Local Staples",
    3: "Local Snacks & Specialty Bites",
    4: "Fast Food & Service Quality",
    5: "Popular Local Favorites",
    6: "Comfort Food & Value Meals"
    # Add more if best_k > 6,
    #7: 'Overall Dining Experience',
    #8: 'Location & Ambiance',
}

# ── Apply labels ──────────────────────────────────────────────────────────────
print("\nApplying topic labels...")

df_full = df.copy()
df_full = df_full.merge(
    df_lda[['name', 'dominant_topic', 'topic_1_pct', 'topic_2_pct', 'topic_3_pct']],
    on='name',
    how='left',
)

df_full['dominant_topic'] = df_full['dominant_topic'].fillna(0).astype(int)
df_full['topic_1_pct']    = df_full['topic_1_pct'].fillna(0.0)
df_full['topic_2_pct']    = df_full['topic_2_pct'].fillna(0.0)
df_full['topic_3_pct']    = df_full['topic_3_pct'].fillna(0.0)
df_full['topic_label']    = df_full['dominant_topic'].map(TOPIC_LABELS).fillna('No Reviews')

# Save topic labels CSV for reference
pd.DataFrame([
    {'topic_id': k, 'topic_label': v, 'keywords': ', '.join(topic_keywords.get(k-1, []))}
    for k, v in TOPIC_LABELS.items()
]).to_csv(f'{OUT_DIR}/lda_topic_labels.csv', index=False)

df_full.to_csv(OUT_FILE, index=False, encoding='utf-8-sig')

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  STEP 3 COMPLETE")
print(f"{'='*55}")
print(f"  Best k (topics)      : {best_k}")
print(f"  Best coherence score : {best_coh:.4f}")
print(f"  Restaurants with LDA : {df_full[df_full['dominant_topic'] > 0]['name'].count()}")
print(f"  Restaurants no LDA   : {df_full[df_full['dominant_topic'] == 0]['name'].count()}")
print(f"\n  Topic distribution:")
for tid, label in TOPIC_LABELS.items():
    count = (df_full['dominant_topic'] == tid).sum()
    print(f"    Topic {tid} ({label}): {count} restaurants")
print(f"\n  Output: {OUT_FILE}")
print(f"{'='*55}")
print(f"\n⚠️  If topic labels look wrong, edit TOPIC_LABELS in this")
print(f"   script and re-run. Then proceed to s4_kbf.py")
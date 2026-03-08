import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
import os
import re
warnings.filterwarnings('ignore')

# Gensim for LDA
import gensim
import gensim.corpora as corpora
from gensim.models import LdaModel, CoherenceModel

# ============================================================
# CONFIGURATION
# ============================================================
INPUT_FILE  = 'master_terengganu_preprocessed.csv'
OUTPUT_DIR  = 'lda_outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOPIC_RANGE_START = 5
TOPIC_RANGE_END   = 21        # tests 5 to 20 inclusive
RANDOM_STATE      = 42
PASSES            = 15        # LDA training passes
TOP_WORDS         = 10        # words per topic to display

# Colour palette
ACCENT = '#C0392B'
BG     = '#FAFAFA'

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
# BILINGUAL STOPWORD LIST (English + Malay)
# Defined at module level so tokenize() works on Windows
# ============================================================
english_stopwords = {
    'the','a','an','and','or','but','in','on','at','to','for',
    'of','with','by','from','is','was','are','were','be','been',
    'being','have','has','had','do','does','did','will','would',
    'could','should','may','might','shall','can','need','dare',
    'this','that','these','those','it','its','i','me','my','we',
    'our','you','your','he','she','they','them','their','his','her',
    'what','which','who','whom','when','where','why','how','all',
    'each','every','both','few','more','most','other','some','such',
    'no','not','only','same','so','than','too','very','just','also',
    'about','above','after','again','against','already','always',
    'am','among','another','any','because','before','between',
    'during','even','ever','get','got','here','however','if',
    'into','like','many','much','never','now','often','out','over',
    'own','re','said','since','still','then','there','though',
    'through','time','under','until','up','use','used','using',
    'via','while','within','without','yet','s','t','don','didn',
    'doesn','isn','wasn','weren','haven','hadn','couldn','wouldn',
    'really','nice','great','good','bad','place','went','go','come',
    'came','back','well','made','make','say','said','know','think',
    'see','look','want','way','one','two','three','first','last',
    'us','around','away','bit','cant','definitely','else','enough',
    'especially','found','give','given','going','let','lot','might',
    'nothing','ok','okay','once','quite','rather','seem','seemed',
    'set','something','sure','take','taken','tell','though','told',
    'tried','try','turn','usually','visit','visited','worth','would',
    'able','actually','almost','along','already','although','anyone',
    'anything','anyway','anywhere','ask','asked','best','better',
    'big','called','coming','day','days','done','everyone','everything',
    'far','feel','felt','full','getting','high','hour','hours',
    'however','keep','kept','least','left','long','maybe','meal',
    'min','minutes','must','near','next','night','nothing','offer',
    'open','ordered','ordering','overall','past','per','perhaps',
    'pretty','probably','put','right','room','seen','side','simply',
    'slightly','small','sometimes','soon','start','started','stop',
    'table','thing','things','thought','times','today','together',
    'took','totally','town','tried','truly','two','unfortunately',
    'unless','upon','usually','wait','waiting','went','whenever',
    'whether','whole','wide','wrong','yet'
}

malay_stopwords = {
    'yang','dan','di','ini','itu','ke','dari','dengan','untuk',
    'pada','adalah','dalam','tidak','ada','juga','saya','kami',
    'kita','mereka','dia','ia','anda','boleh','akan','sudah',
    'sudah','telah','lagi','lebih','atau','oleh','atas','bagi',
    'bila','bukan','dah','dua','tapi','tiga','tiap','tiada',
    'tetapi','selepas','sebelum','semua','satu','sama','pun',
    'pula','pernah','perlu','orang','oleh','nya','namun','mana',
    'macam','lah','kena','kat','kalau','jadi','hanya','harap',
    'dapat','cuma','ber','baru','banyak','baik','antara','ambil',
    'agak','ada','sebab','seperti','supaya','walau','walaupun',
    'melalui','mengenai','tentang','terhadap','selain','selalu',
    'setiap','sebuah','sahaja','sangat','sangatlah','saya',
    'masih','mesti','memang','melainkan','mana','malah','kerana',
    'kepada','ketika','kini','kami','hingga','hampir','hal',
    'harus','berikan','beberapa','bahawa','bahkan','amat'
}

ALL_STOPWORDS = english_stopwords | malay_stopwords


# ============================================================
# TOKENIZER FUNCTION
# Must be defined at module level for Windows multiprocessing
# ============================================================
def tokenize(text):
    if not isinstance(text, str):
        return []
    # Lowercase and keep only alphabetic tokens
    tokens = re.findall(r'[a-zA-Z]+', text.lower())
    # Remove stopwords and very short tokens (< 3 chars)
    tokens = [t for t in tokens if t not in ALL_STOPWORDS and len(t) >= 3]
    return tokens


def get_top3_topics(args):
    final_model, bow = args
    topic_dist = final_model.get_document_topics(bow, minimum_probability=0.0)
    sorted_topics = sorted(topic_dist, key=lambda x: x[1], reverse=True)[:3]
    return sorted_topics


def dominant_topic_for_restaurant(group):
    return group['Dominant_Topic'].mode()[0]


# ============================================================
# MAIN — Required wrapper for Windows multiprocessing
# ============================================================
if __name__ == '__main__':

    # ----------------------------------------------------------
    # STEP 1: Load Data
    # ----------------------------------------------------------
    print("=" * 60)
    print("  LDA TOPIC MODELLING — TERENGGANU RESTAURANT REVIEWS")
    print("=" * 60)

    df = pd.read_csv(INPUT_FILE)
    print(f"\n✅ Loaded: {len(df):,} reviews from {df['Name'].nunique():,} restaurants\n")

    # ----------------------------------------------------------
    # STEP 2: Tokenize Reviews
    # ----------------------------------------------------------
    print("⏳ Tokenizing reviews...")
    df['Tokens'] = df['Cleaned_Text'].apply(tokenize)

    df = df[df['Tokens'].str.len() > 0].reset_index(drop=True)
    print(f"✅ Tokenized: {len(df):,} reviews kept after tokenization\n")

    all_tokens = [t for tokens in df['Tokens'] for t in tokens]
    print(f"   Total tokens in corpus : {len(all_tokens):,}")
    print(f"   Unique tokens          : {len(set(all_tokens)):,}")
    avg_tokens = df['Tokens'].str.len().mean()
    print(f"   Avg tokens per review  : {avg_tokens:.1f}\n")

    # ----------------------------------------------------------
    # STEP 3: Build Gensim Dictionary & Corpus
    # ----------------------------------------------------------
    print("⏳ Building dictionary and corpus...")

    dictionary = corpora.Dictionary(df['Tokens'])
    dictionary.filter_extremes(no_below=5, no_above=0.70)
    corpus = [dictionary.doc2bow(tokens) for tokens in df['Tokens']]

    print(f"✅ Dictionary size (after filtering): {len(dictionary):,} unique tokens")
    print(f"✅ Corpus size                       : {len(corpus):,} documents\n")

    # ----------------------------------------------------------
    # STEP 4: Find Optimal Number of Topics (Coherence Score)
    # ----------------------------------------------------------
    print(f"⏳ Testing topic counts from {TOPIC_RANGE_START} to {TOPIC_RANGE_END - 1}...")
    print("   (This may take a few minutes)\n")

    coherence_values = []
    model_list       = []
    topic_range      = range(TOPIC_RANGE_START, TOPIC_RANGE_END)

    for num_topics in topic_range:
        model = LdaModel(
            corpus          = corpus,
            id2word         = dictionary,
            num_topics      = num_topics,
            random_state    = RANDOM_STATE,
            passes          = PASSES,
            alpha           = 'auto',
            eta             = 'auto',
            per_word_topics = True
        )
        model_list.append(model)

        # processes=1 prevents Windows multiprocessing spawn error
        coherence_model = CoherenceModel(
            model      = model,
            texts      = df['Tokens'].tolist(),
            dictionary = dictionary,
            coherence  = 'c_v',
            processes  = 1
        )
        score = coherence_model.get_coherence()
        coherence_values.append(score)
        print(f"   Topics: {num_topics:2d}  |  Coherence (c_v): {score:.4f}")

    optimal_idx   = coherence_values.index(max(coherence_values))
    optimal_k     = list(topic_range)[optimal_idx]
    optimal_score = coherence_values[optimal_idx]

    print(f"\n✅ Optimal number of topics: {optimal_k}  (coherence = {optimal_score:.4f})\n")

    # ----------------------------------------------------------
    # STEP 5: Plot Coherence Score Graph
    # ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(list(topic_range), coherence_values,
            color=ACCENT, linewidth=2.5, marker='o',
            markersize=7, markerfacecolor='white',
            markeredgecolor=ACCENT, markeredgewidth=2)

    ax.scatter([optimal_k], [optimal_score],
               color=ACCENT, s=150, zorder=5,
               label=f'Optimal k = {optimal_k} (score = {optimal_score:.4f})')
    ax.axvline(optimal_k, color=ACCENT, linestyle='--', linewidth=1.2, alpha=0.5)

    ax.set_title('LDA Coherence Score (c_v) by Number of Topics')
    ax.set_xlabel('Number of Topics (k)')
    ax.set_ylabel('Coherence Score (c_v)')
    ax.set_xticks(list(topic_range))
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/coherence_score_chart.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Chart saved: coherence_score_chart.png\n")

    # ----------------------------------------------------------
    # STEP 6: Train Final LDA Model with Optimal k
    # ----------------------------------------------------------
    print(f"⏳ Training final LDA model with {optimal_k} topics...")
    final_model = model_list[optimal_idx]
    print(f"✅ Final LDA model trained\n")

    # ----------------------------------------------------------
    # STEP 7: Display & Label Topics
    # ----------------------------------------------------------
    print("=" * 60)
    print(f"  DISCOVERED TOPICS (k = {optimal_k})")
    print("=" * 60)

    topics_data = []
    for topic_id in range(optimal_k):
        top_words = final_model.show_topic(topic_id, topn=TOP_WORDS)
        word_list = [w for w, _ in top_words]
        word_str  = ', '.join(word_list)
        label     = f"Topic {topic_id + 1}"

        topics_data.append({
            'Topic_ID' : topic_id,
            'Label'    : label,
            'Top_Words': word_str
        })
        print(f"\n  Topic {topic_id + 1:2d}: {word_str}")

    print("\n" + "=" * 60)
    print("  ⚠️  Please manually rename the labels in the output CSV")
    print("      based on the top words shown above.")
    print("=" * 60 + "\n")

    # ----------------------------------------------------------
    # STEP 8: Assign Top 3 Topics per Review (with percentages)
    # ----------------------------------------------------------
    print("⏳ Assigning top 3 topics per review...")

    df['Topic_Distribution'] = [
        get_top3_topics((final_model, bow)) for bow in corpus
    ]

    df['Topic_1_ID']     = df['Topic_Distribution'].apply(lambda x: x[0][0] + 1 if len(x) > 0 else None)
    df['Topic_1_Pct']    = df['Topic_Distribution'].apply(lambda x: round(x[0][1] * 100, 2) if len(x) > 0 else None)
    df['Topic_2_ID']     = df['Topic_Distribution'].apply(lambda x: x[1][0] + 1 if len(x) > 1 else None)
    df['Topic_2_Pct']    = df['Topic_Distribution'].apply(lambda x: round(x[1][1] * 100, 2) if len(x) > 1 else None)
    df['Topic_3_ID']     = df['Topic_Distribution'].apply(lambda x: x[2][0] + 1 if len(x) > 2 else None)
    df['Topic_3_Pct']    = df['Topic_Distribution'].apply(lambda x: round(x[2][1] * 100, 2) if len(x) > 2 else None)
    df['Dominant_Topic'] = df['Topic_1_ID']

    print("✅ Topic assignment complete\n")

    # ----------------------------------------------------------
    # STEP 9: Aggregate Topics per Restaurant
    # ----------------------------------------------------------
    print("⏳ Aggregating dominant topic per restaurant...")

    restaurant_topics = (
        df.groupby('Name')
        .apply(dominant_topic_for_restaurant)
        .reset_index()
        .rename(columns={0: 'Dominant_Topic'})
    )

    restaurant_avg = (
        df.groupby('Name')[['Topic_1_Pct', 'Topic_2_Pct', 'Topic_3_Pct']]
        .mean()
        .round(2)
        .reset_index()
    )

    restaurant_summary = pd.merge(restaurant_topics, restaurant_avg, on='Name')
    print(f"✅ Restaurant topic summary: {len(restaurant_summary):,} restaurants\n")

    # ----------------------------------------------------------
    # STEP 10: Save Outputs
    # ----------------------------------------------------------
    review_output_cols = [
        'Name', 'Municipality', 'Categories', 'Rating',
        'Review_Text', 'Cleaned_Text',
        'Dominant_Topic',
        'Topic_1_ID', 'Topic_1_Pct',
        'Topic_2_ID', 'Topic_2_Pct',
        'Topic_3_ID', 'Topic_3_Pct'
    ]
    df[review_output_cols].to_csv(f'{OUTPUT_DIR}/lda_reviews_with_topics.csv', index=False)
    print(f"✅ Saved: lda_reviews_with_topics.csv")

    restaurant_summary.to_csv(f'{OUTPUT_DIR}/lda_restaurant_topics.csv', index=False)
    print(f"✅ Saved: lda_restaurant_topics.csv")

    topics_df = pd.DataFrame(topics_data)
    topics_df.to_csv(f'{OUTPUT_DIR}/lda_topic_labels.csv', index=False)
    print(f"✅ Saved: lda_topic_labels.csv")

    final_model.save(f'{OUTPUT_DIR}/lda_final_model')
    print(f"✅ Saved: lda_final_model (gensim model files)\n")

    # ----------------------------------------------------------
    # FINAL PRINTED SUMMARY
    # ----------------------------------------------------------
    print("=" * 60)
    print("  LDA MODELLING SUMMARY")
    print("=" * 60)
    print(f"  Total reviews processed  : {len(df):,}")
    print(f"  Dictionary size          : {len(dictionary):,} tokens")
    print(f"  Topic range tested       : {TOPIC_RANGE_START} – {TOPIC_RANGE_END - 1}")
    print(f"  Optimal topics (k)       : {optimal_k}")
    print(f"  Best coherence score     : {optimal_score:.4f}")
    print(f"  Restaurants with topics  : {len(restaurant_summary):,}")
    print("=" * 60)
    print(f"\n  Dominant topic distribution across restaurants:")
    topic_dist_summary = restaurant_summary['Dominant_Topic'].value_counts().sort_index()
    for topic_id, count in topic_dist_summary.items():
        pct = count / len(restaurant_summary) * 100
        print(f"  Topic {int(topic_id):2d}  →  {count:3d} restaurants ({pct:.1f}%)")
    print("=" * 60)
    print("\n  Output files in /lda_outputs/:")
    print("  - coherence_score_chart.png")
    print("  - lda_reviews_with_topics.csv")
    print("  - lda_restaurant_topics.csv")
    print("  - lda_topic_labels.csv       ← rename labels here!")
    print("  - lda_final_model            (saved model)")
    print("=" * 60)
    print("\n  ➡  Next step: step5_kbf_filtering.py\n")
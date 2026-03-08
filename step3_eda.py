import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from collections import Counter
import re
import os

# ============================================================
# CONFIGURATION
# ============================================================
INPUT_FILE  = 'master_terengganu_preprocessed.csv'
OUTPUT_DIR  = 'eda_outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Colour palette (consistent across all charts)
PALETTE     = 'YlOrRd'
ACCENT      = '#C0392B'
BG          = '#FAFAFA'

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
# LOAD DATA
# ============================================================
df = pd.read_csv(INPUT_FILE)

# Keep only the 8 official Terengganu districts for district-level charts
DISTRICTS = ['Kuala Terengganu', 'Besut', 'Kemaman',
             'Hulu Terengganu', 'Dungun', 'Marang', 'Setiu']
df_district = df[df['Municipality'].isin(DISTRICTS)].copy()

print("=" * 55)
print("  EXPLORATORY DATA ANALYSIS — TERENGGANU RESTAURANTS")
print("=" * 55)
print(f"  Total reviews          : {len(df):,}")
print(f"  Unique restaurants     : {df['Name'].nunique():,}")
print(f"  Districts covered      : {df_district['Municipality'].nunique()}")
print(f"  Unique categories      : {df['Categories'].nunique()}")
print(f"  Rating range           : {df['Rating'].min()} – {df['Rating'].max()}")
print(f"  Avg words per review   : {df['Cleaned_Text'].str.split().str.len().mean():.1f}")
print("=" * 55 + "\n")


# ============================================================
# CHART 1 — Rating Distribution
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
rating_counts = df['Rating'].value_counts().sort_index()
colors = sns.color_palette(PALETTE, len(rating_counts))
bars = ax.bar(rating_counts.index.astype(str), rating_counts.values,
              color=colors, edgecolor='white', linewidth=0.8, width=0.6)

for bar, val in zip(bars, rating_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
            f'{val:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_title('Rating Distribution')
ax.set_xlabel('Star Rating')
ax.set_ylabel('Number of Reviews')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/01_rating_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Chart 1 saved: 01_rating_distribution.png")

# Print summary
print(f"   Rating breakdown:")
for rating, count in rating_counts.items():
    pct = count / len(df) * 100
    print(f"   {rating}★ → {count:,} reviews ({pct:.1f}%)")
print()


# ============================================================
# CHART 2 — Reviews per District
# ============================================================
fig, ax = plt.subplots(figsize=(9, 5))
dist_counts = df_district['Municipality'].value_counts().sort_values(ascending=True)
colors = sns.color_palette(PALETTE, len(dist_counts))
bars = ax.barh(dist_counts.index, dist_counts.values,
               color=colors, edgecolor='white', linewidth=0.8)

for bar, val in zip(bars, dist_counts.values):
    ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height() / 2,
            f'{val:,}', va='center', fontsize=10, fontweight='bold')

ax.set_title('Number of Reviews per District')
ax.set_xlabel('Number of Reviews')
ax.set_ylabel('District')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/02_reviews_per_district.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Chart 2 saved: 02_reviews_per_district.png")

print(f"   District breakdown:")
for dist, count in dist_counts.sort_values(ascending=False).items():
    pct = count / len(df_district) * 100
    print(f"   {dist:<22} → {count:,} reviews ({pct:.1f}%)")
print()


# ============================================================
# CHART 3 — Unique Restaurants per District
# ============================================================
fig, ax = plt.subplots(figsize=(9, 5))
rest_per_dist = (df_district.drop_duplicates(subset='Name')
                 .groupby('Municipality')['Name'].count()
                 .sort_values(ascending=True))
colors = sns.color_palette(PALETTE, len(rest_per_dist))
bars = ax.barh(rest_per_dist.index, rest_per_dist.values,
               color=colors, edgecolor='white', linewidth=0.8)

for bar, val in zip(bars, rest_per_dist.values):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
            f'{val}', va='center', fontsize=10, fontweight='bold')

ax.set_title('Number of Unique Restaurants per District')
ax.set_xlabel('Number of Restaurants')
ax.set_ylabel('District')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/03_restaurants_per_district.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Chart 3 saved: 03_restaurants_per_district.png")
print()


# ============================================================
# CHART 4 — Reviews per Restaurant (Histogram)
# ============================================================
fig, ax = plt.subplots(figsize=(9, 5))
reviews_per_rest = df.groupby('Name').size()
ax.hist(reviews_per_rest.values, bins=30, color=ACCENT,
        edgecolor='white', linewidth=0.8)

ax.axvline(reviews_per_rest.median(), color='#2C3E50', linestyle='--',
           linewidth=1.5, label=f'Median: {reviews_per_rest.median():.0f}')
ax.axvline(reviews_per_rest.mean(), color='#1ABC9C', linestyle='--',
           linewidth=1.5, label=f'Mean: {reviews_per_rest.mean():.1f}')

ax.set_title('Distribution of Reviews per Restaurant')
ax.set_xlabel('Number of Reviews')
ax.set_ylabel('Number of Restaurants')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/04_reviews_per_restaurant_hist.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Chart 4 saved: 04_reviews_per_restaurant_hist.png")

print(f"   Reviews per restaurant stats:")
print(f"   Min    : {reviews_per_rest.min()}")
print(f"   Max    : {reviews_per_rest.max()}")
print(f"   Median : {reviews_per_rest.median():.0f}")
print(f"   Mean   : {reviews_per_rest.mean():.1f}")
low_review = (reviews_per_rest <= 3).sum()
print(f"   Restaurants with ≤ 3 reviews: {low_review} ({low_review/len(reviews_per_rest)*100:.1f}%)")
print()


# ============================================================
# CHART 5 — Top 20 Most Frequent Words
# ============================================================
all_words = ' '.join(df['Cleaned_Text'].dropna()).split()
word_freq  = Counter(all_words).most_common(20)
words, freqs = zip(*word_freq)

fig, ax = plt.subplots(figsize=(10, 6))
colors = sns.color_palette(PALETTE, len(words))
bars = ax.barh(list(words)[::-1], list(freqs)[::-1],
               color=list(colors)[::-1], edgecolor='white', linewidth=0.8)

for bar, val in zip(bars, list(freqs)[::-1]):
    ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height() / 2,
            f'{val:,}', va='center', fontsize=9, fontweight='bold')

ax.set_title('Top 20 Most Frequent Words in Reviews')
ax.set_xlabel('Frequency')
ax.set_ylabel('Word')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/05_top20_words.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Chart 5 saved: 05_top20_words.png")

print(f"   Top 10 words:")
for word, freq in word_freq[:10]:
    print(f"   '{word}' → {freq:,} times")
print()


# ============================================================
# CHART 6 — Category Distribution (Top 10)
# ============================================================
fig, ax = plt.subplots(figsize=(9, 5))
cat_counts = df['Categories'].value_counts().head(10).sort_values(ascending=True)
colors = sns.color_palette(PALETTE, len(cat_counts))
bars = ax.barh(cat_counts.index, cat_counts.values,
               color=colors, edgecolor='white', linewidth=0.8)

for bar, val in zip(bars, cat_counts.values):
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
            f'{val:,}', va='center', fontsize=10, fontweight='bold')

ax.set_title('Top 10 Restaurant Categories')
ax.set_xlabel('Number of Reviews')
ax.set_ylabel('Category')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/06_category_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Chart 6 saved: 06_category_distribution.png")
print()


# ============================================================
# CHART 7 — Average Rating per District
# ============================================================
fig, ax = plt.subplots(figsize=(9, 5))
avg_rating = (df_district.groupby('Municipality')['Rating']
              .mean().sort_values(ascending=True))
colors = sns.color_palette(PALETTE, len(avg_rating))
bars = ax.barh(avg_rating.index, avg_rating.values,
               color=colors, edgecolor='white', linewidth=0.8)

for bar, val in zip(bars, avg_rating.values):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f'{val:.2f}★', va='center', fontsize=10, fontweight='bold')

ax.set_title('Average Rating per District')
ax.set_xlabel('Average Star Rating')
ax.set_ylabel('District')
ax.set_xlim(0, 6)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/07_avg_rating_per_district.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Chart 7 saved: 07_avg_rating_per_district.png")

print(f"   Average rating per district:")
for dist, avg in avg_rating.sort_values(ascending=False).items():
    print(f"   {dist:<22} → {avg:.2f}★")
print()


# ============================================================
# CHART 8 — Review Word Count Distribution
# ============================================================
fig, ax = plt.subplots(figsize=(9, 5))
word_counts = df['Cleaned_Text'].str.split().str.len()
ax.hist(word_counts, bins=40, color=ACCENT, edgecolor='white', linewidth=0.8)
ax.axvline(word_counts.median(), color='#2C3E50', linestyle='--',
           linewidth=1.5, label=f'Median: {word_counts.median():.0f} words')
ax.axvline(word_counts.mean(), color='#1ABC9C', linestyle='--',
           linewidth=1.5, label=f'Mean: {word_counts.mean():.1f} words')

ax.set_title('Review Length Distribution (Word Count)')
ax.set_xlabel('Number of Words per Review')
ax.set_ylabel('Number of Reviews')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/08_review_length_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Chart 8 saved: 08_review_length_distribution.png")

print(f"   Review length stats:")
print(f"   Min    : {word_counts.min()} words")
print(f"   Max    : {word_counts.max()} words")
print(f"   Median : {word_counts.median():.0f} words")
print(f"   Mean   : {word_counts.mean():.1f} words")
print()

# ============================================================
# FINAL SUMMARY
# ============================================================
print("=" * 55)
print("  EDA COMPLETE — All charts saved to /eda_outputs/")
print("=" * 55)
print(f"  01_rating_distribution.png")
print(f"  02_reviews_per_district.png")
print(f"  03_restaurants_per_district.png")
print(f"  04_reviews_per_restaurant_hist.png")
print(f"  05_top20_words.png")
print(f"  06_category_distribution.png")
print(f"  07_avg_rating_per_district.png")
print(f"  08_review_length_distribution.png")
print("=" * 55)
print("\n  ➡  Next step: step4_lda_modeling.py\n")
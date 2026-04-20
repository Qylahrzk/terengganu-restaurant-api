import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import os

# ── CONFIG ────────────────────────────────────────────────────────────────────
IN_FILE     = 'master_terengganu_preprocessed.csv'
OUT_DIR     = 'eda_outputs'
DISTRICTS   = ['Kuala Terengganu', 'Besut', 'Kemaman', 'Hulu Terengganu', 
               'Dungun', 'Marang', 'Setiu', 'Kuala Nerus']

# ── SETUP ─────────────────────────────────────────────────────────────────────
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

print("Loading data...")
if not os.path.exists(IN_FILE):
    print(f"Error: {IN_FILE} not found. Run Step 2 first!")
    exit()

df = pd.read_csv(IN_FILE)

# Standardise column headers (Handle variations in case)
df.columns = [c.title() if c != 'has_sufficient_text' else 'has_sufficient_text' for c in df.columns]

# Map common variations to the expected names
col_map = {'Review_Text': 'Reviews', 'Cleaned_text': 'Cleaned_Text', 'Cleaned_text': 'Cleaned_Text'} 
df = df.rename(columns=col_map)

# FIX: Force Cleaned_Text to be string and handle NaN values
df['Cleaned_Text'] = df['Cleaned_Text'].fillna('').astype(str)

# Calculate word count safely
df['word_count'] = df['Cleaned_Text'].apply(lambda x: len(str(x).split()))

# Filter for Official Districts
df_district = df[df['Municipality'].isin(DISTRICTS)].copy()

# ── GENERATE CHARTS ───────────────────────────────────────────────────────────
sns.set_style("whitegrid")
print(f"Generating charts in /{OUT_DIR}...")

# 1. Rating Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['Rating'].dropna(), bins=10, kde=True, color='orange')
plt.title('01. Rating Distribution')
plt.savefig(f'{OUT_DIR}/01_rating_distribution.png')
plt.close()

# 2. Reviews per District
plt.figure(figsize=(10, 6))
sns.countplot(data=df_district, y='Municipality', order=df_district['Municipality'].value_counts().index, palette='viridis')
plt.title('02. Total Reviews per District')
plt.savefig(f'{OUT_DIR}/02_reviews_per_district.png')
plt.close()

# 3. Restaurants per District
plt.figure(figsize=(10, 6))
dist_counts = df_district.groupby('Municipality')['Name'].nunique().sort_values(ascending=False)
sns.barplot(x=dist_counts.values, y=dist_counts.index, palette='mako')
plt.title('03. Unique Restaurants per District')
plt.savefig(f'{OUT_DIR}/03_restaurants_per_district.png')
plt.close()

# 4. Reviews per Restaurant
review_counts = df.groupby('Name').size()
plt.figure(figsize=(10, 5))
sns.histplot(review_counts, bins=50, color='teal')
plt.title('04. Reviews per Restaurant Distribution')
plt.savefig(f'{OUT_DIR}/04_reviews_per_restaurant_hist.png')
plt.close()

# 5. Top 20 Words
all_text = " ".join(df[df['Cleaned_Text'] != '']['Cleaned_Text'])
word_freq = Counter(all_text.split()).most_common(20)
df_words = pd.DataFrame(word_freq, columns=['Word', 'Frequency'])
plt.figure(figsize=(12, 6))
sns.barplot(data=df_words, x='Frequency', y='Word', palette='plasma')
plt.title('05. Top 20 Most Frequent Words')
plt.savefig(f'{OUT_DIR}/05_top20_words.png')
plt.close()

# 6. Category Distribution
plt.figure(figsize=(10, 8))
top_cats = df['Categories'].value_counts().head(15)
sns.barplot(x=top_cats.values, y=top_cats.index, palette='magma')
plt.title('06. Top 15 Restaurant Categories')
plt.savefig(f'{OUT_DIR}/06_category_distribution.png')
plt.close()

# 7. Avg Rating per District
plt.figure(figsize=(10, 6))
avg_ratings = df_district.groupby('Municipality')['Rating'].mean().sort_values()
sns.barplot(x=avg_ratings.values, y=avg_ratings.index, palette='coolwarm')
plt.xlim(0, 5)
plt.title('07. Average Rating per District')
plt.savefig(f'{OUT_DIR}/07_avg_rating_per_district.png')
plt.close()

# 8. Review Length Distribution
plt.figure(figsize=(10, 5))
sns.histplot(df['word_count'], bins=30, kde=True, color='purple')
plt.title('08. Review Length Distribution')
plt.savefig(f'{OUT_DIR}/08_review_length_distribution.png')
plt.close()

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print("=" * 55)
print("  EXPLORATORY DATA ANALYSIS COMPLETE")
print("=" * 55)
print(f"  Total reviews          : {len(df):,}")
print(f"  Unique restaurants     : {df['Name'].nunique():,}")
print(f"  Avg words per review   : {df['word_count'].mean():.1f}")
print(f"  Charts saved to        : /{OUT_DIR}/")
print("=" * 55)

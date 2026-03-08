import pandas as pd

# ── Run ONCE beforehand ──────────────────────────────
# preprocessing_terengganu.py  →  produces master_terengganu_preprocessed.csv
# ────────────────────────────────────────────────────

# ── Then in your main application ───────────────────

# KBF pool: ALL 1,051 restaurants (with or without reviews)
df_restaurants = pd.read_csv('master_terengganu_unified.csv')
df_restaurants = df_restaurants.drop_duplicates(subset=['Name']).copy()
df_restaurants = df_restaurants[['Name', 'Municipality', 'Categories',
                                  'Latitude', 'Longitude', 'Address', 'Rating']]
print(f"KBF restaurant pool: {len(df_restaurants)} restaurants")

# LDA pool: 857 restaurants with 5,192 valid reviews
df_lda = pd.read_csv('master_terengganu_preprocessed.csv')
print(f"LDA review pool: {df_lda['Name'].nunique()} restaurants, {len(df_lda)} reviews")

# ── Later in your pipeline ───────────────────────────
# df_lda        → feeds into LDA topic modelling
# df_restaurants → feeds into KBF filtering
# Both results  → merged at recommendation stage
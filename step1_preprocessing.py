import pandas as pd
import re

# ============================================================
# STEP 1: Load the Original Unified Data
# ============================================================
df = pd.read_csv('master_terengganu_unified.csv')
print(f"Original rows: {len(df)}")
print(f"Original columns: {df.columns.tolist()}\n")


# ============================================================
# STEP 2: Drop Missing Essential Data
# Remove rows that don't have a restaurant name or review text
# ============================================================
df_clean = df.dropna(subset=['Name', 'Cleaned_Text']).copy()
print(f"After dropping missing Name/Cleaned_Text: {len(df_clean)} rows")


# ============================================================
# STEP 3: Clean Restaurant Names
# ============================================================
def clean_restaurant_name(name):
    if not isinstance(name, str):
        return name

    # Remove "@" location tags (e.g., "KFC @ KTCC" -> "KFC")
    name = re.sub(r'\s*@.*$', '', name)

    # Remove trailing city/state suffixes
    cities = (r'(Dungun|Terengganu|Kemaman|Kuala Besut|Kuala Terengganu'
              r'|Besut|Marang|Setiu|Hulu Terengganu)')
    name = re.sub(rf'[\s,\-]+{cities}$', '', name, flags=re.IGNORECASE)

    # Remove trailing punctuation like dashes or commas left over
    name = re.sub(r'[\s\-,]+$', '', name)

    # Remove non-ASCII characters (weird symbols)
    name = name.encode('ascii', 'ignore').decode('ascii')

    # Standardize whitespace and convert to Title Case
    name = re.sub(r'\s+', ' ', name).strip()
    return name.title()

df_clean['Name'] = df_clean['Name'].apply(clean_restaurant_name)


# ============================================================
# STEP 4: Clean & Standardize Municipality
# Handles: postcodes, junk entries, known district aliases
# ============================================================
# Remove postcodes prefix (e.g., "21000 Kuala Terengganu" -> "Kuala Terengganu")
df_clean['Municipality'] = (
    df_clean['Municipality']
    .astype(str)
    .str.replace(r'^\d{5}\s*', '', regex=True)
    .str.strip()
    .str.title()
)

# Map known aliases and messy municipality names to clean district names
municipality_map = {
    'Kuala Terengganu'  : 'Kuala Terengganu',
    'Terengganu'        : 'Kuala Terengganu',   # generic → default to capital
    'Kuala Berang'      : 'Hulu Terengganu',
    'Kuala Dungun'      : 'Dungun',
    'Dungun'            : 'Dungun',
    'Chukai'            : 'Kemaman',
    'Kemaman'           : 'Kemaman',
    'Kijal'             : 'Kemaman',
    'Kerteh'            : 'Kemaman',
    'Permaisuri'        : 'Besut',
    'Kampung Raja'      : 'Besut',
    'Besut'             : 'Besut',
    'Jerteh'            : 'Besut',
    'Marang'            : 'Marang',
    'Setiu'             : 'Setiu',
    'Hulu Terengganu'   : 'Hulu Terengganu',
}

def map_municipality(muni):
    if not isinstance(muni, str):
        return 'Unknown'
    # Partial match against known keys
    for key, district in municipality_map.items():
        if key.lower() in muni.lower():
            return district
    return muni.strip() if muni.strip() else 'Unknown'

df_clean['Municipality'] = df_clean['Municipality'].apply(map_municipality)


# ============================================================
# STEP 5: Normalize Categories
# Split multi-label categories and take the primary (first) one
# ============================================================
def normalize_category(cat):
    if not isinstance(cat, str) or cat.strip() == '':
        return 'Restaurant'   # default fallback
    # Take only the first label if multiple exist (e.g., "Restaurant, Cafe" -> "Restaurant")
    primary = cat.split(',')[0].strip()
    return primary.title()

df_clean['Categories'] = df_clean['Categories'].apply(normalize_category)


# ============================================================
# STEP 6: Standardize Rating
# Round non-standard ratings (e.g., 3.7, 4.1) to nearest 0.5
# and clip to valid range [1.0, 5.0]
# ============================================================
def round_rating(r):
    if pd.isna(r):
        return r
    # Round to nearest 0.5
    rounded = round(r * 2) / 2
    # Clip to valid scale
    return max(1.0, min(5.0, rounded))

df_clean['Rating'] = df_clean['Rating'].apply(round_rating)

# Fill missing ratings with the median per restaurant, then global median
median_rating = df_clean['Rating'].median()
df_clean['Rating'] = (
    df_clean.groupby('Name')['Rating']
    .transform(lambda x: x.fillna(x.median()))
    .fillna(median_rating)
)


# ============================================================
# STEP 7: Filter Junk / Low-Quality Reviews
# ============================================================
# Remove reviews without at least one letter (emoji-only, numbers-only)
df_clean = df_clean[df_clean['Cleaned_Text'].str.contains('[a-zA-Z]', na=False)]

# Remove extremely short reviews (≤ 2 words) — too noisy for LDA
df_clean = df_clean[df_clean['Cleaned_Text'].str.split().str.len() > 2]


# ============================================================
# STEP 8: Deduplication
# Remove exact duplicate rows, then near-duplicate (Name + Review_Text)
# ============================================================
before_dedup = len(df_clean)
df_clean = df_clean.drop_duplicates()
df_clean = df_clean.drop_duplicates(subset=['Name', 'Review_Text'])
print(f"Rows removed by deduplication: {before_dedup - len(df_clean)}")


# ============================================================
# STEP 9: Reset Index
# ============================================================
df_clean = df_clean.reset_index(drop=True)


# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n========== PREPROCESSING SUMMARY ==========")
print(f"Original rows           : {len(df)}")
print(f"Cleaned rows            : {len(df_clean)}")
print(f"Rows removed            : {len(df) - len(df_clean)}")
print(f"Unique restaurants      : {df_clean['Name'].nunique()}")
print(f"Unique municipalities   : {df_clean['Municipality'].nunique()}")
print(f"Municipality distribution:\n{df_clean['Municipality'].value_counts().to_string()}")
print(f"\nRating distribution:\n{df_clean['Rating'].value_counts().sort_index().to_string()}")
print(f"\nTop 10 Categories:\n{df_clean['Categories'].value_counts().head(10).to_string()}")
print(f"\nCleaned_Text word count stats:")
print(df_clean['Cleaned_Text'].str.split().str.len().describe().to_string())
print("============================================\n")

# Save the preprocessed file
df_clean.to_csv('master_terengganu_preprocessed.csv', index=False)
print("Saved: master_terengganu_preprocessed.csv")
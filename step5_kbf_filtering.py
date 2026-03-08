import pandas as pd
import numpy as np
import os
import re
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
INPUT_PREPROCESSED = 'master_terengganu_preprocessed.csv'
INPUT_UNIFIED      = 'master_terengganu_unified.csv'
INPUT_LDA_TOPICS   = 'lda_outputs/lda_restaurant_topics.csv'
INPUT_LDA_LABELS   = 'lda_outputs/lda_topic_labels.csv'
OUTPUT_DIR         = 'kbf_outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# STEP 1: Load Data
# ============================================================
print("=" * 60)
print("  KBF FILTERING — TERENGGANU RESTAURANT RECOMMENDER")
print("=" * 60)

# Load ALL restaurants (including those with no reviews) for KBF pool
df_all = pd.read_csv(INPUT_UNIFIED)
df_all = df_all.drop_duplicates(subset=['Name']).copy()
df_all = df_all[['Name', 'Municipality', 'Categories',
                  'Latitude', 'Longitude', 'Address', 'Rating']].reset_index(drop=True)

# Load preprocessed reviews for keyword detection
df_reviews = pd.read_csv(INPUT_PREPROCESSED)

# Load LDA topic results
df_lda      = pd.read_csv(INPUT_LDA_TOPICS)
df_labels   = pd.read_csv(INPUT_LDA_LABELS)

print(f"\n✅ KBF restaurant pool     : {len(df_all):,} restaurants")
print(f"✅ Reviews for keyword scan : {len(df_reviews):,} reviews")
print(f"✅ LDA topics loaded        : {len(df_lda):,} restaurants\n")


# ============================================================
# STEP 2: Clean Restaurant Names & Municipality
# ============================================================
def clean_name(name):
    if not isinstance(name, str):
        return name
    name = re.sub(r'\s*@.*$', '', name)
    cities = (r'(Dungun|Terengganu|Kemaman|Kuala Besut|Kuala Terengganu'
              r'|Besut|Marang|Setiu|Hulu Terengganu)')
    name = re.sub(rf'[\s,\-]+{cities}$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'[\s\-,]+$', '', name)
    name = name.encode('ascii', 'ignore').decode('ascii')
    name = re.sub(r'\s+', ' ', name).strip()
    return name.title()

municipality_map = {
    'Kuala Terengganu' : 'Kuala Terengganu',
    'Terengganu'       : 'Kuala Terengganu',
    'Kuala Berang'     : 'Hulu Terengganu',
    'Kuala Dungun'     : 'Dungun',
    'Dungun'           : 'Dungun',
    'Chukai'           : 'Kemaman',
    'Kemaman'          : 'Kemaman',
    'Kijal'            : 'Kemaman',
    'Kerteh'           : 'Kemaman',
    'Permaisuri'       : 'Besut',
    'Kampung Raja'     : 'Besut',
    'Besut'            : 'Besut',
    'Jerteh'           : 'Besut',
    'Marang'           : 'Marang',
    'Setiu'            : 'Setiu',
    'Hulu Terengganu'  : 'Hulu Terengganu',
}

def map_municipality(muni):
    if not isinstance(muni, str):
        return 'Unknown'
    muni = re.sub(r'^\d{5}\s*', '', str(muni)).strip()
    for key, district in municipality_map.items():
        if key.lower() in muni.lower():
            return district
    return muni.strip().title() if muni.strip() else 'Unknown'

df_all['Name']         = df_all['Name'].apply(clean_name)
df_all['Municipality'] = df_all['Municipality'].apply(map_municipality)
df_reviews['Name']     = df_reviews['Name'].apply(clean_name)

print("✅ Names and municipality cleaned\n")


# ============================================================
# STEP 3: Map Cuisine Type
# ============================================================
def normalize_category(cat):
    if not isinstance(cat, str) or cat.strip() == '':
        return 'Restaurant'
    return cat.split(',')[0].strip().title()

cuisine_map = {
    'Malaysian Restaurant'    : 'Malay',
    'Hawker Stall'            : 'Malay',
    'Satay Restaurant'        : 'Malay',
    'Ikan Bakar Restaurant'   : 'Malay',
    'Nasi Goreng Restaurant'  : 'Malay',
    'Rice Restaurant'         : 'Malay',
    'Lunch Restaurant'        : 'Malay',
    'Noodle Shop'             : 'Malay',
    'Soup Restaurant'         : 'Malay',
    'Soup Shop'               : 'Malay',
    'Western Restaurant'      : 'Western',
    'Steak House'             : 'Western',
    'Fish & Chips Restaurant' : 'Western',
    'Bistro'                  : 'Western',
    'Pizza Restaurant'        : 'Western',
    'Hamburger Restaurant'    : 'Western',
    'Sandwich Shop'           : 'Western',
    'Diner'                   : 'Western',
    'Chinese Restaurant'      : 'Chinese',
    'Dim Sum Restaurant'      : 'Chinese',
    'Cantonese Restaurant'    : 'Chinese',
    'Chinese Noodle Restaurant': 'Chinese',
    'Seafood Restaurant'      : 'Seafood',
    'Dried Seafood Store'     : 'Seafood',
    'Fast Food Restaurant'    : 'Fast Food',
    'Chicken Restaurant'      : 'Fast Food',
    'Chicken Wings Restaurant': 'Fast Food',
    'Delivery Restaurant'     : 'Fast Food',
    'Takeout Restaurant'      : 'Fast Food',
    'Cafe'                    : 'Cafe',
    'Coffee Shop'             : 'Cafe',
    'Brunch Restaurant'       : 'Cafe',
    'Breakfast Restaurant'    : 'Cafe',
    'Bakery'                  : 'Cafe',
    'Dessert Shop'            : 'Cafe',
    'Ice Cream Shop'          : 'Cafe',
    'Thai Restaurant'         : 'Thai',
    'Indian Restaurant'       : 'Indian',
    'Japanese Restaurant'     : 'Japanese',
    'Korean Restaurant'       : 'Korean',
    'Indonesian Restaurant'   : 'Indonesian',
    'Asian Restaurant'        : 'Asian',
    'Asian Fusion Restaurant' : 'Asian',
    'Fusion Restaurant'       : 'Asian',
    'Vegetarian Restaurant'   : 'Vegetarian',
    'Italian Restaurant'      : 'Italian',
    'Buffet Restaurant'       : 'Buffet',
    'Family Restaurant'       : 'Family',
    'Middle Eastern Restaurant': 'Middle Eastern',
    'Barbecue Restaurant'     : 'BBQ',
    'Ayam Penyet Restaurant'  : 'Indonesian',
    'Kebab Shop'              : 'Middle Eastern',
}

df_all['Categories']  = df_all['Categories'].apply(normalize_category)
df_all['Cuisine_Type'] = df_all['Categories'].map(cuisine_map).fillna('Other')

print("✅ Cuisine types mapped\n")


# ============================================================
# STEP 4: Standardize Rating
# ============================================================
def round_rating(r):
    if pd.isna(r):
        return r
    rounded = round(r * 2) / 2
    return max(1.0, min(5.0, rounded))

df_all['Rating'] = df_all['Rating'].apply(round_rating)

# Fill missing ratings with global median
median_rating = df_all['Rating'].median()
df_all['Rating'] = df_all['Rating'].fillna(median_rating)

print(f"✅ Ratings standardized (median fallback: {median_rating}★)\n")


# ============================================================
# STEP 5: Build Review Text Pool per Restaurant
# (Aggregate all reviews per restaurant for keyword scanning)
# ============================================================
review_pool = (
    df_reviews.groupby('Name')['Review_Text']
    .apply(lambda x: ' '.join(x.dropna().astype(str)).lower())
    .reset_index()
    .rename(columns={'Review_Text': 'All_Reviews'})
)


def has_keyword(name, keywords):
    """Check if any review for this restaurant contains any keyword."""
    row = review_pool[review_pool['Name'] == name]
    if row.empty:
        return 'Unknown'
    text = row.iloc[0]['All_Reviews']
    return 'Yes' if any(kw in text for kw in keywords) else 'No'


# ============================================================
# STEP 6: Extract KBF Filter Attributes via Keyword Detection
# ============================================================
print("⏳ Extracting KBF filter attributes from reviews...")
print("   (Scanning keywords across all reviews per restaurant)\n")

# --- 6a. Halal ---
halal_keywords = [
    'halal', 'halaal', 'no pork', 'no alcohol', 'muslim friendly',
    'muslim-friendly', 'tiada babi', 'tidak ada babi', 'tanpa babi'
]

# --- 6b. Vegetarian / Vegan ---
vegetarian_keywords = [
    'vegetarian', 'veggie', 'no meat', 'plant based', 'plant-based',
    'sayur', 'sayuran', 'tanpa daging'
]
vegan_keywords = [
    'vegan', 'no animal', 'dairy free', 'dairy-free', 'no egg',
    'no dairy', 'plant based', 'plant-based'
]

# --- 6c. Parking ---
parking_keywords = [
    'parking', 'park', 'car park', 'carpark', 'tempat letak',
    'easy to park', 'ample parking', 'free parking', 'parking available'
]

# --- 6d. Family-Friendly ---
family_keywords = [
    'family', 'families', 'kids', 'children', 'child', 'baby',
    'stroller', 'high chair', 'family friendly', 'bring kids',
    'keluarga', 'kanak-kanak', 'anak'
]

# --- 6e. Romantic / Date ---
romantic_keywords = [
    'romantic', 'date', 'couple', 'anniversary', 'candlelight',
    'intimate', 'honeymoon', 'propose', 'dating', 'pasangan'
]

# --- 6f. Scenic / Beach View ---
scenic_keywords = [
    'beach', 'sea view', 'ocean view', 'river view', 'scenic',
    'waterfront', 'pantai', 'tepi laut', 'view', 'sunset',
    'beautiful view', 'nice view', 'overlooking', 'tepi sungai'
]

# --- 6g. Outdoor Seating ---
outdoor_keywords = [
    'outdoor', 'outside', 'open air', 'al fresco', 'garden',
    'rooftop', 'terrace', 'taman', 'luar', 'open space'
]

# --- 6h. Wi-Fi ---
wifi_keywords = [
    'wifi', 'wi-fi', 'wireless', 'internet', 'free wifi',
    'good wifi', 'wifi available'
]

# Apply keyword detection for each restaurant
print("   Detecting: Halal...")
df_all['Is_Halal'] = df_all['Name'].apply(
    lambda n: 'Yes' if (
        df_all.loc[df_all['Name'] == n, 'Cuisine_Type'].values[0] in
        ['Malay', 'Indian', 'Middle Eastern', 'Indonesian']
        or has_keyword(n, halal_keywords) == 'Yes'
    ) else has_keyword(n, halal_keywords)
)

print("   Detecting: Vegetarian...")
df_all['Is_Vegetarian'] = df_all['Name'].apply(
    lambda n: 'Yes' if (
        df_all.loc[df_all['Name'] == n, 'Cuisine_Type'].values[0] == 'Vegetarian'
        or has_keyword(n, vegetarian_keywords) == 'Yes'
    ) else has_keyword(n, vegetarian_keywords)
)

print("   Detecting: Vegan...")
df_all['Is_Vegan'] = df_all['Name'].apply(
    lambda n: has_keyword(n, vegan_keywords)
)

print("   Detecting: Parking...")
df_all['Has_Parking'] = df_all['Name'].apply(
    lambda n: has_keyword(n, parking_keywords)
)

print("   Detecting: Family-Friendly...")
df_all['Is_Family_Friendly'] = df_all['Name'].apply(
    lambda n: has_keyword(n, family_keywords)
)

print("   Detecting: Romantic/Date...")
df_all['Is_Romantic'] = df_all['Name'].apply(
    lambda n: has_keyword(n, romantic_keywords)
)

print("   Detecting: Scenic/Beach View...")
df_all['Has_Scenic_View'] = df_all['Name'].apply(
    lambda n: has_keyword(n, scenic_keywords)
)

print("   Detecting: Outdoor Seating...")
df_all['Has_Outdoor'] = df_all['Name'].apply(
    lambda n: has_keyword(n, outdoor_keywords)
)

print("   Detecting: Wi-Fi...")
df_all['Has_Wifi'] = df_all['Name'].apply(
    lambda n: has_keyword(n, wifi_keywords)
)

print("\n✅ All KBF attributes extracted\n")


# ============================================================
# STEP 7: Merge LDA Topic Results
# ============================================================
print("⏳ Merging LDA topic assignments...")

# Build topic label dictionary from lda_topic_labels.csv
topic_label_dict = dict(zip(
    df_labels['Topic_ID'] + 1,   # Topic_ID is 0-indexed, our IDs are 1-indexed
    df_labels['Label']
))

df_lda['Topic_Label'] = df_lda['Dominant_Topic'].map(topic_label_dict)

df_all = pd.merge(
    df_all,
    df_lda[['Name', 'Dominant_Topic', 'Topic_Label',
            'Topic_1_Pct', 'Topic_2_Pct', 'Topic_3_Pct']],
    on='Name',
    how='left'
)

# Restaurants with no reviews get 'Unknown' topic
df_all['Dominant_Topic'] = df_all['Dominant_Topic'].fillna(0).astype(int)
df_all['Topic_Label']    = df_all['Topic_Label'].fillna('No Reviews')

print(f"✅ LDA topics merged\n")


# ============================================================
# STEP 8: Add Rating Band (for easy filtering)
# ============================================================
def rating_band(r):
    if pd.isna(r):   return 'Unrated'
    if r >= 4.5:     return 'Excellent (≥4.5★)'
    if r >= 4.0:     return 'Very Good (≥4.0★)'
    if r >= 3.0:     return 'Good (≥3.0★)'
    return 'Below Average (<3.0★)'

df_all['Rating_Band'] = df_all['Rating'].apply(rating_band)

print("✅ Rating bands assigned\n")


# ============================================================
# STEP 9: Final Column Organisation
# ============================================================
final_cols = [
    # Identity
    'Name', 'Address', 'Municipality',
    # Cuisine & Category
    'Categories', 'Cuisine_Type',
    # Rating
    'Rating', 'Rating_Band',
    # Location
    'Latitude', 'Longitude',
    # KBF Keyword Filters
    'Is_Halal', 'Is_Vegetarian', 'Is_Vegan',
    'Has_Parking', 'Is_Family_Friendly',
    'Is_Romantic', 'Has_Scenic_View',
    'Has_Outdoor', 'Has_Wifi',
    # LDA Topic
    'Dominant_Topic', 'Topic_Label',
    'Topic_1_Pct', 'Topic_2_Pct', 'Topic_3_Pct',
]

df_final = df_all[final_cols].copy()


# ============================================================
# STEP 10: Save Output
# ============================================================
df_final.to_csv(f'{OUTPUT_DIR}/kbf_restaurant_profiles.csv', index=False)
print(f"✅ Saved: kbf_restaurant_profiles.csv\n")


# ============================================================
# FINAL SUMMARY
# ============================================================
print("=" * 60)
print("  KBF FILTERING SUMMARY")
print("=" * 60)
print(f"  Total restaurants profiled   : {len(df_final):,}")
print(f"\n  --- Cuisine Type Breakdown ---")
for cuisine, count in df_final['Cuisine_Type'].value_counts().items():
    print(f"  {cuisine:<25} : {count:,}")

print(f"\n  --- District Breakdown ---")
for district, count in df_final['Municipality'].value_counts().head(10).items():
    print(f"  {district:<25} : {count:,}")

print(f"\n  --- Rating Band Breakdown ---")
for band, count in df_final['Rating_Band'].value_counts().items():
    print(f"  {band:<30} : {count:,}")

print(f"\n  --- KBF Keyword Filter Summary ---")
kbf_cols = ['Is_Halal','Is_Vegetarian','Is_Vegan','Has_Parking',
            'Is_Family_Friendly','Is_Romantic','Has_Scenic_View',
            'Has_Outdoor','Has_Wifi']
for col in kbf_cols:
    yes_count = (df_final[col] == 'Yes').sum()
    print(f"  {col:<25} : {yes_count:,} restaurants detected")

print(f"\n  --- LDA Topic Distribution ---")
for label, count in df_final['Topic_Label'].value_counts().items():
    print(f"  {label:<30} : {count:,}")

print("=" * 60)
print("\n  Output files in /kbf_outputs/:")
print("  - kbf_restaurant_profiles.csv")
print("=" * 60)
print("\n  ➡  Next step: step6_hybrid_recommendation.py\n")
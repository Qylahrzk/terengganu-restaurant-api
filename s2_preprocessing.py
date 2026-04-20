import pandas as pd
import re
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('omw-1.4', quiet=True)

# File names based on your scraper output
IN_FILE  = 'master_990_with_reviews.csv'
OUT_FILE = 'master_990_terengganu_preprocessed.csv' # Matching your app's expected name

# ─────────────────────────────────────────────────────────────────────────────

STOP_WORDS = set(stopwords.words('english'))
EXTRA_STOPS = {
    'food', 'place', 'restaurant', 'eat', 'come', 'get', 'go', 'also',
    'would', 'could', 'really', 'very', 'quite', 'well', 'one', 'time',
    'staff', 'service', 'price', 'order', 'ordered', 'came', 'got',
    'try', 'tried', 'recommend', 'good', 'great', 'nice', 'bad',
    'like', 'love', 'best', 'better', 'taste', 'tasty', 'delicious',
    'terengganu', 'malaysia', 'kuala', 'restoran', 'kedai', 'makan', 'mayang', 'mall'
}
STOP_WORDS.update(EXTRA_STOPS)
lemmatizer = WordNetLemmatizer()

def clean_review(text):
    if not text or pd.isna(text) or str(text).strip() == '':
        return ''
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', ' ', text) # URLs
    text = re.sub(r'\S+@\S+', ' ', text)       # Emails
    text = text.encode('ascii', 'ignore').decode('ascii') # Non-ASCII
    text = re.sub(r'[^a-z\s]', ' ', text)      # Punctuation/Numbers
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

# ── Load ──────────────────────────────────────────────────────────────────────
if not os.path.exists(IN_FILE):
    print(f"Error: {IN_FILE} not found. Run Step 1 first!")
    exit()

print("Loading master file...")
df = pd.read_csv(IN_FILE)

# Logic fix: Ensure column names match scraper output
# If scraper used 'reviews', we map it to 'review_text' for consistency
if 'reviews' in df.columns and 'review_text' not in df.columns:
    df = df.rename(columns={'reviews': 'review_text'})

# ── Clean review text ─────────────────────────────────────────────────────────
print("Cleaning review text...")
df['cleaned_text'] = df['review_text'].apply(clean_review)

# Mark restaurants with sufficient text for LDA (at least 5 useful words)
df['has_sufficient_text'] = df['cleaned_text'].apply(
    lambda x: len(str(x).split()) >= 5
)

# ── Clean other fields (KBF Data) ─────────────────────────────────────────────
print("Standardizing municipality and categories...")
municipality_map = {
    'kuala terengganu': 'Kuala Terengganu',
    'besut': 'Besut', 'kemaman': 'Kemaman', 'dungun': 'Dungun',
    'hulu terengganu': 'Hulu Terengganu', 'marang': 'Marang', 'setiu': 'Setiu'
}
df['municipality'] = df['municipality'].str.lower().str.strip().map(
    lambda x: municipality_map.get(x, str(x).title())
)

df['address']      = df['address'].fillna('Address not available')
df['cuisine_type'] = df['cuisine_type'].fillna('Other')
df['categories']   = df['categories'].fillna('Restaurant')
df['rating']       = pd.to_numeric(df['rating'], errors='coerce').fillna(0.0)

# ── Save ──────────────────────────────────────────────────────────────────────
# Save the full preprocessed file
df.to_csv(OUT_FILE, index=False, encoding='utf-8-sig')

# Save a clean version for the KBF pool (Unified list)
df_unified = df[['name', 'municipality', 'categories', 'latitude', 'longitude', 'address', 'rating']]
df_unified.columns = [c.title() for c in df_unified.columns] # Capitalize headers to match old code
df_unified.to_csv('master_990_terengganu_unified.csv', index=False)

print(f"\n{'='*40}")
print(f"  STEP 2 COMPLETE")
print(f"{'='*40}")
print(f"  Total restaurants in pool : {len(df)}")
print(f"  Ready for LDA (with text) : {df['has_sufficient_text'].sum()}")
print(f"  KBF Unified File created  : master_990_terengganu_unified.csv")
print(f"  LDA Preprocessed created  : {OUT_FILE}")

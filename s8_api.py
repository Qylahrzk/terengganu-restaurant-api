"""
STEP 8 — Flask API (Terengganu Restaurant Recommender) v2.1
============================================================
Endpoints:
  GET  /health              → API status check
  GET  /restaurants         → all restaurants (optional filters)
  GET  /restaurants/nearby  → nearby restaurants by GPS
  POST /recommend           → hybrid recommendation (30% KBF + 70% LDA)
  POST /chat                → RAG-powered AI chat (Gemini 1.5 Flash) [NEW v2.1]

Changes vs v2.0:
  1. Added POST /chat endpoint (RAG pattern)
     - Searches restaurant list with keyword + attribute matching
     - Injects top matches as context into Gemini prompt
     - Returns natural language answer + matching restaurants
  2. Added google-generativeai import + model init
  3. GEMINI_API_KEY read from environment variable

Deploy:
  git add step8_api.py requirements.txt
  git commit -m "v2.1: add /chat RAG endpoint with Gemini"
  git push

Run locally:
  pip install flask flask-cors supabase google-generativeai
  GEMINI_API_KEY=your_key python step8_api.py
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from supabase import create_client, Client
from math import radians, sin, cos, sqrt, atan2
import os
import threading
import time
import warnings
warnings.filterwarnings('ignore')

# ── Gemini import (safe — won't crash if key is missing) ──────────────────────
try:
    import google.generativeai as genai
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False
    print("[chat] google-generativeai not installed — /chat will return error")


# ── CONFIGURATION ─────────────────────────────────────────────────────────────
SUPABASE_URL = os.environ.get('SUPABASE_URL', 'https://turaafqegjhsijpaooli.supabase.co')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InR1cmFhZnFlZ2poc2lqcGFvb2xpIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzE5MDM5OTQsImV4cCI6MjA4NzQ3OTk5NH0.jUSbHp7kjVoxcOVQDUKoWGG6h38CbLIqqu2YdtzSPs8')
GEMINI_KEY   = os.environ.get('GEMINI_API_KEY')

KBF_WEIGHT = 0.30
LDA_WEIGHT = 0.70
TOP_N      = 10

# ⚠️  Must match TOPIC_LABELS in s3_lda.py exactly (including spacing)
TOPIC_LABEL_TO_ID = {
    'Casual Dining & Variety'         : 1,
    'Malay Breakfast & Local Staples' : 2,
    'Local Snacks & Specialty Bites'  : 3,
    'Fast Food & Service Quality'     : 4,
    'Popular Local Favorites'         : 5,
    'Comfort Food & Value Meals'      : 6,
}
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ── Initialise Gemini model ───────────────────────────────────────────────────
# Model is initialised once at startup and reused for all /chat requests.
# If GEMINI_API_KEY is not set, _gemini_model stays None and /chat returns
# a helpful error message instead of crashing.
_gemini_model = None
if _GENAI_AVAILABLE and GEMINI_KEY:
    try:
        genai.configure(api_key=GEMINI_KEY)
        _gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        print("[chat] Gemini 1.5 Flash model ready")
    except Exception as e:
        print(f"[chat] Gemini init failed: {e}")
else:
    if not GEMINI_KEY:
        print("[chat] GEMINI_API_KEY not set — /chat disabled")


# ── HELPERS ───────────────────────────────────────────────────────────────────

def haversine(lat1, lon1, lat2, lon2):
    """Distance in km between two GPS coordinates (Haversine formula)."""
    R    = 6371
    dlat = radians(float(lat2) - float(lat1))
    dlon = radians(float(lon2) - float(lon1))
    a    = (sin(dlat / 2) ** 2 +
            cos(radians(float(lat1))) *
            cos(radians(float(lat2))) *
            sin(dlon / 2) ** 2)
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def distance_label(km, coordinate_source):
    """Human-readable distance string for Flutter UI."""
    if coordinate_source == 'original':
        return f"{km:.1f} km away"
    elif coordinate_source == 'geocoded':
        return f"~{km:.1f} km away (estimated)"
    else:
        return "Nearby"


def load_restaurants():
    """Fetch all restaurant profiles from Supabase."""
    try:
        response = supabase.table('restaurant_profiles').select('*').execute()
        return response.data if response.data else []
    except Exception as e:
        print(f"Supabase error: {e}")
        return []


# ── SCORING ───────────────────────────────────────────────────────────────────

def compute_kbf_score(restaurant, preferences):
    """
    Score a restaurant based on KBF preference matches.
    Returns 0.0–1.0.
    Only attributes the user actually set (truthy) contribute to the score.
    """
    score      = 0.0
    max_points = 0

    checks = [
        ('district',        lambda r: r.get('municipality', '').lower() == preferences['district'].lower()),
        ('cuisine',         lambda r: r.get('cuisine_type', '').lower() == preferences['cuisine'].lower()),
        ('min_rating',      lambda r: float(r.get('rating', 0)) >= float(preferences['min_rating'])),
        ('halal',           lambda r: r.get('is_halal') == True),
        ('vegetarian',      lambda r: r.get('is_vegetarian') == True),
        ('vegan',           lambda r: r.get('is_vegan') == True),
        ('parking',         lambda r: r.get('has_parking') == True),
        ('wifi',            lambda r: r.get('has_wifi') == True),
        ('ac',              lambda r: r.get('has_ac') == True),
        ('outdoor',         lambda r: r.get('has_outdoor') == True),
        ('accessible',      lambda r: r.get('is_accessible') == True),
        ('family_friendly', lambda r: r.get('is_family_friendly') == True),
        ('group_friendly',  lambda r: r.get('is_group_friendly') == True),
        ('casual',          lambda r: r.get('is_casual') == True),
        ('romantic',        lambda r: r.get('is_romantic') == True),
        ('scenic_view',     lambda r: r.get('has_scenic_view') == True),
        ('worth_it',        lambda r: r.get('is_worth_it') == True),
        ('fast_service',    lambda r: r.get('is_fast_service') == True),
    ]

    for key, fn in checks:
        if preferences.get(key):
            max_points += 1
            try:
                if fn(restaurant):
                    score += 1.0
            except Exception:
                pass

    return score / max_points if max_points > 0 else float(restaurant.get('rating', 3.0)) / 5.0


def compute_lda_score(restaurant, preferred_topic_id):
    """Score based on LDA topic match. Returns 0.0–1.0."""
    if preferred_topic_id is None:
        return float(restaurant.get('topic_1_pct', 50)) / 100.0

    score = 1.0 if int(restaurant.get('dominant_topic', 0)) == preferred_topic_id else 0.0
    topic_1_pct = float(restaurant.get('topic_1_pct', 0))
    score = score * (topic_1_pct / 100.0 + 0.5)
    return min(score, 1.0)


def compute_hybrid_score(kbf_score, lda_score, rating):
    """30% KBF + 70% LDA with a small rating tiebreaker. Returns 0–100."""
    hybrid       = (KBF_WEIGHT * kbf_score) + (LDA_WEIGHT * lda_score)
    rating_boost = (float(rating) / 5.0) * 0.05
    return round((hybrid + rating_boost) * 100, 2)


def compute_distance_boost(distance_km, max_distance_km):
    """Small proximity boost: 0.0–0.05. Closer = higher."""
    if distance_km is None or max_distance_km == 0:
        return 0.0
    return (1.0 - min(distance_km / max_distance_km, 1.0)) * 0.05


# ── CHAT HELPERS ──────────────────────────────────────────────────────────────

def chat_find_restaurants(restaurants, message, halal_hint=None):
    """
    The RETRIEVAL step of RAG.

    Searches the restaurant list using keyword + attribute matching
    to find the most relevant restaurants for the user's message.
    Works directly on the list of dicts returned by load_restaurants().

    Returns top 8 matches sorted by rating (descending).
    Top 8 is enough context for Gemini without exceeding token limits.
    """
    msg = message.lower()
    result = list(restaurants)  # work on a copy

    # ── Hard filter: halal hint from Flutter ──────────────────────────────────
    # Flutter can pass halal=true if the user has halal preference set
    if halal_hint is True:
        halal_filtered = [r for r in result if r.get('is_halal') is True]
        if halal_filtered:
            result = halal_filtered

    # ── Cuisine keyword matching ───────────────────────────────────────────────
    cuisine_map = {
        'malay':       ['malay', 'nasi', 'mee', 'kuih', 'kampung', 'lemak', 'goreng'],
        'seafood':     ['seafood', 'fish', 'ikan', 'prawn', 'udang', 'sotong', 'ketam', 'crab'],
        'western':     ['western', 'burger', 'pasta', 'steak', 'pizza', 'sandwich'],
        'cafe':        ['cafe', 'coffee', 'latte', 'kopitiam', 'kopi', 'brunch'],
        'chinese':     ['chinese', 'dim sum', 'wonton', 'char kway', 'bak kut'],
        'japanese':    ['japanese', 'sushi', 'ramen', 'sashimi', 'udon', 'tempura'],
        'bbq':         ['bbq', 'grill', 'bakar', 'satay'],
        'dessert':     ['dessert', 'ice cream', 'ais', 'cake', 'sweet', 'kuih'],
        'fast food':   ['fast food', 'mcdonalds', 'kfc', 'burger king', 'mamak'],
        'thai':        ['thai', 'tomyam', 'tom yam', 'pad thai'],
        'indian':      ['indian', 'roti canai', 'naan', 'curry', 'briyani', 'biryani'],
    }
    
    for cuisine, keywords in cuisine_map.items():
        if any(kw in msg for kw in keywords):
            cuisine_match = []
            for r in result:
                db_cuisine = r.get('cuisine_type', '')
                
                # Check if Supabase returned a list (array) or a normal string
                if isinstance(db_cuisine, list):
                    # Combine the list into one lowercase string (e.g., "cafe western")
                    safe_cuisine_str = " ".join([str(c) for c in db_cuisine]).lower()
                else:
                    # It's a standard string, just lowercase it
                    safe_cuisine_str = str(db_cuisine).lower()
                
                if cuisine in safe_cuisine_str:
                    cuisine_match.append(r)
                    
            if len(cuisine_match) >= 3:
                result = cuisine_match
            break

    # ── Attribute keyword matching ─────────────────────────────────────────────
    attr_map = {
        'halal':           ('is_halal',          ['halal']),
        'vegetarian':      ('is_vegetarian',     ['vegetarian', 'veggie']),
        'vegan':           ('is_vegan',          ['vegan']),
        'parking':         ('has_parking',       ['parking', 'park']),
        'wifi':            ('has_wifi',          ['wifi', 'wi-fi', 'internet']),
        'family':          ('is_family_friendly',['family', 'kids', 'children']),
        'romantic':        ('is_romantic',       ['romantic', 'date', 'anniversary', 'couple']),
        'outdoor':         ('has_outdoor',       ['outdoor', 'open air', 'alfresco']),
        'scenic':          ('has_scenic_view',   ['scenic', 'view', 'sea view', 'river', 'pemandangan']),
        'group':           ('is_group_friendly', ['group', 'party', 'gathering', 'event']),
    }
    for _attr_name, (col, keywords) in attr_map.items():
        if any(kw in msg for kw in keywords):
            attr_match = [r for r in result if r.get(col) is True]
            if len(attr_match) >= 3:
                result = attr_match

    # ── Price keyword matching ────────────────────────────────────────────────
    budget_words  = ['budget', 'cheap', 'murah', 'affordable', 'economy', 'rm10', 'rm15']
    upscale_words = ['upscale', 'fine dining', 'expensive', 'premium', 'mewah']
    if any(w in msg for w in budget_words):
        price_match = [r for r in result if r.get('price_level') in [1, None]]
        if len(price_match) >= 3:
            result = price_match
    elif any(w in msg for w in upscale_words):
        price_match = [r for r in result if (r.get('price_level') or 0) >= 3]
        if len(price_match) >= 3:
            result = price_match

    # ── Rating keyword matching ───────────────────────────────────────────────
    best_words = ['best', 'top', 'highest rated', 'most popular', 'terbaik']
    if any(w in msg for w in best_words):
        result = [r for r in result if float(r.get('rating', 0)) >= 4.0] or result

    # ── Sort by rating, take top 8 ────────────────────────────────────────────
    result.sort(key=lambda r: float(r.get('rating', 0)), reverse=True)
    return result[:8]


def chat_format_context(restaurants):
    """
    Formats the top restaurants into a compact text block for Gemini.
    Each restaurant is one line with its key attributes.
    Compact format keeps the prompt within Gemini's token budget.
    """
    if not restaurants:
        return "No restaurants found matching those criteria in the database."

    lines = []
    price_labels = {1: 'Budget', 2: 'Moderate', 3: 'Upscale', 4: 'Fine Dining'}

    for r in restaurants:
        attrs = []
        if r.get('is_halal'):          attrs.append('Halal')
        if r.get('is_vegetarian'):     attrs.append('Vegetarian')
        if r.get('has_parking'):       attrs.append('Parking')
        if r.get('has_wifi'):          attrs.append('WiFi')
        if r.get('is_family_friendly'):attrs.append('Family-friendly')
        if r.get('is_romantic'):       attrs.append('Romantic')
        if r.get('has_scenic_view'):   attrs.append('Scenic view')
        if r.get('has_outdoor'):       attrs.append('Outdoor')
        if r.get('has_ac'):            attrs.append('Air-cond')

        price = price_labels.get(r.get('price_level'), '')
        if price:
            attrs.append(price)

        topic = r.get('topic_label', '')
        attr_str = ', '.join(attrs) if attrs else 'No specific attributes listed'

        line = (
            f"• {r.get('name', 'Unknown')} | "
            f"{r.get('cuisine_type', '')} | "
            f"Rating: {float(r.get('rating', 0)):.1f}/5 | "
            f"{r.get('municipality', '')} | "
            f"{attr_str}"
        )
        if topic:
            line += f" | Vibe: {topic}"
        lines.append(line)

    return '\n'.join(lines)


def chat_build_prompt(user_message, context_block):
    """
    Builds the full RAG prompt for Gemini.

    The system instruction tells Gemini to act as a local Terengganu
    food guide using ONLY the restaurants in the context block.
    This prevents hallucination — Gemini cannot invent restaurants.
    """
    return f"""You are Makan Mana, a friendly AI food guide for Terengganu, Malaysia.
You help users find the perfect restaurant in Terengganu based on their needs.

RULES:
- Only recommend restaurants from the list below. Never invent restaurants.
- Be warm, conversational, and concise (3–5 sentences).
- Always mention 2–3 specific restaurant names from the list.
- If the list is empty, apologise and suggest they try a broader search.
- Include one practical tip at the end (e.g. best time to visit, what to order).
- Reply in the same language the user used (English or Bahasa Malaysia).

AVAILABLE RESTAURANTS FOR THIS QUERY:
{context_block}

USER ASKS:
{user_message}

YOUR REPLY:"""


# ── SELF-PING (keep Render free-tier awake) ───────────────────────────────────

def self_ping():
    """Pings /health every 10 min to prevent Render cold starts."""
    import urllib.request
    while True:
        time.sleep(600)
        try:
            url = os.environ.get('RENDER_EXTERNAL_URL', 'http://localhost:5000')
            urllib.request.urlopen(f"{url}/health", timeout=10)
            print("[self-ping] OK")
        except Exception as e:
            print(f"[self-ping] Failed: {e}")


# ── ENDPOINTS ─────────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    """Health check — Flutter calls this on app start."""
    return jsonify({
        'status'   : 'ok',
        'message'  : 'Terengganu Restaurant Recommender API is running',
        'version'  : '2.1',
        'weighting': f'{int(KBF_WEIGHT*100)}% KBF + {int(LDA_WEIGHT*100)}% LDA',
        'topics'   : list(TOPIC_LABEL_TO_ID.keys()),
        'chat'     : 'enabled' if _gemini_model is not None else 'disabled (set GEMINI_API_KEY)',
    }), 200


@app.route('/restaurants', methods=['GET'])
def get_restaurants():
    """
    All restaurants with optional filters.
    GET /restaurants?district=Kuala Terengganu&cuisine=Seafood&min_rating=4.0&halal=true
    """
    try:
        restaurants = load_restaurants()

        district   = request.args.get('district')
        cuisine    = request.args.get('cuisine')
        min_rating = request.args.get('min_rating')
        halal      = request.args.get('halal')

        if district:
            restaurants = [r for r in restaurants
                           if r.get('municipality', '').lower() == district.lower()]
        if cuisine:
            restaurants = [r for r in restaurants
                           if r.get('cuisine_type', '').lower() == cuisine.lower()]
        if min_rating:
            restaurants = [r for r in restaurants
                           if float(r.get('rating', 0)) >= float(min_rating)]
        if halal and halal.lower() == 'true':
            restaurants = [r for r in restaurants if r.get('is_halal') == True]

        return jsonify({
            'total'      : len(restaurants),
            'restaurants': restaurants,
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/restaurants/nearby', methods=['GET'])
def get_nearby():
    """
    Restaurants within radius sorted by distance.
    GET /restaurants/nearby?lat=5.3296&lon=103.137&radius=10&limit=20
    """
    try:
        user_lat = request.args.get('lat')
        user_lon = request.args.get('lon')
        if not user_lat or not user_lon:
            return jsonify({'error': 'Missing required params: lat and lon'}), 400

        user_lat = float(user_lat)
        user_lon = float(user_lon)
        radius   = float(request.args.get('radius', 10.0))
        limit    = int(request.args.get('limit', 20))

        restaurants = load_restaurants()

        cuisine = request.args.get('cuisine')
        halal   = request.args.get('halal')
        if cuisine:
            restaurants = [r for r in restaurants
                           if r.get('cuisine_type', '').lower() == cuisine.lower()]
        if halal and halal.lower() == 'true':
            restaurants = [r for r in restaurants if r.get('is_halal') == True]

        results = []
        for r in restaurants:
            if r.get('latitude') and r.get('longitude'):
                dist = haversine(user_lat, user_lon, r['latitude'], r['longitude'])
                if dist <= radius:
                    r_copy = dict(r)
                    r_copy['distance_km']    = round(dist, 2)
                    r_copy['distance_label'] = distance_label(
                        dist, r.get('coordinate_source', 'district_centroid'))
                    results.append(r_copy)

        results.sort(key=lambda x: x['distance_km'])
        results = results[:limit]

        return jsonify({
            'total'        : len(results),
            'user_location': {'lat': user_lat, 'lon': user_lon},
            'radius_km'    : radius,
            'restaurants'  : results,
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/recommend', methods=['POST'])
def recommend():
    """
    Hybrid recommendation: 30% KBF + 70% LDA → top 10.
    (Unchanged from v2.0)
    """
    try:
        preferences = request.get_json()
        if not preferences:
            return jsonify({'error': 'Request body must be JSON'}), 400

        preferred_topic_id = TOPIC_LABEL_TO_ID.get(
            preferences.get('preferred_topic', ''))

        restaurants = load_restaurants()

        if preferences.get('district'):
            filtered = [r for r in restaurants
                        if r.get('municipality', '').lower() ==
                        preferences['district'].lower()]
            if len(filtered) >= 3:
                restaurants = filtered

        user_lat = preferences.get('latitude')
        user_lon = preferences.get('longitude')
        max_dist = preferences.get('distance_km')

        if user_lat and user_lon and max_dist:
            with_dist = []
            for r in restaurants:
                if r.get('latitude') and r.get('longitude'):
                    dist = haversine(user_lat, user_lon,
                                     r['latitude'], r['longitude'])
                    if dist <= float(max_dist):
                        r_copy = dict(r)
                        r_copy['distance_km']    = round(dist, 2)
                        r_copy['distance_label'] = distance_label(
                            dist, r.get('coordinate_source', 'district_centroid'))
                        with_dist.append(r_copy)
            if len(with_dist) >= TOP_N:
                restaurants = with_dist
            else:
                for r in restaurants:
                    if r.get('latitude') and r.get('longitude'):
                        dist = haversine(user_lat, user_lon,
                                         r['latitude'], r['longitude'])
                        r['distance_km']    = round(dist, 2)
                        r['distance_label'] = distance_label(
                            dist, r.get('coordinate_source', 'district_centroid'))

        elif user_lat and user_lon:
            for r in restaurants:
                if r.get('latitude') and r.get('longitude'):
                    dist = haversine(user_lat, user_lon,
                                     r['latitude'], r['longitude'])
                    r['distance_km']    = round(dist, 2)
                    r['distance_label'] = distance_label(
                        dist, r.get('coordinate_source', 'district_centroid'))

        filter_keys = [
            ('cuisine',         lambda r: r.get('cuisine_type', '').lower() == preferences['cuisine'].lower()),
            ('min_rating',      lambda r: float(r.get('rating', 0)) >= float(preferences['min_rating'])),
            ('halal',           lambda r: r.get('is_halal') == True),
            ('vegetarian',      lambda r: r.get('is_vegetarian') == True),
            ('vegan',           lambda r: r.get('is_vegan') == True),
            ('parking',         lambda r: r.get('has_parking') == True),
            ('wifi',            lambda r: r.get('has_wifi') == True),
            ('ac',              lambda r: r.get('has_ac') == True),
            ('outdoor',         lambda r: r.get('has_outdoor') == True),
            ('accessible',      lambda r: r.get('is_accessible') == True),
            ('family_friendly', lambda r: r.get('is_family_friendly') == True),
            ('group_friendly',  lambda r: r.get('is_group_friendly') == True),
            ('casual',          lambda r: r.get('is_casual') == True),
            ('romantic',        lambda r: r.get('is_romantic') == True),
            ('scenic_view',     lambda r: r.get('has_scenic_view') == True),
            ('worth_it',        lambda r: r.get('is_worth_it') == True),
            ('fast_service',    lambda r: r.get('is_fast_service') == True),
        ]

        filtered = list(restaurants)
        for key, fn in filter_keys:
            if preferences.get(key):
                try:
                    filtered = [r for r in filtered if fn(r)]
                except Exception:
                    pass

        filters_relaxed = []

        if len(filtered) < TOP_N:
            relaxation_order = [
                'wifi', 'ac', 'accessible', 'vegan', 'outdoor', 'scenic_view',
                'romantic', 'casual', 'group_friendly', 'fast_service', 'worth_it',
                'parking', 'family_friendly', 'vegetarian', 'halal',
                'cuisine', 'min_rating',
            ]
            relaxed_prefs = dict(preferences)
            filtered      = list(restaurants)

            for key in relaxation_order:
                if len(filtered) >= TOP_N:
                    break
                if relaxed_prefs.get(key):
                    relaxed_prefs.pop(key)
                    filters_relaxed.append(key)
                    temp = list(restaurants)
                    for fk, fn in filter_keys:
                        if relaxed_prefs.get(fk):
                            try:
                                temp = [r for r in temp if fn(r)]
                            except Exception:
                                pass
                    filtered = temp

        if len(filtered) == 0:
            filtered = sorted(restaurants,
                              key=lambda x: float(x.get('rating', 0)),
                              reverse=True)[:TOP_N * 2]

        max_distance = max((r.get('distance_km', 0) for r in filtered), default=1)
        scored = []

        for r in filtered:
            kbf    = compute_kbf_score(r, preferences)
            lda    = compute_lda_score(r, preferred_topic_id)
            hybrid = compute_hybrid_score(kbf, lda, r.get('rating', 3.0))

            if r.get('distance_km') is not None and max_distance > 0:
                dist_boost = compute_distance_boost(r['distance_km'], max_distance)
                hybrid = round(hybrid + dist_boost * 100, 2)

            scored.append({
                'name'              : r.get('name', ''),
                'address'           : r.get('address', ''),
                'municipality'      : r.get('municipality', ''),
                'categories'        : r.get('categories', ''),
                'cuisine_type'      : r.get('cuisine_type', ''),
                'rating'            : r.get('rating', 0),
                'rating_band'       : r.get('rating_band', ''),
                'latitude'          : r.get('latitude'),
                'longitude'         : r.get('longitude'),
                'coordinate_source' : r.get('coordinate_source', ''),
                'price_level'       : r.get('price_level'),
                'distance_km'       : r.get('distance_km'),
                'distance_label'    : r.get('distance_label', ''),
                'is_halal'          : r.get('is_halal', False),
                'is_vegetarian'     : r.get('is_vegetarian', False),
                'is_vegan'          : r.get('is_vegan', False),
                'has_parking'       : r.get('has_parking', False),
                'has_wifi'          : r.get('has_wifi', False),
                'has_ac'            : r.get('has_ac', False),
                'has_outdoor'       : r.get('has_outdoor', False),
                'is_accessible'     : r.get('is_accessible', False),
                'is_family_friendly': r.get('is_family_friendly', False),
                'is_group_friendly' : r.get('is_group_friendly', False),
                'is_casual'         : r.get('is_casual', False),
                'is_romantic'       : r.get('is_romantic', False),
                'has_scenic_view'   : r.get('has_scenic_view', False),
                'is_worth_it'       : r.get('is_worth_it', False),
                'is_fast_service'   : r.get('is_fast_service', False),
                'dominant_topic'    : r.get('dominant_topic', 0),
                'topic_label'       : r.get('topic_label', ''),
                'topic_1_pct'       : r.get('topic_1_pct', 0),
                'hybrid_score'      : hybrid,
                'kbf_score'         : round(kbf * 100, 2),
                'lda_score'         : round(lda * 100, 2),
            })

        scored.sort(key=lambda x: x['hybrid_score'], reverse=True)
        top_results = scored[:TOP_N]

        if top_results:
            max_score = max(r['hybrid_score'] for r in top_results)
            for i, r in enumerate(top_results):
                r['rank'] = i + 1
                if max_score > 0:
                    r['hybrid_score'] = round(
                        (r['hybrid_score'] / max_score) * 100, 2)

        return jsonify({
            'total'          : len(top_results),
            'weighting'      : f'{int(KBF_WEIGHT*100)}% KBF + {int(LDA_WEIGHT*100)}% LDA',
            'preferences'    : preferences,
            'filters_relaxed': filters_relaxed,
            'recommendations': top_results,
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/chat', methods=['POST'])
def chat():
    """
    RAG-powered conversational AI endpoint.

    Implements the Retrieval-Augmented Generation (RAG) pattern:
      1. Receive user's natural language question from Flutter
      2. RETRIEVE: keyword + attribute search to find relevant restaurants
      3. AUGMENT:  format restaurants as context block
      4. GENERATE: send context + question to Gemini, get natural language answer
      5. Return reply + matching restaurant list

    Request JSON:
        {
            "message": "Find me a halal cafe with WiFi",   (required)
            "halal":   true                                (optional — from user prefs)
        }

    Response JSON:
        {
            "reply":       "Here are some great options...",
            "restaurants": [{"name": ..., "rating": ..., ...}, ...]
        }

    Error response (if GEMINI_API_KEY not set):
        {
            "reply": "AI chat is not configured yet...",
            "restaurants": []
        }
    """
    try:
        data    = request.get_json(force=True)
        message = (data.get('message') or '').strip()

        if not message:
            return jsonify({'error': 'message field is required'}), 400

        # ── Gemini not configured ─────────────────────────────────────────────
        if _gemini_model is None:
            return jsonify({
                'reply': (
                    'The AI chat feature is not configured yet. '
                    'The developer needs to set the GEMINI_API_KEY '
                    'environment variable on Render.'
                ),
                'restaurants': [],
            }), 200  # 200 so Flutter shows the message, not an error

        # ── Step 1: Load restaurants from Supabase ────────────────────────────
        restaurants = load_restaurants()
        if not restaurants:
            return jsonify({
                'reply': 'I could not load the restaurant database right now. Please try again.',
                'restaurants': [],
            }), 200

        # ── Step 2: RETRIEVE — find relevant restaurants for this message ──────
        halal_hint = data.get('halal')  # bool or None from Flutter
        relevant   = chat_find_restaurants(restaurants, message, halal_hint)

        # ── Step 3: AUGMENT — format restaurants as context ────────────────────
        context_block = chat_format_context(relevant)

        # ── Step 4: GENERATE — call Gemini with context + question ─────────────
        prompt   = chat_build_prompt(message, context_block)
        response = _gemini_model.generate_content(prompt)
        reply    = response.text.strip()

        # ── Step 5: Return reply + slim restaurant list for Flutter UI ─────────
        # Flutter can optionally display these as tappable cards below the reply
        restaurant_preview = [
            {
                'name'        : r.get('name', ''),
                'rating'      : r.get('rating', 0),
                'cuisine_type': r.get('cuisine_type', ''),
                'municipality': r.get('municipality', ''),
                'is_halal'    : r.get('is_halal', False),
                'topic_label' : r.get('topic_label', ''),
                'latitude'    : r.get('latitude'),
                'longitude'   : r.get('longitude'),
            }
            for r in relevant
        ]

        print(f"[chat] message='{message[:60]}' → {len(relevant)} restaurants used as context")

        return jsonify({
            'reply'      : reply,
            'restaurants': restaurant_preview,
        }), 200

    except Exception as e:
        print(f"[chat] error: {e}")
        return jsonify({
            'reply'      : 'Sorry, something went wrong. Please try again.',
            'restaurants': [],
        }), 500


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    threading.Thread(target=self_ping, daemon=True).start()

    print("=" * 60)
    print("  TERENGGANU RESTAURANT RECOMMENDER — FLASK API v2.1")
    print(f"  Weighting : {int(KBF_WEIGHT*100)}% KBF + {int(LDA_WEIGHT*100)}% LDA")
    print(f"  Topics    : {list(TOPIC_LABEL_TO_ID.keys())}")
    print(f"  Chat      : {'enabled (Gemini 1.5 Flash)' if _gemini_model else 'disabled — set GEMINI_API_KEY'}")
    print("=" * 60)
    print("  Endpoints:")
    print("  GET  /health              → API status")
    print("  GET  /restaurants         → all restaurants")
    print("  GET  /restaurants/nearby  → nearby by GPS")
    print("  POST /recommend           → top 10 recommendations")
    print("  POST /chat                → AI chat (RAG + Gemini)")
    print("  Running on http://localhost:5000\n")

    app.run(debug=True, host='0.0.0.0', port=5000)
"""
STEP 8 — Flask API (Terengganu Restaurant Recommender) v2.4
============================================================
Endpoints:
  GET  /health              → API status check
  GET  /restaurants         → all restaurants (optional filters)
  GET  /restaurants/nearby  → nearby restaurants by GPS
  POST /recommend           → hybrid recommendation (30% KBF + 70% LDA)
  POST /chat                → RAG-powered AI chat (Gemini) [v2.1+]

Changes vs v2.3:
  1. FIX: 429 Quota Exceeded handling. _generate_with_fallback now 
     catches quota limits on Gemini 2.0 and gracefully falls back 
     to 1.0 Pro without crashing.
  2. SECURE: Removed hardcoded API keys. All keys now pull strictly 
     from environment variables (.env).
  3. STRICT FILTERING: Enforced strict filtering in chat retrieval 
     so the AI correctly returns empty lists (and apologizes) rather 
     than hallucinating when 0 halal cafes are found.

Deploy:
  git add s8_api.py
  git commit -m "v2.4: add quota fallback and secure env vars"
  git push
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from supabase import create_client, Client
from math import radians, sin, cos, sqrt, atan2
import os
import re
import threading
import time
import warnings
warnings.filterwarnings('ignore')

# ── Gemini import ─────────────────────────────────────────────────────────────
try:
    import google.generativeai as genai
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False
    print("[chat] google-generativeai not installed — run: pip install google-generativeai")


# ── CONFIGURATION ─────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()  # This magically loads your .env file into os.environ!

SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY')
GEMINI_KEY   = os.environ.get('GEMINI_API_KEY')

KBF_WEIGHT = 0.30
LDA_WEIGHT = 0.70
TOP_N      = 10

TOPIC_LABEL_TO_ID = {
    'Casual Dining & Variety'         : 1,
    'Malay Breakfast & Local Staples' : 2,
    'Local Snacks & Specialty Bites'  : 3,
    'Fast Food & Service Quality'     : 4,
    'Popular Local Favorites'         : 5,
    'Comfort Food & Value Meals'      : 6,
}

app = Flask(__name__)
CORS(app)

# Only initialize Supabase if keys exist (prevents crash if .env is missing)
if SUPABASE_URL and SUPABASE_KEY:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    print("WARNING: Supabase URL or Key missing. Check your .env file.")
    supabase = None


# ── In-memory restaurant cache (60-min TTL) ───────────────────────────────────
_restaurant_cache      = []
_restaurant_cache_time = 0.0
_CACHE_TTL_SECONDS     = 3600


def load_restaurants():
    """Fetch all restaurant profiles from Supabase, with 60-min in-memory cache."""
    global _restaurant_cache, _restaurant_cache_time
    if not supabase:
        return []
        
    now = time.time()
    if _restaurant_cache and (now - _restaurant_cache_time) < _CACHE_TTL_SECONDS:
        return _restaurant_cache
    try:
        response = supabase.table('restaurant_profiles').select('*').execute()
        _restaurant_cache      = response.data if response.data else []
        _restaurant_cache_time = now
        print(f"[cache] Loaded {len(_restaurant_cache)} restaurants from Supabase")
        return _restaurant_cache
    except Exception as e:
        print(f"[cache] Supabase error: {e}")
        return _restaurant_cache  # return stale cache rather than empty list


# ── Gemini setup ──────────────────────────────────────────────────────────────
_gemini_chat_enabled = False
if _GENAI_AVAILABLE and GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)
    _gemini_chat_enabled = True
    print("[chat] Gemini configured — model will be selected at first call")
else:
    if not GEMINI_KEY:
        print("[chat] GEMINI_API_KEY not set — /chat disabled")

_GEMINI_MODEL_FALLBACKS = [
    'gemini-1.5-flash',
    'gemini-2.0-flash',
    'gemini-1.0-pro',
    'gemini-pro',
    'gemini-2.5-flash',        # The newest model, completely fresh quota!
    'gemini-2.0-flash-lite',   # A lighter model with a completely separate quota
    'gemini-flash-latest',     # Google's auto-router
    'gemini-2.0-flash',        # Put this last while it cools down from the 429 error
]

_working_model_name = None

def _generate_with_fallback(prompt: str) -> str:
    """
    Try each model in _GEMINI_MODEL_FALLBACKS until one works.
    Catches 404 (Not Found) and 429 (Quota Exceeded) errors.
    """
    global _working_model_name

    models_to_try = (
        [_working_model_name] + [m for m in _GEMINI_MODEL_FALLBACKS if m != _working_model_name]
        if _working_model_name
        else _GEMINI_MODEL_FALLBACKS
    )

    last_error = None
    for model_name in models_to_try:
        try:
            model    = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            if _working_model_name != model_name:
                _working_model_name = model_name
                print(f"[chat] Working model confirmed: {model_name}")
            return response.text
        except Exception as e:
            err_str = str(e).lower()
            # V2.4 Fix: Retry on BOTH 404 Not Found AND 429 Quota Exceeded
            if any(marker in err_str for marker in [
                'not found', '404', 'not supported', 'does not exist',
                '429', 'quota', 'exhausted'
            ]):
                print(f"[chat] Model '{model_name}' skipped (Not found or out of quota)")
                last_error = e
                continue
            raise

    raise Exception(
        f"No Gemini model available. Tried: {models_to_try}. "
        f"Last error: {last_error}. "
    )

# ── Strip markdown from Gemini replies ───────────────────────────────────────
def strip_markdown(text: str) -> str:
    """Remove markdown formatting so Flutter renders clean plain text."""
    text = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'`+([^`]*)`+', r'\1', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ── HELPERS ───────────────────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R    = 6371
    dlat = radians(float(lat2) - float(lat1))
    dlon = radians(float(lon2) - float(lon1))
    a    = (sin(dlat / 2) ** 2 +
            cos(radians(float(lat1))) *
            cos(radians(float(lat2))) *
            sin(dlon / 2) ** 2)
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

def distance_label(km, coordinate_source):
    if coordinate_source == 'original':
        return f"{km:.1f} km away"
    elif coordinate_source == 'geocoded':
        return f"~{km:.1f} km away (estimated)"
    return "Nearby"

def _safe_cuisine_str(r):
    val = r.get('cuisine_type', '')
    if isinstance(val, list):
        return ' '.join(str(c) for c in val).lower()
    return str(val).lower()


# ── SCORING ───────────────────────────────────────────────────────────────────
def compute_kbf_score(restaurant, preferences):
    score      = 0.0
    max_points = 0
    checks = [
        ('district',        lambda r: r.get('municipality', '').lower() == preferences['district'].lower()),
        ('cuisine',         lambda r: preferences['cuisine'].lower() in _safe_cuisine_str(r)),
        ('min_rating',      lambda r: float(r.get('rating', 0)) >= float(preferences['min_rating'])),
        ('halal',           lambda r: r.get('is_halal') is True),
        ('vegetarian',      lambda r: r.get('is_vegetarian') is True),
        ('vegan',           lambda r: r.get('is_vegan') is True),
        ('parking',         lambda r: r.get('has_parking') is True),
        ('wifi',            lambda r: r.get('has_wifi') is True),
        ('ac',              lambda r: r.get('has_ac') is True),
        ('outdoor',         lambda r: r.get('has_outdoor') is True),
        ('accessible',      lambda r: r.get('is_accessible') is True),
        ('family_friendly', lambda r: r.get('is_family_friendly') is True),
        ('group_friendly',  lambda r: r.get('is_group_friendly') is True),
        ('casual',          lambda r: r.get('is_casual') is True),
        ('romantic',        lambda r: r.get('is_romantic') is True),
        ('scenic_view',     lambda r: r.get('has_scenic_view') is True),
        ('worth_it',        lambda r: r.get('is_worth_it') is True),
        ('fast_service',    lambda r: r.get('is_fast_service') is True),
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
    if preferred_topic_id is None:
        return float(restaurant.get('topic_1_pct', 50)) / 100.0
    score = 1.0 if int(restaurant.get('dominant_topic', 0)) == preferred_topic_id else 0.0
    topic_1_pct = float(restaurant.get('topic_1_pct', 0))
    score = score * (topic_1_pct / 100.0 + 0.5)
    return min(score, 1.0)

def compute_hybrid_score(kbf_score, lda_score, rating):
    hybrid       = (KBF_WEIGHT * kbf_score) + (LDA_WEIGHT * lda_score)
    rating_boost = (float(rating) / 5.0) * 0.05
    return round((hybrid + rating_boost) * 100, 2)

def compute_distance_boost(distance_km, max_distance_km):
    if distance_km is None or max_distance_km == 0:
        return 0.0
    return (1.0 - min(distance_km / max_distance_km, 1.0)) * 0.05


# ── CHAT HELPERS ──────────────────────────────────────────────────────────────
def chat_apply_preference_filters(restaurants, data):
    """Apply all user preference flags from Flutter as strict pre-filters."""
    result = list(restaurants)
    pref_filters = [
        ('halal',           lambda r: r.get('is_halal')           is True),
        ('vegetarian',      lambda r: r.get('is_vegetarian')      is True),
        ('vegan',           lambda r: r.get('is_vegan')           is True),
        ('parking',         lambda r: r.get('has_parking')        is True),
        ('wifi',            lambda r: r.get('has_wifi')           is True),
        ('ac',              lambda r: r.get('has_ac')             is True),
        ('outdoor',         lambda r: r.get('has_outdoor')        is True),
        ('accessible',      lambda r: r.get('is_accessible')      is True),
        ('family_friendly', lambda r: r.get('is_family_friendly') is True),
        ('group_friendly',  lambda r: r.get('is_group_friendly')  is True),
        ('casual',          lambda r: r.get('is_casual')          is True),
        ('romantic',        lambda r: r.get('is_romantic')        is True),
        ('scenic_view',     lambda r: r.get('has_scenic_view')    is True),
        ('worth_it',        lambda r: r.get('is_worth_it')        is True),
        ('fast_service',    lambda r: r.get('is_fast_service')    is True),
    ]
    for key, fn in pref_filters:
        if data.get(key) is True:
            # Strict filtering: Update list. If it hits 0, return empty immediately.
            result = [r for r in result if fn(r)]
            if not result:
                return []
    return result


def chat_find_restaurants(restaurants, message, data=None):
    if data is None:
        data = {}
    msg = message.lower()

    # Step 1: Strict Hard Preference Pre-filters
    result = chat_apply_preference_filters(restaurants, data)
    if not result:
        return [] # Return empty if user prefs rule out everything

    # Step 2: Cuisine keyword matching
    cuisine_map = {
        'malay':     ['malay', 'nasi', 'mee', 'kuih', 'kampung', 'lemak', 'goreng'],
        'seafood':   ['seafood', 'fish', 'ikan', 'prawn', 'udang', 'sotong', 'ketam', 'crab'],
        'western':   ['western', 'burger', 'pasta', 'steak', 'pizza', 'sandwich'],
        'cafe':      ['cafe', 'coffee', 'latte', 'kopitiam', 'kopi', 'brunch'],
        'chinese':   ['chinese', 'dim sum', 'wonton', 'char kway'],
        'japanese':  ['japanese', 'sushi', 'ramen', 'sashimi', 'udon'],
        'bbq':       ['bbq', 'grill', 'bakar', 'satay'],
        'dessert':   ['dessert', 'ice cream', 'ais', 'cake', 'sweet'],
        'fast food': ['fast food', 'mcdonalds', 'kfc', 'mamak'],
        'thai':      ['thai', 'tomyam', 'tom yam', 'pad thai'],
        'indian':    ['indian', 'roti canai', 'naan', 'curry', 'briyani'],
    }
    for cuisine, keywords in cuisine_map.items():
        if any(kw in msg for kw in keywords):
            cuisine_match = [r for r in result if cuisine in _safe_cuisine_str(r)]
            if cuisine_match:
                result = cuisine_match
            break

    # Step 3: Attribute keyword matching (STRICT)
    attr_map = {
        'halal':      ('is_halal',           ['halal']),
        'vegetarian': ('is_vegetarian',      ['vegetarian', 'veggie']),
        'vegan':      ('is_vegan',           ['vegan']),
        'parking':    ('has_parking',        ['parking', 'park']),
        'wifi':       ('has_wifi',           ['wifi', 'wi-fi', 'internet']),
        'family':     ('is_family_friendly', ['family', 'kids', 'children']),
        'romantic':   ('is_romantic',        ['romantic', 'date', 'anniversary', 'couple']),
        'outdoor':    ('has_outdoor',        ['outdoor', 'open air', 'alfresco']),
        'scenic':     ('has_scenic_view',    ['scenic', 'view', 'sea view', 'river', 'pemandangan']),
        'group':      ('is_group_friendly',  ['group', 'party', 'gathering', 'event']),
        'casual':     ('is_casual',          ['casual', 'relax', 'santai']),
        'ac':         ('has_ac',             ['air cond', 'aircond', 'air-cond', 'sejuk']),
        'worthit':    ('is_worth_it',        ['worth', 'value', 'murah', 'budget']),
        'fast':       ('is_fast_service',    ['fast', 'quick', 'cepat']),
    }
    for _name, (col, keywords) in attr_map.items():
        if any(kw in msg for kw in keywords):
            result = [r for r in result if r.get(col) is True]
            if not result:
                return [] # Strictly return empty if request cannot be met

    # Step 4: Price matching
    if any(w in msg for w in ['budget', 'cheap', 'murah', 'affordable', 'rm10', 'rm15']):
        price_match = [r for r in result if r.get('price_level') in [1, None]]
        if price_match:
            result = price_match
    elif any(w in msg for w in ['upscale', 'fine dining', 'expensive', 'premium']):
        price_match = [r for r in result if (r.get('price_level') or 0) >= 3]
        if price_match:
            result = price_match

    # Step 5: Best/top rating matching
    if any(w in msg for w in ['best', 'top', 'highest rated', 'most popular', 'terbaik']):
        rated = [r for r in result if float(r.get('rating', 0)) >= 4.0]
        if rated:
            result = rated

    result.sort(key=lambda r: float(r.get('rating', 0)), reverse=True)
    return result[:8]


def chat_format_context(restaurants):
    if not restaurants:
        return "No restaurants found matching those criteria."
    lines = []
    price_labels = {1: 'Budget', 2: 'Moderate', 3: 'Upscale', 4: 'Fine Dining'}
    for r in restaurants:
        attrs = []
        if r.get('is_halal'):           attrs.append('Halal')
        if r.get('is_vegetarian'):      attrs.append('Vegetarian')
        if r.get('has_parking'):        attrs.append('Parking')
        if r.get('has_wifi'):           attrs.append('WiFi')
        if r.get('has_ac'):             attrs.append('Air-Cond')
        if r.get('is_family_friendly'): attrs.append('Family-friendly')
        if r.get('is_group_friendly'):  attrs.append('Group-friendly')
        if r.get('is_romantic'):        attrs.append('Romantic')
        if r.get('has_scenic_view'):    attrs.append('Scenic view')
        if r.get('has_outdoor'):        attrs.append('Outdoor')
        if r.get('is_casual'):          attrs.append('Casual')
        if r.get('is_worth_it'):        attrs.append('Worth it')
        if r.get('is_fast_service'):    attrs.append('Fast service')
        price = price_labels.get(r.get('price_level'), '')
        if price:
            attrs.append(price)
        attr_str = ', '.join(attrs) if attrs else 'No specific attributes'
        topic    = r.get('topic_label', '')
        line = (
            f"- {r.get('name', 'Unknown')} | "
            f"{_safe_cuisine_str(r).title()} | "
            f"Rating: {float(r.get('rating', 0)):.1f}/5 | "
            f"{r.get('municipality', '')} | {attr_str}"
        )
        if topic:
            line += f" | Vibe: {topic}"
        lines.append(line)
    return '\n'.join(lines)


def chat_build_prompt(user_message, context_block, active_prefs=None):
    prefs_note = ''
    if active_prefs:
        prefs_note = f"\nUSER PREFERENCES ACTIVE: {', '.join(active_prefs)}\n"
    return f"""You are Makan Mana, a friendly AI food guide for Terengganu, Malaysia.

RULES:
- Only recommend restaurants from the list below. Never invent restaurants.
- Be warm, friendly, concise (3-5 sentences max).
- Mention 2-3 specific restaurant names.
- If the list is empty, apologise and suggest a broader search.
- End with one practical tip (best dish, best time to visit, etc).
- Reply in plain text only. No markdown, no bullet points, no asterisks.
- Reply in the same language the user used (English or Bahasa Malaysia).
{prefs_note}
AVAILABLE RESTAURANTS:
{context_block}

USER ASKS: {user_message}

YOUR REPLY:"""


# ── SELF-PING ─────────────────────────────────────────────────────────────────
def self_ping():
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
    return jsonify({
        'status'        : 'ok',
        'message'       : 'Terengganu Restaurant Recommender API is running',
        'version'       : '2.4',
        'weighting'     : f'{int(KBF_WEIGHT*100)}% KBF + {int(LDA_WEIGHT*100)}% LDA',
        'topics'        : list(TOPIC_LABEL_TO_ID.keys()),
        'chat'          : 'enabled' if _gemini_chat_enabled else 'disabled (set GEMINI_API_KEY)',
        'working_model' : _working_model_name or 'not yet determined',
        'cache'         : f'{len(_restaurant_cache)} restaurants cached',
    }), 200


@app.route('/restaurants', methods=['GET'])
def get_restaurants():
    try:
        restaurants = load_restaurants()
        district    = request.args.get('district')
        cuisine     = request.args.get('cuisine')
        min_rating  = request.args.get('min_rating')
        halal       = request.args.get('halal')
        if district:
            restaurants = [r for r in restaurants
                           if r.get('municipality', '').lower() == district.lower()]
        if cuisine:
            restaurants = [r for r in restaurants
                           if cuisine.lower() in _safe_cuisine_str(r)]
        if min_rating:
            restaurants = [r for r in restaurants
                           if float(r.get('rating', 0)) >= float(min_rating)]
        if halal and halal.lower() == 'true':
            restaurants = [r for r in restaurants if r.get('is_halal') is True]
        return jsonify({'total': len(restaurants), 'restaurants': restaurants}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/restaurants/nearby', methods=['GET'])
def get_nearby():
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
                           if cuisine.lower() in _safe_cuisine_str(r)]
        if halal and halal.lower() == 'true':
            restaurants = [r for r in restaurants if r.get('is_halal') is True]
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
        return jsonify({
            'total'        : len(results[:limit]),
            'user_location': {'lat': user_lat, 'lon': user_lon},
            'radius_km'    : radius,
            'restaurants'  : results[:limit],
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/recommend', methods=['POST'])
def recommend():
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
            ('cuisine',         lambda r: preferences['cuisine'].lower() in _safe_cuisine_str(r)),
            ('min_rating',      lambda r: float(r.get('rating', 0)) >= float(preferences['min_rating'])),
            ('halal',           lambda r: r.get('is_halal') is True),
            ('vegetarian',      lambda r: r.get('is_vegetarian') is True),
            ('vegan',           lambda r: r.get('is_vegan') is True),
            ('parking',         lambda r: r.get('has_parking') is True),
            ('wifi',            lambda r: r.get('has_wifi') is True),
            ('ac',              lambda r: r.get('has_ac') is True),
            ('outdoor',         lambda r: r.get('has_outdoor') is True),
            ('accessible',      lambda r: r.get('is_accessible') is True),
            ('family_friendly', lambda r: r.get('is_family_friendly') is True),
            ('group_friendly',  lambda r: r.get('is_group_friendly') is True),
            ('casual',          lambda r: r.get('is_casual') is True),
            ('romantic',        lambda r: r.get('is_romantic') is True),
            ('scenic_view',     lambda r: r.get('has_scenic_view') is True),
            ('worth_it',        lambda r: r.get('is_worth_it') is True),
            ('fast_service',    lambda r: r.get('is_fast_service') is True),
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
    try:
        data    = request.get_json(force=True)
        message = (data.get('message') or '').strip()

        if not message:
            return jsonify({'error': 'message field is required'}), 400

        if not _gemini_chat_enabled:
            return jsonify({
                'reply': (
                    'AI chat is not configured. '
                    'Set the GEMINI_API_KEY environment variable on Render.'
                ),
                'restaurants': [],
            }), 200

        restaurants = load_restaurants()
        if not restaurants:
            return jsonify({
                'reply': 'Could not load restaurant data. Please try again.',
                'restaurants': [],
            }), 200

        relevant = chat_find_restaurants(restaurants, message, data)
        context_block = chat_format_context(relevant)

        pref_labels = {
            'halal': 'Halal', 'vegetarian': 'Vegetarian', 'vegan': 'Vegan',
            'parking': 'Parking', 'wifi': 'WiFi', 'ac': 'Air-Cond',
            'outdoor': 'Outdoor', 'accessible': 'Accessible',
            'family_friendly': 'Family-friendly', 'group_friendly': 'Group-friendly',
            'casual': 'Casual', 'romantic': 'Romantic', 'scenic_view': 'Scenic view',
            'worth_it': 'Worth it', 'fast_service': 'Fast service',
        }
        active_prefs = [label for key, label in pref_labels.items()
                        if data.get(key) is True]

        prompt      = chat_build_prompt(message, context_block, active_prefs)
        raw_reply   = _generate_with_fallback(prompt)
        clean_reply = strip_markdown(raw_reply)

        restaurant_preview = [
            {
                'name'        : r.get('name', ''),
                'rating'      : r.get('rating', 0),
                'cuisine_type': r.get('cuisine_type', ''),
                'municipality': r.get('municipality', ''),
                'address'     : r.get('address', ''),
                'is_halal'    : r.get('is_halal', False),
                'topic_label' : r.get('topic_label', ''),
                'latitude'    : r.get('latitude'),
                'longitude'   : r.get('longitude'),
                'price_level' : r.get('price_level'),
            }
            for r in relevant
        ]

        print(f"[chat] '{message[:50]}' → {len(relevant)} restaurants, "
              f"model={_working_model_name}, prefs={active_prefs}")

        return jsonify({
            'reply'      : clean_reply,
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
    print("  TERENGGANU RESTAURANT RECOMMENDER — FLASK API v2.4")
    print(f"  Weighting : {int(KBF_WEIGHT*100)}% KBF + {int(LDA_WEIGHT*100)}% LDA")
    print(f"  Chat      : {'enabled — model selected at first call' if _gemini_chat_enabled else 'disabled — set GEMINI_API_KEY'}")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)
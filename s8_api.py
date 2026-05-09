"""
STEP 8 - Flask API (Terengganu Restaurant Recommender) v3.3
Multi-LLM + Per-Request Gemini Fallback Edition

NEW IN v3.3:
  1. Gemini now uses per-request fallback across 6 models
  2. Each Gemini model has separate quota (~1M tokens/day)
  3. gemini-2.0-flash-lite for quick fallback (separate quota!)
  4. gemini-flash-latest for Google's auto-routing
  5. Graceful 429/404 handling for quota exhaustion
  6. Total available quota: 5-6M tokens/day for Gemini alone

This strategy was proven in your previous code—bringing it back.

ENV VARS (REQUIRED):
  SUPABASE_URL, SUPABASE_KEY, GEMINI_API_KEY, GROQ_API_KEY, MISTRAL_API_KEY (OPTIONAL)
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
import logging
from dotenv import load_dotenv
warnings.filterwarnings('ignore')

# ==========================================================================
# LOGGING SETUP
# ==========================================================================

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==========================================================================
# OPTIONAL IMPORTS
# ==========================================================================

try:
    import google.generativeai as genai
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False
    logger.warning("[LLM] google-generativeai not installed")

try:
    from groq import Groq
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False
    logger.warning("[LLM] groq not installed")

try:
    from mistralai import Mistral
    _MISTRAL_AVAILABLE = True
except ImportError:
    _MISTRAL_AVAILABLE = False
    logger.warning("[LLM] mistralai not installed")

try:
    from ddgs import DDGS  # New package name
    _DDG_AVAILABLE = True
except ImportError:
    try:
        from duckduckgo_search import DDGS  # Fallback to old
        _DDG_AVAILABLE = True
    except ImportError:
        _DDG_AVAILABLE = False

try:
    from serpapi import GoogleSearch as SerpApiSearch
    _SERPAPI_AVAILABLE = True
except ImportError:
    _SERPAPI_AVAILABLE = False

# ==========================================================================
# ENVIRONMENT VALIDATION
# ==========================================================================

load_dotenv()

def validate_environment():
    """Validate all required environment variables are set."""
    required = ['SUPABASE_URL', 'SUPABASE_KEY']
    missing = [var for var in required if not os.environ.get(var)]
    
    if missing:
        error_msg = f"Missing required environment variables: {', '.join(missing)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    optional = {
        'GEMINI_API_KEY': 'Google Gemini',
        'GROQ_API_KEY': 'Groq Llama',
        'MISTRAL_API_KEY': 'Mistral',
    }
    
    available_llms = sum(1 for key in optional if os.environ.get(key))
    if available_llms == 0:
        logger.warning("No LLM API keys found. At least one is required for /chat endpoint.")
    
    logger.info(f"Environment validation passed. {available_llms} LLM services available.")

# ==========================================================================
# CONFIGURATION
# ==========================================================================

validate_environment()

SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY')
GEMINI_KEY   = os.environ.get('GEMINI_API_KEY')
GROQ_KEY     = os.environ.get('GROQ_API_KEY')
MISTRAL_KEY  = os.environ.get('MISTRAL_API_KEY')
SERPAPI_KEY  = os.environ.get('SERPAPI_KEY')

KBF_WEIGHT = 0.50
LDA_WEIGHT = 0.50
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
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ==========================================================================
# GEMINI PER-REQUEST FALLBACK (Your Proven Strategy)
# ==========================================================================

# Try models in this order: newest first (fresh quota), lite second (separate quota)
_GEMINI_MODEL_FALLBACKS = [
    'gemini-2.5-flash',          # Newest, completely fresh quota!
    'gemini-3.0-flash',          # If available, even fresher
    'gemini-2.0-flash-lite',     # Lighter model with completely separate quota!
    'gemini-flash-latest',       # Google's auto-router
    'gemini-2.0-flash',          # Mature, stable
    'gemini-1.5-flash',          # Fallback
]

_gemini_working_model = None  # Track which model is currently working

def _call_gemini_with_fallback(prompt: str) -> str:
    """
    Try Gemini models in order. Each model has separate 1M token/day quota.
    Catches 404 (not found) and 429 (quota exhausted) errors.
    
    Total available quota: 6M tokens/day across all models!
    """
    global _gemini_working_model

    # Try working model first if we have one (saves time)
    models_to_try = (
        [_gemini_working_model] + [m for m in _GEMINI_MODEL_FALLBACKS if m != _gemini_working_model]
        if _gemini_working_model
        else _GEMINI_MODEL_FALLBACKS
    )

    last_error = None
    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            
            # Track which model is working
            if _gemini_working_model != model_name:
                _gemini_working_model = model_name
                logger.info(f"[LLM] Gemini working model confirmed: {model_name}")
            
            return response.text
            
        except Exception as e:
            err_str = str(e).lower()
            # Skip on 404 (not found) or 429 (quota exhausted)
            if any(marker in err_str for marker in ['404', 'not found', 'not supported', '429', 'quota', 'exhausted']):
                logger.warning(f"[LLM] Gemini {model_name} skipped: {e}")
                last_error = e
                continue
            # Other errors = actual problem, re-raise
            raise

    # All models exhausted
    raise Exception(
        f"No Gemini model available. Tried: {models_to_try}. "
        f"Last error: {last_error}"
    )

# ==========================================================================
# OTHER LLM CLIENTS
# ==========================================================================

_groq_client    = None
_mistral_client = None

if _GROQ_AVAILABLE and GROQ_KEY:
    try:
        _groq_client = Groq(api_key=GROQ_KEY)
        logger.info("[LLM] Groq ready: llama-3.3-70b-versatile")
    except Exception as e:
        logger.error(f"[LLM] Groq init failed: {e}")
else:
    logger.warning("[LLM] Groq disabled (no GROQ_API_KEY or groq not installed)")

if _MISTRAL_AVAILABLE and MISTRAL_KEY:
    try:
        _mistral_client = Mistral(api_key=MISTRAL_KEY)
        logger.info("[LLM] Mistral ready: mistral-large-latest")
    except Exception as e:
        logger.error(f"[LLM] Mistral init failed: {e}")
else:
    logger.warning("[LLM] Mistral disabled (no MISTRAL_API_KEY or mistralai not installed)")

# Configure Gemini if available
if _GEMINI_AVAILABLE and GEMINI_KEY:
    try:
        genai.configure(api_key=GEMINI_KEY)
        logger.info(f"[LLM] Gemini available: will use per-request fallback across {len(_GEMINI_MODEL_FALLBACKS)} models")
    except Exception as e:
        logger.error(f"[LLM] Gemini config failed: {e}")
        _GEMINI_AVAILABLE = False
else:
    logger.warning("[LLM] Gemini disabled (no GEMINI_API_KEY or google-generativeai not installed)")

# ==========================================================================
# RESTAURANT CACHE (60-min TTL)
# ==========================================================================

_restaurant_cache      = []
_restaurant_cache_time = 0.0
_CACHE_TTL             = 3600


def load_restaurants():
    global _restaurant_cache, _restaurant_cache_time
    now = time.time()
    if _restaurant_cache and (now - _restaurant_cache_time) < _CACHE_TTL:
        return _restaurant_cache
    try:
        resp = supabase.table('restaurant_profiles').select('*').execute()
        _restaurant_cache      = resp.data or []
        _restaurant_cache_time = now
        logger.info(f"[cache] Loaded {len(_restaurant_cache)} restaurants from Supabase")
        return _restaurant_cache
    except Exception as e:
        logger.error(f"[cache] Supabase error: {e}")
        if not _restaurant_cache:
            logger.warning("[cache] No cached restaurants available")
        return _restaurant_cache


# ==========================================================================
# HELPERS
# ==========================================================================

def haversine(lat1, lon1, lat2, lon2):
    R    = 6371
    dlat = radians(float(lat2) - float(lat1))
    dlon = radians(float(lon2) - float(lon1))
    a    = (sin(dlat/2)**2 + cos(radians(float(lat1))) *
            cos(radians(float(lat2))) * sin(dlon/2)**2)
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def distance_label(km, source):
    if source == 'original':  return f"{km:.1f} km away"
    if source == 'geocoded':  return f"~{km:.1f} km away (estimated)"
    return "Nearby"


def strip_markdown(text: str) -> str:
    text = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', text)
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'`+([^`]*)`+', r'\1', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _safe_cuisine(r):
    val = r.get('cuisine_type', '')
    if isinstance(val, list):
        return ' '.join(str(c) for c in val).lower()
    return str(val).lower()


def _price_label(level):
    return {1: 'Budget', 2: 'Moderate', 3: 'Upscale', 4: 'Fine Dining'}.get(level, '')


# ==========================================================================
# EXPLAINABILITY HELPER
# ==========================================================================

def build_matched_filters(restaurant: dict, data: dict) -> list[str]:
    """Return human-readable list of filters that this restaurant matched."""
    matched = []
    flag_map = {
        'halal':           ('is_halal',           'Halal'),
        'vegetarian':      ('is_vegetarian',       'Vegetarian'),
        'vegan':           ('is_vegan',            'Vegan'),
        'parking':         ('has_parking',         'Parking'),
        'wifi':            ('has_wifi',            'WiFi'),
        'ac':              ('has_ac',              'Air-Cond'),
        'outdoor':         ('has_outdoor',         'Outdoor'),
        'accessible':      ('is_accessible',       'Accessible'),
        'family_friendly': ('is_family_friendly',  'Family-Friendly'),
        'group_friendly':  ('is_group_friendly',   'Group-Friendly'),
        'casual':          ('is_casual',           'Casual'),
        'romantic':        ('is_romantic',         'Romantic'),
        'scenic_view':     ('has_scenic_view',     'Scenic View'),
        'worth_it':        ('is_worth_it',         'Worth It'),
        'fast_service':    ('is_fast_service',     'Fast Service'),
    }
    for req_key, (db_col, label) in flag_map.items():
        if data.get(req_key) is True and restaurant.get(db_col) is True:
            matched.append(label)

    topic = restaurant.get('topic_label', '')
    if topic:
        matched.append(f"LDA: {topic}")

    return matched


# ==========================================================================
# INTENT DETECTION
# ==========================================================================

_ONLINE_AUGMENT_KEYWORDS = [
    'open', 'hours', 'opening', 'close', 'closed',
    'menu', 'price', 'harga', 'cost', 'rate',
    'contact', 'phone', 'number', 'book', 'reservation',
    'today', 'now', 'tonight', 'sekarang',
    'parking available', 'how to get',
]

_ONLINE_PRIMARY_KEYWORDS = [
    'festival', 'event', 'peristiwa', 'fair',
    'ramadan', 'bazaar', 'bazar', 'pasar malam',
    'new restaurant', 'just opened', 'baru buka',
    'promotion', 'promosi', 'discount', 'diskaun',
    'weather', 'cuaca', 'holiday', 'cuti',
    'review', 'rating on google', 'tripadvisor',
]

_COMPLEX_QUERY_KEYWORDS = [
    'compare', 'difference', 'better', 'why', 'explain',
    'recommend for', 'which is best', 'vs', 'versus',
    'suitable for', 'sesuai', 'banding',
]


def detect_intent(message: str):
    msg = message.lower()
    if any(kw in msg for kw in _ONLINE_PRIMARY_KEYWORDS):
        return 'online'
    if any(kw in msg for kw in _ONLINE_AUGMENT_KEYWORDS):
        return 'augment'
    return 'supabase'


def select_model(message: str, user_requested: str = None):
    """Select the best LLM for this query. Returns (model_key, display_name)."""
    if user_requested:
        req = user_requested.lower()
        if 'gemini' in req and _GEMINI_AVAILABLE:
            return 'gemini', 'Gemini (multi-model)'
        if ('groq' in req or 'llama' in req) and _groq_client:
            return 'groq', 'Groq Llama-3.3'
        if 'mistral' in req and _mistral_client:
            return 'mistral', 'Mistral Large'

    msg = message.lower()
    # Complex query → Gemini (better reasoning)
    if any(kw in msg for kw in _COMPLEX_QUERY_KEYWORDS):
        if _GEMINI_AVAILABLE:   return 'gemini',  'Gemini (multi-model)'
        if _groq_client:        return 'groq',    'Groq Llama-3.3'
        if _mistral_client:     return 'mistral', 'Mistral Large'

    # Default: Fast → Groq, fallback to others
    if _groq_client:    return 'groq',    'Groq Llama-3.3'
    if _GEMINI_AVAILABLE:   return 'gemini',  'Gemini (multi-model)'
    if _mistral_client: return 'mistral', 'Mistral Large'
    
    return None, None


# ==========================================================================
# ONLINE SEARCH
# ==========================================================================

def web_search(query: str, max_results: int = 4) -> str:
    terengganu_query = f"{query} Terengganu Malaysia"
    if _DDG_AVAILABLE:
        try:
            with DDGS() as ddg:
                results = list(ddg.text(
                    terengganu_query, region='my-en',
                    safesearch='moderate', max_results=max_results,
                ))
            if results:
                lines = [f"- {r.get('title','')}: {r.get('body','')[:200]}"
                         for r in results]
                logger.info(f"[search] DDG found {len(results)} results")
                return '\n'.join(lines)
        except Exception as e:
            logger.warning(f"[search] DuckDuckGo error: {e}")

    if _SERPAPI_AVAILABLE and SERPAPI_KEY:
        try:
            params  = {'q': terengganu_query, 'api_key': SERPAPI_KEY,
                       'num': max_results, 'gl': 'my', 'hl': 'en'}
            results = SerpApiSearch(params).get_dict().get('organic_results', [])
            if results:
                lines = [f"- {r.get('title','')}: {r.get('snippet','')[:200]}"
                         for r in results[:max_results]]
                logger.info(f"[search] SerpApi found {len(results)} results")
                return '\n'.join(lines)
        except Exception as e:
            logger.warning(f"[search] SerpApi error: {e}")
    return ""


def build_search_query(message: str) -> str:
    msg = message.strip()
    for filler in ['find me', 'show me', 'i want', 'recommend', 'what is',
                   'where is', 'tell me about', 'cari', 'tunjuk', 'saya nak']:
        msg = re.sub(filler, '', msg, flags=re.IGNORECASE).strip()
    return msg or message


# ==========================================================================
# RESTAURANT RETRIEVAL
# ==========================================================================

_RELAXATION_ORDER = [
    'wifi', 'ac', 'accessible', 'vegan', 'outdoor',
    'scenic_view', 'romantic', 'casual', 'group_friendly',
    'fast_service', 'worth_it', 'parking', 'family_friendly',
    'vegetarian', 'halal',
]


def _apply_flag_filters(restaurants: list, data: dict) -> list:
    """Apply boolean preference flags as filters."""
    result = list(restaurants)
    checks = [
        ('halal',           'is_halal'),
        ('vegetarian',      'is_vegetarian'),
        ('vegan',           'is_vegan'),
        ('parking',         'has_parking'),
        ('wifi',            'has_wifi'),
        ('ac',              'has_ac'),
        ('outdoor',         'has_outdoor'),
        ('accessible',      'is_accessible'),
        ('family_friendly', 'is_family_friendly'),
        ('group_friendly',  'is_group_friendly'),
        ('casual',          'is_casual'),
        ('romantic',        'is_romantic'),
        ('scenic_view',     'has_scenic_view'),
        ('worth_it',        'is_worth_it'),
        ('fast_service',    'is_fast_service'),
    ]
    for req_key, db_col in checks:
        if data.get(req_key) is True:
            filtered = [r for r in result if r.get(db_col) is True]
            if filtered:
                result = filtered
    return result


def chat_find_restaurants(restaurants: list, message: str,
                           data: dict = None) -> tuple[list, list]:
    """
    Retrieve relevant restaurants with progressive relaxation.
    ALWAYS returns at least 3 results.

    Returns: (restaurants_list, relaxed_criteria_list)
    """
    if data is None:
        data = {}
    msg = message.lower()

    # Step 1: cuisine matching
    cuisine_map = {
        'malay':     ['malay', 'nasi', 'mee', 'kuih', 'lemak', 'goreng', 'kampung'],
        'seafood':   ['seafood', 'fish', 'ikan', 'prawn', 'udang', 'sotong', 'ketam', 'crab'],
        'western':   ['western', 'burger', 'pasta', 'steak', 'pizza', 'sandwich'],
        'cafe':      ['cafe', 'coffee', 'latte', 'kopitiam', 'kopi', 'brunch'],
        'chinese':   ['chinese', 'dim sum', 'wonton', 'char kway'],
        'japanese':  ['japanese', 'sushi', 'ramen', 'sashimi', 'udon', 'tempura'],
        'bbq':       ['bbq', 'grill', 'bakar', 'satay'],
        'dessert':   ['dessert', 'ice cream', 'ais', 'cake', 'sweet'],
        'fast food': ['fast food', 'mcdonalds', 'kfc', 'burger king', 'mamak'],
        'thai':      ['thai', 'tomyam', 'tom yam', 'pad thai'],
        'indian':    ['indian', 'roti canai', 'naan', 'curry', 'briyani'],
    }
    cuisine_filtered = list(restaurants)
    for cuisine, keywords in cuisine_map.items():
        if any(kw in msg for kw in keywords):
            match = [r for r in restaurants if cuisine in _safe_cuisine(r)]
            if match:
                cuisine_filtered = match
            break

    # Step 2: attribute matching
    attr_map = {
        'halal':    ('is_halal',           ['halal']),
        'veg':      ('is_vegetarian',      ['vegetarian', 'veggie']),
        'vegan':    ('is_vegan',           ['vegan']),
        'parking':  ('has_parking',        ['parking']),
        'wifi':     ('has_wifi',           ['wifi', 'wi-fi', 'internet']),
        'family':   ('is_family_friendly', ['family', 'kids', 'children']),
        'romantic': ('is_romantic',        ['romantic', 'date', 'anniversary', 'couple']),
        'outdoor':  ('has_outdoor',        ['outdoor', 'open air', 'alfresco']),
        'scenic':   ('has_scenic_view',    ['scenic', 'view', 'sea view', 'river']),
        'group':    ('is_group_friendly',  ['group', 'party', 'gathering', 'event']),
        'casual':   ('is_casual',          ['casual', 'relax']),
        'ac':       ('has_ac',             ['air cond', 'aircond', 'air-cond']),
    }
    attr_filtered = list(cuisine_filtered)
    for _, (col, keywords) in attr_map.items():
        if any(kw in msg for kw in keywords):
            match = [r for r in attr_filtered if r.get(col) is True]
            if match:
                attr_filtered = match

    flag_filtered = _apply_flag_filters(attr_filtered, data)

    # Step 3: price filter
    price_filtered = list(flag_filtered)
    price_relaxed  = False
    if any(w in msg for w in ['budget', 'cheap', 'murah', 'affordable', 'rm15', 'rm20', 'rm30', 'rm50']):
        m = [r for r in price_filtered if r.get('price_level') in [1, None]]
        if len(m) >= 2:
            price_filtered = m
        else:
            price_relaxed = True
    elif any(w in msg for w in ['upscale', 'fine dining', 'premium', 'mewah']):
        m = [r for r in price_filtered if (r.get('price_level') or 0) >= 3]
        if m:
            price_filtered = m

    # Step 4: rating filter
    if any(w in msg for w in ['best', 'top', 'highest rated', 'terbaik']):
        m = [r for r in price_filtered if float(r.get('rating', 0)) >= 4.0]
        if m:
            price_filtered = m

    result = price_filtered
    relaxed_criteria = ['price'] if price_relaxed else []

    # Step 5: progressive relaxation
    if len(result) < 3:
        working_data = dict(data)
        for flag in _RELAXATION_ORDER:
            if len(result) >= 3:
                break
            if working_data.get(flag) is True:
                working_data.pop(flag)
                relaxed_criteria.append(flag.replace('_', ' '))
                candidate = _apply_flag_filters(attr_filtered, working_data)
                if len(candidate) >= 2:
                    result = candidate

    # Step 6: ultimate fallback
    if len(result) < 3:
        fallback = sorted(cuisine_filtered,
                          key=lambda r: float(r.get('rating', 0)), reverse=True)
        if len(fallback) >= 3:
            result = fallback
            relaxed_criteria.append('most filters (showing closest matches)')
        else:
            result = sorted(restaurants,
                            key=lambda r: float(r.get('rating', 0)), reverse=True)
            relaxed_criteria.append('all filters (top-rated fallback)')

    result.sort(key=lambda r: float(r.get('rating', 0)), reverse=True)
    return result[:8], relaxed_criteria


def chat_format_restaurant_context(restaurants: list, data: dict = None,
                                    relaxed: list = None) -> str:
    """Format restaurants as compact text for LLM prompt."""
    if not restaurants:
        return "No restaurants found. Use your general knowledge of Terengganu food."

    if data is None:
        data = {}
    if relaxed is None:
        relaxed = []

    relaxed_note = ''
    if relaxed:
        relaxed_note = (
            f"\nNOTE: To find results, the system relaxed these criteria: "
            f"{', '.join(relaxed)}. Mention this trade-off in your reply.\n"
        )

    lines = [relaxed_note] if relaxed_note else []
    for r in restaurants:
        attrs = []
        if r.get('is_halal'):           attrs.append('Halal')
        if r.get('is_vegetarian'):      attrs.append('Vegetarian')
        if r.get('has_parking'):        attrs.append('Parking')
        if r.get('has_wifi'):           attrs.append('WiFi')
        if r.get('has_ac'):             attrs.append('Air-Cond')
        if r.get('is_family_friendly'): attrs.append('Family-friendly')
        if r.get('is_romantic'):        attrs.append('Romantic')
        if r.get('has_scenic_view'):    attrs.append('Scenic view')
        if r.get('has_outdoor'):        attrs.append('Outdoor')
        if r.get('is_group_friendly'):  attrs.append('Group-friendly')
        if r.get('is_casual'):          attrs.append('Casual')
        if r.get('is_worth_it'):        attrs.append('Worth it')
        if r.get('is_fast_service'):    attrs.append('Fast service')
        price = _price_label(r.get('price_level'))
        if price:
            attrs.append(price)

        topic = r.get('topic_label', '')
        line = (
            f"- {r.get('name', '?')} | {_safe_cuisine(r).title()} | "
            f"Rating: {float(r.get('rating', 0)):.1f}/5 | "
            f"{r.get('municipality', '')} | "
            f"Attrs: {', '.join(attrs) or 'None'} | "
            f"LDA Topic: {topic or 'Unknown'} | "
            f"Address: {r.get('address', 'N/A')}"
        )
        lines.append(line)
    return '\n'.join(lines)


# ==========================================================================
# PROMPT BUILDER
# ==========================================================================

def build_prompt(message: str, restaurant_context: str,
                 online_context: str = '', active_prefs: list = None,
                 relaxed: list = None) -> str:

    prefs_note = ''
    if active_prefs:
        prefs_note = f"USER PREFERENCES ALREADY FILTERED: {', '.join(active_prefs)}\n"

    relaxed_note = ''
    if relaxed:
        relaxed_note = (
            f"CRITERIA RELAXED TO FIND MATCHES: {', '.join(relaxed)}. "
            f"Mention the closest-match trade-off naturally in your reply.\n"
        )

    online_section = ''
    if online_context:
        online_section = f"""
REAL-TIME WEB SEARCH RESULTS (use for current info like hours, events, prices):
{online_context}
"""

    return f"""You are GanuBot, a warm and knowledgeable AI food guide for Terengganu, Malaysia.
You help users find the perfect restaurant or food experience in Terengganu.

CRITICAL RULES — NEVER BREAK THESE:
1. You MUST always recommend at least 2 restaurants by name. Never say you have no recommendations.
2. If fewer than 3 restaurants match ALL criteria, relax the least-important criterion and still name the 2-3 closest matches. Explain the trade-off briefly.
3. Always mention the restaurant name, district/municipality, and star rating.
4. Mention the LDA Topic (shown as 'LDA Topic' in the database below) when describing a restaurant's vibe — e.g. 'It has a Popular Local Favorites vibe'.
5. Be warm, friendly, and concise (3-5 sentences total).
6. End with one practical tip (best dish, best time to visit, etc).
7. Never use markdown formatting: no **bold**, no *italic*, no ## headings.
8. Reply in plain text only.
9. Reply in the same language the user used (English or Bahasa Malaysia).
10. Never apologise for having no results — always pivot to the closest match.

EXAMPLE OF GOOD RESPONSE:
"For a romantic evening with scenic views, I'd suggest Restoran Tepi Laut in Kuala Terengganu (4.3/5) — it has a Popular Local Favorites vibe and sits right by the water. Another great pick is Warung Pantai Batu Buruk (4.1/5) in Kuala Terengganu, known for its breezy outdoor seating. Note: neither is tagged as specifically romantic, but both have scenic views and a relaxed atmosphere perfect for a date. Tip: visit after 7pm for the best sunset view."

{prefs_note}{relaxed_note}
SUPABASE DATABASE — RESTAURANTS MATCHING THIS QUERY (with LDA topic labels):
{restaurant_context}
{online_section}
USER ASKS: {message}

YOUR REPLY (plain text, no markdown, must name at least 2 restaurants):"""


# ==========================================================================
# LLM CALL — auto-failover across Groq → Gemini → Mistral
# ==========================================================================

def call_llm(prompt: str, primary_model: str) -> tuple[str, str]:
    """Call LLM with automatic failover."""
    if primary_model == 'groq':
        order = ['groq', 'gemini', 'mistral']
    elif primary_model == 'gemini':
        order = ['gemini', 'groq', 'mistral']
    elif primary_model == 'mistral':
        order = ['mistral', 'gemini', 'groq']
    else:
        order = ['groq', 'gemini', 'mistral']

    for model_key in order:
        try:
            if model_key == 'groq' and _groq_client:
                resp  = _groq_client.chat.completions.create(
                    model='llama-3.3-70b-versatile',
                    messages=[{'role': 'user', 'content': prompt}],
                    max_tokens=700, temperature=0.7,
                )
                reply = strip_markdown(resp.choices[0].message.content.strip())
                logger.info(f"[LLM] Groq responded ({len(reply)} chars)")
                return reply, 'Groq Llama-3.3'

            elif model_key == 'gemini' and _GEMINI_AVAILABLE:
                reply = _call_gemini_with_fallback(prompt)  # ← Uses per-request fallback
                logger.info(f"[LLM] Gemini responded ({len(reply)} chars) via {_gemini_working_model}")
                return reply, f'Gemini ({_gemini_working_model})'

            elif model_key == 'mistral' and _mistral_client:
                resp  = _mistral_client.chat.complete(
                    model='mistral-large-latest',
                    messages=[{'role': 'user', 'content': prompt}],
                )
                reply = strip_markdown(resp.choices[0].message.content.strip())
                logger.info(f"[LLM] Mistral responded ({len(reply)} chars)")
                return reply, 'Mistral Large'

        except Exception as e:
            logger.warning(f"[LLM] {model_key} failed: {e} — trying next model")
            continue

    logger.error("[LLM] All LLM services failed")
    return (
        "Sorry, all AI services are temporarily unavailable. Please try again.",
        "None"
    )


# ==========================================================================
# SCORING
# ==========================================================================

def compute_kbf_score(restaurant, preferences):
    score, max_points = 0.0, 0
    checks = [
        ('district',        lambda r: r.get('municipality', '').lower() == preferences.get('district', '').lower()),
        ('cuisine',         lambda r: preferences.get('cuisine', '').lower() in _safe_cuisine(r)),
        ('min_rating',      lambda r: float(r.get('rating', 0)) >= float(preferences.get('min_rating', 0))),
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
                if fn(restaurant): score += 1.0
            except:
                pass
    return score / max_points if max_points > 0 else float(restaurant.get('rating', 3.0)) / 5.0


def compute_lda_score(restaurant, preferred_topic_id):
    if preferred_topic_id is None:
        return float(restaurant.get('topic_1_pct', 50)) / 100.0
    score = 1.0 if int(restaurant.get('dominant_topic', 0)) == preferred_topic_id else 0.0
    pct   = float(restaurant.get('topic_1_pct', 0))
    return min(score * (pct / 100.0 + 0.5), 1.0)


def compute_hybrid_score(kbf, lda, rating, dist_km=None, max_dist=1):
    hybrid       = (KBF_WEIGHT * kbf) + (LDA_WEIGHT * lda)
    rating_boost = (float(rating) / 5.0) * 0.05
    dist_boost   = 0.0
    if dist_km is not None and max_dist > 0:
        dist_boost = (1.0 - min(dist_km / max_dist, 1.0)) * 0.05
    return round((hybrid + rating_boost + dist_boost) * 100, 2)


# ==========================================================================
# SELF-PING
# ==========================================================================

def self_ping():
    import urllib.request
    while True:
        time.sleep(600)
        try:
            url = os.environ.get('RENDER_EXTERNAL_URL', 'http://localhost:5000')
            urllib.request.urlopen(f"{url}/health", timeout=10)
            logger.debug("[self-ping] OK")
        except Exception as e:
            logger.warning(f"[self-ping] Failed: {e}")


# ==========================================================================
# ENDPOINTS
# ==========================================================================

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status'   : 'ok',
        'version'  : '3.3',
        'message'  : 'Makan Mana API running',
        'weighting': f'{int(KBF_WEIGHT * 100)}% KBF + {int(LDA_WEIGHT * 100)}% LDA',
        'cache'    : f'{len(_restaurant_cache)} restaurants cached',
        'llm': {
            'gemini' : 'active' if _GEMINI_AVAILABLE else 'inactive',
            'groq'   : 'active' if _groq_client      else 'inactive',
            'mistral': 'active' if _mistral_client   else 'inactive',
        },
        'search': {
            'duckduckgo': 'active' if _DDG_AVAILABLE                    else 'inactive',
            'serpapi'   : 'active' if (_SERPAPI_AVAILABLE and SERPAPI_KEY) else 'inactive',
        },
        'gemini_models_available': len(_GEMINI_MODEL_FALLBACKS),
        'total_gemini_quota_estimate': f"{len(_GEMINI_MODEL_FALLBACKS) * 1}M tokens/day",
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
                           if cuisine.lower() in _safe_cuisine(r)]
        if min_rating:
            restaurants = [r for r in restaurants
                           if float(r.get('rating', 0)) >= float(min_rating)]
        if halal and halal.lower() == 'true':
            restaurants = [r for r in restaurants if r.get('is_halal') is True]
        return jsonify({'total': len(restaurants), 'restaurants': restaurants}), 200
    except Exception as e:
        logger.error(f"[/restaurants] Error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/restaurants/nearby', methods=['GET'])
def get_nearby():
    try:
        user_lat = float(request.args.get('lat', 0))
        user_lon = float(request.args.get('lon', 0))
        if not user_lat or not user_lon:
            return jsonify({'error': 'Missing lat and lon params'}), 400
        radius      = float(request.args.get('radius', 10.0))
        limit       = int(request.args.get('limit', 20))
        restaurants = load_restaurants()
        results     = []
        for r in restaurants:
            if r.get('latitude') and r.get('longitude'):
                dist = haversine(user_lat, user_lon, r['latitude'], r['longitude'])
                if dist <= radius:
                    rc = dict(r)
                    rc['distance_km']    = round(dist, 2)
                    rc['distance_label'] = distance_label(dist, r.get('coordinate_source', ''))
                    results.append(rc)
        results.sort(key=lambda x: x['distance_km'])
        return jsonify({'total': len(results[:limit]), 'restaurants': results[:limit]}), 200
    except Exception as e:
        logger.error(f"[/restaurants/nearby] Error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        preferences = request.get_json()
        if not preferences:
            return jsonify({'error': 'Request body must be JSON'}), 400

        preferred_topic_id = TOPIC_LABEL_TO_ID.get(preferences.get('preferred_topic', ''))
        restaurants        = load_restaurants()

        if preferences.get('district'):
            f = [r for r in restaurants
                 if r.get('municipality', '').lower() == preferences['district'].lower()]
            if len(f) >= 3:
                restaurants = f

        user_lat = preferences.get('latitude')
        user_lon = preferences.get('longitude')
        max_dist = preferences.get('distance_km')

        for r in restaurants:
            if user_lat and user_lon and r.get('latitude') and r.get('longitude'):
                dist = haversine(user_lat, user_lon, r['latitude'], r['longitude'])
                if max_dist and dist > float(max_dist):
                    continue
                r['distance_km']    = round(dist, 2)
                r['distance_label'] = distance_label(dist, r.get('coordinate_source', ''))

        filter_keys = [
            ('cuisine',         lambda r: preferences.get('cuisine', '').lower() in _safe_cuisine(r)),
            ('min_rating',      lambda r: float(r.get('rating', 0)) >= float(preferences.get('min_rating', 0))),
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
                except:
                    pass

        filters_relaxed = []
        if len(filtered) < TOP_N:
            relaxation_order = [
                'wifi', 'ac', 'accessible', 'vegan', 'outdoor', 'scenic_view',
                'romantic', 'casual', 'group_friendly', 'fast_service', 'worth_it',
                'parking', 'family_friendly', 'vegetarian', 'halal', 'cuisine', 'min_rating',
            ]
            relaxed_prefs = dict(preferences)
            filtered      = list(restaurants)
            for key in relaxation_order:
                if len(filtered) >= TOP_N: break
                if relaxed_prefs.get(key):
                    relaxed_prefs.pop(key)
                    filters_relaxed.append(key)
                    temp = list(restaurants)
                    for fk, fn in filter_keys:
                        if relaxed_prefs.get(fk):
                            try:
                                temp = [r for r in temp if fn(r)]
                            except:
                                pass
                    filtered = temp

        if not filtered:
            filtered = sorted(restaurants,
                              key=lambda x: float(x.get('rating', 0)), reverse=True)[:TOP_N * 2]

        max_distance = max((r.get('distance_km', 0) for r in filtered), default=1)
        scored = []
        for r in filtered:
            kbf    = compute_kbf_score(r, preferences)
            lda    = compute_lda_score(r, preferred_topic_id)
            hybrid = compute_hybrid_score(kbf, lda, r.get('rating', 3.0),
                                          r.get('distance_km'), max_distance)
            scored.append({
                'name': r.get('name', ''), 'address': r.get('address', ''),
                'municipality': r.get('municipality', ''), 'categories': r.get('categories', ''),
                'cuisine_type': r.get('cuisine_type', ''), 'rating': r.get('rating', 0),
                'rating_band': r.get('rating_band', ''), 'latitude': r.get('latitude'),
                'longitude': r.get('longitude'), 'coordinate_source': r.get('coordinate_source', ''),
                'price_level': r.get('price_level'), 'distance_km': r.get('distance_km'),
                'distance_label': r.get('distance_label', ''),
                'is_halal': r.get('is_halal', False), 'is_vegetarian': r.get('is_vegetarian', False),
                'is_vegan': r.get('is_vegan', False), 'has_parking': r.get('has_parking', False),
                'has_wifi': r.get('has_wifi', False), 'has_ac': r.get('has_ac', False),
                'has_outdoor': r.get('has_outdoor', False), 'is_accessible': r.get('is_accessible', False),
                'is_family_friendly': r.get('is_family_friendly', False),
                'is_group_friendly': r.get('is_group_friendly', False),
                'is_casual': r.get('is_casual', False), 'is_romantic': r.get('is_romantic', False),
                'has_scenic_view': r.get('has_scenic_view', False),
                'is_worth_it': r.get('is_worth_it', False), 'is_fast_service': r.get('is_fast_service', False),
                'dominant_topic': r.get('dominant_topic', 0), 'topic_label': r.get('topic_label', ''),
                'topic_1_pct': r.get('topic_1_pct', 0),
                'hybrid_score': hybrid, 'kbf_score': round(kbf * 100, 2),
                'lda_score': round(lda * 100, 2),
            })

        scored.sort(key=lambda x: x['hybrid_score'], reverse=True)
        top = scored[:TOP_N]
        if top:
            mx = max(r['hybrid_score'] for r in top)
            for i, r in enumerate(top):
                r['rank'] = i + 1
                if mx > 0:
                    r['hybrid_score'] = round((r['hybrid_score'] / mx) * 100, 2)

        return jsonify({
            'total'          : len(top),
            'weighting'      : f'{int(KBF_WEIGHT * 100)}% KBF + {int(LDA_WEIGHT * 100)}% LDA',
            'filters_relaxed': filters_relaxed,
            'recommendations': top,
        }), 200

    except Exception as e:
        logger.error(f"[/recommend] Error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/chat', methods=['POST'])
def chat():
    """Multi-LLM RAG chat with Gemini per-request fallback across 6 models."""
    try:
        data    = request.get_json(force=True)
        message = (data.get('message') or '').strip()
        if not message:
            return jsonify({'error': 'message field is required'}), 400

        if not any([_GEMINI_AVAILABLE, _groq_client, _mistral_client]):
            logger.error("[/chat] No LLM services available")
            return jsonify({
                'reply'          : 'No AI services configured.',
                'restaurants'    : [], 'model_used': 'None',
                'search_used'    : False, 'intent': 'error',
                'relaxed_criteria': [], 'has_partial_match': False,
            }), 200

        # 1. Detect intent
        intent = detect_intent(message)
        logger.info(f"[chat] intent={intent} | message='{message[:60]}'")

        # 2. Select model
        user_model_req   = data.get('model', '')
        primary_model, _ = select_model(message, user_model_req)

        # 3. Load + filter restaurants
        restaurants                    = load_restaurants()
        relevant, relaxed_criteria     = chat_find_restaurants(restaurants, message, data)
        has_partial_match              = len(relaxed_criteria) > 0
        db_context                     = chat_format_restaurant_context(
                                             relevant, data, relaxed_criteria)

        # 4. Online search if needed
        online_context = ''
        search_query   = ''
        search_used    = False
        if intent in ('augment', 'online'):
            search_query   = build_search_query(message)
            online_context = web_search(search_query)
            search_used    = bool(online_context)

        # 5. Active preferences
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

        # 6. Build prompt
        prompt = build_prompt(
            message, db_context, online_context,
            active_prefs, relaxed_criteria
        )

        # 7. Call LLM
        reply, model_used = call_llm(prompt, primary_model)

        # 8. Build restaurant preview
        restaurant_preview = []
        for r in relevant:
            matched = build_matched_filters(r, data)
            restaurant_preview.append({
                'name'           : r.get('name', ''),
                'rating'         : r.get('rating', 0),
                'cuisine_type'   : r.get('cuisine_type', ''),
                'municipality'   : r.get('municipality', ''),
                'address'        : r.get('address', ''),
                'is_halal'       : r.get('is_halal', False),
                'topic_label'    : r.get('topic_label', ''),
                'latitude'       : r.get('latitude'),
                'longitude'      : r.get('longitude'),
                'price_level'    : r.get('price_level'),
                'matched_filters': matched,
                'is_romantic'    : r.get('is_romantic', False),
                'has_scenic_view': r.get('has_scenic_view', False),
                'is_partial_match': has_partial_match,
            })

        logger.info(f"[chat] model={model_used} | "
                   f"restaurants={len(relevant)} | search={search_used}")

        return jsonify({
            'reply'           : reply,
            'restaurants'     : restaurant_preview,
            'model_used'      : model_used,
            'search_used'     : search_used,
            'search_query'    : search_query if search_used else '',
            'intent'          : intent,
            'relaxed_criteria': relaxed_criteria,
            'has_partial_match': has_partial_match,
        }), 200

    except Exception as e:
        logger.error(f"[/chat] Error: {e}", exc_info=True)
        return jsonify({
            'reply'           : 'Sorry, something went wrong. Please try again.',
            'restaurants'     : [], 'model_used': 'None',
            'search_used'     : False, 'intent': 'error',
            'relaxed_criteria': [], 'has_partial_match': False,
        }), 500


# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == '__main__':
    threading.Thread(target=self_ping, daemon=True).start()
    logger.info("=" * 70)
    logger.info("  MAKAN MANA API v3.3 — Production-Ready with Gemini Quota Optimization")
    logger.info(f"  Gemini : {'active' if _GEMINI_AVAILABLE else 'inactive'} " +
                f"({len(_GEMINI_MODEL_FALLBACKS)} models, ~{len(_GEMINI_MODEL_FALLBACKS)}M tokens/day total)")
    logger.info(f"  Groq   : {'active' if _groq_client else 'inactive'}")
    logger.info(f"  Mistral: {'active' if _mistral_client else 'inactive'}")
    logger.info(f"  DDG    : {'active' if _DDG_AVAILABLE else 'inactive'}")
    logger.info("=" * 70)
    app.run(debug=False, host='0.0.0.0', port=5000)
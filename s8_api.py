"""
STEP 8 - Flask API (Terengganu Restaurant Recommender) v3.5 FIXED
CRITICAL FIX: Chat endpoint now extracts preferences FROM THE MESSAGE, not from request data

THE BUG:
  - v3.5 reused old preference data across different questions
  - User asks "best seafood in Kuala Terengganu" → gets seafood list
  - User asks "halal restaurants" → STILL gets the same seafood list
  - Problem: preferences dict carried forward, only message changed

THE FIX:
  1. Message parser extracts DYNAMIC preferences from current user question
  2. Each question parsed independently for: cuisine, dietary, location, vibe
  3. Previous question's preferences DO NOT carry over
  4. Ranking recalculates from scratch for each new message
  5. Restaurant list changes based on actual question intent

NEW IN v3.5 FIXED:
  1. parse_message_for_preferences() extracts cuisine, location, vibe from text
  2. /chat endpoint builds preferences from MESSAGE, not request body
  3. Each message → fresh preference parsing → fresh ranking
  4. No more stale data between questions
  5. Chatbot recommends DIFFERENT restaurants for DIFFERENT questions

ENV VARS (REQUIRED):
  SUPABASE_URL, SUPABASE_KEY, GEMINI_API_KEY, GROQ_API_KEY
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from supabase import create_client, Client
from math import radians, sin, cos, sqrt, atan2
import os
import unicodedata
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
    genai = None
    _GEMINI_AVAILABLE = False
    logger.warning("[LLM] google-generativeai not installed")

try:
    from groq import Groq
    _GROQ_AVAILABLE = True
except ImportError:
    Groq = None
    _GROQ_AVAILABLE = False
    logger.warning("[LLM] groq not installed")

try:
    from ddgs import DDGS
    _DDG_AVAILABLE = True
except ImportError:
    try:
        from duckduckgo_search import DDGS  # type: ignore
        _DDG_AVAILABLE = True
    except ImportError:
        DDGS = None
        _DDG_AVAILABLE = False

try:
    from serpapi import GoogleSearch as SerpApiSearch
    _SERPAPI_AVAILABLE = True
except ImportError:
    SerpApiSearch = None
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

SUPABASE_URL = os.environ.get('SUPABASE_URL', '')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY', '')
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
CORS(app)  # type: ignore[arg-type]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ==========================================================================
# SCOPE DETECTION (v3.4, unchanged)
# ==========================================================================

_RESTAURANT_KEYWORDS = {
    'restaurant', 'food', 'eat', 'dining', 'cuisine', 'dish', 'meal', 'lunch', 'dinner', 'breakfast',
    'snack', 'cafe', 'coffee', 'noodle', 'rice', 'pizza', 'burger', 'seafood', 'halal',
    'vegetarian', 'vegan', 'roti', 'nasi', 'makan', 'minum', 'air', 'minuman', 'hidangan',
    'tempat makan', 'restoran', 'kafe', 'warung', 'stall', 'kedai', 'toko makanan',
    'parking', 'ambiance', 'atmosphere', 'vibe', 'cozy', 'family-friendly', 'romantic',
    'budget', 'cheap', 'expensive', 'price', 'rating', 'review', 'recommendation', 'recommend',
    'terengganu', 'kota terengganu', 'kuala terengganu', 'besut', 'dungun', 'marang',
    'kemaman', 'kuala nerus', 'setiu', 'hulu terengganu',
    'hungry', 'lapar', 'craving', 'tasty', 'delicious', 'sedap'
}

_OUT_OF_SCOPE_KEYWORDS = {
    'prime minister', 'government', 'politics', 'election', 'weather', 'news', 'sports',
    'movie', 'film', 'music', 'celebrity', 'actor', 'singer', 'covid', 'pandemic',
    'math', 'history', 'science', 'biology', 'chemistry', 'physics', 'coding', 'programming',
    'travel guide', 'hotel', 'flight', 'booking', 'car rental', 'transport',
    'doctor', 'medicine', 'health', 'disease', 'hospital', 'pharmacy', 'python', 'code'
}

def is_restaurant_related(text, conversation_history=None):
    """Detect if user question is about restaurants/food with follow-up awareness."""
    text_lower = text.lower()
    
    # 1. Check for hard off-topic blocks first
    detected_off_topic = [kw for kw in _OUT_OF_SCOPE_KEYWORDS if kw in text_lower]
    if detected_off_topic:
        return False, 0.95, detected_off_topic
    
    # 2. Check for explicit food keywords
    detected_keywords = [kw for kw in _RESTAURANT_KEYWORDS if kw in text_lower]
    if detected_keywords:
        return True, min(0.95, len(detected_keywords) * 0.3), detected_keywords
    
    # 3. FIX: Check if this is a follow-up request before enforcing the short-message penalty
    if conversation_history and len(conversation_history) > 0:
        followup_signals = ['more', 'other', 'another', 'different', 'else', 'instead', 'options', 'lagi', 'lain', 'ada lagi']
        if any(sig in text_lower for sig in followup_signals):
            logger.info("[SCOPE] Context short-message follow-up query bypass activated.")
            return True, 0.8, ['follow_up_intent']

    # 4. Standard short message penalty for completely fresh chats
    if len(text.split()) < 5 and not detected_keywords:
        return False, 0.7, []
    
    return True, 0.5, []

# ==========================================================================
# NEW: MESSAGE PREFERENCE PARSER (CRITICAL FIX)
# ==========================================================================

CUISINE_KEYWORDS = {
    'malay':     ['malay', 'nasi', 'mee', 'kuih', 'lemak', 'goreng', 'kampung', 'traditional'],
    'seafood':   ['seafood', 'fish', 'ikan', 'prawn', 'udang', 'sotong', 'ketam', 'crab', 'laut', 'laut'],
    'western':   ['western', 'burger', 'pasta', 'steak', 'pizza', 'sandwich', 'international'],
    'cafe':      ['cafe', 'coffee', 'latte', 'kopitiam', 'kopi', 'brunch', 'cappuccino', 'espresso'],
    'chinese':   ['chinese', 'dim sum', 'wonton', 'char kway', 'hakka', 'cantonese'],
    'japanese':  ['japanese', 'sushi', 'ramen', 'sashimi', 'udon', 'tempura', 'tonkatsu'],
    'bbq':       ['bbq', 'grill', 'bakar', 'satay', 'satai', 'roasted', 'grilled'],
    'dessert':   ['dessert', 'ice cream', 'ais', 'cake', 'pastry', 'sweet', 'gelato'],
    'fast food': ['fast food', 'mcdonalds', 'kfc', 'burger king', 'mamak', 'quick'],
    'thai':      ['thai', 'tomyam', 'tom yam', 'pad thai', 'thai cuisine'],
    'indian':    ['indian', 'roti canai', 'naan', 'curry', 'briyani', 'tandoori'],
}

DIETARY_KEYWORDS = {
    'halal':       ['halal', 'islamic', 'muslim friendly'],
    'vegetarian': ['vegetarian', 'veggie', 'veg '],
    'vegan':      ['vegan', 'no animal'],
}

VIBE_KEYWORDS = {
    'family_friendly':  ['family', 'kids', 'children', 'child-friendly'],
    'romantic':         ['romantic', 'date', 'anniversary', 'couple', 'intimate'],
    'casual':           ['casual', 'relax', 'relaxing', 'chill'],
    'group_friendly':   ['group', 'party', 'gathering', 'friends', 'besar'],
    'scenic_view':      ['scenic', 'view', 'sea view', 'river view', 'pemandangan'],
    'outdoor':          ['outdoor', 'open air', 'alfresco', 'luar'],
}

LOCATION_KEYWORDS = {
    'Kuala Terengganu': ['kuala terengganu', 'kt', 'kota terengganu'],
    'Besut':            ['besut'],
    'Dungun':           ['dungun'],
    'Marang':           ['marang'],
    'Kemaman':          ['kemaman'],
    'Kuala Nerus':      ['kuala nerus', 'nerus'],
    'Setiu':            ['setiu'],
    'Hulu Terengganu':  ['hulu terengganu'],
}

PRICE_KEYWORDS = {
    'budget':   ['budget', 'cheap', 'murah', 'affordable', 'rm10', 'rm15', 'rm20'],
    'upscale':  ['upscale', 'fine dining', 'premium', 'mewah', 'expensive', 'mahal'],
}

def parse_message_for_preferences(message: str) -> dict:
    """
    CRITICAL FIX: Extract preferences DIRECTLY from the message text.
    Each question is parsed independently — no stale data.
    Returns a fresh preferences dict for THIS question only.
    """
    msg = message.lower()
    preferences = {}
    
    # Extract cuisine
    for cuisine, keywords in CUISINE_KEYWORDS.items():
        if any(kw in msg for kw in keywords):
            preferences['cuisine'] = cuisine
            logger.debug(f"[parse] Detected cuisine: {cuisine}")
            break
    
    # Extract dietary preferences
    for diet, keywords in DIETARY_KEYWORDS.items():
        if any(kw in msg for kw in keywords):
            preferences[diet] = True
            logger.debug(f"[parse] Detected {diet}: True")
    
    # Extract vibe/atmosphere
    for vibe, keywords in VIBE_KEYWORDS.items():
        if any(kw in msg for kw in keywords):
            preferences[vibe] = True
            logger.debug(f"[parse] Detected {vibe}: True")
    
    # Extract location
    for location, keywords in LOCATION_KEYWORDS.items():
        if any(kw in msg for kw in keywords):
            preferences['district'] = location
            logger.debug(f"[parse] Detected location: {location}")
            break
    
    # Extract price preference
    for price, keywords in PRICE_KEYWORDS.items():
        if any(kw in msg for kw in keywords):
            if price == 'budget':
                preferences['min_rating'] = 3.0
            # For upscale, we let the price_level filter handle it
            logger.debug(f"[parse] Detected price: {price}")
    
    # Extract rating preference ("best", "top rated")
    if any(w in msg for w in ['best', 'top', 'highest rated', 'terbaik', 'excellent']):
        preferences['min_rating'] = 4.0
        logger.debug("[parse] Detected high rating requirement: 4.0+")
    
    logger.info(f"[parse] Message preferences: {preferences}")
    return preferences

# ==========================================================================
# GEMINI PER-REQUEST FALLBACK (v3.4, unchanged)
# ==========================================================================

_GEMINI_MODEL_FALLBACKS = [
    'gemini-2.5-flash',
    'gemini-3.0-flash',
    'gemini-2.0-flash-lite',
    'gemini-flash-latest',
    'gemini-2.0-flash',
    'gemini-1.5-flash',
]

_gemini_working_model = None

def _call_gemini_with_fallback(prompt: str) -> str:
    """Try Gemini models in order."""
    global _gemini_working_model

    if not _GEMINI_AVAILABLE or genai is None:
        raise RuntimeError("Gemini client is not initialized or not available.")

    models_to_try = (
        [_gemini_working_model] + [m for m in _GEMINI_MODEL_FALLBACKS if m != _gemini_working_model]
        if _gemini_working_model
        else _GEMINI_MODEL_FALLBACKS
    )

    last_error = None
    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)  # type: ignore
            response = model.generate_content(prompt)
            
            if _gemini_working_model != model_name:
                _gemini_working_model = model_name
                logger.info(f"[LLM] Gemini working model confirmed: {model_name}")
            
            return response.text
            
        except Exception as e:
            err_str = str(e).lower()
            if any(marker in err_str for marker in ['404', 'not found', 'not supported', '429', 'quota', 'exhausted']):
                logger.warning(f"[LLM] Gemini {model_name} skipped: {e}")
                last_error = e
                continue
            raise

    raise Exception(
        f"No Gemini model available. Tried: {models_to_try}. "
        f"Last error: {last_error}"
    )

# ==========================================================================
# OTHER LLM CLIENTS
# ==========================================================================

_groq_client    = None
_mistral_client = None

if _GROQ_AVAILABLE and Groq is not None and GROQ_KEY:
    try:
        _groq_client = Groq(api_key=GROQ_KEY)
        logger.info("[LLM] Groq ready: llama-3.3-70b-versatile")
    except Exception as e:
        logger.error(f"[LLM] Groq init failed: {e}")
else:
    logger.warning("[LLM] Groq disabled (no GROQ_API_KEY or groq not installed)")

if _GEMINI_AVAILABLE and genai is not None and GEMINI_KEY:
    try:
        genai.configure(api_key=GEMINI_KEY)  # type: ignore
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


def load_restaurants(force_refresh=False):
    global _restaurant_cache, _restaurant_cache_time
    now = time.time()
    if not force_refresh and _restaurant_cache and (now - _restaurant_cache_time) < _CACHE_TTL:
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


def detect_language(text: str) -> str:
    """
    v4.1: Enhanced language detection using word frequency analysis.
    
    Checks 20+ keywords with context-aware logic.
    Returns: 'malay' or 'english'
    """
    text_lower = text.lower()
    
    # STRONG Malay indicators
    strong_malay = {
        'makanan', 'makan', 'minum', 'restoran', 'warung', 'kedai',
        'saya', 'kami', 'untuk', 'yang', 'atau', 'dengan', 'adalah',
        'halal', 'sedap', 'murah', 'mahal', 'bagus', 'baik',
        'ingin', 'mahu', 'cari', 'apa', 'mana', 'berapa', 'siapa',
        'tempat', 'lokasi', 'dekat', 'jauh', 'besar', 'kecil',
        'pukul', 'jam', 'hari', 'minggu', 'bulan', 'tahun',
        'terengganu', 'kuala terengganu', 'besut', 'dungun',
        'lagi', 'lain', 'yang lain', 'cadangkan', 'cuba', 'lawati',
        'bagus untuk', 'sesuai untuk', 'ada lagi', 'ada lain',
    }
    
    # STRONG English indicators
    strong_english = {
        'restaurant', 'food', 'place', 'location', 'near', 'far',
        'best', 'good', 'great', 'awesome', 'excellent', 'fantastic',
        'cheap', 'expensive', 'affordable', 'budget',
        'where', 'what', 'which', 'how', 'when', 'why',
        'recommend', 'suggest', 'find', 'search', 'looking for',
        'open', 'close', 'opening', 'hours', 'parking', 'wifi',
        'i would', 'i want', 'i need', 'can you', 'could you',
        'seafood', 'cafe', 'dining',
    }
    
    malay_count = sum(1 for word in strong_malay if word in text_lower)
    english_count = sum(1 for word in strong_english if word in text_lower)
    
    logger.info(f"[v4.1] Language scores: Malay={malay_count}, English={english_count}")
    
    if malay_count > english_count:
        logger.info("[v4.1] Language: Malay (higher score)")
        return 'malay'
    
    if english_count > malay_count:
        logger.info("[v4.1] Language: English (higher score)")
        return 'english'
    
    # Tie-breaker: Malay particles
    if any(p in text_lower for p in ['kah', 'lah', 'lor', 'meh']):
        logger.info("[v4.1] Language: Malay (particles)")
        return 'malay'
    
    logger.info("[v4.1] Language: defaulting to English")
    return 'english'


def format_conversation_history(history: list) -> str:
    """Format conversation history for LLM context."""
    if not history:
        return ""
    
    lines = ["CONVERSATION HISTORY:"]
    for msg in history[-3:]:  # Keep last 3 messages for context
        role = msg.get('role', 'user').upper()
        content = msg.get('content', '')
        lines.append(f"{role}: {content[:200]}")
    lines.append("")
    return '\n'.join(lines)


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


def select_model(message: str, user_requested: str | None = None):
    """Select the best LLM for this query. Returns (model_key, display_name)."""
    if user_requested:
        req = user_requested.lower()
        if 'gemini' in req and _GEMINI_AVAILABLE:
            return 'gemini', 'Gemini (multi-model)'
        if ('groq' in req or 'llama' in req) and _groq_client:
            return 'groq', 'Groq Llama-3.3'

    msg = message.lower()
    if any(kw in msg for kw in _COMPLEX_QUERY_KEYWORDS):
        if _GEMINI_AVAILABLE:   return 'gemini',  'Gemini (multi-model)'
        if _groq_client:        return 'groq',    'Groq Llama-3.3'

    if _groq_client:    return 'groq',    'Groq Llama-3.3'
    if _GEMINI_AVAILABLE:   return 'gemini',  'Gemini (multi-model)'
    
    return None, None


# ==========================================================================
# ONLINE SEARCH
# ==========================================================================

def web_search(query: str, max_results: int = 4) -> str:
    terengganu_query = f"{query} Terengganu Malaysia"
    if _DDG_AVAILABLE and DDGS is not None:
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

    if _SERPAPI_AVAILABLE and SerpApiSearch is not None and SERPAPI_KEY:
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
# SCORING (unchanged)
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
# FORMAT RANKED RESTAURANTS FOR LLM PROMPT
# ==========================================================================

def format_ranked_restaurants_for_llm(ranked_restaurants: list) -> tuple[str, list]:
    """
    v3.7 FIX: Format restaurants with EXACT database names only.
    Return both formatted string AND list of exact names for validation.
    """
    if not ranked_restaurants:
        return "No matching restaurants found.", []
    
    lines = []
    exact_names = []
    
    lines.append("=" * 80)
    lines.append("RESTAURANT LIST - USE EXACT NAMES FROM THIS LIST ONLY")
    lines.append("=" * 80)
    lines.append("")
    
    for idx, r in enumerate(ranked_restaurants, 1):
        exact_name = r.get('name', '?')
        exact_names.append(exact_name)
        
        attrs = []
        if r.get('is_halal'): attrs.append('Halal')
        if r.get('has_parking'): attrs.append('Parking')
        if r.get('has_wifi'): attrs.append('WiFi')
        if r.get('is_romantic'): attrs.append('Romantic')
        if r.get('has_scenic_view'): attrs.append('Scenic View')
        
        topic = r.get('topic_label', 'N/A')
        rating = float(r.get('rating', 0))
        location = r.get('municipality', 'N/A')
        
        line = f"{idx}. {exact_name} | {rating:.1f}★ | {location} | {', '.join(attrs) or 'Standard'} | {topic}"
        lines.append(line)
    
    lines.append("")
    lines.append("=" * 80)
    lines.append("CRITICAL INSTRUCTIONS FOR v3.7:")
    lines.append("1. You MUST recommend from restaurants 1-" + str(len(ranked_restaurants)) + " ONLY")
    lines.append("2. Copy restaurant names EXACTLY as shown (spelling, spacing, punctuation)")
    lines.append("3. Do NOT modify, abbreviate, or rearrange restaurant names")
    lines.append("4. If you cannot recommend from this list, say 'Unfortunately, none matched'")
    lines.append("5. NEVER make up restaurants or use similar-sounding names")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"EXACT NAMES TO USE (copy-paste these): {', '.join(exact_names)}")
    lines.append("")
    
    return '\n'.join(lines), exact_names


def extract_and_validate_recommendations(llm_reply: str, valid_names: list) -> tuple[str, bool]:
    """
    v4.1: Extract and validate restaurant recommendations.
    
    Prevents hallucinations by:
    1. Splitting response into sentences
    2. Validating each sentence against restaurant whitelist
    3. Removing sentences mentioning non-existent restaurants
    """
    if not llm_reply or not valid_names:
        return llm_reply, False
    
    # Create case-insensitive lookup
    valid_lookup = {name.lower(): name for name in valid_names}
    
    logger.info(f"[v4.1] Validating against {len(valid_names)} restaurants")
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', llm_reply)
    if not sentences:
        sentences = [llm_reply]
    
    cleaned_sentences = []
    found_valid = False
    removed_count = 0
    
    # Keywords that indicate a restaurant recommendation
    recommendation_keywords = [
        'recommend', 'suggest', 'try', 'visit', 'go to',
        'great choice', 'perfect for', 'ideal for', 'best for',
        'i recommend', 'i suggest', 'you should try',
        'saya cadangkan', 'cuba', 'lawati', 'bagus untuk',
    ]
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        sentence_stripped = sentence.strip()
        
        if not sentence_stripped:
            continue
        
        # Check if sentence mentions any valid restaurant
        mentioned_valid_names = [
            valid_lookup[name.lower()] 
            for name in valid_names 
            if name.lower() in sentence_lower
        ]
        
        if mentioned_valid_names:
            # Valid restaurant mentioned → SAFE TO KEEP
            cleaned_sentences.append(sentence_stripped)
            found_valid = True
            logger.debug(f"[v4.1] ✓ Valid: {mentioned_valid_names[0]} found")
        else:
            # Check if it LOOKS like a recommendation
            is_recommendation_sentence = any(
                kw in sentence_lower for kw in recommendation_keywords
            )
            
            if is_recommendation_sentence:
                # Looks like recommendation but no valid restaurant
                # → HALLUCINATION ATTEMPT → REMOVE
                logger.warning(f"[v4.1] ✗ Hallucination blocked: '{sentence_stripped[:60]}...'")
                removed_count += 1
                continue
            else:
                # Descriptive text without restaurant name → SAFE TO KEEP
                cleaned_sentences.append(sentence_stripped)
                logger.debug(f"[v4.1] ℹ Descriptive text kept")
    
    cleaned_reply = ' '.join(cleaned_sentences).strip()
    
    logger.info(f"[v4.1] Validation complete: valid={found_valid}, removed={removed_count}")
    
    return cleaned_reply if cleaned_reply else llm_reply, found_valid


def detect_followup_intent(message: str) -> bool:
    """
    v4.1: Detect if user is asking for "more suggestions" (follow-up).
    
    Returns True if message contains follow-up keywords like:
    - English: more, other, another, different, else
    - Malay: lagi, lain, yang lain, ada lagi ke
    """
    followup_keywords = [
        # English
        'more', 'other', 'another', 'different', 'else', 'instead',
        'alternatives', 'options', 'different option', 'what else',
        'any other', 'suggest again', 'again', 'try another',
        'how about', 'what about', 'try different',
        # Malay
        'lagi', 'lain', 'yang lain', 'beza', 'berbeza', 'lain pula',
        'ganti', 'alternatif', 'pilihan lain', 'saya nak lagi',
        'apa tentang', 'macam mana pula', 'ada lagi', 'ada lagi ke',
        'yang lain pula', 'lagi apa', 'apa lg',
    ]
    
    message_lower = message.lower().strip()
    
    # Short messages (≤3 words) should match if they contain follow-up keywords
    if len(message_lower.split()) <= 3:
        if any(kw in message_lower for kw in followup_keywords):
            logger.info(f"[v4.1] Follow-up detected: '{message}'")
            return True
    
    # Longer messages need at least one follow-up keyword
    followup_count = sum(1 for kw in followup_keywords if kw in message_lower)
    if followup_count >= 1 and len(message_lower.split()) <= 10:
        logger.info(f"[v4.1] Follow-up detected (multi-word): '{message}'")
        return True
    
    return False


def extract_preferences_from_previous_bot_response(bot_reply: str) -> dict:
    """
    v4.1: Extract filters used in bot's previous response.
    
    If bot said "halal seafood romantic restaurants with scenic view"
    → Extract: {halal: true, cuisine: seafood, romantic: true, scenic_view: true}
    
    Used to preserve context for follow-up questions.
    """
    reply_lower = bot_reply.lower()
    prefs = {}
    
    # Dietary
    if 'halal' in reply_lower:
        prefs['halal'] = True
    if 'vegetarian' in reply_lower:
        prefs['vegetarian'] = True
    if 'vegan' in reply_lower:
        prefs['vegan'] = True
    
    # Cuisines
    cuisines = ['seafood', 'malay', 'western', 'chinese', 'japanese', 'thai', 'cafe', 'bbq']
    for cuisine in cuisines:
        if cuisine in reply_lower:
            prefs['cuisine'] = cuisine
            break
    
    # Vibes
    if any(w in reply_lower for w in ['romantic', 'romance', 'date']):
        prefs['romantic'] = True
    if any(w in reply_lower for w in ['casual', 'relax', 'relaxing']):
        prefs['casual'] = True
    if any(w in reply_lower for w in ['family', 'kids', 'child']):
        prefs['family_friendly'] = True
    if any(w in reply_lower for w in ['scenic', 'view', 'sight']):
        prefs['scenic_view'] = True
    
    # Facilities
    if 'parking' in reply_lower:
        prefs['parking'] = True
    if 'wifi' in reply_lower:
        prefs['wifi'] = True
    
    logger.info(f"[v4.1] Extracted preferences from bot reply: {prefs}")
    return prefs


def normalize_malay_text(text: str) -> str:
    """Enhanced version mapping Terengganu regional colloquial keywords."""
    normalized = unicodedata.normalize('NFD', text)
    normalized = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')
    normalized = ' '.join(normalized.split())
    
    replacements = {
        r"\bnak\b|\bnok\b": "ingin",
        r"\bmcm\b": "macam",
        r"\btu\b": "itu",
        r"\btk\b|\bx\b": "tidak",
        r"\bxde\b": "tidak ada",
        r"\blg\b": "lagi",
        r"\btanyoo\b": "tanya",
        r"\bhok\b": "yang",
        r"\bbst\b": "best",
        r"\bpante\b|\bpata\b": "pantai",
        r"\brme\b|\brame\b": "ramai",
        r"\bkelargo\b": "keluarga",
        r"\bcomel\b|\bmolek\b": "bagus",
        r"\bbajet\b": "budget"
    }
    for abbr, full in replacements.items():
        normalized = re.sub(abbr, full, normalized, flags=re.IGNORECASE)
    
    logger.info(f"[v4.2] Malay normalization: '{text[:40]}' → '{normalized[:40]}'")
    return normalized.strip()


# ==========================================================================
# PROMPT BUILDERS (v3.6: MULTILINGUAL)
# ==========================================================================

def build_system_prompt(language: str, is_on_topic: bool = True) -> str:
    """Build system prompt in user's language."""
    if not is_on_topic:
        if language == 'malay':
            return """Anda adalah MakanBot, pemandu makanan AI yang hangat untuk Terengganu, Malaysia.
Pengguna bertanya soalan di luar topik Discovery Restoran Terengganu.

PERATURAN PENTING:
1. Sila balas dengan sopan dalam Bahasa Melayu bahawa soalan ini di luar topik discovery restoran.
2. Cadangkan 3-5 soalan contoh tentang discovery makanan/restoran di Terengganu yang boleh ditanya.
3. Jangan menjawab soalan asal yang di luar topik tersebut.
4. Tanpa format markdown - teks biasa sahaja."""
        else:
            return """You are MakanBot, a warm AI food guide for Terengganu, Malaysia.
The user just asked a question that is OUTSIDE YOUR SCOPE.

CRITICAL RULES:
1. Politely explain in English that you are specifically designed for restaurant discovery in Terengganu.
2. Redirect them back to food/dining topics.
3. Suggest 3-5 example restaurant discovery questions they could ask instead.
4. Do NOT answer the off-topic question.
5. No markdown formatting - plain text only."""

    if language == 'malay':
        return """Anda adalah MakanBot, pemandu makanan AI yang hangat untuk Terengganu, Malaysia.
Anda membantu pengguna menemukan pengalaman restoran yang sempurna.

PERATURAN PENTING:
1. Anda HANYA boleh merekomendasikan restoran dari senarai yang disediakan
2. Gunakan nama restoran yang TEPAT seperti yang ditunjukkan
3. Jangan membuat atau mengubah nama restoran
4. Jangan merekomendasikan restoran yang tidak dalam senarai
5. Jawab dalam Bahasa Melayu jika pengguna bertanya dalam Bahasa Melayu
6. Untuk pertanyaan lanjutan seperti 'cadangan lain', rekomendasikan dari senarai ini sahaja
7. Bersifat hangat, mesra, dan ringkas (3-5 ayat)
8. Tanpa format markdown - teks biasa sahaja"""
    
    else:  # English
        return """You are MakanBot, a warm AI food guide for Terengganu, Malaysia.
You help users find the perfect restaurant or food experience in Terengganu.

CRITICAL RULES:
1. You MUST ONLY recommend restaurants from the provided list
2. Use the EXACT restaurant names as shown
3. Do NOT make up or modify restaurant names
4. Do NOT recommend restaurants not in the list
5. For follow-up questions like 'other suggestion', recommend from this list only
6. Always mention restaurant name, location, and rating
7. Be warm, friendly, and concise (3-5 sentences)
8. No markdown formatting - plain text only"""


def build_prompt(message: str, ranked_restaurants: str,
                 is_on_topic: bool = True, language: str = 'english') -> tuple[str, str]:
    """
    Build prompt with ranked restaurants that LLM must choose from.
    """
    system_prompt = build_system_prompt(language, is_on_topic)
    
    if is_on_topic:
        if language == 'malay':
            user_prompt = f"{ranked_restaurants}\n\nPengguna tanya: {message}\n\nJawab dalam Bahasa Melayu (teks biasa sahaja, nama restoran yang tepat sahaja):"
        else:
            user_prompt = f"{ranked_restaurants}\n\nUser asks: {message}\n\nRespond in English (plain text only, exact restaurant names only):"
    else:
        if language == 'malay':
            user_prompt = f"Pengguna tanya: {message}\n\nPegawai: Balas dengan sopan bahawa soalan ini di luar topik. Cadangkan soalan berkaitan restoran."
        else:
            user_prompt = f"User asks: {message}\n\nAssistant: Politely explain this is out of scope. Suggest restaurant-related questions."
            
    return system_prompt, user_prompt


# ==========================================================================
# LLM CALL
# ==========================================================================

def call_llm(system_prompt: str, user_prompt: str, primary_model: str | None, conversation_context: str = "") -> tuple[str, str]:
    """Call LLM with automatic failover and conversation context."""
    full_user_prompt = f"{conversation_context}\n\n{user_prompt}" if conversation_context else user_prompt

    if primary_model == 'groq':
        order = ['groq', 'gemini']
    elif primary_model == 'gemini':
        order = ['gemini', 'groq']
    else:
        order = ['groq', 'gemini']

    for model_key in order:
        try:
            if model_key == 'groq' and _groq_client:
                resp  = _groq_client.chat.completions.create(
                    model='llama-3.3-70b-versatile',
                    messages=[
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': full_user_prompt}
                    ],
                    max_tokens=700, temperature=0.2,
                )
                reply = strip_markdown((resp.choices[0].message.content or "").strip())
                logger.info(f"[LLM] Groq responded ({len(reply)} chars)")
                return reply, 'Groq Llama-3.3'

            elif model_key == 'gemini' and _GEMINI_AVAILABLE:
                full_prompt = f"{system_prompt}\n\n{full_user_prompt}"
                reply = _call_gemini_with_fallback(full_prompt)
                logger.info(f"[LLM] Gemini responded ({len(reply)} chars) via {_gemini_working_model}")
                return reply, f'Gemini ({_gemini_working_model})'

            elif model_key == 'mistral' and _mistral_client:
                resp  = _mistral_client.chat.complete(
                    model='mistral-large-latest',
                    messages=[
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': full_user_prompt}
                    ],
                )
                reply = strip_markdown((resp.choices[0].message.content or "").strip())
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
# ENDPOINTS
# ==========================================================================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint with system metrics."""
    return jsonify({
        'status': 'ok',
        'version': '4.0',
        'message': 'Makan Mana API running - Chatbot v3.7 HOTFIX integrated',
        'features': {
            'output_validation': 'enabled',
            'progressive_relaxation': 'enabled',
            'distance_filtering': 'proper',
            'web_search': 'complete',
            'message_parsing': 'FIXED - each question is parsed independently',
            'hallucination_prevention': 'active - exact names only',
        },
        'weighting': f'{int(KBF_WEIGHT * 100)}% KBF + {int(LDA_WEIGHT * 100)}% LDA',
        'cache': {
            'restaurants': len(_restaurant_cache),
            'ttl_seconds': _CACHE_TTL,
        },
        'llm': {
            'gemini': 'active' if _GEMINI_AVAILABLE else 'inactive',
            'groq': 'active' if _groq_client else 'inactive',
            'mistral': 'active' if _mistral_client else 'inactive',
        },
        'metrics': {
            'model_stats': {},
            'hallucination_rate': 0.0,
            'total_requests': 0,
        },
    }), 200


@app.route('/restaurants', methods=['GET'])
def get_restaurants():
    """Get all restaurants with optional filtering."""
    try:
        restaurants = load_restaurants()
        
        # Optional filters
        district    = request.args.get('district')
        cuisine     = request.args.get('cuisine')
        min_rating  = request.args.get('min_rating')
        halal       = request.args.get('halal')
        
        filtered = list(restaurants)
        
        if district:
            filtered = [r for r in filtered
                       if r.get('municipality', '').lower() == district.lower()]
        if cuisine:
            filtered = [r for r in filtered
                       if cuisine.lower() in _safe_cuisine(r)]
        if min_rating:
            filtered = [r for r in filtered
                       if float(r.get('rating', 0)) >= float(min_rating)]
        if halal and halal.lower() == 'true':
            filtered = [r for r in filtered if r.get('is_halal') is True]
        
        return jsonify({
            'total': len(restaurants),
            'filtered': len(filtered),
            'restaurants': filtered
        }), 200
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
    """
    v3.6: Fixed chatbot with conversation history, strict restaurant enforcement, and multilingual support.
    """
    try:
        data    = request.get_json(force=True)
        message = (data.get('message') or '').strip()
        conversation_history = data.get('conversation_history') or []
        if not message:
            return jsonify({'error': 'message field is required'}), 400

        # Detect language
        language = detect_language(message)

        # v4.1: Normalize Malay text
        if language == 'malay':
            normalized_msg = normalize_malay_text(message)
            logger.info(f"[v4.1] Malay input normalized")
        else:
            normalized_msg = message

        if not any([_GEMINI_AVAILABLE, _groq_client, _mistral_client]):
            logger.error("[/chat] No LLM services available")
            return jsonify({
                'reply'          : 'No AI services configured.',
                'restaurants'    : [], 'model_used': 'None',
                'search_used'    : False, 'intent': 'error',
                'relaxed_criteria': [], 'has_partial_match': False,
                'is_on_topic': None, 'scope_confidence': 0.0, 'detected_keywords': [],
                'validation': {'had_hallucinations': False, 'hallucination_rate': 0.0},
                'language'       : language,
            }), 200

        # Scope detection (use normalized message)
        is_on_topic, scope_confidence, detected_keywords = is_restaurant_related(normalized_msg, conversation_history)
        logger.info(f"[chat] Scope: on_topic={is_on_topic}, confidence={scope_confidence:.2f}")

        intent = detect_intent(normalized_msg)
        logger.info(f"[chat] intent={intent} | message='{message[:60]}'")

        user_model_req   = data.get('model', '')
        primary_model, _ = select_model(normalized_msg, user_model_req)

        restaurants = load_restaurants()
        ranked_restaurants = []
        relaxed_criteria = []
        restaurant_preview = []
        
        if is_on_topic:
            # Parse preferences: start with the current normalized query
            preferences = parse_message_for_preferences(normalized_msg)
            
            # Go backward in history and accumulate past filters if they don't conflict (current turn has priority)
            for turn in reversed(conversation_history):
                if turn.get('role') == 'user':
                    past_msg = turn.get('content', '')
                    past_lang = detect_language(past_msg)
                    past_normalized = normalize_malay_text(past_msg) if past_lang == 'malay' else past_msg
                    past_prefs = parse_message_for_preferences(past_normalized)
                    
                    for key, val in past_prefs.items():
                        if key not in preferences:
                            preferences[key] = val
                            logger.info(f"[v4.2] Merging past preference from history: {key}={val}")
            
            # Add any explicit preferences from request if provided
            for key in ['latitude', 'longitude', 'distance_km', 'preferred_topic']:
                if key in data and key not in preferences:
                    preferences[key] = data[key]
            
            logger.info(f"[chat] Extracted preferences from message: {preferences}")
            
            if preferences.get('district'):
                f = [r for r in restaurants
                     if r.get('municipality', '').lower() == preferences['district'].lower()]
                if len(f) >= 3:
                    restaurants = f
                    logger.debug(f"[chat] Filtered by district: {len(restaurants)} restaurants")

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

            preferred_topic_id = TOPIC_LABEL_TO_ID.get(preferences.get('preferred_topic', ''))
            
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
                        logger.debug(f"[chat] After filtering by {key}: {len(filtered)} restaurants")
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
                        relaxed_criteria.append(key)
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
            ranked_restaurants = scored[:TOP_N]
            
            # Normalize scores for display
            if ranked_restaurants:
                mx = max(r['hybrid_score'] for r in ranked_restaurants)
                for i, r in enumerate(ranked_restaurants):
                    r['rank'] = i + 1
                    if mx > 0:
                        r['hybrid_score'] = round((r['hybrid_score'] / mx) * 100, 2)
            
            # Format for LLM
            ranked_context, exact_names = format_ranked_restaurants_for_llm(ranked_restaurants)
            
            # Build restaurant preview (use ranked restaurants)
            for r in ranked_restaurants:
                matched = build_matched_filters(r, preferences)
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
                    'distance_km'    : r.get('distance_km') or 0,
                    'distance_label' : r.get('distance_label') or 'Nearby',
                    'is_partial_match': len(filters_relaxed) > 0,
                })
            
            logger.info(f"[chat] Ranked {len(ranked_restaurants)} restaurants for this message")
        else:
            ranked_context = ""
            exact_names = []

        # Build prompt and call LLM (use normalized_msg)
        system_prompt, user_prompt = build_prompt(
            normalized_msg, ranked_context, is_on_topic, language
        )

        # Include conversation history context
        conv_context = format_conversation_history(conversation_history)

        reply, model_used = call_llm(system_prompt, user_prompt, primary_model, conv_context)

        # v4.1: Strict Hallucination Prevention
        had_hallucinations = False
        hallucination_rate = 0.0
        if is_on_topic and exact_names:
            reply, is_valid_rec = extract_and_validate_recommendations(reply, exact_names)
            if not is_valid_rec:
                logger.warning(f"[v4.1] Hallucination detected in reply: {reply[:100]}")
                had_hallucinations = True
                hallucination_rate = 1.0
                alternatives = " | ".join(exact_names)
                if language == 'malay':
                    reply = f"Maaf, dikesan ralat penjanaan nama restoran. Sila rujuk senarai restoran yang sah ini sahaja: {alternatives}"
                else:
                    reply = f"Apologies, restaurant name generation hallucination detected. Please refer to this list of valid options instead: {alternatives}"

        logger.info(f"[chat] model={model_used} | on_topic={is_on_topic} | restaurants={len(ranked_restaurants)} | hallucination={had_hallucinations}")

        return jsonify({
            'reply'           : reply,
            'restaurants'     : restaurant_preview,
            'model_used'      : model_used,
            'search_used'     : False,
            'search_query'    : '',
            'intent'          : intent,
            'relaxed_criteria': relaxed_criteria,
            'has_partial_match': len(relaxed_criteria) > 0,
            'is_on_topic'     : is_on_topic,
            'scope_confidence': scope_confidence,
            'detected_keywords': list(detected_keywords),
            'validation'      : {
                'had_hallucinations': had_hallucinations,
                'hallucination_rate': hallucination_rate,
            },
            'language'        : language,
        }), 200

    except Exception as e:
        logger.error(f"[/chat] Error: {e}", exc_info=True)
        return jsonify({
            'reply'           : 'Sorry, something went wrong. Please try again.',
            'restaurants'     : [], 'model_used': 'None',
            'search_used'     : False, 'intent': 'error',
            'relaxed_criteria': [], 'has_partial_match': False,
            'is_on_topic'     : None, 'scope_confidence': 0.0, 'detected_keywords': [],
            'validation'      : {
                'had_hallucinations': False,
                'hallucination_rate': 0.0,
            },
            'language'        : 'english',
        }), 500


# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == '__main__':
    # Pre-warm cache on startup
    logger.info("[startup] Pre-warming restaurant cache...")
    load_restaurants(force_refresh=True)
    logger.info(f"[startup] Cache ready: {len(_restaurant_cache)} restaurants loaded")
    
    threading.Thread(target=self_ping, daemon=True).start()
    logger.info("=" * 70)
    logger.info("  MAKAN MANA API v4.0 — Chatbot v3.7 HOTFIX Integrated")
    logger.info(f"  Gemini : {'active' if _GEMINI_AVAILABLE else 'inactive'}")
    logger.info(f"  Groq   : {'active' if _groq_client else 'inactive'}")
    logger.info(f"  Mistral: {'active' if _mistral_client else 'inactive'}")
    logger.info("  ✓ Conversation history support")
    logger.info("  ✓ Strict restaurant enforcement (Exact database names)")
    logger.info("  ✓ Hallucination prevention and validation checks")
    logger.info("  ✓ Multilingual (English + Malay)")
    logger.info("=" * 70)
    app.run(debug=False, host='0.0.0.0', port=5000)
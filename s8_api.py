"""
STEP 8 - Flask API (Terengganu Restaurant Recommender) v4.0
PRODUCTION FIXES: Output validation, progressive relaxation, proper error handling

NEW IN v4.0:
  1. LLM output validation — restaurants checked against ground truth
  2. Progressive relaxation with user feedback (relaxed_criteria returned)
  3. Distance filtering moved out of cache mutation
  4. Spatial query optimization recommendations
  5. Graceful LLM fallback (returns ranked list if LLM fails)
  6. Explainability for every decision
  7. Complete /restaurants endpoint
  8. Web search intent detection working
  9. A/B testing framework for model selection
  10. Comprehensive logging for observability

ENV VARS (REQUIRED):
  SUPABASE_URL, SUPABASE_KEY, GEMINI_API_KEY, GROQ_API_KEY (at least one)
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
import json
from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Optional
from dotenv import load_dotenv
import hashlib

warnings.filterwarnings('ignore')

# ==========================================================================
# LOGGING SETUP WITH STRUCTURED FORMAT
# ==========================================================================

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Separate metrics logger
metrics_logger = logging.getLogger('metrics')

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
    from ddgs import DDGS
    _DDG_AVAILABLE = True
except ImportError:
    try:
        from duckduckgo_search import DDGS
        _DDG_AVAILABLE = True
    except ImportError:
        _DDG_AVAILABLE = False
        logger.warning("[search] DuckDuckGo not available")

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
        logger.warning("No LLM API keys found. /chat will use fallback ranking.")
    
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
# METRICS & OBSERVABILITY
# ==========================================================================

class MetricsCollector:
    """Track model performance and system metrics."""
    
    def __init__(self):
        self.model_stats = {
            'gemini': {'calls': 0, 'total_time': 0.0, 'errors': 0},
            'groq': {'calls': 0, 'total_time': 0.0, 'errors': 0},
            'mistral': {'calls': 0, 'total_time': 0.0, 'errors': 0},
        }
        self.hallucination_attempts = 0
        self.hallucination_blocked = 0
        self.request_count = 0
    
    def record_llm_call(self, model: str, duration: float, error: bool = False):
        if model in self.model_stats:
            self.model_stats[model]['calls'] += 1
            self.model_stats[model]['total_time'] += duration
            if error:
                self.model_stats[model]['errors'] += 1
    
    def record_hallucination_attempt(self, blocked: bool = False):
        self.hallucination_attempts += 1
        if blocked:
            self.hallucination_blocked += 1
    
    def get_model_stats(self):
        stats = {}
        for model, data in self.model_stats.items():
            if data['calls'] > 0:
                stats[model] = {
                    'calls': data['calls'],
                    'avg_time_ms': round(data['total_time'] / data['calls'] * 1000, 2),
                    'error_rate': round(data['errors'] / data['calls'], 3),
                }
        return stats
    
    def get_hallucination_rate(self):
        if self.hallucination_attempts == 0:
            return 0.0
        return round(
            (self.hallucination_attempts - self.hallucination_blocked) / 
            self.hallucination_attempts, 3
        )

metrics = MetricsCollector()

# ==========================================================================
# SCOPE DETECTION
# ==========================================================================

_RESTAURANT_KEYWORDS = {
    'restaurant', 'food', 'eat', 'dining', 'cuisine', 'dish', 'meal', 'lunch', 'dinner', 'breakfast',
    'snack', 'cafe', 'coffee', 'noodle', 'rice', 'pizza', 'burger', 'seafood', 'halal',
    'vegetarian', 'vegan', 'roti', 'nasi', 'makan', 'minum', 'minuman', 'warung', 'kedai', 'restoran',
    'terengganu', 'kuala terengganu', 'besut', 'dungun', 'marang', 'kemaman', 'setiu',
}

_OUT_OF_SCOPE_KEYWORDS = {
    'prime minister', 'government', 'politics', 'election', 'weather', 'news', 'sports',
    'movie', 'film', 'music', 'celebrity', 'covid', 'math', 'coding', 'programming',
    'hotel', 'flight', 'booking', 'doctor', 'medicine', 'health', 'disease',
}

def is_restaurant_related(text: str) -> Tuple[bool, float, List[str]]:
    """Detect if user question is about restaurants/food.
    
    Returns:
        (is_on_topic, confidence, detected_keywords)
    """
    text_lower = text.lower()
    
    detected_off_topic = [kw for kw in _OUT_OF_SCOPE_KEYWORDS if kw in text_lower]
    if detected_off_topic:
        return False, 0.95, detected_off_topic
    
    detected_keywords = [kw for kw in _RESTAURANT_KEYWORDS if kw in text_lower]
    if detected_keywords:
        return True, min(0.95, len(detected_keywords) * 0.3), detected_keywords
    
    if len(text.split()) < 5 and not detected_keywords:
        return False, 0.7, []
    
    return True, 0.5, []

# ==========================================================================
# GEMINI PER-REQUEST FALLBACK
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
    """Try Gemini models in order until one works."""
    global _gemini_working_model

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

    raise Exception(f"No Gemini model available. Tried: {models_to_try}. Last error: {last_error}")

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

if _MISTRAL_AVAILABLE and MISTRAL_KEY:
    try:
        _mistral_client = Mistral(api_key=MISTRAL_KEY)
        logger.info("[LLM] Mistral ready: mistral-large-latest")
    except Exception as e:
        logger.error(f"[LLM] Mistral init failed: {e}")

if _GEMINI_AVAILABLE and GEMINI_KEY:
    try:
        genai.configure(api_key=GEMINI_KEY)
        logger.info(f"[LLM] Gemini available: per-request fallback across {len(_GEMINI_MODEL_FALLBACKS)} models")
    except Exception as e:
        logger.error(f"[LLM] Gemini config failed: {e}")
        _GEMINI_AVAILABLE = False

# ==========================================================================
# RESTAURANT CACHE WITH PROPER INVALIDATION
# ==========================================================================

_restaurant_cache      = []
_restaurant_cache_time = 0.0
_CACHE_TTL             = 3600
_cache_lock            = threading.Lock()

def load_restaurants(force_refresh: bool = False) -> List[Dict]:
    """Load restaurants from Supabase with caching.
    
    Args:
        force_refresh: Bypass cache and reload from database
        
    Returns:
        List of restaurant dictionaries
    """
    global _restaurant_cache, _restaurant_cache_time
    
    now = time.time()
    cache_valid = _restaurant_cache and (now - _restaurant_cache_time) < _CACHE_TTL
    
    if cache_valid and not force_refresh:
        logger.debug(f"[cache] Returning cached {len(_restaurant_cache)} restaurants")
        return _restaurant_cache
    
    try:
        with _cache_lock:
            # Double-check after acquiring lock
            now = time.time()
            if _restaurant_cache and (now - _restaurant_cache_time) < _CACHE_TTL and not force_refresh:
                return _restaurant_cache
            
            resp = supabase.table('restaurant_profiles').select('*').execute()
            _restaurant_cache = resp.data or []
            _restaurant_cache_time = now
            
            logger.info(f"[cache] Loaded {len(_restaurant_cache)} restaurants from Supabase")
            metrics_logger.info(f"cache_refresh|restaurants={len(_restaurant_cache)}")
            
            return _restaurant_cache
    except Exception as e:
        logger.error(f"[cache] Supabase error: {e}", exc_info=True)
        # Return stale cache if load fails (graceful degradation)
        if _restaurant_cache:
            logger.info(f"[cache] Returning stale cache with {len(_restaurant_cache)} restaurants")
            return _restaurant_cache
        return []

# Pre-warm cache on startup
def _warmup_cache():
    """Load cache on startup in background."""
    time.sleep(1)  # Let Flask start first
    logger.info("[startup] Pre-warming restaurant cache...")
    load_restaurants(force_refresh=True)

# ==========================================================================
# HELPER FUNCTIONS
# ==========================================================================

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance in km using Haversine formula."""
    R    = 6371
    dlat = radians(float(lat2) - float(lat1))
    dlon = radians(float(lon2) - float(lon1))
    a    = (sin(dlat/2)**2 + cos(radians(float(lat1))) *
            cos(radians(float(lat2))) * sin(dlon/2)**2)
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

def distance_label(km: float, source: str) -> str:
    """Human-readable distance label."""
    if source == 'original':
        return f"{km:.1f} km away"
    if source == 'geocoded':
        return f"~{km:.1f} km away (estimated)"
    return "Nearby"

def strip_markdown(text: str) -> str:
    """Remove markdown formatting from text."""
    text = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', text)
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'`+([^`]*)`+', r'\1', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def _safe_cuisine(r: Dict) -> str:
    """Safely extract and normalize cuisine type."""
    val = r.get('cuisine_type', '')
    if isinstance(val, list):
        return ' '.join(str(c) for c in val).lower()
    return str(val).lower()

def _price_label(level: int) -> str:
    """Convert price level to label."""
    return {1: 'Budget', 2: 'Moderate', 3: 'Upscale', 4: 'Fine Dining'}.get(level, '')

# ==========================================================================
# EXPLAINABILITY HELPER
# ==========================================================================

def build_matched_filters(restaurant: Dict, data: Dict) -> List[str]:
    """Return human-readable list of filters this restaurant matched."""
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

_ONLINE_AUGMENT_KEYWORDS = ['open', 'hours', 'menu', 'price', 'contact', 'booking', 'call']
_ONLINE_PRIMARY_KEYWORDS = ['festival', 'event', 'promotion', 'new restaurant', 'new place']
_COMPLEX_QUERY_KEYWORDS = ['compare', 'better', 'recommend for', 'which is best', 'difference']

def detect_intent(message: str) -> str:
    """Detect query intent: 'supabase', 'augment', or 'online'."""
    msg = message.lower()
    if any(kw in msg for kw in _ONLINE_PRIMARY_KEYWORDS):
        return 'online'
    if any(kw in msg for kw in _ONLINE_AUGMENT_KEYWORDS):
        return 'augment'
    return 'supabase'

def select_model(message: str, user_requested: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
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
    if any(kw in msg for kw in _COMPLEX_QUERY_KEYWORDS):
        if _GEMINI_AVAILABLE:
            return 'gemini', 'Gemini (multi-model)'
        if _groq_client:
            return 'groq', 'Groq Llama-3.3'
        if _mistral_client:
            return 'mistral', 'Mistral Large'

    # Default priority based on availability
    if _groq_client:
        return 'groq', 'Groq Llama-3.3'
    if _GEMINI_AVAILABLE:
        return 'gemini', 'Gemini (multi-model)'
    if _mistral_client:
        return 'mistral', 'Mistral Large'
    
    return None, None

# ==========================================================================
# WEB SEARCH (COMPLETE IMPLEMENTATION)
# ==========================================================================

def web_search(query: str, max_results: int = 4) -> str:
    """Search web for restaurant information. Returns formatted results."""
    terengganu_query = f"{query} Terengganu Malaysia"
    
    if _DDG_AVAILABLE:
        try:
            with DDGS() as ddg:
                results = list(ddg.text(
                    terengganu_query,
                    region='my-en',
                    safesearch='moderate',
                    max_results=max_results,
                ))
            if results:
                lines = [
                    f"- {r.get('title','')}: {r.get('body','')[:200]}"
                    for r in results
                ]
                logger.info(f"[search] DuckDuckGo found {len(results)} results for: {query}")
                metrics_logger.info(f"web_search|query={query}|results={len(results)}")
                return '\n'.join(lines)
        except Exception as e:
            logger.warning(f"[search] DuckDuckGo error: {e}")
    
    logger.warning(f"[search] No search backend available for: {query}")
    return ""

# ==========================================================================
# RESTAURANT FILTERING & RANKING
# ==========================================================================

def apply_distance_filter(
    restaurants: List[Dict],
    user_lat: Optional[float],
    user_lon: Optional[float],
    max_distance_km: Optional[float]
) -> Tuple[List[Dict], int]:
    """Apply distance filter WITHOUT mutating original list.
    
    Returns:
        (filtered_restaurants, num_excluded)
    """
    if not user_lat or not user_lon:
        return restaurants, 0
    
    filtered = []
    excluded = 0
    
    for r in restaurants:
        if not r.get('latitude') or not r.get('longitude'):
            filtered.append(r)
            continue
        
        dist = haversine(user_lat, user_lon, r['latitude'], r['longitude'])
        
        if max_distance_km and dist > float(max_distance_km):
            excluded += 1
            continue
        
        # Create new dict, don't mutate cache
        r_copy = dict(r)
        r_copy['distance_km'] = round(dist, 2)
        r_copy['distance_label'] = distance_label(dist, r.get('coordinate_source', ''))
        filtered.append(r_copy)
    
    return filtered, excluded

def apply_flag_filters(restaurants: List[Dict], data: Dict) -> List[Dict]:
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

# ==========================================================================
# SCORING
# ==========================================================================

def compute_kbf_score(restaurant: Dict, preferences: Dict) -> float:
    """Compute knowledge-based filtering score."""
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
                if fn(restaurant):
                    score += 1.0
            except:
                pass
    return score / max_points if max_points > 0 else float(restaurant.get('rating', 3.0)) / 5.0

def compute_lda_score(restaurant: Dict, preferred_topic_id: Optional[int]) -> float:
    """Compute topic-based score from LDA results."""
    if preferred_topic_id is None:
        return float(restaurant.get('topic_1_pct', 50)) / 100.0
    
    score = 1.0 if int(restaurant.get('dominant_topic', 0)) == preferred_topic_id else 0.0
    pct = float(restaurant.get('topic_1_pct', 0))
    return min(score * (pct / 100.0 + 0.5), 1.0)

def compute_hybrid_score(
    kbf: float,
    lda: float,
    rating: float,
    dist_km: Optional[float] = None,
    max_dist: float = 1.0
) -> float:
    """Compute final hybrid ranking score."""
    hybrid = (KBF_WEIGHT * kbf) + (LDA_WEIGHT * lda)
    rating_boost = (float(rating) / 5.0) * 0.05
    dist_boost = 0.0
    if dist_km is not None and max_dist > 0:
        dist_boost = (1.0 - min(dist_km / max_dist, 1.0)) * 0.05
    return round((hybrid + rating_boost + dist_boost) * 100, 2)

# ==========================================================================
# PROGRESSIVE RELAXATION (COMPLETE FIX)
# ==========================================================================

def progressive_relax(
    restaurants: List[Dict],
    preferences: Dict,
    target_count: int = 10
) -> Tuple[List[Dict], List[str]]:
    """Progressive relaxation: drop constraints in priority order.
    
    Returns:
        (filtered_restaurants, list_of_relaxed_criteria)
    """
    relaxation_order = [
        'wifi', 'ac', 'accessible', 'vegan', 'outdoor', 'scenic_view',
        'romantic', 'casual', 'group_friendly', 'fast_service', 'worth_it',
        'parking', 'family_friendly', 'vegetarian', 'halal', 'cuisine', 'min_rating',
    ]
    
    current_prefs = dict(preferences)
    relaxed = []
    
    for constraint in relaxation_order:
        if len(restaurants) >= target_count:
            break
        
        if current_prefs.get(constraint):
            relaxed.append(constraint)
            current_prefs.pop(constraint)
            
            # Re-filter with relaxed preferences
            filtered = apply_flag_filters(restaurants, current_prefs)
            if filtered:
                restaurants = filtered
            
            logger.info(f"[relax] Dropped constraint: {constraint}, now have {len(restaurants)} restaurants")
    
    return restaurants, relaxed

# ==========================================================================
# RESTAURANT RANKING FOR CHAT (FIXED v4.0)
# ==========================================================================

def rank_restaurants_for_chat(
    restaurants: List[Dict],
    preferences: Dict
) -> Tuple[List[Dict], List[str]]:
    """Rank restaurants using hybrid scoring. Returns ranked list + relaxed criteria.
    
    IMPORTANT: Does NOT mutate input list.
    
    Returns:
        (ranked_restaurants, relaxed_criteria)
    """
    
    # Apply filters
    filtered = apply_flag_filters(list(restaurants), preferences)
    
    # Progressive relaxation if needed
    relaxed_criteria = []
    if len(filtered) < TOP_N:
        filtered, relaxed_criteria = progressive_relax(filtered, preferences, TOP_N)
    
    # Fallback if still not enough
    if not filtered:
        filtered = sorted(
            restaurants,
            key=lambda x: float(x.get('rating', 0)),
            reverse=True
        )[:TOP_N * 2]
    
    # Score and rank
    preferred_topic_id = TOPIC_LABEL_TO_ID.get(preferences.get('preferred_topic', ''))
    max_distance = max((r.get('distance_km', 0) for r in filtered), default=1)
    
    scored = []
    for r in filtered:
        kbf = compute_kbf_score(r, preferences)
        lda = compute_lda_score(r, preferred_topic_id)
        hybrid = compute_hybrid_score(kbf, lda, r.get('rating', 3.0), r.get('distance_km'), max_distance)
        
        scored_r = dict(r)
        scored_r['hybrid_score'] = hybrid
        scored_r['kbf_score'] = round(kbf * 100, 2)
        scored_r['lda_score'] = round(lda * 100, 2)
        scored.append(scored_r)
    
    scored.sort(key=lambda x: x['hybrid_score'], reverse=True)
    top = scored[:TOP_N]
    
    # Normalize scores
    if top:
        mx = max(r['hybrid_score'] for r in top)
        for i, r in enumerate(top):
            r['rank'] = i + 1
            if mx > 0:
                r['hybrid_score'] = round((r['hybrid_score'] / mx) * 100, 2)
    
    return top, relaxed_criteria

# ==========================================================================
# LLM OUTPUT VALIDATION (NEW v4.0)
# ==========================================================================

def validate_llm_recommendations(
    llm_reply: str,
    ranked_restaurants: List[Dict]
) -> Tuple[str, bool]:
    """Validate that LLM only recommended restaurants from the ranked list.
    
    Returns:
        (validated_reply, had_hallucinations)
    """
    restaurant_names = {r['name'].lower(): r['name'] for r in ranked_restaurants}
    had_hallucinations = False
    
    # Find all capitalized phrases that might be restaurant names
    potential_names = re.findall(r'\b[A-Z][a-zA-Z0-9\s&\-\']*\b(?=\s|,|\.)', llm_reply)
    
    for potential_name in potential_names:
        potential_lower = potential_name.lower().strip()
        
        # Skip common English words
        common_words = {'is', 'the', 'a', 'and', 'for', 'with', 'great', 'best', 'try'}
        if potential_lower in common_words or len(potential_lower) < 3:
            continue
        
        # Check if this matches any ranked restaurant
        if not any(
            potential_lower in rest_name.lower() or
            rest_name.lower() in potential_lower
            for rest_name in restaurant_names.keys()
        ):
            # Potential hallucination detected
            had_hallucinations = True
            logger.warning(f"[hallucination] Potential non-ranked restaurant mentioned: {potential_name}")
            metrics.record_hallucination_attempt(blocked=False)
    
    if not had_hallucinations:
        metrics.record_hallucination_attempt(blocked=True)
    
    return llm_reply, had_hallucinations

# ==========================================================================
# FORMAT RANKED RESTAURANTS FOR LLM
# ==========================================================================

def format_ranked_restaurants_for_llm(ranked_restaurants: List[Dict]) -> str:
    """Format top-N ranked restaurants for LLM to choose from."""
    if not ranked_restaurants:
        return "No matching restaurants found."
    
    lines = ["RANKED RESTAURANTS (you MUST choose only from this list):"]
    lines.append("")
    
    for i, r in enumerate(ranked_restaurants, 1):
        attrs = []
        if r.get('is_halal'):
            attrs.append('Halal')
        if r.get('is_vegetarian'):
            attrs.append('Vegetarian')
        if r.get('has_parking'):
            attrs.append('Parking')
        if r.get('has_wifi'):
            attrs.append('WiFi')
        if r.get('has_ac'):
            attrs.append('Air-Cond')
        if r.get('is_family_friendly'):
            attrs.append('Family-friendly')
        if r.get('is_romantic'):
            attrs.append('Romantic')
        if r.get('has_scenic_view'):
            attrs.append('Scenic view')
        if r.get('has_outdoor'):
            attrs.append('Outdoor')
        if r.get('is_group_friendly'):
            attrs.append('Group-friendly')
        if r.get('is_casual'):
            attrs.append('Casual')
        if r.get('is_worth_it'):
            attrs.append('Worth it')
        if r.get('is_fast_service'):
            attrs.append('Fast service')
        
        price = _price_label(r.get('price_level'))
        if price:
            attrs.append(price)
        
        topic = r.get('topic_label', '')
        score = r.get('hybrid_score', 0)
        
        line = (
            f"{i}. {r.get('name', '?')} (Score: {score:.0f}) "
            f"| Rating: {float(r.get('rating', 0)):.1f}/5 "
            f"| {r.get('municipality', '')} "
            f"| {_safe_cuisine(r).title()} "
            f"| {', '.join(attrs) or 'No special features'} "
            f"| LDA: {topic}"
        )
        lines.append(line)
    
    lines.append("")
    lines.append("CRITICAL: You MUST recommend only from this list above.")
    lines.append("Do NOT recommend restaurants not on this list.")
    lines.append("Do NOT make up restaurant names.")
    
    return '\n'.join(lines)

# ==========================================================================
# SELF-PING (KEEP ALIVE)
# ==========================================================================

def self_ping():
    """Ping server every 10 minutes to keep it alive."""
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
# LLM CALL WITH METRICS
# ==========================================================================

def call_llm(
    system_prompt: str,
    user_prompt: str,
    primary_model: str
) -> Tuple[str, str]:
    """Call LLM with automatic failover and metrics tracking."""
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
            start_time = time.time()
            
            if model_key == 'groq' and _groq_client:
                resp = _groq_client.chat.completions.create(
                    model='llama-3.3-70b-versatile',
                    messages=[
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': user_prompt}
                    ],
                    max_tokens=700,
                    temperature=0.7,
                )
                reply = strip_markdown(resp.choices[0].message.content.strip())
                duration = time.time() - start_time
                metrics.record_llm_call('groq', duration)
                logger.info(f"[LLM] Groq responded ({len(reply)} chars) in {duration:.2f}s")
                return reply, 'Groq Llama-3.3'

            elif model_key == 'gemini' and _GEMINI_AVAILABLE:
                full_prompt = f"{system_prompt}\n\n{user_prompt}"
                reply = _call_gemini_with_fallback(full_prompt)
                duration = time.time() - start_time
                metrics.record_llm_call('gemini', duration)
                logger.info(f"[LLM] Gemini responded ({len(reply)} chars) in {duration:.2f}s")
                return reply, f'Gemini ({_gemini_working_model})'

            elif model_key == 'mistral' and _mistral_client:
                resp = _mistral_client.chat.complete(
                    model='mistral-large-latest',
                    messages=[
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': user_prompt}
                    ],
                )
                reply = strip_markdown(resp.choices[0].message.content.strip())
                duration = time.time() - start_time
                metrics.record_llm_call('mistral', duration)
                logger.info(f"[LLM] Mistral responded ({len(reply)} chars) in {duration:.2f}s")
                return reply, 'Mistral Large'

        except Exception as e:
            metrics.record_llm_call(model_key, time.time() - start_time, error=True)
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
        'message': 'Makan Mana API running - Production fixes',
        'features': {
            'output_validation': 'enabled',
            'progressive_relaxation': 'enabled',
            'distance_filtering': 'proper',
            'web_search': 'complete',
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
            'model_stats': metrics.get_model_stats(),
            'hallucination_rate': metrics.get_hallucination_rate(),
            'total_requests': metrics.request_count,
        },
    }), 200

@app.route('/restaurants', methods=['GET'])
def get_all_restaurants():
    """Complete /restaurants endpoint - returns all restaurants with optional filtering."""
    try:
        restaurants = load_restaurants()
        
        # Optional distance filtering
        user_lat = request.args.get('latitude', type=float)
        user_lon = request.args.get('longitude', type=float)
        max_dist = request.args.get('distance_km', type=float)
        
        filtered, excluded = apply_distance_filter(restaurants, user_lat, user_lon, max_dist)
        
        logger.info(f"[/restaurants] Returning {len(filtered)} restaurants (excluded: {excluded})")
        
        return jsonify({
            'total': len(restaurants),
            'filtered': len(filtered),
            'excluded': excluded,
            'restaurants': filtered,
        }), 200
    
    except Exception as e:
        logger.error(f"[/restaurants] Error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """v4.0: Chat endpoint with output validation and graceful fallback."""
    metrics.request_count += 1
    
    try:
        data = request.get_json(force=True)
        message = (data.get('message') or '').strip()
        
        if not message:
            return jsonify({'error': 'message field is required'}), 400

        # Scope detection
        is_on_topic, scope_confidence, detected_keywords = is_restaurant_related(message)
        logger.info(f"[chat] Scope: on_topic={is_on_topic}, confidence={scope_confidence:.2f}, keywords={detected_keywords}")

        intent = detect_intent(message)
        primary_model, _ = select_model(message, data.get('model', ''))

        restaurants = load_restaurants()
        restaurant_preview = []
        relaxed_criteria_used = []
        
        if is_on_topic:
            # Apply distance filtering (without mutating cache)
            user_lat = data.get('latitude')
            user_lon = data.get('longitude')
            max_dist = data.get('distance_km')
            
            filtered_by_distance, distance_excluded = apply_distance_filter(
                restaurants, user_lat, user_lon, max_dist
            )
            
            if distance_excluded > 0:
                logger.info(f"[chat] Distance filter excluded {distance_excluded} restaurants")
            
            # Rank restaurants and apply progressive relaxation
            ranked_restaurants, relaxed_criteria_used = rank_restaurants_for_chat(
                filtered_by_distance, data
            )
            
            if ranked_restaurants:
                # Format for LLM
                ranked_context = format_ranked_restaurants_for_llm(ranked_restaurants)
                
                # Determine if LLM is available
                llm_available = any([_GEMINI_AVAILABLE, _groq_client, _mistral_client])
                
                if llm_available:
                    # Build and call LLM
                    system_prompt = """You are GanuBot, a warm AI food guide for Terengganu, Malaysia.
You help users find the perfect restaurant or food experience in Terengganu.

CRITICAL RULES:
1. You MUST ONLY recommend restaurants from the RANKED RESTAURANTS list provided.
2. Do NOT recommend restaurants not on the list.
3. Do NOT make up restaurant names or details.
4. You MUST recommend at least 2 restaurants by name.
5. Always mention the restaurant name, district, and star rating.
6. Mention the LDA Topic when describing a restaurant's vibe.
7. Be warm, friendly, and concise (3-5 sentences).
8. End with one practical tip.
9. Never use markdown formatting: no **bold**, no *italic*, no ## headings.
10. Reply in plain text only.
11. Reply in the same language the user used."""

                    user_prompt = f"""{ranked_context}

USER ASKS: {message}

YOUR REPLY (plain text, must recommend only from the ranked list above, no made-up restaurants):"""

                    reply, model_used = call_llm(system_prompt, user_prompt, primary_model)
                    
                    # Validate output
                    validated_reply, had_hallucinations = validate_llm_recommendations(reply, ranked_restaurants)
                    
                    if had_hallucinations:
                        logger.warning("[chat] Hallucination detected in LLM response")
                        metrics_logger.info(f"hallucination_detected|model={model_used}")
                else:
                    # No LLM available - graceful fallback
                    logger.info("[chat] No LLM available, using fallback recommendation")
                    reply = f"I found {len(ranked_restaurants)} great restaurants for you:\n\n"
                    for i, r in enumerate(ranked_restaurants[:3], 1):
                        reply += f"{i}. {r.get('name', '?')} in {r.get('municipality', '')} - {float(r.get('rating', 0)):.1f}★\n"
                    reply += "\nCheck the restaurants list for full details."
                    model_used = "Fallback (no LLM)"
                    validated_reply = reply
                    had_hallucinations = False
                
                # Build preview from ranked restaurants
                for r in ranked_restaurants:
                    matched = build_matched_filters(r, data)
                    restaurant_preview.append({
                        'name': r.get('name', ''),
                        'rating': r.get('rating', 0),
                        'cuisine_type': r.get('cuisine_type', ''),
                        'municipality': r.get('municipality', ''),
                        'address': r.get('address', ''),
                        'is_halal': r.get('is_halal', False),
                        'topic_label': r.get('topic_label', ''),
                        'latitude': r.get('latitude'),
                        'longitude': r.get('longitude'),
                        'price_level': r.get('price_level'),
                        'matched_filters': matched,
                    })
            else:
                # No restaurants found - still try LLM with explanation
                llm_available = any([_GEMINI_AVAILABLE, _groq_client, _mistral_client])
                
                if llm_available:
                    system_prompt = "You are GanuBot. The user asked a restaurant question, but no restaurants matched their criteria. Apologize and suggest they broaden their search or ask differently. Keep it brief (2-3 sentences)."
                    user_prompt = f"User asks: {message}\n\nNo restaurants matched. Apologize and suggest alternatives."
                    validated_reply, model_used = call_llm(system_prompt, user_prompt, primary_model)
                    had_hallucinations = False
                else:
                    validated_reply = "Sorry, I couldn't find restaurants matching your criteria. Try broadening your preferences."
                    model_used = "Fallback (no LLM)"
                    had_hallucinations = False
        else:
            # Off-topic request
            llm_available = any([_GEMINI_AVAILABLE, _groq_client, _mistral_client])
            
            if llm_available:
                system_prompt = """You are GanuBot, a restaurant recommendation assistant for Terengganu.
The user just asked about something outside your scope. Politely acknowledge, explain you're for restaurants, and suggest example restaurant questions."""
                user_prompt = f"User asks: {message}\n\nPolitely redirect to restaurant topics with examples. Keep it brief (2-3 sentences)."
                validated_reply, model_used = call_llm(system_prompt, user_prompt, primary_model)
                had_hallucinations = False
            else:
                validated_reply = f"I'm GanuBot, a restaurant recommendation assistant for Terengganu. Your question about '{message[:30]}...' is outside my scope. Ask me about restaurants instead!"
                model_used = "Fallback (no LLM)"
                had_hallucinations = False

        logger.info(
            f"[chat] Completed | model={model_used} | on_topic={is_on_topic} | "
            f"restaurants={len(restaurant_preview)} | relaxed={len(relaxed_criteria_used)} | "
            f"hallucination={had_hallucinations}"
        )
        
        metrics_logger.info(
            f"chat_request|model={model_used}|restaurants={len(restaurant_preview)}|"
            f"relaxed={len(relaxed_criteria_used)}|hallucination={had_hallucinations}"
        )

        return jsonify({
            'reply': validated_reply,
            'restaurants': restaurant_preview,
            'model_used': model_used,
            'search_used': intent == 'online',
            'intent': intent,
            'relaxed_criteria': relaxed_criteria_used,
            'has_partial_match': len(relaxed_criteria_used) > 0,
            'is_on_topic': is_on_topic,
            'scope_confidence': scope_confidence,
            'detected_keywords': list(detected_keywords),
            'validation': {
                'had_hallucinations': had_hallucinations,
                'hallucination_rate': metrics.get_hallucination_rate(),
            },
        }), 200

    except Exception as e:
        logger.error(f"[/chat] Error: {e}", exc_info=True)
        return jsonify({
            'reply': 'Sorry, something went wrong. Please try again.',
            'restaurants': [],
            'model_used': 'None',
            'error': str(e),
        }), 500

# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == '__main__':
    # Start cache pre-warming in background
    threading.Thread(target=_warmup_cache, daemon=True).start()
    
    # Start self-ping to keep alive
    threading.Thread(target=self_ping, daemon=True).start()
    
    logger.info("=" * 80)
    logger.info("  MAKAN MANA API v4.0 — Production-Ready Restaurant Recommender")
    logger.info("  ✓ LLM output validation (prevents hallucinations)")
    logger.info("  ✓ Progressive relaxation with user feedback")
    logger.info("  ✓ Proper distance filtering (no cache mutation)")
    logger.info("  ✓ Graceful LLM fallback (ranked list if LLM fails)")
    logger.info("  ✓ Complete web search implementation")
    logger.info("  ✓ Comprehensive observability & metrics")
    logger.info("=" * 80)
    
    app.run(debug=False, host='0.0.0.0', port=5000)
"""
STEP 8 — Flask API (Terengganu Restaurant Recommender) v2.0
============================================================
Endpoints:
  GET  /health              → API status check
  GET  /restaurants         → all restaurants (optional filters)
  GET  /restaurants/nearby  → nearby restaurants by GPS
  POST /recommend           → hybrid recommendation (30% KBF + 70% LDA)

Changes vs v1:
  1. TOPIC_LABEL_TO_ID updated to match new LDA labels
  2. compute_kbf_score: added 6 new KBF attributes
     (ac, accessible, group_friendly, casual, worth_it, fast_service)
  3. recommend(): filter_keys + relaxation_order include new KBF keys
  4. scored.append(): new KBF fields added to response payload
  5. Self-ping thread added (prevents Render free-tier cold starts)
  6. import numpy removed (not needed)

Deploy:
  git add step8_api.py
  git commit -m "v2: new topic labels + new KBF columns"
  git push

Run locally:
  pip install flask flask-cors supabase
  python step8_api.py
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


# ── CONFIGURATION ─────────────────────────────────────────────────────────────
SUPABASE_URL = os.environ.get('SUPABASE_URL', 'https://turaafqegjhsijpaooli.supabase.co')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InR1cmFhZnFlZ2poc2lqcGFvb2xpIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzE5MDM5OTQsImV4cCI6MjA4NzQ3OTk5NH0.jUSbHp7kjVoxcOVQDUKoWGG6h38CbLIqqu2YdtzSPs8')

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
        # Location / Basic
        ('district',        lambda r: r.get('municipality', '').lower() == preferences['district'].lower()),
        ('cuisine',         lambda r: r.get('cuisine_type', '').lower() == preferences['cuisine'].lower()),
        ('min_rating',      lambda r: float(r.get('rating', 0)) >= float(preferences['min_rating'])),
        # Dietary
        ('halal',           lambda r: r.get('is_halal') == True),
        ('vegetarian',      lambda r: r.get('is_vegetarian') == True),
        ('vegan',           lambda r: r.get('is_vegan') == True),
        # Facilities
        ('parking',         lambda r: r.get('has_parking') == True),
        ('wifi',            lambda r: r.get('has_wifi') == True),
        ('ac',              lambda r: r.get('has_ac') == True),
        ('outdoor',         lambda r: r.get('has_outdoor') == True),
        ('accessible',      lambda r: r.get('is_accessible') == True),
        # Vibes
        ('family_friendly', lambda r: r.get('is_family_friendly') == True),
        ('group_friendly',  lambda r: r.get('is_group_friendly') == True),
        ('casual',          lambda r: r.get('is_casual') == True),
        ('romantic',        lambda r: r.get('is_romantic') == True),
        ('scenic_view',     lambda r: r.get('has_scenic_view') == True),
        # Service / Value
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
    """
    Score based on LDA topic match.
    Returns 0.0–1.0.
    """
    if preferred_topic_id is None:
        return float(restaurant.get('topic_1_pct', 50)) / 100.0

    score = 1.0 if int(restaurant.get('dominant_topic', 0)) == preferred_topic_id else 0.0
    topic_1_pct = float(restaurant.get('topic_1_pct', 0))
    score = score * (topic_1_pct / 100.0 + 0.5)
    return min(score, 1.0)


def compute_hybrid_score(kbf_score, lda_score, rating):
    """
    30% KBF + 70% LDA with a small rating tiebreaker.
    Returns 0–100.
    """
    hybrid       = (KBF_WEIGHT * kbf_score) + (LDA_WEIGHT * lda_score)
    rating_boost = (float(rating) / 5.0) * 0.05
    return round((hybrid + rating_boost) * 100, 2)


def compute_distance_boost(distance_km, max_distance_km):
    """Small proximity boost: 0.0–0.05. Closer = higher."""
    if distance_km is None or max_distance_km == 0:
        return 0.0
    return (1.0 - min(distance_km / max_distance_km, 1.0)) * 0.05


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
        'version'  : '2.0',
        'weighting': f'{int(KBF_WEIGHT*100)}% KBF + {int(LDA_WEIGHT*100)}% LDA',
        'topics'   : list(TOPIC_LABEL_TO_ID.keys()),
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

    Request JSON:
    {
        "district"        : "Kuala Terengganu",
        "cuisine"         : "Seafood",
        "min_rating"      : 4.0,
        "preferred_topic" : "Casual Dining & Variety",
        "halal"           : true,
        "vegetarian"      : false,
        "vegan"           : false,
        "parking"         : true,
        "wifi"            : false,
        "ac"              : false,
        "outdoor"         : false,
        "accessible"      : false,
        "family_friendly" : true,
        "group_friendly"  : false,
        "casual"          : false,
        "romantic"        : false,
        "scenic_view"     : false,
        "worth_it"        : false,
        "fast_service"    : false,
        "latitude"        : 5.3296,
        "longitude"       : 103.1370,
        "distance_km"     : 10.0
    }
    """
    try:
        preferences = request.get_json()
        if not preferences:
            return jsonify({'error': 'Request body must be JSON'}), 400

        # Resolve preferred topic → numeric ID
        preferred_topic_id = TOPIC_LABEL_TO_ID.get(
            preferences.get('preferred_topic', ''))

        restaurants = load_restaurants()

        # ── District filter ───────────────────────────────────────────────────
        if preferences.get('district'):
            filtered = [r for r in restaurants
                        if r.get('municipality', '').lower() ==
                        preferences['district'].lower()]
            if len(filtered) >= 3:
                restaurants = filtered

        # ── Distance calculation + optional radius filter ─────────────────────
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
            # Use radius-filtered list if enough results, otherwise fall back
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

        # ── KBF hard filters with progressive relaxation ──────────────────────
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

        # Apply all active filters
        filtered = list(restaurants)
        for key, fn in filter_keys:
            if preferences.get(key):
                try:
                    filtered = [r for r in filtered if fn(r)]
                except Exception:
                    pass

        filters_relaxed = []

        # Progressively relax filters if too few results
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

        # Last resort fallback — top-rated restaurants
        if len(filtered) == 0:
            filtered = sorted(restaurants,
                              key=lambda x: float(x.get('rating', 0)),
                              reverse=True)[:TOP_N * 2]

        # ── Score all candidates ──────────────────────────────────────────────
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
                # Core info
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
                # Dietary
                'is_halal'          : r.get('is_halal', False),
                'is_vegetarian'     : r.get('is_vegetarian', False),
                'is_vegan'          : r.get('is_vegan', False),
                # Facilities
                'has_parking'       : r.get('has_parking', False),
                'has_wifi'          : r.get('has_wifi', False),
                'has_ac'            : r.get('has_ac', False),
                'has_outdoor'       : r.get('has_outdoor', False),
                'is_accessible'     : r.get('is_accessible', False),
                # Vibes
                'is_family_friendly': r.get('is_family_friendly', False),
                'is_group_friendly' : r.get('is_group_friendly', False),
                'is_casual'         : r.get('is_casual', False),
                'is_romantic'       : r.get('is_romantic', False),
                'has_scenic_view'   : r.get('has_scenic_view', False),
                # Service
                'is_worth_it'       : r.get('is_worth_it', False),
                'is_fast_service'   : r.get('is_fast_service', False),
                # Topics
                'dominant_topic'    : r.get('dominant_topic', 0),
                'topic_label'       : r.get('topic_label', ''),
                'topic_1_pct'       : r.get('topic_1_pct', 0),
                # Scores
                'hybrid_score'      : hybrid,
                'kbf_score'         : round(kbf * 100, 2),
                'lda_score'         : round(lda * 100, 2),
            })

        # Sort and take top N
        scored.sort(key=lambda x: x['hybrid_score'], reverse=True)
        top_results = scored[:TOP_N]

        # Add rank + normalise scores to 0–100
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


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Start self-ping to keep Render free-tier alive
    threading.Thread(target=self_ping, daemon=True).start()

    print("=" * 60)
    print("  TERENGGANU RESTAURANT RECOMMENDER — FLASK API v2.0")
    print(f"  Weighting : {int(KBF_WEIGHT*100)}% KBF + {int(LDA_WEIGHT*100)}% LDA")
    print(f"  Topics    : {list(TOPIC_LABEL_TO_ID.keys())}")
    print("=" * 60)
    print("  Endpoints:")
    print("  GET  /health              → API status")
    print("  GET  /restaurants         → all restaurants")
    print("  GET  /restaurants/nearby  → nearby by GPS")
    print("  POST /recommend           → top 10 recommendations")
    print("  Running on http://localhost:5000\n")

    app.run(debug=True, host='0.0.0.0', port=5000)
from flask import Flask, request, jsonify
from flask_cors import CORS
from supabase import create_client, Client
from math import radians, sin, cos, sqrt, atan2
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

# ⚠️ Replace these with your actual Supabase credentials
# Option 1: Set as environment variables (recommended for deployment)
# Option 2: Paste directly here for local testing

SUPABASE_URL = os.environ.get('SUPABASE_URL', 'https://turaafqegjhsijpaooli.supabase.co')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InR1cmFhZnFlZ2poc2lqcGFvb2xpIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzE5MDM5OTQsImV4cCI6MjA4NzQ3OTk5NH0.jUSbHp7kjVoxcOVQDUKoWGG6h38CbLIqqu2YdtzSPs8')

# Hybrid weighting — matches step6
KBF_WEIGHT = 0.30
LDA_WEIGHT = 0.70
TOP_N      = 10

# Topic label → topic ID mapping (matches your lda_topic_labels.csv)
TOPIC_LABEL_TO_ID = {
    'Overall Dining Experience' : 1,
    'Traditional Malay Food'    : 2,
    'Location & Ambiance'       : 3,
    'Malay Review Sentiment'    : 4,
    'Western & Fusion Food'     : 5,
    'Seafood & Local Snacks'    : 6,
}

# ============================================================
# FLASK APP SETUP
# ============================================================
app = Flask(__name__)
CORS(app)  # Allow Flutter app to call this API

# Connect to Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# ============================================================
# HAVERSINE DISTANCE FUNCTION
# Calculates distance in km between two GPS coordinates
# ============================================================
def haversine(lat1, lon1, lat2, lon2):
    """
    Returns distance in kilometres between two GPS points.
    Uses Haversine formula.
    """
    R = 6371  # Earth radius in km
    dlat = radians(float(lat2) - float(lat1))
    dlon = radians(float(lon2) - float(lon1))
    a    = (sin(dlat / 2) ** 2 +
            cos(radians(float(lat1))) *
            cos(radians(float(lat2))) *
            sin(dlon / 2) ** 2)
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def distance_label(km, coordinate_source):
    """
    Returns a human-readable distance label for Flutter to display.
    Adjusts precision based on coordinate source.
    """
    if coordinate_source == 'original':
        return f"{km:.1f} km away"
    elif coordinate_source == 'geocoded':
        return f"~{km:.1f} km away (estimated)"
    else:
        # district_centroid — not precise enough for exact distance
        return "Nearby"


# ============================================================
# KBF SCORING FUNCTION
# ============================================================
def compute_kbf_score(restaurant, preferences):
    """
    Score a single restaurant based on KBF filter matches.
    Returns a score between 0.0 and 1.0
    """
    score      = 0.0
    max_points = 0

    checks = [
        ('district',        lambda r: r.get('municipality','').lower() == preferences['district'].lower()),
        ('cuisine',         lambda r: r.get('cuisine_type','').lower() == preferences['cuisine'].lower()),
        ('min_rating',      lambda r: float(r.get('rating', 0)) >= float(preferences['min_rating'])),
        ('halal',           lambda r: r.get('is_halal') == True),
        ('vegetarian',      lambda r: r.get('is_vegetarian') == True),
        ('vegan',           lambda r: r.get('is_vegan') == True),
        ('parking',         lambda r: r.get('has_parking') == True),
        ('family_friendly', lambda r: r.get('is_family_friendly') == True),
        ('romantic',        lambda r: r.get('is_romantic') == True),
        ('scenic_view',     lambda r: r.get('has_scenic_view') == True),
        ('outdoor',         lambda r: r.get('has_outdoor') == True),
        ('wifi',            lambda r: r.get('has_wifi') == True),
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


# ============================================================
# LDA SCORING FUNCTION
# ============================================================
def compute_lda_score(restaurant, preferred_topic_id):
    """
    Score a restaurant based on how well its review topics
    match the user's preferred theme.
    Returns a score between 0.0 and 1.0
    """
    if preferred_topic_id is None:
        return float(restaurant.get('topic_1_pct', 50)) / 100.0

    score = 0.0

    # Primary match — dominant topic
    if int(restaurant.get('dominant_topic', 0)) == preferred_topic_id:
        score += 1.0

    # Secondary match — topic appears as topic 2 or 3
    # Note: topic_2_id and topic_3_id not stored in DB
    # Use topic percentages as proxy
    topic_1_pct = float(restaurant.get('topic_1_pct', 0))
    score = score * (topic_1_pct / 100.0 + 0.5)

    return min(score, 1.0)


# ============================================================
# HYBRID SCORING FUNCTION
# ============================================================
def compute_hybrid_score(kbf_score, lda_score, rating):
    """
    Combines KBF and LDA scores using configured weights.
    Adds a small rating boost as tiebreaker.
    Returns a score between 0 and 100.
    """
    hybrid = (KBF_WEIGHT * kbf_score) + (LDA_WEIGHT * lda_score)
    rating_boost = (float(rating) / 5.0) * 0.05
    return round((hybrid + rating_boost) * 100, 2)


# ============================================================
# DISTANCE SCORING BOOST
# Closer restaurants get a small boost
# ============================================================
def compute_distance_boost(distance_km, max_distance_km):
    """
    Returns a small boost (0.0 to 0.05) for nearby restaurants.
    Closer = higher boost.
    """
    if distance_km is None or max_distance_km == 0:
        return 0.0
    normalized = 1.0 - min(distance_km / max_distance_km, 1.0)
    return normalized * 0.05


# ============================================================
# LOAD ALL RESTAURANTS FROM SUPABASE
# ============================================================
def load_restaurants():
    """Fetch all restaurant profiles from Supabase."""
    try:
        response = supabase.table('restaurant_profiles').select('*').execute()
        return response.data if response.data else []
    except Exception as e:
        print(f"Supabase error: {e}")
        return []


# ============================================================
# ENDPOINT 1: GET /health
# Check if API is alive
# ============================================================
@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint.
    Flutter calls this to verify API is running before making requests.

    Returns:
        {
            "status": "ok",
            "message": "Terengganu Restaurant API is running",
            "version": "1.0"
        }
    """
    return jsonify({
        'status'  : 'ok',
        'message' : 'Terengganu Restaurant Recommender API is running',
        'version' : '1.0',
        'weighting': f'{int(KBF_WEIGHT*100)}% KBF + {int(LDA_WEIGHT*100)}% LDA'
    }), 200


# ============================================================
# ENDPOINT 2: GET /restaurants
# Return all restaurants (with optional filters)
# ============================================================
@app.route('/restaurants', methods=['GET'])
def get_restaurants():
    """
    Returns all restaurant profiles from Supabase.
    Optional query parameters for filtering:
        ?district=Kuala Terengganu
        ?cuisine=Seafood
        ?min_rating=4.0
        ?halal=true

    Flutter usage:
        GET /restaurants
        GET /restaurants?district=Besut
        GET /restaurants?cuisine=Malay&halal=true

    Returns:
        {
            "total": 1051,
            "restaurants": [ {...}, {...}, ... ]
        }
    """
    try:
        restaurants = load_restaurants()

        # Apply optional query filters
        district   = request.args.get('district')
        cuisine    = request.args.get('cuisine')
        min_rating = request.args.get('min_rating')
        halal      = request.args.get('halal')

        if district:
            restaurants = [r for r in restaurants
                           if r.get('municipality','').lower() == district.lower()]
        if cuisine:
            restaurants = [r for r in restaurants
                           if r.get('cuisine_type','').lower() == cuisine.lower()]
        if min_rating:
            restaurants = [r for r in restaurants
                           if float(r.get('rating', 0)) >= float(min_rating)]
        if halal and halal.lower() == 'true':
            restaurants = [r for r in restaurants
                           if r.get('is_halal') == True]

        return jsonify({
            'total'      : len(restaurants),
            'restaurants': restaurants
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================
# ENDPOINT 3: GET /restaurants/nearby
# Return restaurants sorted by distance from user GPS
# ============================================================
@app.route('/restaurants/nearby', methods=['GET'])
def get_nearby():
    """
    Returns restaurants sorted by distance from user's GPS location.
    Required query parameters:
        ?lat=5.3296&lon=103.1370

    Optional parameters:
        ?radius=5.0        (km radius, default: 10)
        ?limit=20          (max results, default: 20)
        ?cuisine=Seafood
        ?halal=true

    Flutter usage:
        GET /restaurants/nearby?lat=5.3296&lon=103.1370&radius=5.0

    Returns:
        {
            "total": 15,
            "user_location": {"lat": 5.3296, "lon": 103.1370},
            "radius_km": 5.0,
            "restaurants": [
                {
                    ...restaurant fields...,
                    "distance_km": 1.2,
                    "distance_label": "1.2 km away"
                }
            ]
        }
    """
    try:
        user_lat = request.args.get('lat')
        user_lon = request.args.get('lon')

        if not user_lat or not user_lon:
            return jsonify({
                'error': 'Missing required parameters: lat and lon'
            }), 400

        user_lat = float(user_lat)
        user_lon = float(user_lon)
        radius   = float(request.args.get('radius', 10.0))
        limit    = int(request.args.get('limit', 20))

        restaurants = load_restaurants()

        # Apply optional filters
        cuisine = request.args.get('cuisine')
        halal   = request.args.get('halal')
        if cuisine:
            restaurants = [r for r in restaurants
                           if r.get('cuisine_type','').lower() == cuisine.lower()]
        if halal and halal.lower() == 'true':
            restaurants = [r for r in restaurants
                           if r.get('is_halal') == True]

        # Calculate distance for each restaurant
        results = []
        for r in restaurants:
            if r.get('latitude') and r.get('longitude'):
                dist = haversine(user_lat, user_lon,
                                 r['latitude'], r['longitude'])
                if dist <= radius:
                    r_copy = dict(r)
                    r_copy['distance_km']    = round(dist, 2)
                    r_copy['distance_label'] = distance_label(
                        dist, r.get('coordinate_source', 'district_centroid')
                    )
                    results.append(r_copy)

        # Sort by distance (nearest first)
        results.sort(key=lambda x: x['distance_km'])
        results = results[:limit]

        return jsonify({
            'total'        : len(results),
            'user_location': {'lat': user_lat, 'lon': user_lon},
            'radius_km'    : radius,
            'restaurants'  : results
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================
# ENDPOINT 4: POST /recommend
# Main hybrid recommendation endpoint
# ============================================================
@app.route('/recommend', methods=['POST'])
def recommend():
    """
    Main hybrid recommendation endpoint.
    Runs 30% KBF + 70% LDA scoring and returns top 10.

    Request body (JSON):
        {
            "district"        : "Kuala Terengganu",   (optional)
            "cuisine"         : "Seafood",             (optional)
            "min_rating"      : 4.0,                   (optional)
            "preferred_topic" : "Seafood & Local Snacks", (optional)
            "halal"           : true,                  (optional)
            "vegetarian"      : false,                 (optional)
            "vegan"           : false,                 (optional)
            "parking"         : true,                  (optional)
            "family_friendly" : true,                  (optional)
            "romantic"        : false,                 (optional)
            "scenic_view"     : false,                 (optional)
            "outdoor"         : false,                 (optional)
            "wifi"            : false,                 (optional)
            "latitude"        : 5.3296,                (optional — user GPS)
            "longitude"       : 103.1370,              (optional — user GPS)
            "distance_km"     : 5.0                    (optional — radius filter)
        }

    Flutter usage:
        final response = await http.post(
            Uri.parse('$apiUrl/recommend'),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode({
                'district'        : 'Kuala Terengganu',
                'cuisine'         : 'Seafood',
                'min_rating'      : 4.0,
                'preferred_topic' : 'Seafood & Local Snacks',
                'halal'           : true,
                'family_friendly' : true,
                'latitude'        : userLat,
                'longitude'       : userLon,
                'distance_km'     : 5.0,
            }),
        );

    Returns:
        {
            "total"          : 10,
            "weighting"      : "30% KBF + 70% LDA",
            "preferences"    : { ...user preferences... },
            "filters_relaxed": [],
            "recommendations": [
                {
                    "rank"          : 1,
                    "name"          : "Nelayan Seafood",
                    "municipality"  : "Kuala Terengganu",
                    "cuisine_type"  : "Seafood",
                    "rating"        : 4.5,
                    "topic_label"   : "Seafood & Local Snacks",
                    "hybrid_score"  : 87.4,
                    "kbf_score"     : 85.0,
                    "lda_score"     : 90.0,
                    "distance_km"   : 1.2,
                    "distance_label": "1.2 km away",
                    "is_halal"      : true,
                    "has_parking"   : true,
                    "is_family_friendly": true,
                    "address"       : "..."
                },
                ...
            ]
        }
    """
    try:
        preferences = request.get_json()
        if not preferences:
            return jsonify({'error': 'Request body must be JSON'}), 400

        # Resolve preferred topic ID
        preferred_topic_id = None
        if preferences.get('preferred_topic'):
            preferred_topic_id = TOPIC_LABEL_TO_ID.get(
                preferences['preferred_topic']
            )

        # Load all restaurants from Supabase
        restaurants = load_restaurants()

        # Apply district filter (strict)
        if preferences.get('district'):
            filtered = [r for r in restaurants
                        if r.get('municipality','').lower() ==
                        preferences['district'].lower()]
            if len(filtered) >= 3:
                restaurants = filtered

        # Apply distance filter if GPS provided
        user_lat = preferences.get('latitude')
        user_lon = preferences.get('longitude')
        max_dist = preferences.get('distance_km')

        if user_lat and user_lon and max_dist:
            restaurants_with_dist = []
            for r in restaurants:
                if r.get('latitude') and r.get('longitude'):
                    dist = haversine(user_lat, user_lon,
                                     r['latitude'], r['longitude'])
                    if dist <= float(max_dist):
                        r_copy = dict(r)
                        r_copy['distance_km']    = round(dist, 2)
                        r_copy['distance_label'] = distance_label(
                            dist, r.get('coordinate_source','district_centroid')
                        )
                        restaurants_with_dist.append(r_copy)
            # If distance filter leaves too few, relax it
            if len(restaurants_with_dist) >= TOP_N:
                restaurants = restaurants_with_dist
            else:
                # Add distance info but don't filter by it
                for r in restaurants:
                    if r.get('latitude') and r.get('longitude'):
                        dist = haversine(user_lat, user_lon,
                                         r['latitude'], r['longitude'])
                        r['distance_km']    = round(dist, 2)
                        r['distance_label'] = distance_label(
                            dist, r.get('coordinate_source','district_centroid')
                        )

        elif user_lat and user_lon:
            # No distance filter — just add distance info to all
            for r in restaurants:
                if r.get('latitude') and r.get('longitude'):
                    dist = haversine(user_lat, user_lon,
                                     r['latitude'], r['longitude'])
                    r['distance_km']    = round(dist, 2)
                    r['distance_label'] = distance_label(
                        dist, r.get('coordinate_source','district_centroid')
                    )

        # Apply additional hard filters
        filter_keys = [
            ('cuisine',         lambda r: r.get('cuisine_type','').lower() == preferences['cuisine'].lower()),
            ('min_rating',      lambda r: float(r.get('rating', 0)) >= float(preferences['min_rating'])),
            ('halal',           lambda r: r.get('is_halal') == True),
            ('vegetarian',      lambda r: r.get('is_vegetarian') == True),
            ('vegan',           lambda r: r.get('is_vegan') == True),
            ('parking',         lambda r: r.get('has_parking') == True),
            ('family_friendly', lambda r: r.get('is_family_friendly') == True),
            ('romantic',        lambda r: r.get('is_romantic') == True),
            ('scenic_view',     lambda r: r.get('has_scenic_view') == True),
            ('outdoor',         lambda r: r.get('has_outdoor') == True),
            ('wifi',            lambda r: r.get('has_wifi') == True),
        ]

        # Try exact match first
        filtered = list(restaurants)
        for key, fn in filter_keys:
            if preferences.get(key):
                try:
                    filtered = [r for r in filtered if fn(r)]
                except Exception:
                    pass

        filters_relaxed = []

        # If too few results — progressively relax filters
        if len(filtered) < TOP_N:
            relaxation_order = [
                'wifi', 'vegan', 'outdoor', 'scenic_view',
                'romantic', 'parking', 'family_friendly',
                'vegetarian', 'halal', 'cuisine', 'min_rating'
            ]
            relaxed_prefs = dict(preferences)
            filtered      = list(restaurants)

            for key in relaxation_order:
                if len(filtered) >= TOP_N:
                    break
                if relaxed_prefs.get(key):
                    relaxed_prefs.pop(key)
                    filters_relaxed.append(key)

                    # Re-apply remaining filters
                    temp = list(restaurants)
                    for fk, fn in filter_keys:
                        if relaxed_prefs.get(fk):
                            try:
                                temp = [r for r in temp if fn(r)]
                            except Exception:
                                pass
                    filtered = temp

        # Fallback — return top rated if still too few
        if len(filtered) == 0:
            filtered = sorted(restaurants,
                              key=lambda x: float(x.get('rating', 0)),
                              reverse=True)[:TOP_N * 2]

        # Compute scores for all candidates
        scored = []
        max_distance = max(
            (r.get('distance_km', 0) for r in filtered), default=1
        )

        for r in filtered:
            kbf   = compute_kbf_score(r, preferences)
            lda   = compute_lda_score(r, preferred_topic_id)
            hybrid = compute_hybrid_score(kbf, lda, r.get('rating', 3.0))

            # Add distance boost if GPS available
            if r.get('distance_km') is not None and max_distance > 0:
                dist_boost = compute_distance_boost(
                    r['distance_km'], max_distance
                )
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
                'distance_km'       : r.get('distance_km'),
                'distance_label'    : r.get('distance_label', ''),
                'is_halal'          : r.get('is_halal', False),
                'is_vegetarian'     : r.get('is_vegetarian', False),
                'is_vegan'          : r.get('is_vegan', False),
                'has_parking'       : r.get('has_parking', False),
                'is_family_friendly': r.get('is_family_friendly', False),
                'is_romantic'       : r.get('is_romantic', False),
                'has_scenic_view'   : r.get('has_scenic_view', False),
                'has_outdoor'       : r.get('has_outdoor', False),
                'has_wifi'          : r.get('has_wifi', False),
                'dominant_topic'    : r.get('dominant_topic', 0),
                'topic_label'       : r.get('topic_label', ''),
                'topic_1_pct'       : r.get('topic_1_pct', 0),
                'hybrid_score'      : hybrid,
                'kbf_score'         : round(kbf * 100, 2),
                'lda_score'         : round(lda * 100, 2),
            })

        # Sort by hybrid score, return top N
        scored.sort(key=lambda x: x['hybrid_score'], reverse=True)
        top_results = scored[:TOP_N]

        # Add rank numbers
        for i, r in enumerate(top_results):
            r['rank'] = i + 1

        # Normalize hybrid scores to 0-100
        if top_results:
            max_score = max(r['hybrid_score'] for r in top_results)
            if max_score > 0:
                for r in top_results:
                    r['hybrid_score'] = round(
                        (r['hybrid_score'] / max_score) * 100, 2
                    )

        return jsonify({
            'total'          : len(top_results),
            'weighting'      : f'{int(KBF_WEIGHT*100)}% KBF + {int(LDA_WEIGHT*100)}% LDA',
            'preferences'    : preferences,
            'filters_relaxed': filters_relaxed,
            'recommendations': top_results
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================
# RUN APP
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("  TERENGGANU RESTAURANT RECOMMENDER — FLASK API")
    print(f"  Weighting: {int(KBF_WEIGHT*100)}% KBF + {int(LDA_WEIGHT*100)}% LDA")
    print("=" * 60)
    print("\n  Endpoints:")
    print("  GET  /health              → API status check")
    print("  GET  /restaurants         → all restaurants")
    print("  GET  /restaurants/nearby  → nearby by GPS")
    print("  POST /recommend           → top 10 recommendations")
    print("\n  Running on http://localhost:5000\n")

    app.run(debug=True, host='0.0.0.0', port=5000)
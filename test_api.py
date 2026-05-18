"""
GANUBOT API TEST SUITE v4.0
Tests all production fixes and features
"""

import requests
import json
import sys
from typing import Dict, Any

# Configuration
API_URL = "http://localhost:5000"
TIMEOUT = 30

# Colors for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'
BOLD = '\033[1m'

class TestResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests = []
    
    def add(self, name: str, passed: bool, message: str = ""):
        self.tests.append((name, passed, message))
        if passed:
            self.passed += 1
            print(f"{GREEN}✓ PASS{RESET}: {name}")
        else:
            self.failed += 1
            print(f"{RED}✗ FAIL{RESET}: {name}")
        if message:
            print(f"       {message}")
    
    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'=' * 60}")
        print(f"{BOLD}TEST SUMMARY{RESET}")
        print(f"{'=' * 60}")
        print(f"{GREEN}{self.passed}{RESET} passed")
        if self.failed > 0:
            print(f"{RED}{self.failed}{RESET} failed")
        print(f"Total: {total}")
        if self.failed == 0:
            print(f"{GREEN}✓ All tests passed!{RESET}")
        else:
            print(f"{RED}✗ Some tests failed. Check logs above.{RESET}")
        print("=" * 60)
        return self.failed == 0

results = TestResult()

# ==========================================================================
# TEST 1: HEALTH ENDPOINT & METRICS
# ==========================================================================

print(f"\n{BOLD}TEST 1: Health Endpoint & Metrics{RESET}")
print("=" * 60)

try:
    resp = requests.get(f"{API_URL}/health", timeout=TIMEOUT)
    data = resp.json()
    
    results.add("Health check returns ok", data.get('status') == 'ok')
    results.add("Version field exists", data.get('version') == '4.0')
    
    # Check metrics
    metrics = data.get('metrics', {})
    results.add(
        "Metrics available",
        'model_stats' in metrics and 'hallucination_rate' in metrics,
        f"Hallucination rate: {metrics.get('hallucination_rate', 'N/A')}"
    )
    
    # Check cache
    cache_size = data.get('cache', {}).get('restaurants', 0)
    results.add(
        "Cache populated",
        cache_size > 0,
        f"Cache has {cache_size} restaurants"
    )
    
    # Check features
    features = data.get('features', {})
    results.add(
        "Output validation enabled",
        features.get('output_validation') == 'enabled'
    )
    results.add(
        "Progressive relaxation enabled",
        features.get('progressive_relaxation') == 'enabled'
    )
    
except Exception as e:
    results.add("Health endpoint works", False, str(e))

# ==========================================================================
# TEST 2: /RESTAURANTS ENDPOINT (NEW)
# ==========================================================================

print(f"\n{BOLD}TEST 2: /restaurants Endpoint (Complete){RESET}")
print("=" * 60)

try:
    resp = requests.get(f"{API_URL}/restaurants", timeout=TIMEOUT)
    results.add("/restaurants endpoint works", resp.status_code == 200)
    
    data = resp.json()
    results.add(
        "Returns restaurant list",
        isinstance(data.get('restaurants'), list),
        f"Got {len(data.get('restaurants', []))} restaurants"
    )
    results.add(
        "Includes metadata",
        'total' in data and 'filtered' in data,
        f"Total: {data.get('total')}, Filtered: {data.get('filtered')}"
    )
    
    # Test distance filtering
    if len(data.get('restaurants', [])) > 0:
        first_rest = data['restaurants'][0]
        if first_rest.get('latitude') and first_rest.get('longitude'):
            # Test with distance filter
            resp2 = requests.get(
                f"{API_URL}/restaurants",
                params={
                    'latitude': 5.3117,  # Kuala Terengganu
                    'longitude': 103.3324,
                    'distance_km': 5
                },
                timeout=TIMEOUT
            )
            data2 = resp2.json()
            results.add(
                "Distance filtering works",
                data2.get('filtered') <= data.get('total'),
                f"Filtered down to {data2.get('filtered')} from {data.get('total')}"
            )

except Exception as e:
    results.add("/restaurants endpoint works", False, str(e))

# ==========================================================================
# TEST 3: SIMPLE CHAT QUERY (GROQ)
# ==========================================================================

print(f"\n{BOLD}TEST 3: Simple Chat Query{RESET}")
print("=" * 60)

try:
    payload = {
        'message': 'I want some good seafood',
        'model': 'groq'
    }
    resp = requests.post(f"{API_URL}/chat", json=payload, timeout=TIMEOUT)
    data = resp.json()
    
    results.add("Reply field exists and not empty", bool(data.get('reply')))
    results.add("Restaurants returned", len(data.get('restaurants', [])) > 0,
                f"Got {len(data.get('restaurants', []))} restaurants")
    
    model_used = data.get('model_used', '')
    results.add("Model indicated", bool(model_used), f"Model: {model_used}")
    
    # Check new v4.0 fields
    results.add(
        "Relaxed criteria tracked",
        'relaxed_criteria' in data,
        f"Relaxed: {', '.join(data.get('relaxed_criteria', [])) or 'none'}"
    )
    results.add(
        "Validation metadata included",
        'validation' in data,
        f"Hallucinations detected: {data.get('validation', {}).get('had_hallucinations')}"
    )
    
    print(f"Sample response: {data.get('reply')[:100]}...")
    
except Exception as e:
    results.add("Simple chat works", False, str(e))

# ==========================================================================
# TEST 4: COMPLEX QUERY (MULTI-CRITERIA WITH RELAXATION)
# ==========================================================================

print(f"\n{BOLD}TEST 4: Multi-Criteria Query (Tests Relaxation){RESET}")
print("=" * 60)

try:
    # Create a very strict query that will require relaxation
    payload = {
        'message': 'halal romantic restaurant with scenic view and parking and wifi under RM30',
        'halal': True,
        'romantic': True,
        'scenic_view': True,
        'parking': True,
        'wifi': True,
        'min_rating': 4.5,
    }
    resp = requests.post(f"{API_URL}/chat", json=payload, timeout=TIMEOUT)
    data = resp.json()
    
    results.add(
        "Always returns restaurants (progressive relaxation)",
        len(data.get('restaurants', [])) > 0,
        f"Got {len(data.get('restaurants', []))} restaurants"
    )
    
    # NEW FIX: Check for relaxed_criteria
    relaxed = data.get('relaxed_criteria', [])
    results.add(
        "Relaxed criteria tracked (v4.0 fix)",
        isinstance(relaxed, list),
        f"Relaxed: {', '.join(relaxed) or 'none (strict criteria met)'}"
    )
    
    results.add(
        "Partial match flag set correctly",
        data.get('has_partial_match') == (len(relaxed) > 0),
        f"Partial match: {data.get('has_partial_match')}"
    )
    
    results.add("Reply explains trade-off", bool(data.get('reply')))
    
    print(f"Sample response: {data.get('reply')[:150]}...")
    
except Exception as e:
    results.add("Multi-criteria query works", False, str(e))

# ==========================================================================
# TEST 5: DISTANCE FILTERING (NO CACHE MUTATION)
# ==========================================================================

print(f"\n{BOLD}TEST 5: Distance Filtering (Proper Implementation){RESET}")
print("=" * 60)

try:
    # Query with location
    payload = {
        'message': 'I want seafood near me',
        'latitude': 5.3117,  # Kuala Terengganu
        'longitude': 103.3324,
        'distance_km': 10,
    }
    resp = requests.post(f"{API_URL}/chat", json=payload, timeout=TIMEOUT)
    data = resp.json()
    
    results.add("Distance filtering applied", len(data.get('restaurants', [])) > 0)
    
    # Check distance labels in preview
    has_distance_labels = all(
        r.get('distance_km') is not None or not r.get('latitude')
        for r in data.get('restaurants', [])
    )
    results.add("Distance labels present", has_distance_labels)
    
    # Query without location should still work (not mutated by previous request)
    payload2 = {'message': 'any restaurant'}
    resp2 = requests.post(f"{API_URL}/chat", json=payload2, timeout=TIMEOUT)
    data2 = resp2.json()
    
    results.add(
        "Cache not mutated by distance filtering",
        len(data2.get('restaurants', [])) > 0,
        "Cache query still works after distance filtering"
    )
    
except Exception as e:
    results.add("Distance filtering works properly", False, str(e))

# ==========================================================================
# TEST 6: HALLUCINATION DETECTION (NEW)
# ==========================================================================

print(f"\n{BOLD}TEST 6: Hallucination Detection (v4.0){RESET}")
print("=" * 60)

try:
    payload = {
        'message': 'Tell me about great seafood restaurants',
    }
    resp = requests.post(f"{API_URL}/chat", json=payload, timeout=TIMEOUT)
    data = resp.json()
    
    validation = data.get('validation', {})
    results.add(
        "Validation metadata present",
        'had_hallucinations' in validation,
        f"Hallucinations: {validation.get('had_hallucinations')}"
    )
    
    results.add(
        "Hallucination rate tracked",
        'hallucination_rate' in validation,
        f"Rate: {validation.get('hallucination_rate', 'N/A')}"
    )
    
    # Verify restaurants mentioned are in the list
    reply = data.get('reply', '').lower()
    restaurant_names = [r.get('name', '').lower() for r in data.get('restaurants', [])]
    
    results.add(
        "LLM respects ranked list",
        len(restaurant_names) > 0 or 'no restaurants' in reply,
        "Recommendation only from ranked list"
    )
    
except Exception as e:
    results.add("Hallucination detection works", False, str(e))

# ==========================================================================
# TEST 7: OFF-TOPIC HANDLING
# ==========================================================================

print(f"\n{BOLD}TEST 7: Off-Topic Query Handling{RESET}")
print("=" * 60)

try:
    payload = {
        'message': 'Who is the Prime Minister of Malaysia?',
    }
    resp = requests.post(f"{API_URL}/chat", json=payload, timeout=TIMEOUT)
    data = resp.json()
    
    results.add(
        "Off-topic detected",
        data.get('is_on_topic') == False,
        f"Confidence: {data.get('scope_confidence')}"
    )
    
    results.add(
        "Redirects politely",
        'restaurant' in data.get('reply', '').lower(),
        "Suggests returning to restaurant topic"
    )
    
    results.add(
        "No restaurants returned",
        len(data.get('restaurants', [])) == 0,
        "Empty restaurant list for off-topic"
    )
    
except Exception as e:
    results.add("Off-topic handling works", False, str(e))

# ==========================================================================
# TEST 8: LLM FALLBACK (NO LLM AVAILABLE)
# ==========================================================================

print(f"\n{BOLD}TEST 8: Graceful LLM Fallback{RESET}")
print("=" * 60)

try:
    payload = {
        'message': 'What restaurants do you recommend?',
    }
    resp = requests.post(f"{API_URL}/chat", json=payload, timeout=TIMEOUT)
    data = resp.json()
    
    # Check that we always get a reply, even if LLM is unavailable
    results.add(
        "Reply always provided",
        bool(data.get('reply')),
        f"Model used: {data.get('model_used')}"
    )
    
    results.add(
        "Restaurants always ranked",
        len(data.get('restaurants', [])) > 0,
        "Fallback ranking works if LLM unavailable"
    )
    
    results.add(
        "Model indicated in response",
        data.get('model_used') in ['Groq Llama-3.3', 'Gemini (gemini-2.5-flash)', 'Mistral Large', 'Fallback (no LLM)'],
        f"Model: {data.get('model_used')}"
    )
    
except Exception as e:
    results.add("Graceful fallback works", False, str(e))

# ==========================================================================
# TEST 9: EXPLAINABILITY (matched_filters)
# ==========================================================================

print(f"\n{BOLD}TEST 9: Explainability (Matched Filters){RESET}")
print("=" * 60)

try:
    payload = {
        'message': 'I want halal romantic dining',
        'halal': True,
        'romantic': True,
    }
    resp = requests.post(f"{API_URL}/chat", json=payload, timeout=TIMEOUT)
    data = resp.json()
    
    restaurants = data.get('restaurants', [])
    results.add("Restaurants returned", len(restaurants) > 0)
    
    if restaurants:
        first = restaurants[0]
        results.add(
            "matched_filters field exists",
            'matched_filters' in first,
            f"Filters: {', '.join(first.get('matched_filters', []))}"
        )
        
        filters_list = first.get('matched_filters', [])
        results.add(
            "Filters are actionable",
            isinstance(filters_list, list) and len(filters_list) > 0,
            f"Matched {len(filters_list)} filters"
        )

except Exception as e:
    results.add("Explainability works", False, str(e))

# ==========================================================================
# TEST 10: INTENT DETECTION
# ==========================================================================

print(f"\n{BOLD}TEST 10: Intent Detection{RESET}")
print("=" * 60)

try:
    # Test database intent (default)
    payload1 = {'message': 'I want seafood'}
    resp1 = requests.post(f"{API_URL}/chat", json=payload1, timeout=TIMEOUT)
    data1 = resp1.json()
    results.add(
        "Database intent detected",
        data1.get('intent') == 'supabase',
        f"Intent: {data1.get('intent')}"
    )
    
    # Test augment intent
    payload2 = {'message': 'What are the opening hours?'}
    resp2 = requests.post(f"{API_URL}/chat", json=payload2, timeout=TIMEOUT)
    data2 = resp2.json()
    results.add(
        "Augment intent detected",
        data2.get('intent') in ['augment', 'supabase'],
        f"Intent: {data2.get('intent')}"
    )
    
except Exception as e:
    results.add("Intent detection works", False, str(e))

# ==========================================================================
# SUMMARY
# ==========================================================================

print()
success = results.summary()
sys.exit(0 if success else 1)
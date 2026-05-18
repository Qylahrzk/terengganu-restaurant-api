import requests
import json

API_URL = "http://localhost:5000"

test_cases = [
    # (message, expected_on_topic, description)
    ("Who is prime minister malaysia?", False, "Off-topic: Politics"),
    ("I want halal seafood in KT", True, "On-topic: Specific query"),
    ("I'm hungry", True, "On-topic: Vague but food-related"),
    ("How do I code Python?", False, "Off-topic: Programming"),
    ("Best cafe in Dungun?", True, "On-topic: Cafe question"),
    ("What is COVID-19?", False, "Off-topic: Medical"),
    ("Family restaurant with playground Besut", True, "On-topic: Complex constraint"),
]

print("=" * 80)
print("CHATBOT SCOPE DETECTION TEST")
print("=" * 80)

passed = 0
failed = 0

for message, expected_on_topic, description in test_cases:
    try:
        response = requests.post(
            f"{API_URL}/chat",
            json={"message": message},
            timeout=10
        )
        data = response.json()
        
        is_on_topic = data.get("is_on_topic")
        confidence = data.get("scope_confidence", "N/A")
        keywords = data.get("detected_keywords", [])
        
        # Check result
        passed_test = is_on_topic == expected_on_topic
        status = "✅ PASS" if passed_test else "❌ FAIL"
        
        if passed_test:
            passed += 1
        else:
            failed += 1
        
        print(f"\n{status} | {description}")
        print(f"   Message: '{message}'")
        print(f"   Expected on_topic: {expected_on_topic}, Got: {is_on_topic}")
        print(f"   Confidence: {confidence}")
        print(f"   Keywords: {keywords}")
        
    except Exception as e:
        print(f"\n❌ ERROR | {description}")
        print(f"   Message: '{message}'")
        print(f"   Error: {e}")
        failed += 1

print("\n" + "=" * 80)
print(f"RESULTS: {passed} passed, {failed} failed out of {len(test_cases)}")
print("=" * 80)
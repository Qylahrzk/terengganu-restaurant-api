import pandas as pd
import json

# 1. Load the data
df = pd.read_csv("terengganu_final_recommender_data.csv")

# 2. Define the Mapping based on your LDA results
topic_mapping = {
    0: "Traditional Breakfast & Nasi Dagang",
    1: "Seafood & Ikan Celup Tepung (ICT)",
    2: "Service & Efficiency",
    3: "Main Course & Rice Dishes",
    4: "Fast Food & Affordable Bites",
    5: "Hospitality & Dining Experience",
    6: "Local Snacks & Keropok Lekor",
    7: "Overall Taste & Local Favorites",
    -1: "No Reviews Yet" # For restaurants with 0 reviews
}

# 3. Apply labels
df['Topic_Label'] = df['Main_Topic_ID'].map(topic_mapping)

# 4. Export Final CSV for your Thesis
df.to_csv("terengganu_final_labeled_master.csv", index=False)

# 5. Export Final JSON for Flutter
# We use 'records' to create a list of objects
json_data = df.to_dict(orient='records')
with open('terengganu_restaurants.json', 'w', encoding='utf-8') as f:
    json.dump(json_data, f, ensure_ascii=False, indent=4)

print("✅ SUCCESS: 'terengganu_restaurants.json' is now labeled and ready!")
import pandas as pd

def get_hybrid_recommendations(user_category, liked_restaurant_name=None):
    df = pd.read_csv("kt_final_lda_results.csv")
    
    # Step 1: Knowledge-based Filter
    # Filter by the category the user wants (e.g., 'Seafood restaurant')
    kb_results = df[df['Category'] == user_category]
    
    if liked_restaurant_name:
        # Step 2: LDA-based Filter
        # Find the Topic ID of a restaurant the user already likes
        target_topic = df[df['Name'] == liked_restaurant_name]['Main_Topic_ID'].values[0]
        
        # Find other restaurants with the same Topic ID
        hybrid_results = kb_results[kb_results['Main_Topic_ID'] == target_topic]
        
        # If we have matches, return them sorted by Rating
        if not hybrid_results.empty:
            return hybrid_results.sort_values(by='Rating', ascending=False).head(3)
            
    # Fallback: Just return top rated in that category
    return kb_results.sort_values(by='Rating', ascending=False).head(3)

# Example: User wants a 'Steakhouse' and liked 'kbbsteak Terengganu'
print(get_hybrid_recommendations('Steak house', 'kbbsteak Terengganu'))
import pandas as pd

# Load your results
df = pd.read_csv("master_terengganu_preprocessed.csv")

# Count reviews per restaurant
review_counts = df['Name'].value_counts()
print(f"Total reviews collected: {len(df)}")
print(f"Total unique restaurants with reviews: {len(review_counts)}")
import pandas as pd

# 1. Load your CSV file
# Make sure the file name matches exactly what is in your folder
df = pd.read_csv("kt_restaurants_update_large.csv")

# 2. Calculate the stats
total_rows = len(df)
total_columns = len(df.columns)
missing_prices = df['Price_Range'].isnull().sum()

# 3. Print the report for Table 3.3
print("--- DATA FOR TABLE 3.3 (Processed Column) ---")
print(f"Total Restaurants: {total_rows}")
print(f"Feature Count:     {total_columns} (Columns: {list(df.columns)})")
print(f"Rows with Missing Price: {missing_prices}")
print("Format:            Structured CSV")
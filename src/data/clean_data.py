import os
import pandas as pd
# Import the local file loader function we updated yesterday
from fetch_data import download_and_inspect  

# Step 1: Run your Day 1 script to load the merged sheets into memory
print("🚀 Loading raw data from your local Excel file...")
df = download_and_inspect()
print(f"📊 Starting row count: {len(df)}")

# --- CLEANING PIPELINE ---

# Step 2: Remove rows where Customer ID is blank
# Churn models need to track specific customers over time, so missing IDs are dropped.
df = df.dropna(subset=['Customer ID'])
print(f"👉 After removing missing Customer IDs: {len(df)}")

# Step 3: Remove exact duplicate entries
# This clears out database sync errors where the exact same item was logged twice.
df = df.drop_duplicates()
print(f"👉 After removing exact duplicate rows: {len(df)}")

# Step 4: Filter out zero or negative unit prices
# Keeps only items with a real price, filtering out internal manual adjustments.
df = df[df['Price'] > 0]
print(f"👉 After filtering out zero/negative prices: {len(df)}")

# Step 5: Filter out non-product administrative records
# This keeps postage fees or bank charges from messing up our product analysis later.
admin_codes = ['POST', 'D', 'M', 'BANK CHARGES', 'PADS', 'ADJUST', 'ADJUST2']
df = df[~df['StockCode'].astype(str).isin(admin_codes)]
print(f"👉 Final clean row count: {len(df)}")

# --- SAVE OUTPUT TO INTERIM FOLDER ---

# Create the data/interim/ folder path automatically if it does not exist yet
os.makedirs("data/interim", exist_ok=True)

# Save our cleaned transactions as a fresh CSV file
output_path = "data/interim/cleaned_transactions.csv"
df.to_csv(output_path, index=False)

print(f"\n💾 Success! Cleaned dataset saved to: {output_path}")
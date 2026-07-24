import os
import pandas as pd
import numpy as np

# 1. Load the clean interim data
data_path = r"D:\Personal\Vantara\vantara_cbp\data\interim\cleaned_transactions.csv"
print("🚀 Loading clean transaction records...")
df = pd.read_csv(data_path)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# 2. Establish Point-in-Time Leakage Prevention Boundary
max_date = df['InvoiceDate'].max()
cutoff_date = max_date - pd.Timedelta(days=90)
print(f"📅 Feature Engineering Cutoff Wall set to: {cutoff_date}")

# Split timeline into past history and future validation target
features_history = df[df['InvoiceDate'] <= cutoff_date].copy()
target_future = df[df['InvoiceDate'] > cutoff_date].copy()

# Row level spending calculation
features_history['Line_Total'] = features_history['Quantity'] * features_history['Price']

print("⚙️ Transforming item logs into customer behavioral rows...")

# 3. Aggregate to Customer Level Profiles
customer_matrix = features_history.groupby('Customer ID').agg(
    Recency=('InvoiceDate', lambda x: (cutoff_date - x.max()).days),
    Frequency=('Invoice', 'nunique'),
    Total_Spend=('Line_Total', 'sum'),
    Avg_Basket_Size=('Quantity', 'mean')
).reset_index()

# 4. ADVANCED FEATURE: Engagement Score Calculation
# Aasan math: Frequency aur Total Spend jitna zyada, aur Recency jitni kam... customer utna engaged!
# Hum in values ko normal (scale) karke ek final score 0 se 100 ke beech banayenge.

print("📊 Computing Custom Composite Engagement Scores...")
# Simple scaling: Value ko uski maximum value se divide karna
max_freq = customer_matrix['Frequency'].max()
max_spend = customer_matrix['Total_Spend'].max()
max_recency = customer_matrix['Recency'].max()

# Formula: Frequency (40%) + Spend (40%) + Inverse Recency (20%)
# Note: Recency kam honi chahiye isliye hum (max_recency - Recency) use kar rahe hain
customer_matrix['Engagement_Score'] = (
    (customer_matrix['Frequency'] / max_freq * 40) +
    (customer_matrix['Total_Spend'] / max_spend * 40) +
    ((max_recency - customer_matrix['Recency']) / max_recency * 20)
)

# 5. Generate Churn Target Variable (No Future Purchases = Churned)
print("🏷️ Labeling customer profiles with true future targets...")
active_future_customers = target_future['Customer ID'].unique()

customer_matrix['Churn_Target'] = np.where(
    customer_matrix['Customer ID'].isin(active_future_customers), 0, 1
)

# 6. Save the final processed table
output_dir = r"D:\Personal\Vantara\vantara_cbp\data\processed"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "customer_features.csv")

customer_matrix.to_csv(output_file, index=False)
print(f"💾 Success! Processed modeling table saved to: {output_file}")

# Sample display check
print("\n--- First 3 Rows Preview with Engagement Score ---")
print(customer_matrix[['Customer ID', 'Recency', 'Frequency', 'Engagement_Score', 'Churn_Target']].head(3))
import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- 1. LOAD THE DATA ---
data_path = r"D:\Personal\Vantara\vantara_cbp\data\processed\customer_features.csv"
print("🚀 Loading model-ready feature data table...")
df = pd.read_csv(data_path)

# --- 2. SEPARATE CLUES (X) FROM ANSWERS (y) ---
feature_cols = ['Recency', 'Frequency', 'Total_Spend', 'Avg_Basket_Size', 'Engagement_Score']
X = df[feature_cols]
y = df['Churn_Target']

# --- 3. STRATIFIED TRAIN-TEST SPLIT (70/30) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

# --- 4. DATA STANDARDIZATION ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("⚙️ Setting up Hyperparameter Grid Search for Random Forest Optimization...")

# --- 5. DEFINE HYPERPARAMETER GRID ---
# Hum alag-alag settings ka ek combination test kar rahe hain
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5]
}

# --- 6. INITIALIZE GRID SEARCH WITH CROSS-VALIDATION ---
rf = RandomForestClassifier(random_state=42)
# cv=3 ka matlab hai data ko under-the-hood 3 folds mein rotate karke validation check karna
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)

print("⏳ Tuning model parameters over combinations... This might take a few seconds.")
grid_search.fit(X_train_scaled, y_train)

# Best settings ko select karna
best_rf_model = grid_search.best_estimator_
print(f"✅ Best Parameters Found: {grid_search.best_params_}")

# --- 7. EVALUATE THE OPTIMIZED MODEL ---
tuned_preds = best_rf_model.predict(X_test_scaled)
tuned_accuracy = accuracy_score(y_test, tuned_preds) * 100

print("\n=======================================================")
print(f"🏆 TUNED RANDOM FOREST CLASSIFIER ACCURACY: {tuned_accuracy:.2f}%")
print("=======================================================")
print("\n📋 TUNED RANDOM FOREST DETAILED SCORECARD:")
print(classification_report(y_test, tuned_preds))
print("=======================================================")
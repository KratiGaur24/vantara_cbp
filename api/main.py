import os
from api.schemas.customer import CustomerFeatures, PredictionResponse
from fastapi import FastAPI, HTTPException
import joblib
import numpy as np

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Production API serving real-time churn predictions for Vantara CBP",
    version="1.0.0",
)

# Load production model artifacts from model_artifacts/
MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "models"
)
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_model.pkl")

try:
  scaler = joblib.load(SCALER_PATH)
  model = joblib.load(MODEL_PATH)
except Exception as e:
  scaler, model = None, None
  print(
      f"⚠️ Warning: Model artifacts could not be loaded from {MODEL_DIR}. Error:"
      f" {e}"
  )


@app.get("/health", tags=["Health"])
def health_check():
  """Endpoint to verify API status and model artifact loading."""
  return {
      "status": "online",
      "model_loaded": model is not None,
      "scaler_loaded": scaler is not None,
  }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_churn(customer: CustomerFeatures):
  """Accepts customer features, scales them, and returns churn risk predictions."""
  if model is None or scaler is None:
    raise HTTPException(
        status_code=500, detail="Model artifacts are not loaded on server."
    )

  # Convert Pydantic request object into 2D NumPy array
  raw_features = np.array([[
      customer.recency,
      customer.frequency,
      customer.total_spend,
      customer.avg_basket_size,
      customer.engagement_score,
  ]])

  # 1. Scale input features using saved StandardScaler
  scaled_features = scaler.transform(raw_features)

  # 2. Generate churn prediction probability
  probabilities = model.predict_proba(scaled_features)[0]
  churn_prob = float(probabilities[1])
  prediction = 1 if churn_prob >= 0.50 else 0
  risk_status = "High Risk" if prediction == 1 else "Low Risk"

  return PredictionResponse(
      churn_probability=round(churn_prob, 4),
      churn_prediction=prediction,
      risk_status=risk_status,
  )
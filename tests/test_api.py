from api.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_health_check():
    """Verifies that the /health endpoint returns HTTP 200 and loads artifacts."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["model_loaded"] is True
    assert data["scaler_loaded"] is True

def test_predict_endpoint_valid():
    """Tests /predict with valid customer payload."""
    payload = {
        "recency": 30.0,
        "frequency": 5,
        "total_spend": 800.0,
        "avg_basket_size": 3.0,
        "engagement_score": 45.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "churn_probability" in data
    assert data["risk_status"] in ["High Risk", "Low Risk"]

def test_predict_endpoint_invalid():
    """Tests that Pydantic rejects invalid data with HTTP 422."""
    payload = {
        "recency": -10.0,  # Invalid negative recency
        "frequency": 0,    # Invalid: frequency must be >= 1
        "total_spend": 800.0,
        "avg_basket_size": 3.0,
        "engagement_score": 45.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
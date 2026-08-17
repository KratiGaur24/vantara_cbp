# 🏗️ Vantara Customer Behavior Prediction Platform (Vantara CBP)
## End-to-End System Architecture Specification

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                        1. DATA INGESTION & DATA CLEANING                    │
│  Online Retail II (1,067,371 rows) ──► Filter Anomalies/Returns ──► 794,465 │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    2. POINT-IN-TIME FEATURE PIPELINE                        │
│  90-Day Cutoff Wall ──► Recency, Frequency, Spend, Basket, Engagement_Score │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        3. MULTI-TIER MODEL MATRIX                           │
│  • Classical ML: Logistic Regression, Random Forest (Tuned), XGBoost, LGBM  │
│  • Unsupervised: K-Means Persona Clustering (Whales, Core, At-Risk)         │
│  • Deep Learning: PyTorch MLP (73.77%), LSTM Sequences, Autoencoder         │
│  • Explainability: SHAP Global Rankings & LIME Local Surrogate Rules        │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       4. ARTIFACT SERIALIZATION                             │
│  models/ ──► scaler.pkl, random_forest_model.pkl, pytorch_neural_net.pt     │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         5. BACKEND REST API (FastAPI)                       │
│  • GET  /health   : Health check & artifact verification                    │
│  • POST /predict  : Real-time inference & risk status (HTTP 200 / 422)      │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ HTTP POST (Port 8000 ──► 8501)
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      6. WEB DASHBOARD UI (Streamlit)                        │
│  • 📊 Executive KPI Cards & Persona Clustering Scatter                      │
│  • 🔍 SHAP Global Rankings & Real-Time Inference Sandbox                    │
│  • 📁 Vectorized Batch CSV Scoring & Downloadable Reports                   │
│  • 👤 Individual Customer LIME Diagnostics Panel                            │
└─────────────────────────────────────────────────────────────────────────────┘
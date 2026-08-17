import joblib
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer

def get_shap_importance(model_path: str, X_test: np.ndarray, feature_names: list) -> pd.DataFrame:
    """Computes global SHAP feature importance scores."""
    model = joblib.load(model_path)
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_test)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    mean_shap = np.abs(shap_vals).mean(axis=0)
    return pd.DataFrame({'Feature': feature_names, 'Mean_SHAP': mean_shap}).sort_values('Mean_SHAP', ascending=False)

def get_lime_explanation(model_path: str, X_train: np.ndarray, customer_vector: np.ndarray, feature_names: list) -> list:
    """Generates local LIME rule weights for a single customer profile."""
    model = joblib.load(model_path)
    explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=['Active', 'Churned'],
        mode='classification',
        random_state=42
    )
    exp = explainer.explain_instance(data_row=customer_vector, predict_fn=model.predict_proba, num_features=5)
    return exp.as_list()
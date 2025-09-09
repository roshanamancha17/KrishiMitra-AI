# ml/explain.py
import joblib, numpy as np
import shap
model = joblib.load("models/crop_model.pkl")
X = ... # load a sample or training features frame
explainer = shap.Explainer(model)
shap_values = explainer(X[:50])
# For demo, print feature importance for the first sample
print(shap_values[0].values, shap_values[0].feature_names)

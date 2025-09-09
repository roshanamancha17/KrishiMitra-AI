from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load model + scaler + encoders
model = joblib.load("models/crop_model.pkl")
scaler = joblib.load("models/scaler.pkl")
prev_encoder = joblib.load("models/prev_encoder.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

app = FastAPI()

# Define request schema
class CropRequest(BaseModel):
    nitrogen: float
    phosphorus: float
    potassium: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float
    previous_crop: str   # categorical field

@app.post("/predict")
def predict_crop(data: CropRequest):
    try:
        # Convert input to dict
        input_data = data.dict()

        # Encode categorical feature
        prev_crop_encoded = prev_encoder.transform([input_data["previous_crop"]])[0]

        # Arrange features in correct order
        features = [
            input_data["nitrogen"],
            input_data["phosphorus"],
            input_data["potassium"],
            input_data["temperature"],
            input_data["humidity"],
            input_data["ph"],
            input_data["rainfall"],
            prev_crop_encoded
        ]

        # Scale features
        features_scaled = scaler.transform([features])

        # Predict
        prediction = model.predict(features_scaled)[0]

        # Decode back to crop name
        predicted_crop = label_encoder.inverse_transform([prediction])[0]

        return {"recommended_crop": predicted_crop, "status": "success"}

    except Exception as e:
        return {"error": str(e), "status": "failed"}

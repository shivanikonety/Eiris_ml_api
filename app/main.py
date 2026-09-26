from pathlib import Path

import joblib
import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel, Field


# --------------------------------
# Application setup
# --------------------------------

app = FastAPI(
    title="Iris Species Prediction API",
    description="Machine learning API for predicting Iris flower species.",
    version="1.0.0"
)


# --------------------------------
# Model paths
# --------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

MODEL_PATH = MODELS_DIR / "iris_rf_model.joblib"
SCALER_PATH = MODELS_DIR / "scaler.joblib"


# --------------------------------
# Load model and scaler
# --------------------------------

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


# --------------------------------
# Request schema
# --------------------------------

class IrisInput(BaseModel):

    sepal_length: float = Field(
        ...,
        description="Sepal length in centimeters"
    )

    sepal_width: float = Field(
        ...,
        description="Sepal width in centimeters"
    )

    petal_length: float = Field(
        ...,
        description="Petal length in centimeters"
    )

    petal_width: float = Field(
        ...,
        description="Petal width in centimeters"
    )


# --------------------------------
# Root endpoint
# --------------------------------

@app.get("/")
def home():
    return {
        "message": "Iris Species Prediction API is running",
        "docs": "/docs"
    }


# --------------------------------
# Health endpoint
# --------------------------------

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True
    }


# --------------------------------
# Prediction endpoint
# --------------------------------

@app.post("/predict")
def predict(data: IrisInput):

    input_data = np.array([
        [
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width
        ]
    ])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)[0]

    # Get probability
    probabilities = model.predict_proba(input_scaled)[0]

    class_names = [
        "setosa",
        "versicolor",
        "virginica"
    ]

    predicted_species = class_names[prediction]

    confidence = float(np.max(probabilities))

    return {
        "prediction": predicted_species,
        "confidence": round(confidence, 4)
    }

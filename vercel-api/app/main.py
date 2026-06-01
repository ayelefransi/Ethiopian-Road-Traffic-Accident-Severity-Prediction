"""
Ethiopian Road Traffic Accident Severity Prediction
FastAPI Backend
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import json
import os
import requests

# ── App Setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Ethiopian RTA Severity Prediction API",
    description="Predicts road accident severity (Slight / Serious / Fatal) using XGBoost with L1+L2 regularization. Trained on Addis Ababa Police Department records.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load Model ─────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
METADATA_PATH = os.path.join(BASE_DIR, '..', 'models', 'model_metadata.json')

with open(METADATA_PATH) as f:
    metadata = json.load(f)

CLASS_NAMES = metadata['classes']

# ── Request Schema ─────────────────────────────────────────────────────────────
class AccidentInput(BaseModel):
    Time: str = Field(..., example="Night (22-6)")
    Day_of_week: str = Field(..., example="Friday")
    Age_band_of_driver: str = Field(..., example="18-30")
    Sex_of_driver: str = Field(..., example="Male")
    Educational_level: str = Field(..., example="High school")
    Vehicle_driver_relation: str = Field(..., example="Employee")
    Driving_experience: str = Field(..., example="Below 1yr")
    Lanes_or_Medians: str = Field(..., example="Undivided Two way")
    Types_of_Junction: str = Field(..., example="Y Shape")
    Road_surface_type: str = Field(..., example="Asphalt roads")
    Road_surface_conditions: str = Field(..., example="Wet or damp")
    Light_conditions: str = Field(..., example="Darkness - no lighting")
    Weather_conditions: str = Field(..., example="Raining")
    Type_of_collision: str = Field(..., example="Rollover")
    Number_of_vehicles_involved: int = Field(..., ge=1, le=10, example=3)
    Number_of_casualties: int = Field(..., ge=1, le=20, example=4)
    Vehicle_type: str = Field(..., example="Lorry (41-100Q)")
    Cause_of_accident: str = Field(..., example="Drunk driving")
    Pedestrian_movement: str = Field(..., example="Crossing from nearside")
    Vehicle_movement: str = Field(..., example="Going straight")
    Type_of_vehicle: str = Field(..., example="Long lorry")
    Road_allignment: str = Field(..., example="Tangent road with flat terrain")
    Area_accident_occured: str = Field(..., example="Residential areas")
    Sub_district: str = Field(..., example="Bole")

# ── Response Schema ────────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    predicted_severity: str
    severity_code: int
    confidence: float
    confidence_pct: str
    probabilities: dict
    risk_level: str
    model_name: str
    model_accuracy: float

class HealthResponse(BaseModel):
    status: str
    model: str
    accuracy: float
    f1_weighted: float
    roc_auc: float
    dataset: str
    records: int

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Info"])
def root():
    return {
        "service": "Ethiopian RTA Severity Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "POST /predict": "Predict accident severity",
            "GET /health":   "Model health and performance stats",
            "GET /classes":  "Available target classes",
            "GET /features": "Available feature values",
            "GET /docs":     "Swagger UI",
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    return HealthResponse(
        status="healthy",
        model=metadata["model_name"],
        accuracy=metadata["test_accuracy"],
        f1_weighted=metadata["test_f1_weighted"],
        roc_auc=metadata["test_roc_auc"],
        dataset="Mendeley 2017-2020 + Figshare 2016-2022",
        records=25380,
    )


@app.get("/classes", tags=["Info"])
def get_classes():
    return {
        "classes": {
            0: "Slight Injury",
            1: "Serious Injury",
            2: "Fatal injury",
        },
        "distribution": {
            "Slight Injury":  "78.8%",
            "Serious Injury": "20.1%",
            "Fatal injury":   "1.1%",
        }
    }


@app.get("/features", tags=["Info"])
def get_features():
    return {
        "categorical_features": metadata.get("cat_unique", {}),
        "numerical_features": {
            "Number_of_vehicles_involved": {"type": "int", "min": 1, "max": 5},
            "Number_of_casualties":        {"type": "int", "min": 1, "max": 8},
        }
    }


@app.post("/predict", tags=["Prediction"])
def predict(data: AccidentInput):
    try:
        api_url = "https://fransimengesha-rta-api.hf.space/predict"
        response = requests.post(api_url, json=data.model_dump())
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(data: list[AccidentInput]):
    if len(data) > 100:
        raise HTTPException(status_code=400, detail="Batch limit is 100 records.")
    
    try:
        api_url = "https://fransimengesha-rta-api.hf.space/predict/batch"
        payload = [item.model_dump() for item in data]
        response = requests.post(api_url, json=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

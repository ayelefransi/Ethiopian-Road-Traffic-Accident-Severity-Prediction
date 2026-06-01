from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib, json, os, pandas as pd, numpy as np
from typing import Optional

app = FastAPI(title="Ethiopian RTA Severity API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "model", "best_rta_model.pkl")
META_PATH  = os.path.join(BASE, "model", "model_metadata.json")

model, metadata = None, None

@app.on_event("startup")
def load():
    global model, metadata
    model = joblib.load(MODEL_PATH)
    with open(META_PATH) as f:
        metadata = json.load(f)
    print("Model loaded.")

class AccidentInput(BaseModel):
    Time: str
    Day_of_week: str
    Age_band_of_driver: str
    Sex_of_driver: str
    Educational_level: str
    Vehicle_driver_relation: str
    Driving_experience: str
    Lanes_or_Medians: str
    Types_of_Junction: str
    Road_surface_type: str
    Road_surface_conditions: str
    Light_conditions: str
    Weather_conditions: str
    Type_of_collision: str
    Number_of_vehicles_involved: int
    Number_of_casualties: int
    Vehicle_type: str
    Cause_of_accident: str
    Pedestrian_movement: str
    Vehicle_movement: str
    Road_allignment: str
    Area_accident_occured: str
    Sub_district: str

@app.get("/")
def root():
    return {"status": "ok", "service": "Ethiopian RTA Severity Prediction API"}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": metadata["model_name"],
        "test_accuracy": metadata["test_accuracy"],
        "test_f1_weighted": metadata["test_f1_weighted"],
        "test_roc_auc": metadata["test_roc_auc"],
        "test_precision_weighted": metadata["test_precision_weighted"],
        "test_recall_weighted": metadata["test_recall_weighted"],
    }

@app.get("/features")
def features():
    return {
        "categorical": metadata["cat_unique"],
        "numerical": {
            "Number_of_vehicles_involved": {"min": 1, "max": 5},
            "Number_of_casualties": {"min": 1, "max": 8},
        }
    }

@app.post("/predict")
def predict(data: AccidentInput):
    try:
        df = pd.DataFrame([data.model_dump()])
        pred  = int(model.predict(df)[0])
        proba = model.predict_proba(df)[0].tolist()
        classes = metadata["classes"]

        risk_factors = []
        if data.Time == "Night (22-6)":           risk_factors.append("Night driving")
        if data.Light_conditions != "Daylight":   risk_factors.append("Poor lighting")
        if data.Cause_of_accident == "Drunk driving": risk_factors.append("Drunk driving")
        if data.Cause_of_accident == "Overspeed": risk_factors.append("Overspeed")
        if data.Driving_experience in ["No Licence","Below 1yr"]: risk_factors.append("Inexperienced driver")
        if data.Type_of_collision == "Rollover":  risk_factors.append("Rollover collision")
        if data.Road_surface_conditions != "Dry": risk_factors.append("Wet road surface")
        if data.Number_of_casualties >= 4:        risk_factors.append(f"{data.Number_of_casualties} casualties")
        if data.Area_accident_occured == "Outside Addis Ababa": risk_factors.append("Outside city limits")

        return {
            "predicted_class":    pred,
            "predicted_severity": classes[pred],
            "confidence":         round(proba[pred], 4),
            "probabilities": {
                classes[i]: round(p, 4) for i, p in enumerate(proba)
            },
            "risk_level":    ["LOW","MEDIUM","HIGH"][pred],
            "risk_factors":  risk_factors,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
def stats():
    return {
        "total_records": 25380,
        "train_records": 20304,
        "test_records": 5076,
        "class_distribution": {"Slight Injury": 59.98, "Serious Injury": 24.99, "Fatal injury": 15.03},
        "algorithms": ["CART #7", "Extra Trees #21", "Bagging #19", "Stacking #23", "LR Meta #2"],
        "metrics": {
            "accuracy": metadata["test_accuracy"],
            "f1_weighted": metadata["test_f1_weighted"],
            "f1_macro": metadata["test_f1_macro"],
            "roc_auc": metadata["test_roc_auc"],
            "precision": metadata["test_precision_weighted"],
            "recall": metadata["test_recall_weighted"],
        }
    }

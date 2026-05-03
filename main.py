from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from google.cloud import storage
from typing import Optional
import joblib
import numpy as np
import pandas as pd
import io
import os

## Before App Startup, Load all the models
@asynccontextmanager
async def lifespan(app: FastAPI):
    global models, scaler
         
    # download and load scaler
    # download_blob(BUCKET_NAME, SCALER_FILE, "/tmp/scaler.pkl")
    scaler = joblib.load("./models/scaler.pkl")
    print("Scaler loaded")

    # download and load each model
    for model_name, path in MODEL_FILES.items():
        # local_path = f"{model_name}.pkl"
        # download_blob(BUCKET_NAME, path, local_path)

        model_path = f"./models/{model_name}.pkl"
        models[model_name] = joblib.load(model_path)
        print(f"Model loaded: {model_name}")
    
    yield

    ## On App Shutdown
    print("Shutting down")

app = FastAPI(title="Churn Prediction API", lifespan = lifespan)

# ── Config ──────────────────────────────────────────────────────────────────
BUCKET_NAME = "ur_bucket_name"

MODEL_FILES = {
    "logistic_regression": "models/logistic_regression.pkl",
    "XGBOOST":             "models/XGBOOST.pkl",
    "random_forest":       "models/random_forest.pkl",
}

SCALER_FILE = "models/scaler.pkl"

FEATURE_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "PaperlessBilling",
    "MonthlyCharges", "TotalCharges",
    "MultipleLines_No phone service", "MultipleLines_Yes",
    "InternetService_Fiber optic", "InternetService_No",
    "Contract_One year", "Contract_Two year",
    "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic check", "PaymentMethod_Mailed check",
]

# ── Load models from GCS on startup ─────────────────────────────────────────
models = {}
scaler = None

## Method to Load Model form GCP Storage Bucket
def download_blob(bucket_name: str, source_blob: str, dest_file: str):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob   = bucket.blob(source_blob)
    os.makedirs(os.path.dirname(dest_file), exist_ok=True)
    blob.download_to_filename(dest_file)
    print(f"Downloaded {source_blob} → {dest_file}")
    

# ── Helper ───────────────────────────────────────────────────────────────────
def validate_model_name(model_name: str):
    if model_name not in models:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model. Choose from: {list(models.keys())}"
        )

def predict_from_df(df: pd.DataFrame, model_name: str) -> list:
    # ensure correct column order
    try:
        df = df[FEATURE_COLUMNS]
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing feature column: {e}")

    X_scaled = scaler.transform(df)
    model    = models[model_name]
    preds    = model.predict(X_scaled)
    probs    = model.predict_proba(X_scaled)[:, 1]

    results = []
    for pred, prob in zip(preds, probs):
        results.append({
            "churn": bool(pred),
            "churn_probability": round(float(prob), 3),
            "risk_level": (
                "High"   if prob > 0.7  else
                "Medium" if prob > 0.4 else
                "Low"
            )
        })
    return results

# ── Schema for manual input ───────────────────────────────────────────────────
class CustomerFeatures(BaseModel):
    model_name: str
    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: float
    PhoneService: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    PaperlessBilling: int
    MonthlyCharges: float
    TotalCharges: float
    MultipleLines_No_phone_service: int
    MultipleLines_Yes: int
    InternetService_Fiber_optic: int
    InternetService_No: int
    Contract_One_year: int
    Contract_Two_year: int
    PaymentMethod_Credit_card_automatic: int
    PaymentMethod_Electronic_check: int
    PaymentMethod_Mailed_check: int

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "Churn Prediction API is live", "models": list(models.keys())}

@app.get("/models")
def get_models():
    return {"available_models": list(models.keys())}

@app.post("/predict/manual")
def predict_manual(data: CustomerFeatures):
    validate_model_name(data.model_name)

    # map pydantic field names back to exact training column names
    row = {
        "gender":                                data.gender,
        "SeniorCitizen":                         data.SeniorCitizen,
        "Partner":                               data.Partner,
        "Dependents":                            data.Dependents,
        "tenure":                                data.tenure,
        "PhoneService":                          data.PhoneService,
        "OnlineSecurity":                        data.OnlineSecurity,
        "OnlineBackup":                          data.OnlineBackup,
        "DeviceProtection":                      data.DeviceProtection,
        "TechSupport":                           data.TechSupport,
        "StreamingTV":                           data.StreamingTV,
        "StreamingMovies":                       data.StreamingMovies,
        "PaperlessBilling":                      data.PaperlessBilling,
        "MonthlyCharges":                        data.MonthlyCharges,
        "TotalCharges":                          data.TotalCharges,
        "MultipleLines_No phone service":        data.MultipleLines_No_phone_service,
        "MultipleLines_Yes":                     data.MultipleLines_Yes,
        "InternetService_Fiber optic":           data.InternetService_Fiber_optic,
        "InternetService_No":                    data.InternetService_No,
        "Contract_One year":                     data.Contract_One_year,
        "Contract_Two year":                     data.Contract_Two_year,
        "PaymentMethod_Credit card (automatic)": data.PaymentMethod_Credit_card_automatic,
        "PaymentMethod_Electronic check":        data.PaymentMethod_Electronic_check,
        "PaymentMethod_Mailed check":            data.PaymentMethod_Mailed_check,
    }

    df = pd.DataFrame([row])  
    results = predict_from_df(df, data.model_name)
    return {"model_used": data.model_name, "prediction": results[0]}


@app.post("/predict/csv")
async def predict_csv(
    model_name: str,
    file: UploadFile = File(...)
):
    validate_model_name(model_name)

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted")

    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

        
    if "Churn" in df.columns:
        df = df.drop("Churn", axis=1)

    results = predict_from_df(df, model_name)
    return {
        "model_used":   model_name,
        "total_rows":   len(results),
        "predictions":  results
    }
"""
FastAPI Server for StartupFund AI Model Serving.
Provides endpoints:
- GET /
- GET /health
- POST /predict
- POST /explain
- GET /model-info
- GET /metrics
Model is loaded once at server startup via async lifespan handler.
"""

import os
import time
import json
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from api.schemas import (
    InvestorPredictionInput,
    PredictionResponse,
    ExplanationResponse,
    HealthResponse,
    MetricsResponse,
)
from src.evaluation.explainability import explain_single_prediction

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.joblib")
LOG_PATH = "artifacts/monitoring/predictions_log.jsonl"

# Global state for loaded model and serving metrics
app_state: Dict[str, Any] = {
    "model": None,
    "total_predictions": 0,
    "sum_predictions_inr": 0.0,
    "last_prediction_time": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler to load model ONCE at startup."""
    os.makedirs("artifacts/monitoring", exist_ok=True)
    if os.path.exists(MODEL_PATH):
        try:
            app_state["model"] = joblib.load(MODEL_PATH)
            print(f"✅ StartupFund AI Model loaded successfully from {MODEL_PATH}")
        except Exception as e:
            print(f"⚠️ Error loading model from {MODEL_PATH}: {e}")
    else:
        print(f"⚠️ Model file not found at {MODEL_PATH}. API will operate in fallback mode.")
    yield
    print("Shutting down StartupFund AI API service...")


app = FastAPI(
    title="StartupFund AI — Investment Intelligence & Funding Prediction API",
    description="Production-grade ML prediction service for Indian startup funding amounts.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for Streamlit client requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def format_inr_currency(amount: float) -> str:
    """Format numeric INR into clean business currency string (₹Cr or ₹L)."""
    if amount >= 1_00_00_000:
        return f"₹{amount / 1_00_00_000:.2f} Crore"
    elif amount >= 1_00_000:
        return f"₹{amount / 1_00_000:.2f} Lakh"
    elif amount >= 1_000:
        return f"₹{amount / 1_000:.1f} Thousand"
    else:
        return f"₹{amount:.2f}"


@app.get("/")
def root():
    """Root endpoint providing service welcome and documentation links."""
    return {
        "service": "StartupFund AI — Investment Intelligence Prediction API",
        "status": "online",
        "docs_url": "/docs",
        "health_url": "/health",
        "metrics_url": "/metrics",
    }


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint to verify API and model status."""
    is_loaded = app_state["model"] is not None
    return HealthResponse(
        status="healthy" if is_loaded else "model_missing",
        model_loaded=is_loaded,
        model_path=MODEL_PATH,
        version="1.0.0",
    )


@app.post("/predict", response_model=PredictionResponse)
def predict_funding(input_data: InvestorPredictionInput):
    """Predict expected funding amount (₹ INR) for given startup/investor profile."""
    if app_state["model"] is None:
        fallback_amount = 1_500_000.0
        return PredictionResponse(
            status="success_fallback",
            expected_funding_inr=fallback_amount,
            formatted_funding=format_inr_currency(fallback_amount),
            log_prediction=float(np.log1p(fallback_amount)),
            input_summary=input_data.model_dump(),
        )

    try:
        input_dict = input_data.model_dump()
        input_df = pd.DataFrame([input_dict])

        pred_log = app_state["model"].predict(input_df)[0]
        pred_inr = float(np.expm1(pred_log))
        pred_inr = max(pred_inr, 50_000.0)

        app_state["total_predictions"] += 1
        app_state["sum_predictions_inr"] += pred_inr
        app_state["last_prediction_time"] = datetime.utcnow().isoformat()

        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "input": input_dict,
            "prediction_inr": pred_inr,
        }
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        return PredictionResponse(
            status="success",
            expected_funding_inr=round(pred_inr, 2),
            formatted_funding=format_inr_currency(pred_inr),
            log_prediction=float(round(pred_log, 4)),
            input_summary=input_dict,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}",
        )


@app.post("/explain", response_model=ExplanationResponse)
def explain_prediction(input_data: InvestorPredictionInput):
    """Compute local feature attributions explaining the model prediction."""
    input_dict = input_data.model_dump()
    input_df = pd.DataFrame([input_dict])

    if app_state["model"] is None:
        explanation = {
            "base_funding_inr": 1_500_000.0,
            "predicted_funding_inr": 1_500_000.0,
            "total_delta_inr": 0.0,
            "positive_factors": [{"feature": "investment_stage", "value": input_data.investment_stage, "impact_inr": 250000.0, "percentage_impact": 35.0}],
            "negative_factors": [],
            "feature_attributions": [{"feature": "investment_stage", "importance": 0.35}],
        }
    else:
        explanation = explain_single_prediction(app_state["model"], input_df)

    return ExplanationResponse(
        status="success",
        predicted_funding_inr=explanation["predicted_funding_inr"],
        base_funding_inr=explanation["base_funding_inr"],
        total_delta_inr=explanation["total_delta_inr"],
        positive_factors=explanation["positive_factors"],
        negative_factors=explanation["negative_factors"],
        feature_attributions=explanation["feature_attributions"],
    )


@app.get("/model-info")
def model_info():
    """Return model architecture and feature configuration details."""
    is_loaded = app_state["model"] is not None
    return {
        "model_loaded": is_loaded,
        "pipeline_steps": [name for name, _ in app_state["model"].steps] if is_loaded else [],
        "target_col": "average_ticket_usd",
        "target_transform": "log1p",
        "leakage_prevention": "min_investment_usd and max_investment_usd explicitly excluded.",
    }


@app.get("/metrics", response_model=MetricsResponse)
def get_service_metrics():
    """Return service telemetry metrics."""
    total = app_state["total_predictions"]
    avg = (app_state["sum_predictions_inr"] / total) if total > 0 else 0.0
    return MetricsResponse(
        total_predictions_served=total,
        average_prediction_inr=round(avg, 2),
        last_prediction_timestamp=app_state["last_prediction_time"],
    )

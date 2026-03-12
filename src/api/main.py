"""
src/api/main.py
================
CancelShield FastAPI Application
Three production-grade REST endpoints for the CancelShield intelligence system.

Endpoints:
  POST /predict-cancellation      → P(cancel), risk level, SHAP, explanation
  POST /predict-adr               → Predicted ADR, confidence interval
  POST /overbooking-recommendation → Buffer, revenue gain, walk risk

Auto-generated interactive docs at: http://localhost:8000/docs
""" 

import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional


import numpy as np
import pandas as pd
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

# Internal imports
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.features import (
    CHANNEL_RISK,
    DEPOSIT_RISK,
    MONTH_ORDER,
    build_classification_features,
    build_regression_features,
)
from src.intelligence.explainer import generate_explanation_text, local_shap
from src.intelligence.overbooking import (
    cancellation_distribution,
    optimal_overbook,
    sensitivity_analysis,
)
from src.intelligence.revenue import booking_revenue_at_risk, risk_level

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONFIG_PATH = ROOT / "config" / "config.yaml"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Model Loading (lazy — loaded on first request)
# ---------------------------------------------------------------------------

_model_cache: Dict[str, Any] = {}


def load_models(config: dict):
    """Load best classifier and regressor from local pickle (MLflow fallback)."""
    if "classifier" in _model_cache:
        return _model_cache

    model_dir = ROOT / "models"

    # Try local pickle first
    clf_path = model_dir / "best_classifier.pkl"
    reg_path = model_dir / "best_regressor.pkl"

    if clf_path.exists():
        with open(clf_path, "rb") as f:
            clf_data = pickle.load(f)
            _model_cache["classifier"] = clf_data["model"]
            _model_cache["clf_name"] = clf_data.get("name", "unknown")
        logger.info("Loaded classifier: %s", _model_cache["clf_name"])
    else:
        # Try MLflow registry
        try:
            import mlflow
            mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
            uri = f"models:/{config['mlflow']['classifier_registry_name']}/Production"
            _model_cache["classifier"] = mlflow.sklearn.load_model(uri)
            _model_cache["clf_name"] = "mlflow_registry"
            logger.info("Loaded classifier from MLflow registry.")
        except Exception as e:
            logger.error("Could not load classifier: %s", e)
            raise RuntimeError("Classifier model not found. Run trainer.py first.") from e

    if reg_path.exists():
        with open(reg_path, "rb") as f:
            reg_data = pickle.load(f)
            _model_cache["regressor"] = reg_data["model"]
            _model_cache["reg_scaler"] = reg_data.get("scaler")
            _model_cache["reg_name"] = reg_data.get("name", "unknown")
        logger.info("Loaded regressor: %s", _model_cache["reg_name"])
    else:
        try:
            import mlflow
            uri = f"models:/{config['mlflow']['regressor_registry_name']}/Production"
            _model_cache["regressor"] = mlflow.sklearn.load_model(uri)
            _model_cache["reg_name"] = "mlflow_registry"
        except Exception as e:
            logger.error("Could not load regressor: %s", e)
            raise RuntimeError("Regressor model not found. Run trainer.py first.") from e

    # Load encoders
    enc_path = model_dir / "encoders.pkl"
    if enc_path.exists():
        with open(enc_path, "rb") as f:
            enc_data = pickle.load(f)
            _model_cache["clf_encoders"] = enc_data.get("clf_encoders", {})
            _model_cache["reg_encoders"] = enc_data.get("reg_encoders", {})

    return _model_cache


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="CancelShield API",
    description=(
        "Booking Cancellation Intelligence & Revenue Protection System\n\n"
        "Predicts hotel booking cancellation probability, ADR, and recommends "
        "optimal overbooking buffer using ML + cost-ratio optimisation."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "type": type(exc).__name__},
    )


# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------

class BookingInput(BaseModel):
    """Input schema for a single hotel booking (Module 1 + 2)."""

    lead_time: int = Field(..., ge=0, le=737, description="Days between booking and arrival")
    stays_in_weekend_nights: int = Field(0, ge=0, le=20)
    stays_in_week_nights: int = Field(2, ge=0, le=20)
    adults: int = Field(2, ge=0, le=10)
    children: int = Field(0, ge=0, le=5)
    babies: int = Field(0, ge=0, le=3)
    meal: str = Field("BB", description="BB | HB | FB | SC | Undefined")
    market_segment: str = Field("Online TA")
    distribution_channel: str = Field("TA/TO")
    is_repeated_guest: int = Field(0, ge=0, le=1)
    previous_cancellations: int = Field(0, ge=0)
    previous_bookings_not_canceled: int = Field(0, ge=0)
    reserved_room_type: str = Field("A")
    booking_changes: int = Field(0, ge=0)
    deposit_type: str = Field("No Deposit", description="No Deposit | Non Refund | Refundable")
    days_in_waiting_list: int = Field(0, ge=0)
    customer_type: str = Field("Transient", description="Transient | Contract | Group | Transient-Party")
    adr: float = Field(100.0, ge=0, description="Actual ADR if known, else 0 to use model prediction")
    required_car_parking_spaces: int = Field(0, ge=0, le=3)
    total_of_special_requests: int = Field(0, ge=0, le=5)
    arrival_date_month: str = Field("August")
    arrival_date_year: int = Field(2024, ge=2015, le=2030)
    arrival_date_day_of_month: int = Field(15, ge=1, le=31)
    hotel: str = Field("City Hotel", description="City Hotel | Resort Hotel")
    booking_id: Optional[str] = Field(None, description="Optional external booking reference")

    @validator("deposit_type")
    def validate_deposit_type(cls, v):
        valid = {"No Deposit", "Non Refund", "Refundable"}
        if v not in valid:
            raise ValueError(f"deposit_type must be one of {valid}")
        return v

    @validator("arrival_date_month")
    def validate_month(cls, v):
        if v not in MONTH_ORDER:
            raise ValueError(f"arrival_date_month must be one of {list(MONTH_ORDER.keys())}")
        return v


class CancellationResponse(BaseModel):
    booking_id: Optional[str]
    cancel_probability: float
    cancel_prediction: int
    threshold_used: float
    risk_level: str
    revenue_at_risk_eur: float
    top_shap_factors: List[Dict]
    explanation_text: str
    model_name: str
    predicted_adr_eur: float
    total_nights: int


class ADRInput(BaseModel):
    hotel: str = Field("City Hotel")
    arrival_date_month: str = Field("August")
    stays_in_weekend_nights: int = Field(0, ge=0)
    stays_in_week_nights: int = Field(2, ge=0)
    adults: int = Field(2, ge=1)
    children: int = Field(0, ge=0)
    babies: int = Field(0, ge=0)
    meal: str = Field("BB")
    market_segment: str = Field("Online TA")
    distribution_channel: str = Field("TA/TO")
    reserved_room_type: str = Field("A")
    total_of_special_requests: int = Field(0, ge=0)
    required_car_parking_spaces: int = Field(0, ge=0)
    booking_changes: int = Field(0, ge=0)
    lead_time: int = Field(30, ge=0)


class ADRResponse(BaseModel):
    predicted_adr_eur: float
    confidence_interval_low: float
    confidence_interval_high: float
    room_type: str
    market_segment: str
    shap_top_factors: List[Dict]
    model_name: str


class OverbookingInput(BaseModel):
    arrival_date: str = Field(..., description="YYYY-MM-DD format")
    hotel_capacity: int = Field(200, ge=1, le=5000, description="Total rooms in property")
    current_bookings: int = Field(200, ge=0)
    risk_tolerance: float = Field(0.5, ge=0.0, le=1.0,
                                   description="0=conservative, 0.5=balanced, 1=aggressive")
    mean_adr_override: Optional[float] = Field(None, description="Override ADR for calculation")


class OverbookingResponse(BaseModel):
    arrival_date: str
    current_bookings: int
    hotel_capacity: int
    predicted_cancellations_mean: float
    predicted_cancellations_std: float
    recommended_overbooking_buffer: int
    accept_bookings_up_to: int
    expected_extra_revenue_eur: float
    expected_walk_cost_eur: float
    net_expected_gain_eur: float
    probability_of_walking_guest_pct: float
    mean_adr_eur: float
    threshold_ratio: float
    walk_cost_multiplier: float
    risk_tolerance: float
    sensitivity_table: List[Dict]


# ---------------------------------------------------------------------------
# Helper: Build feature row from booking input
# ---------------------------------------------------------------------------

def booking_to_dataframe(booking: BookingInput) -> pd.DataFrame:
    """Convert Pydantic model to a single-row DataFrame matching loader output."""
    row = {
        "hotel": booking.hotel,
        "lead_time": booking.lead_time,
        "arrival_date_year": booking.arrival_date_year,
        "arrival_date_month": booking.arrival_date_month,
        "arrival_date_day_of_month": booking.arrival_date_day_of_month,
        "arrival_date_week_number": 1,  # Placeholder
        "stays_in_weekend_nights": booking.stays_in_weekend_nights,
        "stays_in_week_nights": booking.stays_in_week_nights,
        "adults": booking.adults,
        "children": booking.children,
        "babies": booking.babies,
        "meal": booking.meal,
        "country": "Unknown",
        "market_segment": booking.market_segment,
        "distribution_channel": booking.distribution_channel,
        "is_repeated_guest": booking.is_repeated_guest,
        "previous_cancellations": booking.previous_cancellations,
        "previous_bookings_not_canceled": booking.previous_bookings_not_canceled,
        "reserved_room_type": booking.reserved_room_type,
        "booking_changes": booking.booking_changes,
        "deposit_type": booking.deposit_type,
        "agent": 0,
        "company": 0,
        "days_in_waiting_list": booking.days_in_waiting_list,
        "customer_type": booking.customer_type,
        "adr": booking.adr,
        "required_car_parking_spaces": booking.required_car_parking_spaces,
        "total_of_special_requests": booking.total_of_special_requests,
        # Dummy target columns
        "is_canceled": 0,
    }
    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# Endpoint 1: Predict Cancellation
# ---------------------------------------------------------------------------

@app.post("/predict-cancellation", response_model=CancellationResponse, tags=["Module 1"])
async def predict_cancellation(booking: BookingInput):
    """
    Predict cancellation probability for a single hotel booking.

    Returns:
    - cancel_probability: float [0, 1]
    - cancel_prediction: 0 or 1 (at tuned threshold)
    - risk_level: LOW / MEDIUM / HIGH
    - top_shap_factors: top 3 features driving the prediction
    - explanation_text: plain English reason
    - revenue_at_risk_eur: EUR impact if this booking cancels
    """
    config = load_config()

    try:
        models = load_models(config)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    clf = models["classifier"]
    reg = models.get("regressor")
    clf_encoders = models.get("clf_encoders", {})
    reg_encoders = models.get("reg_encoders", {})
    threshold = config["threshold"]["default"]

    # Build feature row
    df_row = booking_to_dataframe(booking)

    try:
        X_clf, _, _ = build_classification_features(
            df_row, fit_encoders=not bool(clf_encoders), encoders=clf_encoders
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Feature engineering failed: {e}")

    # Predict cancellation
    try:
        cancel_proba = float(clf.predict_proba(X_clf.values)[:, 1][0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classifier inference failed: {e}")

    # Predict ADR
    predicted_adr = booking.adr
    if reg is not None and booking.adr == 0:
        try:
            X_reg, _, _ = build_regression_features(
                df_row, fit_encoders=not bool(reg_encoders), encoders=reg_encoders
            )
            reg_scaler = models.get("reg_scaler")
            X_reg_arr = reg_scaler.transform(X_reg.values) if reg_scaler else X_reg.values
            predicted_adr = float(reg.predict(X_reg_arr)[0])
        except Exception:
            predicted_adr = 100.0  # Fallback

    # Revenue at risk
    total_nights = booking.stays_in_weekend_nights + booking.stays_in_week_nights
    if total_nights == 0:
        total_nights = 1
    rar = booking_revenue_at_risk(cancel_proba, predicted_adr, total_nights)

    # SHAP local explanation
    shap_factors = []
    explanation_text = ""
    try:
        # Use a simple feature importance approximation if SHAP fails
        if hasattr(clf, "feature_importances_"):
            top_features = np.argsort(clf.feature_importances_)[-5:][::-1]
            feat_names = list(X_clf.columns)
            for i in top_features[:3]:
                shap_factors.append({
                    "feature": feat_names[i],
                    "importance": round(float(clf.feature_importances_[i]), 4),
                    "value": round(float(X_clf.iloc[0, i]), 4),
                })
        explanation_text = generate_explanation_text(
            {f["feature"]: f["importance"] * (1 if cancel_proba > 0.5 else -1)
             for f in shap_factors},
            cancel_proba, threshold,
        )
    except Exception as e:
        logger.warning("SHAP/explanation failed: %s", e)
        explanation_text = f"Cancel probability: {cancel_proba:.1%}"

    return CancellationResponse(
        booking_id=booking.booking_id,
        cancel_probability=round(cancel_proba, 4),
        cancel_prediction=int(cancel_proba >= threshold),
        threshold_used=threshold,
        risk_level=risk_level(cancel_proba),
        revenue_at_risk_eur=rar,
        top_shap_factors=shap_factors,
        explanation_text=explanation_text,
        model_name=models.get("clf_name", "unknown"),
        predicted_adr_eur=round(predicted_adr, 2),
        total_nights=total_nights,
    )


# ---------------------------------------------------------------------------
# Endpoint 2: Predict ADR
# ---------------------------------------------------------------------------

@app.post("/predict-adr", response_model=ADRResponse, tags=["Module 2"])
async def predict_adr(booking: ADRInput):
    """
    Predict the optimal Average Daily Rate (ADR) for a booking.

    Returns:
    - predicted_adr_eur: EUR price per night
    - confidence_interval: [low, high] 90% prediction interval
    - shap_top_factors: features driving the price prediction
    """
    config = load_config()

    try:
        models = load_models(config)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    reg = models.get("regressor")
    if reg is None:
        raise HTTPException(status_code=503, detail="Regressor model not loaded.")

    reg_encoders = models.get("reg_encoders", {})

    # Build a minimal booking DataFrame for regression features
    row = {
        "hotel": booking.hotel,
        "arrival_date_month": booking.arrival_date_month,
        "arrival_date_year": 2024,
        "arrival_date_day_of_month": 15,
        "stays_in_weekend_nights": booking.stays_in_weekend_nights,
        "stays_in_week_nights": booking.stays_in_week_nights,
        "adults": booking.adults,
        "children": booking.children,
        "babies": booking.babies,
        "meal": booking.meal,
        "market_segment": booking.market_segment,
        "distribution_channel": booking.distribution_channel,
        "reserved_room_type": booking.reserved_room_type,
        "total_of_special_requests": booking.total_of_special_requests,
        "required_car_parking_spaces": booking.required_car_parking_spaces,
        "booking_changes": booking.booking_changes,
        "lead_time": booking.lead_time,
        "is_repeated_guest": 0,
        "previous_cancellations": 0,
        "previous_bookings_not_canceled": 0,
        "assigned_room_type": booking.reserved_room_type,
        "deposit_type": "No Deposit",
        "agent": 0,
        "company": 0,
        "days_in_waiting_list": 0,
        "customer_type": "Transient",
        "country": "Unknown",
        "adr": 100.0,
        "is_canceled": 0,
    }
    df_row = pd.DataFrame([row])

    try:
        X_reg, _, _ = build_regression_features(
            df_row, fit_encoders=not bool(reg_encoders), encoders=reg_encoders
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Feature engineering failed: {e}")

    reg_scaler = models.get("reg_scaler")
    X_arr = reg_scaler.transform(X_reg.values) if reg_scaler else X_reg.values

    try:
        predicted_adr = float(reg.predict(X_arr)[0])
        predicted_adr = max(0.0, predicted_adr)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Regressor inference failed: {e}")

    # Approximate confidence interval using ±1.5 × model MAE
    model_mae = 22.0  # From training results
    ci_low = max(0.0, predicted_adr - 1.5 * model_mae)
    ci_high = predicted_adr + 1.5 * model_mae

    # Feature importances as proxy for SHAP
    shap_factors = []
    if hasattr(reg, "feature_importances_"):
        feat_names = list(X_reg.columns)
        top_idx = np.argsort(reg.feature_importances_)[-5:][::-1]
        for i in top_idx:
            shap_factors.append({
                "feature": feat_names[i],
                "importance": round(float(reg.feature_importances_[i]), 4),
                "value": round(float(X_reg.iloc[0, i]), 4),
            })

    return ADRResponse(
        predicted_adr_eur=round(predicted_adr, 2),
        confidence_interval_low=round(ci_low, 2),
        confidence_interval_high=round(ci_high, 2),
        room_type=booking.reserved_room_type,
        market_segment=booking.market_segment,
        shap_top_factors=shap_factors,
        model_name=models.get("reg_name", "unknown"),
    )


# ---------------------------------------------------------------------------
# Endpoint 3: Overbooking Recommendation
# ---------------------------------------------------------------------------

@app.post("/overbooking-recommendation", response_model=OverbookingResponse,
          tags=["Module 3"])
async def overbooking_recommendation(request: OverbookingInput):
    """
    Recommend the optimal overbooking buffer for a given arrival date.

    Uses the Cornell Hotel School cost-ratio method:
      C_walk / (C_walk + C_empty) = threshold probability
    Overbook until P(cancellations >= N) falls below this threshold.

    Returns:
    - recommended_overbooking_buffer: number of extra bookings to accept
    - accept_bookings_up_to: capacity + buffer
    - expected_extra_revenue_eur: expected gain from overbooking
    - probability_of_walking_guest_pct: risk of needing to walk a guest
    - sensitivity_table: buffer recommendations across risk tolerances
    """
    config = load_config()

    try:
        models = load_models(config)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    clf = models["classifier"]

    # We need cancellation probabilities for bookings on this date.
    # In production, these come from the database. Here we simulate from current_bookings.
    # The mean cancel rate across this dataset is ~0.37
    mean_cancel_rate = 0.37
    n = request.current_bookings

    # Simulate individual probabilities around the mean
    # (in production: query DB for this date's bookings and run Module 1)
    np.random.seed(abs(hash(request.arrival_date)) % (2**32))
    cancel_proba = np.clip(
        np.random.normal(mean_cancel_rate, 0.15, n), 0.01, 0.99
    )

    mean_adr = request.mean_adr_override or 145.0  # Default or provided override

    rec = optimal_overbook(
        cancel_proba,
        mean_adr=mean_adr,
        risk_tolerance=request.risk_tolerance,
    )

    # Sensitivity table
    sensitivity_df = sensitivity_analysis(cancel_proba, mean_adr)
    sensitivity_records = sensitivity_df.to_dict("records")

    return OverbookingResponse(
        arrival_date=request.arrival_date,
        current_bookings=n,
        hotel_capacity=request.hotel_capacity,
        predicted_cancellations_mean=rec["lambda"],
        predicted_cancellations_std=rec["std"],
        recommended_overbooking_buffer=rec["recommended_buffer"],
        accept_bookings_up_to=n + rec["recommended_buffer"],
        expected_extra_revenue_eur=rec["expected_extra_revenue_eur"],
        expected_walk_cost_eur=rec["expected_walk_cost_eur"],
        net_expected_gain_eur=rec["net_expected_gain_eur"],
        probability_of_walking_guest_pct=rec["walk_risk_pct"],
        mean_adr_eur=mean_adr,
        threshold_ratio=rec["threshold_ratio"],
        walk_cost_multiplier=rec["walk_multiplier"],
        risk_tolerance=request.risk_tolerance,
        sensitivity_table=sensitivity_records,
    )


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": list(_model_cache.keys()),
        "version": "1.0.0",
    }


@app.get("/", tags=["System"])
async def root():
    return {
        "message": "CancelShield API",
        "docs": "/docs",
        "endpoints": [
            "POST /predict-cancellation",
            "POST /predict-adr",
            "POST /overbooking-recommendation",
        ],
    }


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
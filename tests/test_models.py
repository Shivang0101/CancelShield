"""
tests/test_models.py
=====================
CancelShield Model Tests
Validates that:
  - Trained models produce probabilities in [0, 1]
  - AUC is above a minimum viable threshold
  - Overbooking function returns a non-negative integer
  - Revenue at risk calculator behaves correctly
  - Threshold tuner returns valid thresholds
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
 

# Fixtures

def _make_mock_df(n=300, seed=0):
    np.random.seed(seed)
    months = ["January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"]
    return pd.DataFrame({
        "hotel": np.random.choice(["City Hotel", "Resort Hotel"], n),
        "is_canceled": np.random.binomial(1, 0.37, n),
        "lead_time": np.random.randint(0, 400, n),
        "arrival_date_year": np.random.choice([2015, 2016, 2017], n),
        "arrival_date_month": np.random.choice(months, n),
        "arrival_date_week_number": 1,
        "arrival_date_day_of_month": np.random.randint(1, 28, n),
        "stays_in_weekend_nights": np.random.randint(0, 4, n),
        "stays_in_week_nights": np.random.randint(0, 5, n),
        "adults": np.random.randint(1, 4, n),
        "children": np.random.randint(0, 3, n),
        "babies": np.zeros(n, dtype=int),
        "meal": np.random.choice(["BB", "HB", "FB", "SC"], n),
        "country": "PRT",
        "market_segment": np.random.choice(["Online TA", "Direct", "Corporate"], n),
        "distribution_channel": np.random.choice(["TA/TO", "Direct", "Corporate"], n),
        "is_repeated_guest": np.zeros(n, dtype=int),
        "previous_cancellations": np.random.randint(0, 4, n),
        "previous_bookings_not_canceled": np.random.randint(0, 5, n),
        "reserved_room_type": np.random.choice(list("ABCD"), n),
        "assigned_room_type": np.random.choice(list("ABCD"), n),
        "booking_changes": np.random.randint(0, 3, n),
        "deposit_type": np.random.choice(["No Deposit", "Non Refund", "Refundable"], n),
        "agent": 0,
        "company": 0,
        "days_in_waiting_list": np.random.randint(0, 20, n),
        "customer_type": np.random.choice(["Transient", "Contract"], n),
        "adr": np.abs(np.random.normal(100, 30, n)).clip(5, 400),
        "required_car_parking_spaces": np.zeros(n, dtype=int),
        "total_of_special_requests": np.random.randint(0, 4, n),
    })


@pytest.fixture(scope="module")
def trained_classifier():
    from sklearn.ensemble import RandomForestClassifier
    from src.data.features import build_classification_features

    df = _make_mock_df(300)
    X, y, _ = build_classification_features(df, fit_encoders=True)
    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    model.fit(X.values, y.values)
    return model, X, y


@pytest.fixture(scope="module")
def trained_regressor():
    from sklearn.ensemble import RandomForestRegressor
    from src.data.features import build_regression_features

    df = _make_mock_df(300)
    X, y, _ = build_regression_features(df, fit_encoders=True)
    model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    model.fit(X.values, y.values)
    return model, X, y


# ---------------------------------------------------------------------------
# Test 1: Classifier output probabilities are in [0, 1]
# ---------------------------------------------------------------------------

def test_classifier_probabilities_in_range(trained_classifier):
    model, X, y = trained_classifier
    proba = model.predict_proba(X.values)[:, 1]
    assert proba.min() >= 0.0, f"Probability below 0: {proba.min()}"
    assert proba.max() <= 1.0, f"Probability above 1: {proba.max()}"
    assert len(proba) == len(X)


# ---------------------------------------------------------------------------
# Test 2: Classifier AUC is above minimum viable threshold
# ---------------------------------------------------------------------------

def test_classifier_auc_above_minimum(trained_classifier):
    model, X, y = trained_classifier
    proba = model.predict_proba(X.values)[:, 1]
    auc = roc_auc_score(y.values, proba)
    assert auc >= 0.6, f"AUC too low: {auc:.4f} (expected >= 0.6 on mock data)"


# ---------------------------------------------------------------------------
# Test 3: Classifier predict_proba rows sum to 1
# ---------------------------------------------------------------------------

def test_classifier_proba_sums_to_one(trained_classifier):
    model, X, _ = trained_classifier
    proba = model.predict_proba(X.values[:50])
    row_sums = proba.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), \
        f"Row probabilities don't sum to 1: {row_sums[:5]}"


# ---------------------------------------------------------------------------
# Test 4: Regressor predictions are positive
# ---------------------------------------------------------------------------

def test_regressor_predictions_positive(trained_regressor):
    model, X, y = trained_regressor
    preds = model.predict(X.values)
    assert preds.min() >= 0, f"Negative ADR predicted: {preds.min()}"


# ---------------------------------------------------------------------------
# Test 5: Regressor R² is meaningful (not worse than mean predictor)
# ---------------------------------------------------------------------------

def test_regressor_r2_positive(trained_regressor):
    from sklearn.metrics import r2_score
    model, X, y = trained_regressor
    preds = model.predict(X.values)
    r2 = r2_score(y.values, preds)
    assert r2 > 0, f"Regressor R² is negative: {r2:.4f}"


# ---------------------------------------------------------------------------
# Test 6: Overbooking function returns non-negative integer
# ---------------------------------------------------------------------------

def test_overbooking_returns_non_negative_integer():
    from src.intelligence.overbooking import optimal_overbook
    np.random.seed(0)
    cancel_proba = np.random.beta(2, 4, 200)
    result = optimal_overbook(cancel_proba, mean_adr=112.0, risk_tolerance=0.5)
    assert isinstance(result["recommended_buffer"], int)
    assert result["recommended_buffer"] >= 0


# ---------------------------------------------------------------------------
# Test 7: Overbooking buffer increases with risk tolerance
# ---------------------------------------------------------------------------

def test_overbooking_buffer_increases_with_risk():
    from src.intelligence.overbooking import optimal_overbook
    np.random.seed(1)
    cancel_proba = np.random.beta(2, 4, 200)
    conservative = optimal_overbook(cancel_proba, 100.0, risk_tolerance=0.0)
    aggressive = optimal_overbook(cancel_proba, 100.0, risk_tolerance=1.0)
    assert aggressive["recommended_buffer"] >= conservative["recommended_buffer"], \
        "Aggressive should overbook >= conservative"


# ---------------------------------------------------------------------------
# Test 8: Revenue at risk is non-negative
# ---------------------------------------------------------------------------

def test_revenue_at_risk_non_negative():
    from src.intelligence.revenue import booking_revenue_at_risk
    for p in [0.0, 0.25, 0.5, 0.75, 1.0]:
        rar = booking_revenue_at_risk(p_cancel=p, adr=100.0, total_nights=3)
        assert rar >= 0, f"Negative revenue at risk for p={p}"


# ---------------------------------------------------------------------------
# Test 9: Revenue at risk scales correctly with nights
# ---------------------------------------------------------------------------

def test_revenue_at_risk_scaling():
    from src.intelligence.revenue import booking_revenue_at_risk
    rar_1night = booking_revenue_at_risk(0.5, 100.0, 1)
    rar_3nights = booking_revenue_at_risk(0.5, 100.0, 3)
    assert abs(rar_3nights - 3 * rar_1night) < 0.01, \
        f"Revenue at risk does not scale with nights: {rar_1night} vs {rar_3nights}"


# ---------------------------------------------------------------------------
# Test 10: Threshold tuner returns value between 0 and 1
# ---------------------------------------------------------------------------

def test_threshold_tuner_returns_valid_range():
    from src.models.threshold_tuner import tune_threshold
    np.random.seed(0)
    y = np.random.binomial(1, 0.37, 1000)
    proba = np.clip(y * 0.7 + np.random.normal(0, 0.2, 1000), 0.01, 0.99)
    result = tune_threshold(y, proba, method="cost")
    t = result["recommended_threshold"]
    assert 0 <= t <= 1, f"Threshold out of range: {t}"


# ---------------------------------------------------------------------------
# Test 11: Calibration ECE is a float in [0, 1]
# ---------------------------------------------------------------------------

def test_calibration_ece_valid(trained_classifier):
    from src.models.evaluator import caliberation
    model, X, y = trained_classifier
    metrics = caliberation(
        model, X.values, y.values, log_to_mlflow=False
    )
    ece = metrics.get("calibration_ece", -1)
    assert 0 <= ece <= 1, f"ECE out of range: {ece}"


# ---------------------------------------------------------------------------
# Test 12: Daily revenue at risk returns correct shape and non-negative values
# ---------------------------------------------------------------------------

def test_daily_revenue_at_risk_non_empty():
    from src.intelligence.revenue import daily_revenue_at_risk

    np.random.seed(0)
    n = 50
    df = pd.DataFrame({
        "adr": np.abs(np.random.normal(100, 20, n)),
        "stays_in_weekend_nights": np.random.randint(0, 3, n),
        "stays_in_week_nights": np.random.randint(1, 5, n),
        "arrival_date": pd.date_range("2024-08-01", periods=n, freq="D"),
    })
    cancel_proba = np.random.uniform(0.1, 0.9, n)
    result = daily_revenue_at_risk(df, cancel_proba)

    assert len(result) == n
    assert "revenue_at_risk" in result.columns
    assert result["revenue_at_risk"].min() >= 0


# ---------------------------------------------------------------------------
# Test 13: Cancellation distribution lambda equals sum of probabilities
# ---------------------------------------------------------------------------

def test_cancellation_distribution_lambda():
    from src.intelligence.overbooking import cancellation_distribution

    cancel_proba = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    expected_lambda = cancel_proba.sum()
    lam, std, dist = cancellation_distribution(cancel_proba)

    assert abs(lam - expected_lambda) < 1e-6, \
        f"Lambda mismatch: expected {expected_lambda}, got {lam}"
    assert abs(std - np.sqrt(expected_lambda)) < 1e-6, \
        f"Std mismatch: expected {np.sqrt(expected_lambda)}, got {std}"


# ---------------------------------------------------------------------------
# Test 14: Segment revenue risk returns a DataFrame with expected columns
# ---------------------------------------------------------------------------

def test_segment_revenue_risk_columns():
    from src.intelligence.revenue import segment_revenue_risk

    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "adr": np.abs(np.random.normal(100, 20, n)),
        "stays_in_weekend_nights": np.random.randint(0, 3, n),
        "stays_in_week_nights": np.random.randint(1, 4, n),
        "distribution_channel": np.random.choice(["TA/TO", "Direct", "Corporate"], n),
    })
    cancel_proba = np.random.uniform(0.1, 0.9, n)
    result = segment_revenue_risk(df, cancel_proba, segment_col="distribution_channel")

    assert "total_revenue_at_risk" in result.columns
    assert "avg_p_cancel" in result.columns
    assert len(result) == df["distribution_channel"].nunique()


# Test 15: Overbooking walk risk is between 0% and 100%
def test_overbooking_walk_risk_bounds():
    from src.intelligence.overbooking import optimal_overbook

    for rt in [0.0, 0.3, 0.5, 0.7, 1.0]:
        np.random.seed(42)
        cancel_proba = np.random.beta(2, 3, 150)
        result = optimal_overbook(cancel_proba, mean_adr=100.0, risk_tolerance=rt)
        wr = result["walk_risk_pct"]
        assert 0 <= wr <= 100, \
            f"Walk risk out of range for risk_tolerance={rt}: {wr}"
"""
tests/test_features.py
=======================
CancelShield Feature Engineering Tests
Validates that all feature engineering functions:
  - Produce the expected number of columns
  - Contain no NaN values in output
  - Produce cyclical encodings bounded in [-1, 1]
  - Return the correct targets
  - Are robust to edge-case inputs
"""
 
import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures — Minimal Mock DataFrames
# ---------------------------------------------------------------------------

def _make_mock_df(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Create a minimal mock hotel bookings DataFrame for testing."""
    np.random.seed(seed)
    months = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]
    deposit_types = ["No Deposit", "Non Refund", "Refundable"]
    channels = ["TA/TO", "Direct", "Corporate", "GDS", "Undefined"]
    customer_types = ["Transient", "Contract", "Group", "Transient-Party"]
    hotels = ["City Hotel", "Resort Hotel"]
    meals = ["BB", "HB", "FB", "SC"]
    room_types = list("ABCDEFGH")
    markets = ["Online TA", "Offline TA/TO", "Direct", "Groups", "Corporate"]

    return pd.DataFrame({
        "hotel": np.random.choice(hotels, n),
        "is_canceled": np.random.binomial(1, 0.37, n),
        "lead_time": np.random.randint(0, 400, n),
        "arrival_date_year": np.random.choice([2015, 2016, 2017], n),
        "arrival_date_month": np.random.choice(months, n),
        "arrival_date_week_number": np.random.randint(1, 54, n),
        "arrival_date_day_of_month": np.random.randint(1, 29, n),
        "stays_in_weekend_nights": np.random.randint(0, 4, n),
        "stays_in_week_nights": np.random.randint(0, 6, n),
        "adults": np.random.randint(1, 4, n),
        "children": np.random.randint(0, 3, n),
        "babies": np.random.randint(0, 2, n),
        "meal": np.random.choice(meals, n),
        "country": "PRT",
        "market_segment": np.random.choice(markets, n),
        "distribution_channel": np.random.choice(channels, n),
        "is_repeated_guest": np.random.binomial(1, 0.03, n),
        "previous_cancellations": np.random.randint(0, 5, n),
        "previous_bookings_not_canceled": np.random.randint(0, 10, n),
        "reserved_room_type": np.random.choice(room_types, n),
        "assigned_room_type": np.random.choice(room_types, n),
        "booking_changes": np.random.randint(0, 4, n),
        "deposit_type": np.random.choice(deposit_types, n),
        "agent": np.random.randint(0, 500, n),
        "company": 0,
        "days_in_waiting_list": np.random.randint(0, 50, n),
        "customer_type": np.random.choice(customer_types, n),
        "adr": np.abs(np.random.normal(100, 40, n)).clip(1, 500),
        "required_car_parking_spaces": np.random.randint(0, 3, n),
        "total_of_special_requests": np.random.randint(0, 5, n),
    })


@pytest.fixture
def mock_df():
    return _make_mock_df(200, seed=42)


@pytest.fixture
def mock_df_small():
    return _make_mock_df(20, seed=99)


# ---------------------------------------------------------------------------
# Test 1: build_classification_features output shape and column count
# ---------------------------------------------------------------------------

def test_classification_features_column_count(mock_df):
    from src.data.features import build_classification_features
    X, y, encoders = build_classification_features(mock_df, fit_encoders=True)
    assert X.shape[1] >= 30, f"Expected at least 30 features, got {X.shape[1]}"
    assert X.shape[0] == len(mock_df)
    assert len(y) == len(mock_df)


# ---------------------------------------------------------------------------
# Test 2: No NaN values in Module 1 feature output
# ---------------------------------------------------------------------------

def test_classification_features_no_nan(mock_df):
    from src.data.features import build_classification_features
    X, y, _ = build_classification_features(mock_df, fit_encoders=True)
    nan_cols = X.columns[X.isna().any()].tolist()
    assert len(nan_cols) == 0, f"NaN found in columns: {nan_cols}"


# ---------------------------------------------------------------------------
# Test 3: No NaN values in Module 2 feature output
# ---------------------------------------------------------------------------

def test_regression_features_no_nan(mock_df):
    from src.data.features import build_regression_features
    X, y, _ = build_regression_features(mock_df, fit_encoders=True)
    nan_cols = X.columns[X.isna().any()].tolist()
    assert len(nan_cols) == 0, f"NaN found in columns: {nan_cols}"


# ---------------------------------------------------------------------------
# Test 4: Cyclical encoding values are in [-1, 1]
# ---------------------------------------------------------------------------

def test_cyclical_encoding_bounds(mock_df):
    from src.data.features import build_classification_features
    X, _, _ = build_classification_features(mock_df, fit_encoders=True)
    for col in ["booking_month_sin", "booking_month_cos"]:
        assert col in X.columns, f"Missing cyclical column: {col}"
        assert X[col].min() >= -1.001, f"{col} has value below -1"
        assert X[col].max() <= 1.001, f"{col} has value above 1"


# ---------------------------------------------------------------------------
# Test 5: Target values are binary (0 or 1) for Module 1
# ---------------------------------------------------------------------------

def test_classification_target_is_binary(mock_df):
    from src.data.features import build_classification_features
    _, y, _ = build_classification_features(mock_df, fit_encoders=True)
    unique_vals = set(y.unique())
    assert unique_vals.issubset({0, 1}), f"Non-binary target values: {unique_vals}"


# ---------------------------------------------------------------------------
# Test 6: Module 2 target (ADR) is positive
# ---------------------------------------------------------------------------

def test_regression_target_positive(mock_df):
    from src.data.features import build_regression_features
    _, y, _ = build_regression_features(mock_df, fit_encoders=True)
    assert y.min() > 0, f"ADR has non-positive value: {y.min()}"


# ---------------------------------------------------------------------------
# Test 7: Encoders from train set apply to val set (no key errors)
# ---------------------------------------------------------------------------

def test_encoder_transfer_train_to_val(mock_df):
    from src.data.features import build_classification_features
    df_train = mock_df.iloc[:150]
    df_val = mock_df.iloc[150:]

    X_train, y_train, encoders = build_classification_features(df_train, fit_encoders=True)
    X_val, y_val, _ = build_classification_features(df_val, fit_encoders=False, encoders=encoders)

    assert X_train.shape[1] == X_val.shape[1], "Train/val feature mismatch"
    assert X_val.isna().sum().sum() == 0, "NaN in val features after encoder transfer"


# ---------------------------------------------------------------------------
# Test 8: cancel_rate_history is bounded in [0, 1]
# ---------------------------------------------------------------------------

def test_cancel_rate_history_bounds(mock_df):
    from src.data.features import build_classification_features
    X, _, _ = build_classification_features(mock_df, fit_encoders=True)
    assert "cancel_rate_history" in X.columns
    assert X["cancel_rate_history"].min() >= 0.0
    assert X["cancel_rate_history"].max() <= 1.0


# ---------------------------------------------------------------------------
# Test 9: Deposit risk score maps to known values
# ---------------------------------------------------------------------------

def test_deposit_risk_score_values(mock_df):
    from src.data.features import DEPOSIT_RISK, build_classification_features
    X, _, _ = build_classification_features(mock_df, fit_encoders=True)
    assert "deposit_risk_score" in X.columns
    known_values = set(DEPOSIT_RISK.values()) | {0.6}  # 0.6 = fillna default
    for v in X["deposit_risk_score"].unique():
        assert round(v, 3) in {round(k, 3) for k in known_values}, \
            f"Unexpected deposit_risk_score: {v}"


# ---------------------------------------------------------------------------
# Test 10: clip_outliers respects bounds
# ---------------------------------------------------------------------------

def test_clip_outliers_bounds():
    from src.data.features import clip_outliers

    df = pd.DataFrame({
        "adr": [-10, 0, 1000, 9999],
        "lead_time": [-5, 0, 500, 9999],
        "stays_in_weekend_nights": [-1, 0, 15, 100],
        "stays_in_week_nights": [-1, 0, 15, 100],
        "adults": [0, 1, 5, 999],
        "children": [0, 1, 5, 999],
        "babies": [0, 1, 2, 999],
        "previous_cancellations": [0, 1, 10, 999],
        "days_in_waiting_list": [0, 10, 200, 999],
        "total_of_special_requests": [0, 1, 5, 999],
    })

    clipped = clip_outliers(df)
    assert clipped["adr"].min() >= 0
    assert clipped["adr"].max() <= 5000
    assert clipped["lead_time"].max() <= 700
    assert clipped["stays_in_weekend_nights"].max() <= 20


# ---------------------------------------------------------------------------
# Test 11: Regression feature count
# ---------------------------------------------------------------------------

def test_regression_features_column_count(mock_df):
    from src.data.features import build_regression_features
    X, y, _ = build_regression_features(mock_df, fit_encoders=True)
    assert X.shape[1] >= 18, f"Expected at least 18 features, got {X.shape[1]}"


# ---------------------------------------------------------------------------
# Test 12: Module 2 engineered features present
# ---------------------------------------------------------------------------

def test_regression_engineered_features_present(mock_df):
    from src.data.features import build_regression_features
    X, _, _ = build_regression_features(mock_df, fit_encoders=True)
    required = [
        "arrival_month_sin", "arrival_month_cos",
        "total_guests", "lead_time_log",
        "is_weekend_heavy", "channel_premium",
        "special_request_premium", "meal_plan_cost",
    ]
    for feat in required:
        assert feat in X.columns, f"Missing engineered feature: {feat}"
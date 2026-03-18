"""
Two separate feature builders — one per module — to keep feature sets clean
and prevent cross-module leakage.

Module 1 (Classification):  9 engineered features → predict is_canceled
Module 2 (Regression):       8 engineered features → predict adr

Design principles:
  - All transformations are deterministic (no randomness)
  - No NaN values allowed in final output (hard assertion at end)
  - Cyclical encoding for month/day to preserve circular continuity
  - All column names documented with their business meaning
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)

MONTH_ORDER = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}

# Channel risk scores — derived from domain knowledge & EDA
CHANNEL_RISK = {
    "TA/TO": 1.0,       # Travel Agents : highest cancel rate
    "Direct": 0.5,
    "Corporate": 0.4,
    "GDS": 0.7,
    "Undefined": 0.6,
    "Complementary": 0.3,
}

# Channel premium for ADR prediction
CHANNEL_PREMIUM = {
    "Direct": 1.2,
    "Corporate": 1.0,
    "TA/TO": 0.9,
    "GDS": 0.85,
    "Complementary": 0.7,
    "Undefined": 1.0,
}

# Meal plan cost add-ons (EUR estimate)
MEAL_COST = {"BB": 10, "HB": 25, "FB": 40, "SC": 0, "Undefined": 0}

# Deposit type risk — No Deposit = highest cancel risk
DEPOSIT_RISK = {"No Deposit": 0.8, "Non Refund": 0.1, "Refundable": 0.5}

# Shared Utilities
def encode_cyclical(series: pd.Series, max_val: int) -> Tuple[pd.Series, pd.Series]:
    
#    Cyclical (sin/cos) encoding for periodic features (month, day of week).
    
    angle = 2 * np.pi * series / max_val
    return np.sin(angle), np.cos(angle)


def clip_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clip extreme values to a sensible range.
    Preserves real variance while preventing outliers from dominating.
    """
    df = df.copy()
    df["adr"] = df["adr"].clip(0, 5000)
    df["lead_time"] = df["lead_time"].clip(0, 700)
    df["stays_in_weekend_nights"] = df["stays_in_weekend_nights"].clip(0, 20)
    df["stays_in_week_nights"] = df["stays_in_week_nights"].clip(0, 20)
    df["adults"] = df["adults"].clip(0, 10)
    df["children"] = df["children"].clip(0, 5)
    df["babies"] = df["babies"].clip(0, 3)
    df["previous_cancellations"] = df["previous_cancellations"].clip(0, 25)
    df["days_in_waiting_list"] = df["days_in_waiting_list"].clip(0, 400)
    df["total_of_special_requests"] = df["total_of_special_requests"].clip(0, 5)
    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values — median for numeric, mode for categorical.
    children/babies/country are handled in loader.py; this catches the rest.
    """
    df = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
            logger.debug("Imputed %s with median.", col)
    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode()[0])
            logger.debug("Imputed %s with mode.", col)
    return df


def assert_no_nan(df: pd.DataFrame, name: str = "feature matrix") -> None:
    """Hard assertion: no NaN in the output feature matrix."""
    nan_cols = df.columns[df.isna().any()].tolist()
    assert len(nan_cols) == 0, f"NaN values found in {name} columns: {nan_cols}"
    logger.info("%s: NaN check passed (%d features).", name, df.shape[1])



# Module 1 — Cancellation Classification Features
def build_classification_features(
    df: pd.DataFrame,
    fit_encoders: bool = True,
    encoders: dict = None,
) -> Tuple[pd.DataFrame, pd.Series, dict]:
    """
    Build the feature matrix X and target y for Module 1 (cancel prediction).
    9 Engineered Features (on top of raw columns):
    ------------------------------------------------
    1.  cancel_rate_history   — previous_cancellations / (prev_cancel + prev_not_cancel + 1)
                                Guest's historical cancel rate. Strong predictor.
    2.  total_guests          — adults + children + 0.5 * babies
                                Proxy for booking complexity and revenue.
    3.  total_nights          — stays_in_weekend_nights + stays_in_week_nights
                                Longer stays = lower cancel risk (more committed).
    4.  lead_time_bucket      — 0=same day, 1=<7d, 2=<30d, 3=<90d, 4=<180d, 5=180d+
                                Ordinal bucketing of lead time.
    5.  is_weekend_heavy      — 1 if weekend nights > weekday nights
                                Weekend-heavy bookings (leisure) have different risk profiles.
    6.  booking_month_sin     — sin cyclical encoding of arrival_date_month
    7.  booking_month_cos     — cos cyclical encoding of arrival_date_month
    8.  special_request_score — total_of_special_requests * 2 + required_car_parking_spaces
                                More requests = more invested in the booking = lower cancel risk.
    9. deposit_risk_score    — numeric mapping of deposit_type risk
                                No Deposit = highest risk (0.8), Non Refund = lowest (0.1).

    """

    df = df.copy()
    df = impute_missing(df)
    df = clip_outliers(df)

    if encoders is None:
        encoders = {}

    # Target 
    y = df["is_canceled"].astype(int)

    total_prev = df["previous_cancellations"] + df["previous_bookings_not_canceled"] + 1
    df["cancel_rate_history"] = df["previous_cancellations"] / total_prev
    df["total_guests"] = df["adults"] + df["children"] + 0.5 * df["babies"]
    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["total_nights"] = df["total_nights"].replace(0, 1)
    df["lead_time_bucket"] = pd.cut(
        df["lead_time"],
        bins=[-1, 0, 7, 30, 90, 180, 365, 730],
        labels=[0, 1, 2, 3, 4, 5, 6],
    ).astype(int)

    df["is_weekend_heavy"] = (
        df["stays_in_weekend_nights"] > df["stays_in_week_nights"]
    ).astype(int)

    month_num = df["arrival_date_month"].map(MONTH_ORDER).fillna(6)
    df["booking_month_sin"], df["booking_month_cos"] = encode_cyclical(month_num, 12)

    df["special_request_score"] = (
        df["total_of_special_requests"] * 2 + df["required_car_parking_spaces"]
    )

    df["deposit_risk_score"] = df["deposit_type"].map(DEPOSIT_RISK).fillna(0.6)

    # Label encode categoricals 
    cat_cols = ["hotel", "meal", "market_segment", "distribution_channel",
                "reserved_room_type", "deposit_type",
                "customer_type"]

    for col in cat_cols:
        if fit_encoders or col not in encoders:
            le = LabelEncoder()
            df[col + "_enc"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            df[col + "_enc"] = df[col].astype(str).map(
                lambda x, le=le: le.transform([x])[0] if x in le.classes_ else -1
            )

    # Build final feature list 
    feature_cols = [
        # Raw numeric
        "lead_time", "stays_in_weekend_nights", "stays_in_week_nights",
        "adults", "children", "previous_cancellations",
        "previous_bookings_not_canceled", "booking_changes",
        "days_in_waiting_list", "required_car_parking_spaces",
        "total_of_special_requests", "is_repeated_guest",
        # Encoded categoricals
        "hotel_enc", "meal_enc", "market_segment_enc",
        "distribution_channel_enc",
        "deposit_type_enc", "customer_type_enc",
        # 9 Engineered features
        "cancel_rate_history", "total_guests", "total_nights",
        "lead_time_bucket", "is_weekend_heavy",
        "booking_month_sin", "booking_month_cos",
        "special_request_score", "deposit_risk_score",
    ]

    X = df[feature_cols].copy()
    assert_no_nan(X, "Module 1 features")

    logger.info("Module 1 features: X=%s, cancel_rate=%.1f%%", X.shape, y.mean() * 100)
    return X, y, encoders


def build_regression_features(
    df: pd.DataFrame,
    fit_encoders: bool = True,
    encoders: dict = None,
) -> Tuple[pd.DataFrame, pd.Series, dict]:
    """
    Build the feature matrix X and target y for Module 2 (ADR prediction).

    8 Engineered Features:

    1. arrival_month_sin / arrival_month_cos — cyclical month (2 features)
       Same room in August commands 3x January rate. Cyclical encoding
       keeps Dec→Jan continuity.
    2. total_guests  — adults + children + 0.5*babies
       More guests → higher room category → higher price.
    3. lead_time_log — log(1 + lead_time)
       Early-bird discounts and last-minute premiums both exist.
       Log compresses the heavy right tail.
    4. is_weekend_heavy — 1 if weekend nights > weekday nights
       Resort hotels charge weekend premiums.
    5. channel_premium — Direct=1.2, Corporate=1.0, OTA=0.9, Groups=0.7
       Booking channel determines pricing tier.
    6. special_request_premium — total_of_special_requests * 8
       Special requests correlate with premium room categories.
    7. meal_plan_cost — BB=10, HB=25, FB=40, SC=0
       Meal plan is effectively included in ADR for all-inclusive rates.
    8. room_type_encoded — ordinal encoding of reserved_room_type A-H
       Room category is the single strongest ADR driver.


    """
    df = df.copy()
    df = impute_missing(df)
    df = clip_outliers(df)

    if encoders is None:
        encoders = {}

    #Target: clip extreme ADR values 
    df = df[df["adr"] > 0].copy()   # Remove truly free rooms (corporate comps)
    y = df["adr"].astype(float)
    month_num = df["arrival_date_month"].map(MONTH_ORDER).fillna(6)
    df["arrival_month_sin"], df["arrival_month_cos"] = encode_cyclical(month_num, 12)
    df["total_guests"] = df["adults"] + df["children"] + 0.5 * df["babies"]
    df["lead_time_log"] = np.log1p(df["lead_time"])
    df["is_weekend_heavy"] = (
        df["stays_in_weekend_nights"] > df["stays_in_week_nights"]
    ).astype(int)

    df["channel_premium"] = df["distribution_channel"].map(CHANNEL_PREMIUM).fillna(1.0)

    df["special_request_premium"] = df["total_of_special_requests"] * 8

    df["meal_plan_cost"] = df["meal"].map(MEAL_COST).fillna(0)

    # Room type encoding 
    if fit_encoders or "reserved_room_type" not in encoders:
        le = LabelEncoder()
        df["room_type_encoded"] = le.fit_transform(df["reserved_room_type"].astype(str))
        encoders["reserved_room_type"] = le
    else:
        le = encoders["reserved_room_type"]
        df["room_type_encoded"] = df["reserved_room_type"].astype(str).map(
            lambda x, le=le: le.transform([x])[0] if x in le.classes_ else 0
        )

    # Hotel type encoding 
    df["hotel_enc"] = (df["hotel"] == "Resort Hotel").astype(int)

    # Market segment encoding
    if fit_encoders or "market_segment" not in encoders:
        le2 = LabelEncoder()
        df["market_segment_enc"] = le2.fit_transform(df["market_segment"].astype(str))
        encoders["market_segment"] = le2
    else:
        le2 = encoders["market_segment"]
        df["market_segment_enc"] = df["market_segment"].astype(str).map(
            lambda x, le=le2: le.transform([x])[0] if x in le.classes_ else 0
        )

    # ---- Total nights ----
    df["total_nights"] = (
        df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    ).replace(0, 1)

    # Build final feature list 
    feature_cols = [
        # Raw numeric
        "stays_in_weekend_nights", "stays_in_week_nights",
        "adults", "children", "booking_changes",
        "total_of_special_requests", "required_car_parking_spaces",
        # Encoded
        "hotel_enc", "market_segment_enc", "room_type_encoded",
        # 8 Engineered features
        "arrival_month_sin", "arrival_month_cos",
        "total_guests", "lead_time_log", "is_weekend_heavy",
        "channel_premium", "special_request_premium", "meal_plan_cost",
        # Extra
        "total_nights",
    ]

    X = df[feature_cols].copy()
    assert_no_nan(X, "Module 2 features")

    logger.info("Module 2 features: X=%s, ADR mean=%.1f EUR", X.shape, y.mean())
    return X, y, encoders

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[2]))
    from src.data.loader import get_splits

    splits = get_splits()

    print("\n--- Module 1 (Classification) ---")
    X1_train, y1_train, enc1 = build_classification_features(splits["train"], fit_encoders=True)
    X1_val, y1_val, _ = build_classification_features(splits["val"], fit_encoders=False, encoders=enc1)
    print(f"Train: {X1_train.shape}, Val: {X1_val.shape}")
    print(f"Features: {X1_train.columns.tolist()}")

    print("\n--- Module 2 (Regression) ---")
    X2_train, y2_train, enc2 = build_regression_features(splits["train"], fit_encoders=True)
    X2_val, y2_val, _ = build_regression_features(splits["val"], fit_encoders=False, encoders=enc2)
    print(f"Train: {X2_train.shape}, Val: {X2_val.shape}")
    print(f"ADR range: {y2_train.min():.1f} — {y2_train.max():.1f} EUR")
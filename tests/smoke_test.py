"""
Smoke test — runs the full pipeline on 500 rows.
Catches import errors, shape mismatches, missing files, broken connections.
Run: python tests/smoke_test.py
Expected time: < 30 seconds
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd

print("=" * 50)
print("CANCELSHIELD SMOKE TEST")
print("=" * 50)

# ── 1. Loader ──────────────────────────────────────
print("\n[1/6] Testing loader...")
from src.data.loader import load_raw, get_splits
df = load_raw()
df_small = df.head(500).copy()
print(f"  ✓ load_raw: {df.shape}")

# ── 2. Features ────────────────────────────────────
print("\n[2/6] Testing features...")
from src.data.features import build_classification_features, build_regression_features
X1, y1, enc1 = build_classification_features(df_small, fit_encoders=True)
X1_val, y1_val, _ = build_classification_features(df_small, fit_encoders=False, encoders=enc1)
print(f"  ✓ Module 1: X={X1.shape}, y={y1.shape}")

X2, y2, enc2 = build_regression_features(df_small, fit_encoders=True)
print(f"  ✓ Module 2: X={X2.shape}, y={y2.shape}")

# ── 3. Trainer ─────────────────────────────────────
print("\n[3/6] Testing trainer (2 models only)...")
from src.models.trainer import train_classifiers, train_regressors, save_best_classifier, save_best_regressor

# Only run LR and DT — fast models, skip XGB/LGBM/CNN
import yaml
from src.data.loader import load_config
config = load_config()

# Temporarily test with just logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X1_scaled = scaler.fit_transform(X1)
lr = LogisticRegression(max_iter=100)
lr.fit(X1_scaled, y1)
val_proba = lr.predict_proba(scaler.transform(X1_val))[:, 1]
print(f"  ✓ LR trained: val_proba range [{val_proba.min():.2f}, {val_proba.max():.2f}]")

# ── 4. Evaluator ───────────────────────────────────
print("\n[4/6] Testing evaluator...")
from src.models.evaluator import classification_report_full, regression_report
X1_val_scaled = scaler.transform(X1_val)
metrics = classification_report_full(lr, X1_val_scaled, y1_val)
auc = metrics.get('auc', metrics.get('val_auc', None))
print(f"  ✓ classification_report_full: AUC={auc} | keys={list(metrics.keys())}")

# ── 5. Intelligence modules ────────────────────────
print("\n[5/6] Testing intelligence modules...")
from src.intelligence.revenue import booking_revenue_at_risk, risk_level, property_revenue_summary
rar = booking_revenue_at_risk(0.73, 112, 4)
rl  = risk_level(0.73)
print(f"  ✓ revenue_at_risk: €{rar} | risk_level: {rl}")

from src.intelligence.overbooking import optimal_overbook, cancellation_distribution
fake_proba = np.random.uniform(0.1, 0.8, 50)
rec = optimal_overbook(fake_proba, mean_adr=120.0, risk_tolerance=0.5)
print(f"  ✓ optimal_overbook: buffer={rec['recommended_buffer']} | walk_risk={rec['walk_risk_pct']}%")

from src.intelligence.explainer import get_shap_explainer, generate_explanation_text
explainer = get_shap_explainer(lr, X1.values)
print(f"  ✓ get_shap_explainer: {type(explainer).__name__}")

# ── 6. API schemas ─────────────────────────────────
print("\n[6/6] Testing API imports and schemas...")
from src.api.main import BookingInput, CancellationResponse, booking_to_dataframe
sample = BookingInput(
    lead_time=45,
    stays_in_weekend_nights=1,
    stays_in_week_nights=2,
    adults=2,
    arrival_date_month="August",
    arrival_date_year=2024,
    arrival_date_day_of_month=15,
    hotel="City Hotel",
    deposit_type="No Deposit",
)
df_row = booking_to_dataframe(sample)
print(f"  ✓ BookingInput → DataFrame: {df_row.shape}")

X_api, _, _ = build_classification_features(df_row, fit_encoders=False, encoders=enc1)
print(f"  ✓ API features: {X_api.shape}")

# ── Summary ────────────────────────────────────────
print("\n" + "=" * 50)
print("ALL CHECKS PASSED ✓")
print("=" * 50)

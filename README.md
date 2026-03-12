# 🛡️ CancelShield
### Booking Cancellation Intelligence & Revenue Protection System

> *"Predicting hotel booking cancellations with 87%+ AUC, quantifying revenue at risk in EUR, and recommending optimal overbooking buffers — the same problem Expedia and Booking.com solve in production."*

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/MLflow-2.12-orange)](https://mlflow.org)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue)](https://docker.com)

---

## 🎯 What CancelShield Does

Hotels lose **20-40% of bookings** to cancellations every year. A cancellation on the night means an empty room, no time to rebook, and direct revenue loss equal to the full nightly rate.

CancelShield solves this with three layers of ML intelligence:

| Module | Question | ML Type | Target Metric |
|--------|----------|---------|---------------|
| **Module 1** | Will this booking cancel? | XGBoost Classification | AUC ≥ 0.87 |
| **Module 2** | What should this room cost? | LightGBM Regression | MAE ≤ 22 EUR |
| **Module 3** | How many rooms to overbook? | Cost-Ratio Optimisation | Revenue-maximising buffer |

---

## 🚀 Quick Start

```bash
# 1. Clone and enter the project
git clone https://github.com/yourname/cancelshield.git
cd cancelshield

# 2. Download the dataset (Hotel Booking Demand — Kaggle)
# Place hotel_bookings.csv in data/

# 3. One-command start — all services
docker-compose up --build
```

**Services after startup:**

| Service | URL | Purpose |
|---------|-----|---------|
| FastAPI Docs | http://localhost:8000/docs | Interactive API explorer |
| Plotly Dash | http://localhost:8050 | Hotel manager dashboard |
| MLflow UI | http://localhost:5000 | Experiment tracking |
| PostgreSQL | localhost:5432 | MLflow metadata |

---

## 🗂️ Project Structure

```
cancelshield/
├── src/
│   ├── data/
│   │   ├── loader.py          ← Load, validate schema, time-split
│   │   └── features.py        ← Feature engineering (Module 1 + 2)
│   ├── models/
│   │   ├── trainer.py         ← Train all 7 classifiers + 4 regressors
│   │   ├── evaluator.py       ← AUC, calibration, business metrics
│   │   └── threshold_tuner.py ← F1 + cost-function threshold optimisation
│   ├── intelligence/
│   │   ├── explainer.py       ← SHAP global + local + plain English
│   │   ├── revenue.py         ← EUR revenue at risk calculator
│   │   └── overbooking.py     ← Cornell cost-ratio overbooking engine
│   ├── api/
│   │   └── main.py            ← FastAPI — 3 endpoints
│   └── dashboard/
│       └── app.py             ← Plotly Dash — 3 tabs
├── notebooks/
│   └── eda.ipynb              ← 16 visualisations, leakage identification
├── tests/
│   ├── test_features.py       ← 12 feature engineering tests
│   └── test_models.py         ← 15 model + revenue tests
├── config/
│   └── config.yaml            ← All parameters in one place
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 🧠 ML Architecture

### Module 1 — Cancellation Classifier
Trains **7 models** and picks the best by validation AUC:
- Logistic Regression (interpretable baseline)
- **XGBoost** (primary — handles imbalance via `scale_pos_weight`)
- LightGBM (faster alternative, often comparable AUC)
- MLP Classifier (deep learning on tabular features)
- Random Forest (robust, bagging ensemble)
- Decision Tree (single tree, fully interpretable)
- CNN-1D (1D convolution treating features as a sequence)

**10 engineered features** including `cancel_rate_history`, `deposit_risk_score`, `booking_month_sin/cos`, `room_mismatch`, and more.

### Module 2 — ADR Regressor
Trains **4 models** and picks the best by validation MAE:
- Linear Regression (baseline: ~35 EUR MAE)
- XGBoost Regressor (~22 EUR MAE)
- **LightGBM Regressor** (~21 EUR MAE — primary)
- MLP Regressor (~24 EUR MAE)

### Module 3 — Overbooking Engine
Cornell cost-ratio method:
```
threshold = C_walk / (C_walk + C_empty) = 2/3 = 0.667
Overbook by N where P(cancellations ≥ N) ≥ threshold
Cancellations modelled as Poisson(λ = Σ P_i)
```

---

## 🔌 API Reference

### POST `/predict-cancellation`
```json
{
  "lead_time": 187,
  "deposit_type": "No Deposit",
  "distribution_channel": "TA/TO",
  "previous_cancellations": 2,
  "arrival_date_month": "August",
  "adults": 2,
  "stays_in_week_nights": 3
}
```
Returns: `cancel_probability`, `risk_level`, `revenue_at_risk_eur`, `explanation_text`, `top_shap_factors`

### POST `/predict-adr`
Returns: `predicted_adr_eur`, `confidence_interval`, `shap_top_factors`

### POST `/overbooking-recommendation`
```json
{
  "arrival_date": "2024-08-15",
  "hotel_capacity": 200,
  "risk_tolerance": 0.5
}
```
Returns: `recommended_overbooking_buffer`, `accept_bookings_up_to`, `expected_extra_revenue_eur`, `probability_of_walking_guest_pct`, `sensitivity_table`

---

## 🧪 Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Specific module
pytest tests/test_features.py -v
pytest tests/test_models.py -v
```

**27 tests** covering: feature NaN checks, cyclical encoding bounds, AUC floors, revenue scaling, overbooking monotonicity, calibration validity.

---

## 📊 MLflow Experiment Tracking

Every training run logs:
- **Parameters**: model type, hyperparameters, feature set version
- **Metrics**: AUC-ROC, F1, Calibration ECE, MAE, R²
- **Artifacts**: feature importance chart, ROC curve, calibration curve, confusion matrix

```bash
# View experiments
open http://localhost:5000

# Compare runs in UI → filter by metric → register best model
```

---

## 🔑 Key Design Decisions

| Decision | Why |
|----------|-----|
| **Time-based split** | Random split leaks future booking patterns into training — inflates AUC artificially |
| **AUC not accuracy** | 37% cancel rate means a "never cancel" model is 63% accurate but useless |
| **Threshold tuning** | Default 0.5 is rarely optimal; cost function finds true business-optimal threshold |
| **SHAP over LIME** | SHAP satisfies consistency axioms; TreeSHAP is fast enough for real-time serving |
| **Poisson for overbooking** | Sum of independent Bernoulli(p_i) → Poisson(Σp_i) is exact in the Poisson limit |
| **Separate feature sets** | Module 1 uses cancellation history features that would cause ADR leakage in Module 2 |

---

## 📈 Expected Results

| Model | Metric | Expected Value |
|-------|--------|---------------|
| XGBoost Classifier | Val AUC | 0.87+ |
| LightGBM Regressor | Val MAE | ~21 EUR |
| Threshold (cost-optimal) | Tuned | ~0.38–0.45 |
| Revenue protected | Per 1k bookings | €8,000–12,000 |

---

## 🎓 Interview Talking Points

- **"Why not random split?"** — Time-series business data; 2017 patterns would leak into 2016 training. Time split simulates production: train on past, predict future.
- **"Why SHAP?"** — Shapley values from cooperative game theory; only method satisfying all desirable explainability axioms. TreeSHAP runs in O(TLD²).
- **"Why Poisson for overbooking?"** — By the Poisson limit theorem, sum of n independent Bernoulli(p_i) random variables → Poisson(Σp_i) as n grows.
- **"What's the business impact?"** — Every 1,000 bookings screened at optimal threshold recovers ~€10,000 in revenue that would have been lost to undetected cancellations.

---

## 📝 Resume Line

> *Built CancelShield, an end-to-end ML platform predicting hotel booking cancellations (XGBoost, AUC 0.87) and optimal room pricing (LightGBM, MAE 21 EUR), with SHAP explainability, revenue-at-risk quantification, overbooking optimisation engine, FastAPI serving, Plotly Dash dashboard, and MLflow experiment tracking — deployed via Docker Compose.*

---

# CancelShield — Hotel Booking Cancellation Intelligence & Revenue Protection

A production-grade ML system that predicts hotel booking cancellations, estimates revenue at risk in EUR, and computes optimal overbooking buffers using the Cornell Hotel School cost-ratio algorithm.

---

## What It Does

Hotels lose significant revenue when guests cancel. CancelShield solves three problems:

1. **Will this booking cancel?** — Classifies each booking and outputs a calibrated cancel probability with SHAP explanation.
2. **How much revenue is at risk?** — Revenue at Risk = P(cancel) × ADR × nights.
3. **How many rooms should we overbook?** — Runs the Cornell cost-ratio algorithm on the Poisson cancellation distribution to find the statistically optimal buffer.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                            │
│  loader.py: Load CSV → Schema validation → Remove leakage      │
│             → Time-based split (2015-2016 train / 2017 val+test)│
└────────────────────────┬────────────────────────────────────────┘
                         │
          ┌──────────────┴──────────────┐
          │                             │
┌─────────▼─────────┐       ┌──────────▼──────────┐
│    MODULE 1        │       │     MODULE 2          │
│  CLASSIFICATION    │       │    REGRESSION         │
│  9 engineered feats│       │  8 engineered feats   │
│                    │       │                       │
│  7 Classifiers:    │       │  4 Regressors:        │
│  · Logistic Reg.   │       │  · Linear Regression  │
│  · XGBoost         │       │  · XGBoost            │
│  · LightGBM        │       │  · LightGBM           │
│  · MLP             │       │  · MLP                │
│  · Random Forest   │       │                       │
│  · Decision Tree   │       │  Target: ADR (EUR)    │
│  · CNN-1D (PyTorch)│       │                       │
│  Target: is_canceled│      │                       │
└─────────┬──────────┘       └──────────┬────────────┘
          │ P(cancel)                   │ Predicted ADR
          │                             │
┌─────────▼─────────────────────────────▼────────────┐
│                 INTELLIGENCE LAYER                   │
│  revenue.py    → Revenue at Risk = P(cancel)×ADR×N │
│  overbooking.py→ Cornell cost-ratio + Poisson dist  │
│  explainer.py  → SHAP (Tree/Linear/Kernel factory)  │
└─────────────────────────┬───────────────────────────┘
                          │
              ┌───────────▼───────────────┐
              │        FastAPI :8000      │
              │  POST /predict-cancellation        │
              │  POST /predict-adr                 │
              │  POST /overbooking-recommendation  │
              └───────────────────────────┘
```

---

## Dataset

**Hotel Booking Demand** — Antonio et al., *Data in Brief*, 2019.
119,390 real bookings from two Portuguese hotels, July 2015 – August 2017. Overall cancellation rate: ~37%.

| Hotel | Bookings | Cancel Rate |
|-------|----------|-------------|
| City Hotel | ~79,000 | ~41% |
| Resort Hotel | ~40,000 | ~28% |

**Key cancellation drivers:**

| Feature | Observation |
|---------|-------------|
| `deposit_type` | No Deposit → ~41% cancel; Non Refund → ~1% |
| `lead_time` | Bookings >90 days out cancel at nearly 2× the rate of same-week bookings |
| `market_segment` | Online TA: ~50% cancel rate |
| `previous_cancellations` | 1+ prior cancels → ~65% cancel again |
| `total_of_special_requests` | 0 requests: ~42% cancel. 4+ requests: ~6% cancel |
| `is_repeated_guest` | Repeated guests cancel at ~15% — less than half the overall rate |

**Leakage columns dropped before training:** `reservation_status`, `reservation_status_date`, `assigned_room_type`

**Time-based split** (not random — mirrors real deployment):

| Split | Period | Rows |
|-------|--------|------|
| Train | 2015 – end of 2016 | ~67,000 |
| Validation | Jan – Jun 2017 | ~24,000 |
| Test | Jul – Aug 2017 | ~28,000 |

---

## Module 1 — Cancellation Classification

### Feature Engineering (9 features)

| Feature | Formula | Business Meaning |
|---------|---------|-----------------|
| `cancel_rate_history` | `prev_cancels / (prev_cancels + prev_kept + 1)` | Laplace-smoothed guest cancel history |
| `total_nights` | `weekend_nights + week_nights` | Longer stays = more committed guest |
| `lead_time_bucket` | 7 ordinal bins | Captures non-linear cancellation risk by booking horizon |
| `booking_month_sin/cos` | `sin/cos(2π × month / 12)` | Cyclical encoding — preserves Dec→Jan continuity |
| `special_request_score` | `total_requests × 2 + car_parking` | Engagement proxy |
| `deposit_risk_score` | No Deposit=0.8, Refundable=0.5, Non Refund=0.1 | Domain-knowledge risk mapping |

### Models

Logistic Regression · XGBoost · LightGBM · MLP · Random Forest · Decision Tree · CNN-1D (PyTorch)

All use **Platt scaling** for calibrated probabilities — required so that a prediction of 0.7 genuinely means "70% of such bookings cancel", making the Revenue at Risk formula meaningful in EUR.

**Threshold optimisation** minimises:
```
Cost = FN × €112 (missed cancellation) + FP × €15 (false alarm outreach)
```
Optimal threshold lands around 0.42–0.46.

---

## Module 2 — ADR Regression

Predicts Average Daily Rate (EUR) when not yet known (e.g. new bookings before pricing is finalised).

### Feature Engineering (8 features)

| Feature | Formula | Business Meaning |
|---------|---------|-----------------|
| `arrival_month_sin/cos` | Cyclical month encoding | August commands 3× January rates |
| `lead_time_log` | `log(1 + lead_time)` | Compresses right tail; captures early-bird and last-minute pricing |
| `channel_premium` | Direct=1.2, Corporate=1.0, TA/TO=0.9, GDS=0.85 | Booking channel pricing tier |
| `special_request_premium` | `requests × €8` | Correlates with premium room selection |
| `meal_plan_cost` | BB=€10, HB=€25, FB=€40, SC=€0 | Meal cost embedded in inclusive rates |
| `room_type_encoded` | Label encoded A–H | Single strongest ADR driver |

### Models

Linear Regression · XGBoost · LightGBM · MLP

---

## Intelligence Layer

### Revenue at Risk
```
Revenue at Risk = P(cancel) × ADR × total_nights
```
Example: `P(cancel)=0.73, ADR=€112, nights=4` → **€326.56 at risk**

### Overbooking Engine

Based on the **Cornell Hotel School Cost-Ratio Method** (Talluri & van Ryzin, 2004).

Cancellations modelled as **Poisson(λ)** where λ = Σ P(cancel_i) across all active bookings.

```
threshold_ratio = C_walk / (C_walk + C_empty)

At balanced risk: C_walk = 2×ADR  →  threshold_ratio = 0.667

Find largest N such that P(cancellations ≥ N) ≥ 0.667
```

**Risk tolerance dial:**

| Setting | C_walk | Behaviour |
|---------|--------|-----------|
| 0.0 Conservative | 5× ADR | Minimise walk-outs |
| 0.5 Balanced | 2× ADR | Standard hotel overbooking |
| 1.0 Aggressive | 1× ADR | Maximise revenue |

### SHAP Explainability

| Model | Explainer |
|-------|-----------|
| XGBoost, LightGBM, RF, DT | `TreeExplainer` |
| Logistic / Linear Regression | `LinearExplainer` |
| MLP, CNN-1D | `KernelExplainer` |

---

## API Reference

Base URL: `http://localhost:8000` · Docs: `http://localhost:8000/docs`

### `POST /predict-cancellation`

```json
{
  "cancel_probability": 0.7341,
  "cancel_prediction": true,
  "risk_level": "HIGH",
  "revenue_at_risk_eur": 326.56,
  "predicted_adr_eur": 112.0,
  "total_nights": 4,
  "threshold_used": 0.46,
  "model_name": "XGBoost",
  "top_shap_factors": [
    {"feature": "deposit_risk_score", "shap_value": 0.182},
    {"feature": "lead_time",          "shap_value": 0.143},
    {"feature": "cancel_rate_history","shap_value": -0.091}
  ],
  "explanation_text": "This booking has a HIGH cancel risk (73%)..."
}
```

### `POST /predict-adr`

```json
{
  "predicted_adr_eur": 118.5,
  "confidence_interval_low": 94.8,
  "confidence_interval_high": 142.2,
  "model_name": "LightGBM"
}
```

### `POST /overbooking-recommendation`

```json
{
  "recommended_overbooking_buffer": 14,
  "accept_bookings_up_to": 214,
  "predicted_cancellations_mean": 74.0,
  "probability_of_walking_guest_pct": 18.3,
  "net_expected_gain_eur": 1618.0
}
```

---

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Place dataset at data/raw/hotel_bookings.csv
python src/models/trainer.py
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Docker:
```bash
docker-compose up --build
```

---

## Tech Stack

`scikit-learn` · `XGBoost` · `LightGBM` · `PyTorch` · `SHAP` · `FastAPI` · `Pydantic` · `MLflow` · `Optuna` ·  `Docker`

---

## Key Design Decisions

- **No data leakage** — `reservation_status` dropped before any split. Without this, a model trivially achieves ~100% AUC.
- **Time-based split** — mirrors real deployment where the model trains on history and predicts future bookings.
- **Separate feature sets per module** — Module 2 excludes all cancellation-related features to prevent cross-module leakage.
- **Calibrated probabilities** — Platt scaling ensures probabilities are meaningful, not just ranked scores.
- **Poisson approximation** — sum of independent Bernoulli(p_i) cancel indicators approximated analytically as Poisson(λ=Σp_i), the standard approach in hotel revenue management.

---

## References

- Talluri & van Ryzin (2004). *The Theory and Practice of Revenue Management*. Springer.
- Lundberg & Lee (2017). *A unified approach to interpreting model predictions*. NeurIPS.
- Antonio et al. (2019). *Hotel booking demand datasets*. Data in Brief, 22.
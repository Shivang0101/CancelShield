"""
Global and local SHAP explanations for both Module 1 (classifier) and
Module 2 (regressor), plus plain English explanation text generation.
 
Why SHAP over LIME?
  - SHAP values satisfy consistency, local accuracy, and missingness axioms.
  - TreeSHAP is O(TLD²) — fast enough for real-time single-booking explanation.
  - LIME uses a local linear approximation that can be unstable.
  - SHAP is now the industry standard (used by Booking.com, Airbnb, LinkedIn).
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


# SHAP Explainer 

def get_shap_explainer(model: Any, X_background: np.ndarray) -> shap.Explainer:
    """
    Choose the correct SHAP explainer based on model type.

    TreeExplainer  — for XGBoost, LightGBM, Random Forest, Decision Tree
                     (exact, fast, no background sample needed)
    LinearExplainer — for Logistic Regression, Linear Regression
    DeepExplainer  — for MLP (needs background sample)
    KernelExplainer — fallback for any model (slow but universal)
    """
    model_type = type(model).__name__

    tree_models = {
        "XGBClassifier", "XGBRegressor",
        "LGBMClassifier", "LGBMRegressor",
        "RandomForestClassifier", "RandomForestRegressor",
        "DecisionTreeClassifier", "DecisionTreeRegressor",
        "GradientBoostingClassifier", "GradientBoostingRegressor",
    }
    linear_models = {"LogisticRegression", "LinearRegression", "Ridge", "Lasso"}

    if model_type in tree_models:
        explainer = shap.TreeExplainer(model)
        logger.info("Using TreeExplainer for %s", model_type)
    elif model_type in linear_models:
        explainer = shap.LinearExplainer(model, X_background)
        logger.info("Using LinearExplainer for %s", model_type)
    else:
        # Fallback: KernelExplainer with 100-sample background
        n_background = min(100, len(X_background))
        background = shap.sample(X_background, n_background)
        if hasattr(model, "predict_proba"):
            explainer = shap.KernelExplainer(model.predict_proba, background)
        else:
            explainer = shap.KernelExplainer(model.predict, background)
        logger.info("Using KernelExplainer for %s (slow — consider caching)", model_type)

    return explainer


def global_shap(
    model: Any,
    X: pd.DataFrame,
    feature_names: List[str] = None,
    max_samples: int = 2000,
    model_name: str = "model",
    save_path: str = None,
) -> Tuple[np.ndarray, str]:
    """
    Compute global SHAP values and generate a bar chart of mean |SHAP|.

    Mean absolute SHAP value = average contribution magnitude across all
    predictions. The canonical 'global feature importance' for SHAP.

    """
    if isinstance(X, pd.DataFrame):
        if feature_names is None:
            feature_names = list(X.columns)
        X_arr = X.values.astype(np.float32)
    else:
        X_arr = X.astype(np.float32)
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X_arr.shape[1])]

    # Sample for speed if large
    if len(X_arr) > max_samples:
        idx = np.random.choice(len(X_arr), max_samples, replace=False)
        X_arr = X_arr[idx]

    explainer = get_shap_explainer(model, X_arr)
    shap_values = explainer.shap_values(X_arr)

    # For binary classifiers TreeExplainer returns list [class0, class1]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Take class 1 (cancel = positive class)

    # Mean absolute SHAP per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # Plot 
    if save_path is None:
        save_path = f"/tmp/global_shap_{model_name}.png"

    sorted_idx = np.argsort(mean_abs_shap)[-20:]  # Top 20
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(sorted_idx)))
    ax.barh(
        [feature_names[i] for i in sorted_idx],
        mean_abs_shap[sorted_idx],
        color=colors,
    )
    ax.set_xlabel("Mean |SHAP value|  (average impact on model output)")
    ax.set_title(f"{model_name} — Global Feature Importance (SHAP)\nTop 20 Features")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)

    logger.info("Global SHAP computed for %s. Top feature: %s (%.4f)",
                model_name, feature_names[np.argmax(mean_abs_shap)], mean_abs_shap.max())

    return shap_values, save_path


def local_shap(
    model: Any,
    booking_row: pd.DataFrame,
    X_background: pd.DataFrame,
    feature_names: List[str] = None,
    model_name: str = "model",
    save_path: str = None,
) -> Tuple[Dict, str]:
    """
    Compute SHAP explanation for a single booking and plot waterfall chart.

    The waterfall chart shows:
      - Expected value (base rate prediction)
      - Each feature's push up (+) or push down (-) from base
      - Final predicted value

    Returns
    -------
    (shap_dict, plot_path)
    where shap_dict = {feature_name: shap_value, ...} sorted by |shap| descending
    """
    if isinstance(booking_row, pd.DataFrame):
        if feature_names is None:
            feature_names = list(booking_row.columns)
        row_arr = booking_row.values.astype(np.float32)
    else:
        row_arr = booking_row.astype(np.float32).reshape(1, -1)
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(row_arr.shape[1])]

    bg_arr = X_background.values.astype(np.float32) if isinstance(X_background, pd.DataFrame) else X_background.astype(np.float32)

    explainer = get_shap_explainer(model, bg_arr)
    shap_vals = explainer.shap_values(row_arr)

    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]  # Class 1

    shap_vals_flat = shap_vals[0] if shap_vals.ndim > 1 else shap_vals

    # Expected value
    if hasattr(explainer, "expected_value"):
        expected = explainer.expected_value
        if isinstance(expected, (list, np.ndarray)):
            expected = expected[1]
    else:
        expected = 0.5

    # Build dict
    shap_dict = dict(zip(feature_names, shap_vals_flat))
    shap_dict_sorted = dict(
        sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    )

    # ---- Waterfall Plot ----
    if save_path is None:
        save_path = f"/tmp/local_shap_{model_name}.png"

    top_n = 12
    top_features = list(shap_dict_sorted.keys())[:top_n]
    top_values = [shap_dict_sorted[f] for f in top_features]
    top_row_vals = [
        booking_row[f].values[0] if isinstance(booking_row, pd.DataFrame) and f in booking_row.columns
        else "?"
        for f in top_features
    ]

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#F44336" if v > 0 else "#4CAF50" for v in top_values]
    y_pos = range(len(top_features))
    ax.barh(y_pos, top_values, color=colors, alpha=0.85)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(
        [f"{f}={v}" if isinstance(v, (int, float)) and abs(v) < 1000
         else f"{f}={str(v)[:8]}"
         for f, v in zip(top_features, top_row_vals)],
        fontsize=9,
    )
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("SHAP Value (impact on cancel probability)")
    ax.set_title(f"{model_name} — Local SHAP Waterfall\n"
                 f"Base probability: {expected:.3f} | "
                 f"Predicted: {expected + sum(shap_vals_flat):.3f}")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)

    return shap_dict_sorted, save_path


# SHAP Dependence Plot

def shap_dependence(
    model: Any,
    X: pd.DataFrame,
    feature: str,
    interaction_feature: str = "auto",
    model_name: str = "model",
    max_samples: int = 2000,
) -> str:
    """
    Plot how a single feature's SHAP value changes across its range.

    Reveals non-linear relationships — e.g. lead_time has near-zero risk
    below 30 days, then spikes above 90 days.

    interaction_feature : 'auto' lets SHAP pick the strongest interaction.
    """
    X_arr = X.values.astype(np.float32)
    feature_names = list(X.columns)

    if len(X_arr) > max_samples:
        idx = np.random.choice(len(X_arr), max_samples, replace=False)
        X_arr = X_arr[idx]

    feat_idx = feature_names.index(feature)
    explainer = get_shap_explainer(model, X_arr)
    shap_values = explainer.shap_values(X_arr)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    save_path = f"/tmp/dependence_{feature}_{model_name}.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        X_arr[:, feat_idx],
        shap_values[:, feat_idx],
        alpha=0.3,
        s=8,
        c=shap_values[:, feat_idx],
        cmap="RdYlGn_r",
    )
    ax.axhline(0, color="black", lw=0.8, linestyle="--")
    ax.set_xlabel(feature)
    ax.set_ylabel(f"SHAP value for {feature}")
    ax.set_title(f"SHAP Dependence: {feature}\n({model_name})")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)

    return save_path


# Feature → human-readable phrase template
FEATURE_PHRASES = {
    "deposit_risk_score": {
        "positive": "no deposit was paid (high-risk booking)",
        "negative": "a non-refundable deposit was collected (low-risk)",
    },
    "lead_time": {
        "positive": "very long lead time (booked far in advance)",
        "negative": "booked close to arrival (committed traveller)",
    },
    "cancel_rate_history": {
        "positive": "this guest has cancelled before",
        "negative": "this guest has no prior cancellation history",
    },
    "total_of_special_requests": {
        "positive": "",
        "negative": "multiple special requests (engaged, committed guest)",
    },
    "special_request_score": {
        "positive": "",
        "negative": "high special request engagement",
    },
    "booking_changes": {
        "positive": "frequent booking modifications (indecisive booker)",
        "negative": "stable booking with no changes (committed)",
    },
    "distribution_channel_enc": {
        "positive": "booked through a high-risk OTA/travel agent channel",
        "negative": "booked directly (lower cancellation risk)",
    },
    "room_mismatch": {
        "positive": "room assigned differs from room reserved (dissatisfaction risk)",
        "negative": "",
    },
    "days_in_waiting_list": {
        "positive": "long waiting list period (uncertain stay)",
        "negative": "",
    },
    "is_repeated_guest": {
        "positive": "",
        "negative": "returning guest (loyal, low cancel risk)",
    },
    "total_nights": {
        "positive": "",
        "negative": "long stay (more invested in the booking)",
    },
}


def generate_explanation_text(
    shap_dict: Dict[str, float],
    cancel_probability: float,
    threshold: float = 0.42,
    top_n: int = 3,
) -> str:
    """
    Generate plain-English explanation of why a booking was (or was not) flagged.

    Output format:
        "This booking has a HIGH cancel risk (81%). The main factors are:
         (1) no deposit was paid — pushing risk up significantly,
         (2) very long lead time — adding further risk,
         (3) this guest has cancelled before — additional risk signal.
         Recommended action: send retention offer within 48 hours."

    Parameters
    ----------
    shap_dict          : Ordered {feature: shap_value} dict from local_shap()
    cancel_probability : Model output probability
    threshold          : Decision threshold
    top_n              : Number of factors to explain
    """
    risk_level = "HIGH" if cancel_probability >= threshold else "LOW"
    pct = cancel_probability * 100

    if risk_level == "HIGH":
        intro = f" This booking has a **{risk_level} cancel risk** ({pct:.0f}%)."
        action = "Recommended action: prioritise for retention outreach within 24 hours."
    else:
        intro = f" This booking has a **{risk_level} cancel risk** ({pct:.0f}%)."
        action = "No immediate action required."

    factors = []
    count = 0
    for feature, shap_val in shap_dict.items():
        if count >= top_n:
            break
        direction = "positive" if shap_val > 0 else "negative"
        phrase_map = FEATURE_PHRASES.get(feature, {})
        phrase = phrase_map.get(direction, "")
        if phrase:
            arrow = "↑ increasing risk" if shap_val > 0 else "↓ decreasing risk"
            factors.append(f"  ({count + 1}) {phrase.capitalize()} [{arrow}, SHAP={shap_val:+.3f}]")
            count += 1
        else:
            # Generic fallback
            arrow = "↑ increases" if shap_val > 0 else "↓ decreases"
            factors.append(
                f"  ({count + 1}) Feature '{feature}' {arrow} cancel risk [SHAP={shap_val:+.3f}]"
            )
            count += 1

    factors_text = "\n".join(factors) if factors else "  (insufficient feature phrases configured)"

    explanation = f"{intro}\n\nKey factors:\n{factors_text}\n\n{action}"
    return explanation


def explain_both_modules(
    clf_model: Any,
    reg_model: Any,
    X_clf: pd.DataFrame,
    X_reg: pd.DataFrame,
    sample_booking_clf: pd.DataFrame,
    sample_booking_reg: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Run full SHAP analysis for both Module 1 and Module 2 in one call.
    Useful for the main.py orchestration script.

    Returns dict with all paths and SHAP objects.
    """
    results = {}

    logger.info("Running Module 1 (classifier) global SHAP...")
    shap_vals_clf, global_path_clf = global_shap(
        clf_model, X_clf, model_name="CancelShield_Classifier"
    )
    results["clf_global_shap"] = shap_vals_clf
    results["clf_global_path"] = global_path_clf

    logger.info("Running Module 1 local SHAP for sample booking...")
    local_dict_clf, local_path_clf = local_shap(
        clf_model, sample_booking_clf, X_clf.sample(200),
        model_name="CancelShield_Classifier",
    )
    results["clf_local_dict"] = local_dict_clf
    results["clf_local_path"] = local_path_clf

    logger.info("Running Module 2 (regressor) global SHAP...")
    shap_vals_reg, global_path_reg = global_shap(
        reg_model, X_reg, model_name="CancelShield_ADR"
    )
    results["reg_global_shap"] = shap_vals_reg
    results["reg_global_path"] = global_path_reg

    logger.info("Running Module 2 local SHAP for sample booking...")
    local_dict_reg, local_path_reg = local_shap(
        reg_model, sample_booking_reg, X_reg.sample(200),
        model_name="CancelShield_ADR",
    )
    results["reg_local_dict"] = local_dict_reg
    results["reg_local_path"] = local_path_reg

    # Dependence plots for key features
    logger.info("Running dependence plots...")
    results["dep_lead_time"] = shap_dependence(
        clf_model, X_clf, "lead_time", model_name="CancelShield_Classifier"
    )
    results["dep_deposit_risk"] = shap_dependence(
        clf_model, X_clf, "deposit_risk_score", model_name="CancelShield_Classifier"
    )

    return results
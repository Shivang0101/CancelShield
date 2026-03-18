"""
Full evaluation beyond the standard sklearn defaults:
  - Classification: AUC, F1, confusion matrix, calibration curve, business metric
  - Regression: MAE, RMSE, R², residual analysis, price band accuracy
  - Business metric: revenue protected per 1,000 bookings screened
"""

import logging
from typing import Any, Dict, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def classification_report_full(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
    feature_names: list = None,
    model_name: str = "model",
    log_to_mlflow: bool = True,
) -> Dict[str, float]:

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:, 1]
    else:
        proba = model.decision_function(X_test)
        proba = (proba - proba.min()) / (proba.max() - proba.min())

    preds = (proba >= threshold).astype(int)

    auc_score = roc_auc_score(y_test, proba)
    f1 = f1_score(y_test, preds, zero_division=0)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    acc = (preds == y_test).mean()
    avg_prec = average_precision_score(y_test, proba)
    brier = brier_score_loss(y_test, proba)

    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "test_auc": auc_score,
        "test_f1": f1,
        "test_precision": prec,
        "test_recall": rec,
        "test_accuracy": acc,
        "test_avg_precision": avg_prec,
        "test_brier_score": brier,
        "test_tp": int(tp),
        "test_fp": int(fp),
        "test_fn": int(fn),
        "test_tn": int(tn),
        "threshold": threshold,
    }

    logger.info(
        "[%s] AUC=%.4f | F1=%.4f | Precision=%.4f | Recall=%.4f | Brier=%.4f",
        model_name, auc_score, f1, prec, rec, brier,
    )
    print(classification_report(y_test, preds, target_names=["Not Cancelled", "Cancelled"]))

    if log_to_mlflow:
        for k, v in metrics.items():
            try:
                mlflow.log_metric(k, float(v))
            except Exception:
                pass
    return metrics


def caliberation(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "model",
    n_bins: int = 10,
    log_to_mlflow: bool = True,
) -> Dict[str, float]:
    """

    A perfectly calibrated model: if it says 70% cancel probability,
    exactly 70% of those bookings actually cancel.
    XGBoost typically needs Platt scaling (sigmoid) or isotonic regression.

    Returns
    -------
    dict: mean_calibration_error, expected_calibration_error
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:, 1]
    else:
        return {}

    prob_true, prob_pred = calibration_curve(y_test, proba, n_bins=n_bins, strategy="uniform")

    # Expected Calibration Error (ECE)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (proba >= bin_edges[i]) & (proba < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_acc = y_test[mask].mean()
            bin_conf = proba[mask].mean()
            ece += (mask.sum() / len(y_test)) * abs(bin_acc - bin_conf)

    mce = float(np.mean(np.abs(prob_true - prob_pred)))

    metrics = {"calibration_ece": float(ece), "calibration_mce": mce}
    logger.info("[%s] ECE=%.4f | MCE=%.4f", model_name, ece, mce)

    if log_to_mlflow:
        for k, v in metrics.items():
            try:
                mlflow.log_metric(k, v)
            except Exception:
                pass

    return metrics


def business_metric(
    y_test: np.ndarray,
    proba: np.ndarray,
    adr_series: pd.Series,
    threshold: float = 0.5,
    total_nights_series: pd.Series = None,
    cost_of_missed_cancel: float = 112.0,
    cost_of_false_alarm: float = 15.0,
    model_name: str = "model",
    log_to_mlflow: bool = True,
) -> Dict[str, float]:
    """
    Translate model predictions into business revenue metrics.

    Revenue Protected:
      A correctly predicted cancellation (TP) gives the hotel time to rebook.
      Assumed recovery value = ADR * total_nights.

    False Alarm Cost:
      A falsely flagged booking triggers a retention action (discount offer, phone call).
      Estimated cost = $15 per intervention.

    Business Metric = Revenue Protected - False Alarm Costs
    Per 1,000 bookings screened at the chosen threshold.

    Returns dict with keys: revenue_protected_per_1k, net_value_per_1k,
                             false_alarm_cost_per_1k, cancellations_caught_rate
    """
    preds = (proba >= threshold).astype(int)
    adr = adr_series.values if isinstance(adr_series, pd.Series) else adr_series
    nights = total_nights_series.values if total_nights_series is not None else np.ones(len(y_test)) * 2

    tp_mask = (preds == 1) & (y_test == 1)
    fp_mask = (preds == 1) & (y_test == 0)

    revenue_recovered = (adr[tp_mask] * nights[tp_mask]).sum()
    false_alarm_cost = fp_mask.sum() * cost_of_false_alarm
    net_value = revenue_recovered - false_alarm_cost

    n = len(y_test)
    scale = 1000.0 / n

    metrics = {
        "revenue_protected_per_1k": float(revenue_recovered * scale),
        "false_alarm_cost_per_1k": float(false_alarm_cost * scale),
        "net_value_per_1k": float(net_value * scale),
        "cancellations_caught_rate": float(tp_mask.sum() / max(y_test.sum(), 1)),
        "precision_at_threshold": float(preds[y_test == 1].mean()) if preds.sum() > 0 else 0.0,
    }

    logger.info(
        "[%s] Revenue protected/1k: €%.0f | Net value/1k: €%.0f | Catch rate: %.1f%%",
        model_name, metrics["revenue_protected_per_1k"],
        metrics["net_value_per_1k"], metrics["cancellations_caught_rate"] * 100,
    )

    if log_to_mlflow:
        for k, v in metrics.items():
            try:
                mlflow.log_metric(k, v)
            except Exception:
                pass

    return metrics

def regression_report(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "model",
    log_to_mlflow: bool = True,
) -> Dict[str, float]:
    """
    Full regression evaluation: MAE, RMSE, R², MAPE, price-band accuracy.

    Price-band accuracy: how often is the prediction within ±20 EUR of actual?
    Directly interpretable for hotel pricing decisions.
    """
    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0, None)  # ADR can't be negative

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # MAPE (exclude zero actuals to avoid division by zero)
    nonzero = y_test > 0
    mape = float(np.mean(np.abs((y_test[nonzero] - y_pred[nonzero]) / y_test[nonzero])) * 100)

    # Price-band accuracy
    band_10 = float((np.abs(y_test - y_pred) <= 10).mean() * 100)
    band_20 = float((np.abs(y_test - y_pred) <= 20).mean() * 100)
    band_30 = float((np.abs(y_test - y_pred) <= 30).mean() * 100)

    metrics = {
        "test_mae": float(mae),
        "test_rmse": float(rmse),
        "test_r2": float(r2),
        "test_mape": mape,
        "band_accuracy_10eur": band_10,
        "band_accuracy_20eur": band_20,
        "band_accuracy_30eur": band_30,
    }

    logger.info(
        "[%s] MAE=%.2f | RMSE=%.2f | R²=%.4f | MAPE=%.1f%% | Within ±20: %.1f%%",
        model_name, mae, rmse, r2, mape, band_20,
    )

    
    if log_to_mlflow:
        for k, v in metrics.items():
            try:
                mlflow.log_metric(k, float(v))
            except Exception:
                pass
    return metrics


def comparison_table(results: Dict[str, Dict], metric: str = "val_auc") -> pd.DataFrame:
    """
    Build a ranked comparison DataFrame of all trained models.

    Parameters
    ----------
    results : dict from train_classifiers() or train_regressors()
    metric  : column to sort by (default: val_auc for classifiers, val_mae for regressors)
    """
    rows = []
    for name, r in results.items():
        row = {"model": name}
        row.update(r.get("metrics", {}))
        rows.append(row)

    df = pd.DataFrame(rows).set_index("model")
    ascending = metric in ("val_mae", "val_rmse", "test_mae", "test_rmse")
    df = df.sort_values(metric, ascending=ascending)
    return df
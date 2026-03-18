"""

Finds the optimal decision threshold for a binary classifier using two modes:

Mode 1 (F1 optimisation):    Maximise F1 score on validation set.
Mode 2 (Business cost function): Minimise total cost of missed cancellations
                                  + cost of unnecessary retention interventions.

Why not use 0.5?
In this dataset, the cost of a missed cancellation (empty room = 1x ADR)
is much higher than the cost of a false alarm (sending a retention email = ~€15).

"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)


def find_f1_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_thresholds: int = 200,
    beta: float = 1.0,
) -> Tuple[float, float]:

#    Find the threshold that maximises F-beta score on the provided data.
    
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    best_t, best_f = 0.5, 0.0

    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        tp = ((preds == 1) & (y_true == 1)).sum()
        fp = ((preds == 1) & (y_true == 0)).sum()
        fn = ((preds == 0) & (y_true == 1)).sum()

        if tp + fp == 0 or tp + fn == 0:
            continue

        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        denom = beta ** 2 * prec + rec
        if denom == 0:
            continue
        f_score = (1 + beta ** 2) * prec * rec / denom

        if f_score > best_f:
            best_f = f_score
            best_t = t

    logger.info("F%.1f-optimal threshold: %.4f (F%.1f=%.4f)", beta, best_t, beta, best_f)
    return float(best_t), float(best_f)


def find_cost_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    mean_adr: float = 112.0,
    mean_nights: float = 2.5,
    cost_retention_action: float = 15.0,
    n_thresholds: int = 200,
) -> Tuple[float, float, Dict]:
    """
    Find the threshold that minimises total business cost.

    Cost Model:
    -----------
    False Negative (missed cancellation): hotel doesn't rebook the room.
      → Cost = mean_adr * mean_nights (full revenue lost for that booking)

    False Positive (unnecessary retention action): email/call/discount offer
      → Cost = cost_retention_action (estimate €15 per intervention)

    Total Cost = (FN * cost_per_miss) + (FP * cost_per_false_alarm)

    The optimal threshold is where marginal benefit of catching one more
    cancellation equals the marginal cost of one more false alarm.

    Returns
    -------
    (best_threshold, min_cost_per_booking, full_sweep_dict)
    """
    cost_per_miss = mean_adr * mean_nights
    thresholds = np.linspace(0.01, 0.99, n_thresholds)

    total_costs = []
    fn_costs = []
    fp_costs = []

    n = len(y_true)

    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        fp = ((preds == 1) & (y_true == 0)).sum()
        fn = ((preds == 0) & (y_true == 1)).sum()
        tc = fn * cost_per_miss + fp * cost_retention_action
        total_costs.append(tc / n)
        fn_costs.append(fn * cost_per_miss / n)
        fp_costs.append(fp * cost_retention_action / n)

    best_idx = int(np.argmin(total_costs))
    best_t = float(thresholds[best_idx])
    min_cost = float(total_costs[best_idx])

    logger.info(
        "Cost-optimal threshold: %.4f | Min cost/booking: €%.2f | "
        "(cost_per_miss=€%.1f, cost_per_false_alarm=€%.1f)",
        best_t, min_cost, cost_per_miss, cost_retention_action,
    )

    sweep = {
        "thresholds": thresholds.tolist(),
        "total_cost_per_booking": total_costs,
        "fn_cost_per_booking": fn_costs,
        "fp_cost_per_booking": fp_costs,
        "best_threshold": best_t,
        "min_cost_per_booking": min_cost,
        "cost_per_miss": cost_per_miss,
        "cost_per_false_alarm": cost_retention_action,
    }

    return best_t, min_cost, sweep


# Combined Threshold Analysis
def tune_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    method: str = "cost",
    mean_adr: float = 112.0,
    mean_nights: float = 2.5,
    cost_retention_action: float = 15.0,
    model_name: str = "model",
) -> Dict:
    """
    Full threshold tuning analysis. Runs both F1 and cost-based optimisation
    and returns a comprehensive report.

    Parameters
    ----------
    method : 'f1', 'cost', or 'both'

    Returns
    -------
    dict with:
      - recommended_threshold  (from chosen method)
      - f1_optimal             (from F1 method)
      - cost_optimal           (from cost method)
      - metrics_at_recommended (precision, recall, f1, at chosen threshold)
    """
    result = {}

    # F1 optimisation
    f1_t, best_f1 = find_f1_optimal_threshold(y_true, y_proba, beta=1.0)
    result["f1_optimal"] = {"threshold": f1_t, "f1_score": best_f1}

    # Cost optimisation
    cost_t, min_cost, sweep = find_cost_optimal_threshold(
        y_true, y_proba, mean_adr, mean_nights, cost_retention_action
    )
    result["cost_optimal"] = {"threshold": cost_t, "min_cost_per_booking": min_cost}
    result["sweep_data"] = sweep

    # Choose recommended
    if method == "f1":
        result["recommended_threshold"] = f1_t
    elif method == "cost":
        result["recommended_threshold"] = cost_t
    else:  # both — use cost-optimal as primary
        result["recommended_threshold"] = cost_t

    # Compute metrics at recommended threshold
    t_rec = result["recommended_threshold"]
    preds = (y_proba >= t_rec).astype(int)
    result["metrics_at_recommended"] = {
        "threshold": t_rec,
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "cancellations_caught": int(((preds == 1) & (y_true == 1)).sum()),
        "false_alarms": int(((preds == 1) & (y_true == 0)).sum()),
        "missed_cancellations": int(((preds == 0) & (y_true == 1)).sum()),
    }

    logger.info(
        "Recommended threshold: %.4f | F1=%.4f | Precision=%.4f | Recall=%.4f",
        t_rec,
        result["metrics_at_recommended"]["f1"],
        result["metrics_at_recommended"]["precision"],
        result["metrics_at_recommended"]["recall"],
    )

    return result



if __name__ == "__main__":
    import sys
    import pickle
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from src.data.loader import get_splits
    from src.data.features import build_classification_features

    # Load actual model
    model_path = Path(__file__).resolve().parents[2] / "models" / "best_classifier.pkl"
    with open(model_path, "rb") as f:
        clf_data = pickle.load(f)
    clf = clf_data["model"]
    print(f"Loaded: {clf_data['name']}")

    # Load validation data
    splits = get_splits()
    X_tr, y_tr, enc = build_classification_features(splits["train"], fit_encoders=True)
    X_v, y_v, _     = build_classification_features(splits["val"], fit_encoders=False, encoders=enc)

    # Get real predictions
    y_proba = clf.predict_proba(X_v.values)[:, 1]

    # Tune
    result = tune_threshold(
        y_v.values, y_proba,
        method="both",
        mean_adr=112.0,
        mean_nights=2.5,
        cost_retention_action=15.0,
        model_name=clf_data["name"],
    )

    print("\nThreshold tuning result:")
    for k, v in result.items():
        if k != "sweep_data":
            print(f"  {k}: {v}")

    print(f"\n→ Update config.yaml: threshold.default: {result['f1_optimal']['threshold']:.2f}")
"""
Tunes Random Forest, XGBoost, and LightGBM using Optuna.
Overwrites best_classifier.pkl if the tuned model beats the current best.

Usage:
    python -m src.models.hpo                        # tune all 3, 50 trials each
    python -m src.models.hpo --models xgboost lgbm  # tune specific models
    python -m src.models.hpo --trials 100           # more trials

After running:
    - Best tuned model saved to models/best_classifier.pkl (if better than current)
    - All trials logged to MLflow under experiment CancelShield_HPO
"""

import argparse
import logging
import pickle
import sys
import warnings
from pathlib import Path

import mlflow
import numpy as np
import optuna
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.loader import get_splits, load_config
from src.data.features import build_classification_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# Load data once

def load_data():
    splits = get_splits()
    X_tr, y_tr, enc = build_classification_features(splits["train"], fit_encoders=True)
    X_v,  y_v,  _   = build_classification_features(splits["val"],   fit_encoders=False, encoders=enc)
    return X_tr.values, y_tr.values, X_v.values, y_v.values


def xgboost_objective(trial, X_tr, y_tr, X_v, y_v, scale_pos_weight):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 200, 800),
        "max_depth":        trial.suggest_int("max_depth", 3, 9),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda":       trial.suggest_float("reg_lambda", 0.0, 2.0),
        "scale_pos_weight": scale_pos_weight,
        "random_state":     42,
        "verbosity":        0,
        "eval_metric":      "auc",
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], verbose=False)
    return roc_auc_score(y_v, model.predict_proba(X_v)[:, 1])


def lightgbm_objective(trial, X_tr, y_tr, X_v, y_v):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 200, 800),
        "max_depth":        trial.suggest_int("max_depth", 3, 9),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves":       trial.suggest_int("num_leaves", 20, 150),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples":trial.suggest_int("min_child_samples", 5, 100),
        "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda":       trial.suggest_float("reg_lambda", 0.0, 2.0),
        "class_weight":     "balanced",
        "verbose":          -1,
        "random_state":     42,
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_v, y_v)],
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(period=-1)],
    )
    return roc_auc_score(y_v, model.predict_proba(X_v)[:, 1])


def random_forest_objective(trial, X_tr, y_tr, X_v, y_v):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 600),
        "max_depth":         trial.suggest_int("max_depth", 5, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.7]),
        "class_weight":      "balanced",
        "n_jobs":            -1,
        "random_state":      42,
    }
    model = RandomForestClassifier(**params)
    model.fit(X_tr, y_tr)
    return roc_auc_score(y_v, model.predict_proba(X_v)[:, 1])


# Refit best model on full train+val 

def refit_best(model_name: str, best_params: dict, X_tr, y_tr, X_v, y_v, scale_pos_weight):
    """Refit the winner on train data and return fitted model + val_auc."""
    logger.info("Refitting best %s with params: %s", model_name, best_params)

    if model_name == "xgboost":
        model = xgb.XGBClassifier(
            **best_params,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            verbosity=0,
            eval_metric="auc",
        )
        model.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], verbose=False)

    elif model_name == "lightgbm":
        model = lgb.LGBMClassifier(
            **best_params,
            class_weight="balanced",
            verbose=-1,
            random_state=42,
        )
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_v, y_v)],
            callbacks=[lgb.early_stopping(20), lgb.log_evaluation(period=-1)],
        )

    elif model_name == "random_forest":
        model = RandomForestClassifier(
            **best_params,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )
        model.fit(X_tr, y_tr)

    val_proba = model.predict_proba(X_v)[:, 1]
    val_auc   = roc_auc_score(y_v, val_proba)
    val_f1    = f1_score(y_v, (val_proba >= 0.5).astype(int), zero_division=0)
    return model, val_auc, val_f1


#  Save if better than current best 

def save_if_better(model_name: str, model, val_auc: float, config: dict):
    """Compare with existing best_classifier.pkl and overwrite if better."""
    save_path = ROOT / "models" / "best_classifier.pkl"
    current_auc = 0.0

    if save_path.exists():
        with open(save_path, "rb") as f:
            current = pickle.load(f)
        # Try to get stored AUC — may not exist in older pkl files
        current_auc = current.get("val_auc", 0.0)
        current_name = current.get("name", "unknown")
        logger.info(
            "Current best: %s (AUC=%.4f) | Tuned %s (AUC=%.4f)",
            current_name, current_auc, model_name, val_auc,
        )

    if val_auc > current_auc:
        with open(save_path, "wb") as f:
            pickle.dump({
                "name":    model_name,
                "model":   model,
                "scaler":  None,   
                "val_auc": val_auc,
            }, f)
        logger.info("✓ New best saved: %s (AUC=%.4f) → models/best_classifier.pkl", model_name, val_auc)

        # Log to MLflow
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        mlflow.set_experiment("CancelShield_HPO")
        try:
            with mlflow.start_run(run_name=f"hpo_best_{model_name}"):
                mlflow.log_param("model_name", model_name)
                mlflow.log_metric("val_auc", val_auc)
                mlflow.sklearn.log_model(
                    model,
                    artifact_path="model",
                    registered_model_name=config["mlflow"]["classifier_registry_name"],
                )
        except Exception as e:
            logger.warning("MLflow registry failed: %s", e)

        return True
    else:
        logger.info(
            "✗ Tuned model (AUC=%.4f) did not beat current best (AUC=%.4f). pkl unchanged.",
            val_auc, current_auc,
        )
        return False


# Pretty print best params for config.yaml

def print_yaml_snippet(model_name: str, best_params: dict):
    print(f"\n{'='*55}")
    print(f"  Paste into config.yaml → models → classifiers → {model_name}:")
    print(f"{'='*55}")
    print(f"    {model_name}:")
    for k, v in best_params.items():
        print(f"      {k}: {v}")
    print(f"{'='*55}\n")


# Main 

def run_hpo(models_to_tune: list, n_trials: int):
    config = load_config()
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])

    logger.info("Loading data...")
    X_tr, y_tr, X_v, y_v = load_data()
    scale_pos_weight = float((y_tr == 0).sum()) / max(float((y_tr == 1).sum()), 1)
    logger.info("Train: %d | Val: %d | scale_pos_weight: %.2f", len(X_tr), len(X_v), scale_pos_weight)

    overall_best = {"name": None, "auc": 0.0, "model": None}

    for model_name in models_to_tune:
        logger.info("\n%s", "─" * 55)
        logger.info("Tuning: %s (%d trials)", model_name.upper(), n_trials)
        logger.info("─" * 55)

        mlflow.set_experiment("CancelShield_HPO")

        study = optuna.create_study(
            direction="maximize",
            study_name=f"hpo_{model_name}",
            sampler=optuna.samplers.TPESampler(seed=42),
        )

        def make_objective(mn):
            def objective(trial):
                with mlflow.start_run(run_name=f"hpo_{mn}_trial_{trial.number}", nested=True):
                    if mn == "xgboost":
                        auc = xgboost_objective(trial, X_tr, y_tr, X_v, y_v, scale_pos_weight)
                    elif mn == "lightgbm":
                        auc = lightgbm_objective(trial, X_tr, y_tr, X_v, y_v)
                    elif mn == "random_forest":
                        auc = random_forest_objective(trial, X_tr, y_tr, X_v, y_v)
                    mlflow.log_metric("val_auc", auc)
                    for k, v in trial.params.items():
                        mlflow.log_param(k, v)
                return auc
            return objective

        with mlflow.start_run(run_name=f"hpo_{model_name}_study"):
            study.optimize(
                make_objective(model_name),
                n_trials=n_trials,
                show_progress_bar=True,
            )

        logger.info(
            "%s best trial: AUC=%.4f | params=%s",
            model_name, study.best_value, study.best_params,
        )

        # Refit on full train data with best params
        model, val_auc, val_f1 = refit_best(
            model_name, study.best_params, X_tr, y_tr, X_v, y_v, scale_pos_weight
        )

        logger.info(
            "%s refit → val_AUC=%.4f | val_F1=%.4f",
            model_name, val_auc, val_f1,
        )

        # Track overall best across all models
        if val_auc > overall_best["auc"]:
            overall_best = {"name": model_name, "auc": val_auc, "model": model}

        # Save if better than current pkl
        save_if_better(model_name, model, val_auc, config)

        # Print yaml snippet
        print_yaml_snippet(model_name, study.best_params)

    # Final summary
    logger.info("\n%s", "=" * 55)
    logger.info("HPO COMPLETE")
    logger.info("Best model across all tuned: %s (AUC=%.4f)", overall_best["name"], overall_best["auc"])
    logger.info("Check models/best_classifier.pkl for the winner.")
    logger.info("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CancelShield HPO")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["xgboost", "lightgbm", "random_forest"],
        choices=["xgboost", "lightgbm", "random_forest"],
        help="Which models to tune (default: all three)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Number of Optuna trials per model (default: 50)",
    )
    args = parser.parse_args()

    run_hpo(models_to_tune=args.models, n_trials=args.trials)
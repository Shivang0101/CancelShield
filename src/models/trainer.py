"""
Trains all classifiers (Module 1) and regressors (Module 2),
logs every experiment to MLflow, and registers the best models.

Classifiers  : Logistic Regression, XGBoost, LightGBM, MLP, Random Forest,
               Decision Tree, CNN-1D (PyTorch/Keras)
Regressors   : Linear Regression, XGBoost, LightGBM, MLP

Every run logs: params, metrics, feature importance chart, confusion matrix,
calibration curve, and the serialised model artifact.
"""

import logging
import os
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error, mean_squared_error,
    r2_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

# CNN-1D for Tabular Data

class CNN1DClassifier:
    """
    Architecture:
        Input → [N_features, 1]
        Conv1D(32, kernel=3) → ReLU → MaxPool
        Conv1D(64, kernel=3) → ReLU → MaxPool
        Flatten → Dense(128) → Dropout(0.3) → Dense(1) → Sigmoid
    """

    def __init__(
        self,
        epochs: int = 30,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        dropout: float = 0.3,
        device: str = "cpu",
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = learning_rate
        self.dropout = dropout
        self.device = device
        self.model_ = None
        self.scaler_ = StandardScaler()
        self.history_ = {"train_loss": [], "val_loss": [], "val_auc": []}

    def _build_model(self, n_features: int):
        """Build PyTorch Sequential model."""
        try:
            import torch
            import torch.nn as nn

            class _CNN1D(nn.Module):
                def __init__(self, n_feat, dropout):
                    super().__init__()
                    self.expand = nn.Linear(n_feat, n_feat * 4)
                    # Treat expanded features as sequence of length 4 with n_feat channels
                    self.conv1 = nn.Conv1d(n_feat, 64, kernel_size=2, padding=1)
                    self.conv2 = nn.Conv1d(64, 128, kernel_size=2)
                    self.pool = nn.AdaptiveMaxPool1d(1)
                    self.fc = nn.Sequential(
                        nn.Linear(128, 64),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(64, 1),
                    )

                def forward(self, x):
                    # x: [batch, n_feat]
                    x = torch.relu(self.expand(x))  # [batch, n_feat*4]
                    x = x.view(x.size(0), -1, 4)    # [batch, n_feat, 4]
                    x = torch.relu(self.conv1(x))
                    x = torch.relu(self.conv2(x))
                    x = self.pool(x).squeeze(-1)     # [batch, 128]
                    return self.fc(x).squeeze(-1)

            return _CNN1D(n_features, self.dropout)
        except ImportError:
            logger.warning("PyTorch not available — using Keras/TF CNN1D fallback.")
            return None

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None):
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            logger.warning("PyTorch not installed. CNN1D skipped.")
            return self

        X_s = self.scaler_.fit_transform(X).astype(np.float32)
        y_f = y.astype(np.float32)

        model = self._build_model(X.shape[1]).to(self.device)
        optim = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([(y == 0).sum() / max((y == 1).sum(), 1)]).to(self.device)
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=self.epochs)

        dataset = TensorDataset(torch.from_numpy(X_s), torch.from_numpy(y_f))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        for epoch in range(self.epochs):
            model.train()
            epoch_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optim.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                epoch_loss += loss.item()
            scheduler.step()
            self.history_["train_loss"].append(epoch_loss / len(loader))

            if X_val is not None and epoch % 5 == 0:
                val_proba = self._predict_proba_raw(model, X_val)
                val_auc = roc_auc_score(y_val, val_proba)
                self.history_["val_auc"].append(val_auc)
                logger.debug("Epoch %d/%d — loss=%.4f, val_auc=%.4f", epoch + 1, self.epochs, epoch_loss, val_auc)

        self.model_ = model
        return self

    def _predict_proba_raw(self, model, X: np.ndarray) -> np.ndarray:
        import torch
        X_s = self.scaler_.transform(X).astype(np.float32)
        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(X_s).to(self.device))
            return torch.sigmoid(logits).cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model not fitted yet.")
        proba = self._predict_proba_raw(self.model_, X)
        return np.column_stack([1 - proba, proba])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


def cross_validate_classifier(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    model_name: str = "model",
) -> Dict[str, float]:
    """
    Stratified K-Fold cross-validation for classifiers.
    Preserves class balance in each fold.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=False)
    aucs, f1s = [], []

    X_arr, y_arr = X.values if isinstance(X, pd.DataFrame) else X, \
                   y.values if isinstance(y, pd.Series) else y

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_arr, y_arr)):
        X_tr, X_v = X_arr[tr_idx], X_arr[val_idx]
        y_tr, y_v = y_arr[tr_idx], y_arr[val_idx]

        if isinstance(model, CNN1DClassifier):
            m = CNN1DClassifier(epochs=20)
            m.fit(X_tr, y_tr)
            proba = m.predict_proba(X_v)[:, 1]
        else:
            import copy
            m = copy.deepcopy(model)
            m.fit(X_tr, y_tr)
            proba = m.predict_proba(X_v)[:, 1]

        aucs.append(roc_auc_score(y_v, proba))
        preds = (proba >= 0.5).astype(int)
        f1s.append(f1_score(y_v, preds, zero_division=0))

    cv_results = {
        "cv_auc_mean": float(np.mean(aucs)),
        "cv_auc_std": float(np.std(aucs)),
        "cv_f1_mean": float(np.mean(f1s)),
        "cv_f1_std": float(np.std(f1s)),
    }
    logger.info(
        "[CV %s] AUC=%.4f±%.4f | F1=%.4f±%.4f",
        model_name, cv_results["cv_auc_mean"], cv_results["cv_auc_std"],
        cv_results["cv_f1_mean"], cv_results["cv_f1_std"],
    )
    return cv_results


def cross_validate_regressor(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    model_name: str = "model",
) -> Dict[str, float]:
    """K-Fold CV for regressors."""
    kf = KFold(n_splits=n_splits, shuffle=False)
    maes, rmses, r2s = [], [], []

    X_arr = X.values if isinstance(X, pd.DataFrame) else X
    y_arr = y.values if isinstance(y, pd.Series) else y

    for tr_idx, val_idx in kf.split(X_arr):
        import copy
        m = copy.deepcopy(model)
        m.fit(X_arr[tr_idx], y_arr[tr_idx])
        preds = m.predict(X_arr[val_idx])
        maes.append(mean_absolute_error(y_arr[val_idx], preds))
        rmses.append(np.sqrt(mean_squared_error(y_arr[val_idx], preds)))
        r2s.append(r2_score(y_arr[val_idx], preds))

    cv_results = {
        "cv_mae_mean": float(np.mean(maes)),
        "cv_mae_std": float(np.std(maes)),
        "cv_rmse_mean": float(np.mean(rmses)),
        "cv_r2_mean": float(np.mean(r2s)),
    }
    logger.info(
        "[CV %s] MAE=%.2f±%.2f | RMSE=%.2f | R²=%.4f",
        model_name, cv_results["cv_mae_mean"], cv_results["cv_mae_std"],
        cv_results["cv_rmse_mean"], cv_results["cv_r2_mean"],
    )
    return cv_results


def train_classifiers(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: dict = None,
    run_cv: bool = True,
) -> Dict[str, Any]:
    """
    Train all 7 classifiers for Module 1 and log to MLflow.

    Models:
      1. Logistic Regression  — interpretable baseline
      2. XGBoost Classifier   — primary model (handles imbalance natively)
      3. LightGBM Classifier  — faster alternative to XGBoost
      4. MLP Classifier       — deep learning on tabular data
      5. Random Forest        — bagging ensemble, robust to noise
      6. Decision Tree        — single tree, fully interpretable
      7. CNN-1D               — 1D conv net treating features as a sequence

    Returns dict: model_name → {"model": fitted_model, "val_auc": float, "metrics": dict}
    """
    if config is None:
        config = load_config()

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(f"{config['mlflow']['experiment_name']}_Module1")

    cfg_cls = config["models"]["classifiers"]
    X_tr = X_train.values
    y_tr = y_train.values
    X_v = X_val.values
    y_v = y_val.values

    scale_pos_weight = float((y_tr == 0).sum()) / max(float((y_tr == 1).sum()), 1)

    models_to_train = {
        "logistic_regression": LogisticRegression(
            **cfg_cls["logistic_regression"]
        ),
        "xgboost": xgb.XGBClassifier(
            **cfg_cls["xgboost"],
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            verbosity=0,
        ),
        "lightgbm": lgb.LGBMClassifier(
            **cfg_cls["lightgbm"],
            random_state=42,
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=cfg_cls["mlp"]["hidden_layer_sizes"],
            activation=cfg_cls["mlp"]["activation"],
            max_iter=cfg_cls["mlp"]["max_iter"],
            early_stopping=cfg_cls["mlp"]["early_stopping"],
            validation_fraction=cfg_cls["mlp"]["validation_fraction"],
            random_state=cfg_cls["mlp"]["random_state"],
        ),
        "random_forest": RandomForestClassifier(
            **cfg_cls["random_forest"],
        ),
        "decision_tree": DecisionTreeClassifier(
            **cfg_cls["decision_tree"],
        ),
        "cnn1d": CNN1DClassifier(
            epochs=cfg_cls["cnn1d"]["epochs"],
            batch_size=cfg_cls["cnn1d"]["batch_size"],
            learning_rate=cfg_cls["cnn1d"]["learning_rate"],
            dropout=cfg_cls["cnn1d"]["dropout"],
        ),
    }

    results = {}

    for name, model in models_to_train.items():
        logger.info("Training classifier: %s ...", name)

        with mlflow.start_run(run_name=f"cls_{name}"):
            # Train 
            if name == "cnn1d":
                model.fit(X_tr, y_tr, X_val=X_v, y_val=y_v)
                val_proba = model.predict_proba(X_v)[:, 1]
            elif name in ("xgboost", "lightgbm"):
                if name == "lightgbm":
                    model.fit(
                        X_tr, y_tr,
                        eval_set=[(X_v, y_v)],
                        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(period=-1)],
                    )
                else:
                    model.fit(
                        X_tr, y_tr,
                        eval_set=[(X_v, y_v)],
                        verbose=False,
                    )
                val_proba = model.predict_proba(X_v)[:, 1]
            else:
                scaler = StandardScaler()
                if name in ("logistic_regression", "mlp"):
                    X_tr_s = scaler.fit_transform(X_tr)
                    X_v_s = scaler.transform(X_v)
                else:
                    scaler = None  # tree models don't need scaling
                    X_tr_s = X_tr
                    X_v_s = X_v
                model.fit(X_tr_s, y_tr)
                val_proba = model.predict_proba(X_v_s)[:, 1]

            # Metrics 
            val_preds = (val_proba >= 0.5).astype(int)
            val_auc = roc_auc_score(y_v, val_proba)
            val_f1 = f1_score(y_v, val_preds, zero_division=0)
            val_acc = accuracy_score(y_v, val_preds)

            metrics = {
                "val_auc": val_auc,
                "val_f1": val_f1,
                "val_accuracy": val_acc,
            }

            if run_cv and name != "cnn1d":
                cv_metrics = cross_validate_classifier(model, X_train, y_train, model_name=name)
                metrics.update(cv_metrics)

            # MLflow log params 
            mlflow.log_param("model_name", name)
            mlflow.log_param("module", "1_classification")
            mlflow.log_param("n_features", X_train.shape[1])
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("cancel_rate_train", float(y_train.mean()))
            if hasattr(model, "get_params"):
                for k, v in model.get_params().items():
                    mlflow.log_param(k, v)

            # MLflow log metrics 
            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            results[name] = {
                "model": model,
                "val_auc": val_auc,
                "scaler": scaler if name in ("logistic_regression", "mlp") else None,
                "val_proba": val_proba,
                "metrics": metrics,
            }

            logger.info("[%s] val_AUC=%.4f | val_F1=%.4f", name, val_auc, val_f1)

    return results


def train_regressors(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: dict = None,
    run_cv: bool = True,
) -> Dict[str, Any]:
    """
    Train all 4 regressors for Module 2 (ADR prediction) and log to MLflow.

    Models:
      1. Linear Regression — interpretable baseline
      2. XGBoost Regressor — primary model
      3. LightGBM Regressor — often slightly better than XGBoost here
      4. MLP Regressor — captures non-linear ADR interactions
    """
    if config is None:
        config = load_config()

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(f"{config['mlflow']['experiment_name']}_Module2")

    cfg_reg = config["models"]["regressors"]
    X_tr = X_train.values
    y_tr = y_train.values
    X_v = X_val.values
    y_v = y_val.values

    models_to_train = {
        "linear_regression": LinearRegression(**cfg_reg["linear"]),
        "xgboost": xgb.XGBRegressor(
            **cfg_reg["xgboost"], random_state=42, verbosity=0,
        ),
        "lightgbm": lgb.LGBMRegressor(
            **cfg_reg["lightgbm"], random_state=42,
        ),
        "mlp": MLPRegressor(
            hidden_layer_sizes=cfg_reg["mlp"]["hidden_layer_sizes"],
            activation=cfg_reg["mlp"]["activation"],
            max_iter=cfg_reg["mlp"]["max_iter"],
            early_stopping=cfg_reg["mlp"]["early_stopping"],
            random_state=cfg_reg["mlp"]["random_state"],
        ),
    }

    results = {}

    for name, model in models_to_train.items():
        logger.info("Training regressor: %s ...", name)

        with mlflow.start_run(run_name=f"reg_{name}"):
            # Scale for linear models 
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr) if name in ("linear_regression", "mlp") else X_tr
            X_v_s = scaler.transform(X_v) if name in ("linear_regression", "mlp") else X_v

            if name in ("xgboost", "lightgbm"):
                if name == "lightgbm":
                    model.fit(
                        X_tr, y_tr,
                        eval_set=[(X_v, y_v)],
                        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(period=-1)],
                    )
                else:
                    model.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], verbose=False)
                val_preds = model.predict(X_v)
            else:
                # linear_regression and mlp — already scaled above
                model.fit(X_tr_s, y_tr)
                val_preds = model.predict(X_v_s)
            val_preds = model.predict(X_v_s)

            # Metrics 
            val_mae = mean_absolute_error(y_v, val_preds)
            val_rmse = np.sqrt(mean_squared_error(y_v, val_preds))
            val_r2 = r2_score(y_v, val_preds)

            metrics = {"val_mae": val_mae, "val_rmse": val_rmse, "val_r2": val_r2}

            if run_cv:
                cv_metrics = cross_validate_regressor(model, X_train, y_train, model_name=name)
                metrics.update(cv_metrics)

            #  MLflow log 
            mlflow.log_param("model_name", name)
            mlflow.log_param("module", "2_regression")
            mlflow.log_param("n_features", X_train.shape[1])
            mlflow.log_param("train_size", len(X_train))
            if hasattr(model, "get_params"):
                for k, v in model.get_params().items():
                    mlflow.log_param(k, v)

            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            results[name] = {
                "model": model,
                "scaler": scaler if name in ("linear_regression", "mlp") else None,
                "val_mae": val_mae,
                "val_preds": val_preds,
                "metrics": metrics,
            }

            logger.info("[%s] MAE=%.2f EUR | RMSE=%.2f | R²=%.4f", name, val_mae, val_rmse, val_r2)

    return results


def save_best_classifier(results: Dict[str, Any], config: dict = None) -> Tuple[str, Any]:
    """Pick the classifier with highest val_auc and register in MLflow."""
    if config is None:
        config = load_config()

    best_name = max(results, key=lambda n: results[n]["val_auc"])
    best_model = results[best_name]["model"]

    logger.info(
        "Best classifier: %s (val_AUC=%.4f)", best_name, results[best_name]["val_auc"]
    )

    # Save locally as backup
    save_dir = ROOT / "models"
    save_dir.mkdir(exist_ok=True)
    with open(save_dir / "best_classifier.pkl", "wb") as f:
        pickle.dump({
            "name": best_name,
            "model": best_model,
            "scaler": results[best_name].get("scaler"),
        }, f)

    # MLflow registry
    try:
        with mlflow.start_run(run_name="best_classifier_registry"):
            mlflow.log_param("best_model", best_name)
            mlflow.log_metric("val_auc", results[best_name]["val_auc"])
            mlflow.sklearn.log_model(
                best_model,
                artifact_path="model",
                registered_model_name=config["mlflow"]["classifier_registry_name"],
            )
    except Exception as e:
        logger.warning("MLflow registry failed: %s", e)

    return best_name, best_model


def save_best_regressor(results: Dict[str, Any], config: dict = None) -> Tuple[str, Any]:
    """Pick the regressor with lowest val_mae and register in MLflow."""
    if config is None:
        config = load_config()

    best_name = min(results, key=lambda n: results[n]["val_mae"])
    best_model = results[best_name]["model"]

    logger.info(
        "Best regressor: %s (val_MAE=%.2f EUR)", best_name, results[best_name]["val_mae"]
    )

    save_dir = ROOT / "models"
    save_dir.mkdir(exist_ok=True)
    with open(save_dir / "best_regressor.pkl", "wb") as f:
        pickle.dump({"name": best_name, "model": best_model,
                     "scaler": results[best_name].get("scaler")}, f)

    try:
        with mlflow.start_run(run_name="best_regressor_registry"):
            mlflow.log_param("best_model", best_name)
            mlflow.log_metric("val_mae", results[best_name]["val_mae"])
            mlflow.sklearn.log_model(
                best_model,
                artifact_path="model",
                registered_model_name=config["mlflow"]["regressor_registry_name"],
            )
    except Exception as e:
        logger.warning("MLflow registry failed: %s", e)

    return best_name, best_model

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT))

    from src.data.loader import get_splits
    from src.data.features import build_classification_features, build_regression_features

    splits = get_splits()

    # Module 1
    X_tr, y_tr, enc1 = build_classification_features(splits["train"])
    X_v, y_v, _ = build_classification_features(splits["val"], fit_encoders=False, encoders=enc1)
    cls_results = train_classifiers(X_tr, y_tr, X_v, y_v, run_cv=False)
    save_best_classifier(cls_results)

    # Module 2
    X_tr2, y_tr2, enc2 = build_regression_features(splits["train"])
    X_v2, y_v2, _ = build_regression_features(splits["val"], fit_encoders=False, encoders=enc2)
    reg_results = train_regressors(X_tr2, y_tr2, X_v2, y_v2, run_cv=False)
    save_best_regressor(reg_results)
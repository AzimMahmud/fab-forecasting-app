"""
================================================================================
  FABRIC CONSUMPTION FORECASTING SYSTEM — MODEL TRAINING SCRIPT
  Version  : 3.0.0
  Developer: Azim Mahmud
  Date     : January 2026
  Python   : 3.10.x
  Unit     : yards (all targets and outputs are in yards)

  Models trained:
    1. Linear Regression   (OLS baseline)
    2. Random Forest       (300 estimators)
    3. XGBoost             (300 estimators, eta=0.08)
    4. LSTM                (requires TensorFlow 2.13 — skipped gracefully if absent)
    5. Ensemble            (weighted average of the above)

  Usage:
    pip install -r requirements.txt
    python data_generation_script.py   # generate dataset first
    python train_models.py             # train and save all models
================================================================================
"""

# ============================================================================
# STANDARD LIBRARY
# ============================================================================
import os
import sys
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

# ============================================================================
# THIRD-PARTY IMPORTS
# ============================================================================
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    make_scorer,
)
import xgboost as xgb

# TensorFlow / Keras — optional; LSTM is skipped if not installed
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks  # type: ignore

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print(
        "\n  TensorFlow not found — LSTM will be skipped.\n"
        "  Install with:  pip install tensorflow==2.13.0\n"
    )


# ============================================================================
# LOGGING
# ============================================================================
def _make_logger(name: str = "ModelTraining") -> logging.Logger:
    log = logging.getLogger(name)
    if log.handlers:
        return log
    log.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    log.addHandler(ch)
    return log


logger = _make_logger()


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
class TrainingConfig:
    """
    All hyper-parameters and constants in one place.
    Edit only this class to tune the models.
    """

    # ── Paths ────────────────────────────────────────────────────────────────
    DATA_PATH = Path("generated_data")
    TRAINING_DATA = "training_dataset_5000_orders_yards.csv"
    MODEL_PATH = Path("models")

    # ── Reproducibility ──────────────────────────────────────────────────────
    RANDOM_STATE = 42
    TEST_SIZE = 0.20  # 20 % held-out test set
    VAL_SIZE = 0.20  # 20 % of remaining 80 % => 16 % overall

    # ── XGBoost ─────────────────────────────────────────────────────────────
    XGB_PARAMS = {
        "objective": "reg:squarederror",
        "n_estimators": 300,
        "max_depth": 8,
        "learning_rate": 0.08,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbosity": 0,
    }

    # ── Random Forest ────────────────────────────────────────────────────────
    RF_PARAMS = {
        "n_estimators": 300,
        "max_depth": 12,
        "min_samples_split": 4,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "bootstrap": True,
        "random_state": 42,
        "n_jobs": -1,
    }

    # ── Linear Regression ────────────────────────────────────────────────────
    LR_PARAMS = {"fit_intercept": True}

    # ── LSTM ─────────────────────────────────────────────────────────────────
    LSTM_PARAMS = {
        "units_layer1": 64,
        "units_layer2": 32,
        "dense_units": 16,
        "dropout_rate": 0.20,
        "batch_size": 32,
        "epochs": 200,
        "patience": 15,
    }

    # ── Ensemble weights ─────────────────────────────────────────────────────
    # Computed dynamically at training time using inverse-RMSE² weighting.
    # No hardcoded weights needed.

    # ── Column mapping  (CSV header => internal name) ────────────────────────
    COLUMN_MAPPING = {
        "Order_Quantity": "order_quantity",
        "Fabric_Width_cm": "fabric_width_cm",
        "Marker_Efficiency_%": "marker_efficiency",
        "Expected_Defect_Rate_%": "defect_rate",
        "Operator_Experience_Years": "operator_experience",
        "Garment_Type": "garment_type",
        "Fabric_Type": "fabric_type",
        "Pattern_Complexity": "pattern_complexity",
        "Season": "season",
        "Actual_Consumption_yards": "fabric_consumption_yards",
    }

    # ── Feature set (9 numeric / encoded features) ───────────────────────────
    FEATURES = [
        "order_quantity",
        "fabric_width_cm",
        "marker_efficiency",
        "defect_rate",
        "operator_experience",
        "garment_type_encoded",
        "fabric_type_encoded",
        "pattern_complexity_encoded",
        "season_encoded",
    ]
    TARGET = "fabric_consumption_yards"

    # Alphabetical ordering must match sklearn LabelEncoder default behaviour
    CATEGORICAL_FEATURES = {
        "garment_type": ["Dress", "Jacket", "Pants", "Shirt", "T-Shirt"],
        "fabric_type": ["Cotton", "Cotton-Blend", "Denim", "Polyester", "Silk"],
        "pattern_complexity": ["Complex", "Medium", "Simple"],
        "season": ["Fall", "Spring", "Summer", "Winter"],
    }

    # Default CI fractions (overwritten by empirical test-set values at runtime)
    CI_FRACTION_DEFAULTS = {
        "ensemble": 0.019,
        "xgboost": 0.023,
        "random_forest": 0.027,
        "lstm": 0.031,
        "linear_regression": 0.062,
    }

    # Garment base consumption (metres at 160 cm std width)
    # Kept in metres so physics matches data_generation_script.py
    GARMENT_BASE_M = {
        "T-Shirt": 1.20,
        "Shirt": 1.80,
        "Pants": 2.50,
        "Dress": 3.00,
        "Jacket": 3.50,
    }

    METERS_TO_YARDS = 1.0936132983

    COMPLEXITY_MULTIPLIER = {
        "Simple": 1.00,
        "Medium": 1.15,
        "Complex": 1.35,
    }


# ============================================================================
# DATA PREPROCESSING
# ============================================================================
class DataPreprocessor:
    """Load, clean, label-encode and scale the training dataset."""

    def __init__(self, config):
        self.config = config
        self.label_encoders = {}
        self.scaler = None

    def load_data(self):
        path = self.config.DATA_PATH / self.config.TRAINING_DATA
        logger.info(f"  Loading data from: {path}")
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {path}\n  Run:  python data_generation_script.py"
            )
        df = pd.read_csv(path)
        logger.info(f"  Loaded {df.shape[0]:,} rows x {df.shape[1]} columns")
        return df

    def preprocess(self, df):
        logger.info("  Preprocessing ...")

        # Rename CSV headers to internal names
        df = df.rename(columns=self.config.COLUMN_MAPPING)

        # Drop rows with nulls in ML-relevant columns
        raw_cats = list(self.config.CATEGORICAL_FEATURES.keys())
        numeric_cols = [f for f in self.config.FEATURES if not f.endswith("_encoded")]
        drop_cols = list(dict.fromkeys(numeric_cols + raw_cats + [self.config.TARGET]))
        drop_cols = [c for c in drop_cols if c in df.columns]
        before = len(df)
        df = df.dropna(subset=drop_cols)
        if len(df) < before:
            logger.warning(f"  Dropped {before - len(df)} rows with nulls")

        # Label-encode categoricals (alphabetical classes => 0, 1, 2 ...)
        for feat, categories in self.config.CATEGORICAL_FEATURES.items():
            if feat not in df.columns:
                logger.warning(f"  Column '{feat}' missing — encoding skipped")
                continue
            enc = LabelEncoder()
            enc.fit(categories)
            df[f"{feat}_encoded"] = enc.transform(df[feat])
            self.label_encoders[feat] = enc
            logger.info(f"    Encoded '{feat}' => {categories}")

        # Verify all 9 features exist
        missing = set(self.config.FEATURES) - set(df.columns)
        if missing:
            raise ValueError(f"Features missing after encoding: {missing}")

        logger.info(f"  Preprocessing complete — {len(df):,} rows retained")
        return df

    def scale(self, X, fit=False):
        if fit or self.scaler is None:
            self.scaler = StandardScaler()
            result = self.scaler.fit_transform(X)
            logger.info("  StandardScaler fitted")
        else:
            result = self.scaler.transform(X)
        return result


# ============================================================================
# MODEL TRAINER
# ============================================================================
class ModelTrainer:
    """Train, evaluate, and store all five ML models."""

    def __init__(self, config):
        self.config = config
        self.models = {}
        self.perf = {}
        self.history = {}

    # ── Individual trainers ──────────────────────────────────────────────────

    def _train_linear_regression(self, Xtr, ytr):
        logger.info("    Training Linear Regression ...")
        m = LinearRegression(**self.config.LR_PARAMS)
        m.fit(Xtr, ytr)
        self.models["linear_regression"] = m
        logger.info("    Linear Regression done")
        return m

    def _train_random_forest(self, Xtr, ytr):
        logger.info("    Training Random Forest (300 trees) ...")
        m = RandomForestRegressor(**self.config.RF_PARAMS)
        m.fit(Xtr, ytr)
        self.models["random_forest"] = m
        logger.info("    Random Forest done")
        return m

    def _train_xgboost(self, Xtr, ytr):
        logger.info("    Training XGBoost (300 trees, eta=0.08) ...")
        m = xgb.XGBRegressor(**self.config.XGB_PARAMS)
        m.fit(Xtr, ytr)
        self.models["xgboost"] = m
        logger.info("    XGBoost done")
        return m

    def _train_lstm(self, Xtr, ytr, Xv, yv):
        if not TENSORFLOW_AVAILABLE:
            logger.warning("    TensorFlow not available — LSTM skipped")
            return None

        p = self.config.LSTM_PARAMS
        logger.info(
            f"    Training LSTM (up to {p['epochs']} epochs, "
            f"patience={p['patience']}) ..."
        )
        n_feat = Xtr.shape[1]

        # Reshape to (samples, timesteps=1, features) for Keras LSTM
        Xtr_3d = Xtr.reshape(-1, 1, n_feat)
        Xv_3d = Xv.reshape(-1, 1, n_feat)

        model = keras.Sequential(
            [
                layers.LSTM(
                    p["units_layer1"],
                    activation="relu",
                    input_shape=(1, n_feat),
                    return_sequences=True,
                ),
                layers.Dropout(p["dropout_rate"]),
                layers.LSTM(p["units_layer2"], activation="relu"),
                layers.Dropout(p["dropout_rate"]),
                layers.Dense(p["dense_units"], activation="relu"),
                layers.Dense(1),
            ],
            name="lstm_fabric_forecast",
        )

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="mse",
            metrics=["mae"],
        )

        cb = [
            callbacks.EarlyStopping(
                monitor="val_loss",
                patience=p["patience"],
                restore_best_weights=True,
                verbose=0,
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=6,
                min_lr=1e-5,
                verbose=0,
            ),
        ]

        hist = model.fit(
            Xtr_3d,
            ytr,
            validation_data=(Xv_3d, yv),
            epochs=p["epochs"],
            batch_size=p["batch_size"],
            callbacks=cb,
            verbose=0,
        )

        epochs_run = len(hist.history["loss"])
        best_idx = int(np.argmin(hist.history["val_loss"]))
        best_val = hist.history["val_loss"][best_idx]
        logger.info(
            f"    LSTM done  (best_epoch={best_idx + 1}/{epochs_run}, "
            f"best_val_loss={best_val:.2f})"
        )

        self.models["lstm"] = model
        self.history["lstm"] = hist.history
        return model

    # ── Ensemble ─────────────────────────────────────────────────────────────

    def _build_ensemble(self, Xv, yv, Xte, yte):
        """Weighted-average ensemble with dynamic inverse-RMSE² weights on validation set."""
        val_rmses = {}
        for name in ["xgboost", "random_forest", "lstm", "linear_regression"]:
            m = self.models.get(name)
            if m is None:
                continue
            
            if name == "lstm" and TENSORFLOW_AVAILABLE:
                Xl_v = Xv.reshape(-1, 1, Xv.shape[1])
                val_preds = m.predict(Xl_v, verbose=0).flatten()
            else:
                val_preds = m.predict(Xv)
            
            yv_arr = yv.values if hasattr(yv, "values") else np.asarray(yv)
            rmse_val = float(np.sqrt(mean_squared_error(yv_arr, val_preds)))
            val_rmses[name] = rmse_val
            
        if not val_rmses:
            raise RuntimeError("No sub-models available for ensemble")

        inv_rmse_sq = {n: 1.0 / (rmse ** 2) for n, rmse in val_rmses.items()}
        total = sum(inv_rmse_sq.values())
        weights = {n: round(v / total, 4) for n, v in inv_rmse_sq.items()}

        weight_sum = sum(weights.values())
        first_key = list(weights.keys())[0]
        weights[first_key] = round(weights[first_key] + (1.0 - weight_sum), 4)

        spec = {"model_names": list(weights.keys()), "weights": weights}
        self.models["ensemble"] = spec

        preds = np.zeros(len(yte))
        for name, w in weights.items():
            m = self.models.get(name)
            if m is None:
                logger.warning(
                    f"    Ensemble: sub-model '{name}' not available — skipped"
                )
                continue
            if name == "lstm" and TENSORFLOW_AVAILABLE:
                Xl = Xte.reshape(-1, 1, Xte.shape[1])
                p = m.predict(Xl, verbose=0).flatten()
            else:
                p = m.predict(Xte)
            preds += w * p

        logger.info(f"    Ensemble built (eval on Val)  weights={weights}")
        return spec, preds

    # ── Evaluation ───────────────────────────────────────────────────────────

    def _evaluate(self, model, Xte, yte, name, ypred=None):
        """Compute RMSE, MAE, R2, MAPE and empirical 90th-pct CI fraction."""
        if ypred is None:
            if name == "ensemble":
                raise ValueError("Pass ypred= for the ensemble")
            if name == "lstm" and TENSORFLOW_AVAILABLE:
                Xl = Xte.reshape(-1, 1, Xte.shape[1])
                ypred = model.predict(Xl, verbose=0).flatten()
            else:
                ypred = model.predict(Xte)

        yte_arr = yte.values if hasattr(yte, "values") else np.asarray(yte)

        rmse = float(np.sqrt(mean_squared_error(yte_arr, ypred)))
        mae = float(mean_absolute_error(yte_arr, ypred))
        r2 = float(r2_score(yte_arr, ypred))
        mape = float(
            np.mean(np.abs((yte_arr - ypred) / np.where(yte_arr == 0, 1, yte_arr)))
            * 100
        )

        residuals = np.abs(yte_arr - ypred)
        safe_yte = np.where(yte_arr == 0, 1, yte_arr)
        ci_frac_emp = float(np.percentile(residuals / safe_yte, 90))
        ci_frac = round(
            max(ci_frac_emp, self.config.CI_FRACTION_DEFAULTS.get(name, 0.05)), 4
        )

        return {
            "rmse": round(rmse, 3),
            "mae": round(mae, 3),
            "r2": round(r2, 4),
            "mape": round(mape, 2),
            "ci_bounds": {"ci_fraction": ci_frac},
        }

    # ── Orchestrator ─────────────────────────────────────────────────────────

    def train_all(self, Xtr, ytr, Xv, yv, Xte, yte):
        """Train all five models, evaluate, return (models, performance)."""
        sep = "-" * 60

        logger.info(f"\n{sep}")
        logger.info("  [1/4] Linear Regression")
        self._train_linear_regression(Xtr, ytr)

        logger.info(f"\n{sep}")
        logger.info("  [2/4] Random Forest")
        self._train_random_forest(Xtr, ytr)

        logger.info(f"\n{sep}")
        logger.info("  [3/4] XGBoost")
        self._train_xgboost(Xtr, ytr)

        logger.info(f"\n{sep}")
        logger.info("  [4/4] LSTM Neural Network")
        self._train_lstm(Xtr, ytr, Xv, yv)

        logger.info(f"\n{sep}")
        logger.info("  Evaluating individual models ...")
        for name in ["linear_regression", "random_forest", "xgboost", "lstm"]:
            m = self.models.get(name)
            if m is None:
                continue
            self.perf[name] = self._evaluate(m, Xte, yte, name)
            p = self.perf[name]
            logger.info(
                f"    {name:<22}  RMSE={p['rmse']:>9.3f} yd  "
                f"MAE={p['mae']:>8.3f} yd  R2={p['r2']:.4f}  MAPE={p['mape']:.2f}%"
            )

        logger.info(f"\n{sep}")
        logger.info("  [5/5] Ensemble (weighted average)")
        _, ens_preds = self._build_ensemble(Xv, yv, Xte, yte)
        self.perf["ensemble"] = self._evaluate(
            None, Xte, yte, "ensemble", ypred=ens_preds
        )
        p = self.perf["ensemble"]
        logger.info(
            f"    {'ensemble':<22}  RMSE={p['rmse']:>9.3f} yd  "
            f"MAE={p['mae']:>8.3f} yd  R2={p['r2']:.4f}  MAPE={p['mape']:.2f}%"
        )

        return self.models, self.perf


# ============================================================================
# MODEL SAVER
# ============================================================================
class ModelSaver:
    """Persist trained models and metadata to disk."""

    def __init__(self, config):
        self.config = config
        config.MODEL_PATH.mkdir(parents=True, exist_ok=True)

    def save_pkl(self, obj, filename):
        path = self.config.MODEL_PATH / filename
        joblib.dump(obj, path)
        kb = path.stat().st_size / 1024
        logger.info(f"    Saved {filename:<38} ({kb:>8.1f} KB)")
        return path

    def save_keras(self, model, filename):
        path = self.config.MODEL_PATH / filename
        model.save(str(path))
        kb = path.stat().st_size / 1024
        logger.info(f"    Saved {filename:<38} ({kb:>8.1f} KB)")
        return path

    def save_metadata(
        self,
        performance,
        preprocessor,
        ensemble_weights,
        training_history,
        training_date,
        cv_scores=None,
        feature_importance=None,
        bom_baseline=None,
    ):
        """Write model_metadata.json consumed by app.py."""
        cv = cv_scores or {}
        fi = feature_importance or {}

        _type_map = {
            "xgboost": "xgboost.XGBRegressor",
            "random_forest": "sklearn.ensemble.RandomForestRegressor",
            "linear_regression": "sklearn.linear_model.LinearRegression",
            "lstm": "tensorflow.keras.Sequential",
            "ensemble": "weighted_average_ensemble",
        }

        meta = {
            "version": "3.0.0",
            "training_date": training_date,
            "unit": "yards",
            "n_training_samples": 5000,
            "features": self.config.FEATURES,
            "target": self.config.TARGET,
            "tensorflow_available": TENSORFLOW_AVAILABLE,
            "models": {},
            "bom_baseline": bom_baseline or {"rmse": 0, "mae": 0, "r2": 0, "mape": 0},
            "cv_scores": cv,
            "feature_importance": fi,
        }

        for name, perf in performance.items():
            entry = {
                "file": f"{name}_model.pkl" if name != "lstm" else "lstm_model.h5",
                "type": _type_map.get(name, "unknown"),
                "n_features": len(self.config.FEATURES),
                "rmse": perf["rmse"],
                "mae": perf["mae"],
                "r2": perf["r2"],
                "mape": perf["mape"],
                "ci_bounds": perf.get("ci_bounds", {"ci_fraction": 0.05}),
            }
            if name == "ensemble":
                entry["weights"] = ensemble_weights
                entry["model_names"] = list(ensemble_weights.keys())
            if name == "lstm" and "lstm" in training_history:
                h = training_history["lstm"]
                best_idx = int(np.argmin(h["val_loss"]))
                entry["training_history"] = {
                    "epochs_trained": len(h["loss"]),
                    "best_epoch": best_idx + 1,
                    "best_train_loss": round(float(h["loss"][best_idx]), 4),
                    "best_val_loss": round(float(h["val_loss"][best_idx]), 4),
                }
            if name in fi:
                entry["feature_importance"] = fi[name]
            if name in cv:
                entry["cross_validation"] = cv[name]
            meta["models"][name] = entry

        out = self.config.MODEL_PATH / "model_metadata.json"
        with open(out, "w") as fh:
            json.dump(meta, fh, indent=2)
        logger.info("    Saved model_metadata.json")
        return out


# ============================================================================
# PRETTY PRINT
# ============================================================================
def _print_table(performance):
    print(
        "\n"
        "+---------------------+------------+----------+---------+----------+\n"
        "| Model               |  RMSE (yd) |  MAE(yd) |   R2    |   MAPE   |\n"
        "+---------------------+------------+----------+---------+----------+"
    )
    for name, m in performance.items():
        label = name.replace("_", " ").title()
        print(
            f"| {label:<19} | {m['rmse']:>10.3f} | "
            f"{m['mae']:>8.3f} | {m['r2']:>7.4f} | {m['mape']:>7.2f}% |"
        )
    print("+---------------------+------------+----------+---------+----------+")


# ============================================================================
# CROSS-VALIDATION HELPER
# ============================================================================
def _mape_scorer(y_true, y_pred):
    safe = np.where(y_true == 0, 1, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / safe)) * 100)


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def main():
    bar = "=" * 70
    logger.info(bar)
    logger.info("  FABRIC CONSUMPTION FORECASTING — MODEL TRAINING  v3.0.0")
    logger.info(
        f"  TensorFlow : "
        f"{'available (LSTM enabled)' if TENSORFLOW_AVAILABLE else 'NOT found (LSTM skipped)'}"
    )
    logger.info(f"  Python     : {sys.version.split()[0]}")
    logger.info(f"  NumPy      : {np.__version__}")
    logger.info(f"  Pandas     : {pd.__version__}")
    logger.info(f"  XGBoost    : {xgb.__version__}")
    logger.info(bar)

    try:
        cfg = TrainingConfig()
        preprocessor = DataPreprocessor(cfg)
        trainer = ModelTrainer(cfg)
        saver = ModelSaver(cfg)

        # STEP 1  Load & preprocess
        logger.info("\n  STEP 1 — Load & Preprocess")
        df = preprocessor.load_data()
        df = preprocessor.preprocess(df)
        X = df[cfg.FEATURES].values.astype(np.float32)
        y = df[cfg.TARGET].values.astype(np.float32)
        logger.info(f"  X shape: {X.shape}   y shape: {y.shape}")

        # STEP 2  Split
        logger.info("\n  STEP 2 — Split  (64% train / 16% val / 20% test)")
        indices = np.arange(len(df))
        X_tmp, X_te, y_tmp, y_te, idx_tmp, idx_te = train_test_split(
            X,
            y,
            indices,
            test_size=cfg.TEST_SIZE,
            random_state=cfg.RANDOM_STATE,
        )
        X_tr, X_v, y_tr, y_v, idx_tr, idx_v = train_test_split(
            X_tmp,
            y_tmp,
            idx_tmp,
            test_size=cfg.VAL_SIZE,
            random_state=cfg.RANDOM_STATE,
        )
        logger.info(f"  Train {len(X_tr):,}  Val {len(X_v):,}  Test {len(X_te):,}")

        # STEP 2b  Empirical BOM baseline
        test_df_raw = df.iloc[idx_te]
        bom_base_m = test_df_raw["garment_type"].map(cfg.GARMENT_BASE_M)
        bom_base_m = bom_base_m * (160.0 / test_df_raw["fabric_width_cm"])
        bom_base_m = bom_base_m * test_df_raw["pattern_complexity"].map(
            cfg.COMPLEXITY_MULTIPLIER
        )
        bom_pred_yd = (
            test_df_raw["order_quantity"] * bom_base_m * 1.05 * cfg.METERS_TO_YARDS
        ).values
        bom_actual = y_te.astype(np.float64)
        bom_rmse = float(np.sqrt(mean_squared_error(bom_actual, bom_pred_yd)))
        bom_mae = float(mean_absolute_error(bom_actual, bom_pred_yd))
        bom_r2 = float(r2_score(bom_actual, bom_pred_yd))
        bom_mape = float(
            np.mean(
                np.abs(
                    (bom_actual - bom_pred_yd)
                    / np.where(bom_actual == 0, 1, bom_actual)
                )
            )
            * 100
        )
        bom_baseline = {
            "rmse": round(bom_rmse, 1),
            "mae": round(bom_mae, 1),
            "r2": round(bom_r2, 4),
            "mape": round(bom_mape, 2),
        }
        logger.info(
            f"  BOM Baseline  RMSE={bom_rmse:.1f} yd  MAE={bom_mae:.1f} yd  "
            f"R2={bom_r2:.4f}  MAPE={bom_mape:.2f}%"
        )

        # STEP 3  Scale
        logger.info("\n  STEP 3 — StandardScaler")
        X_tr_s = preprocessor.scale(X_tr, fit=True)
        X_v_s = preprocessor.scale(X_v)
        X_te_s = preprocessor.scale(X_te)

        # STEP 4  Train
        logger.info("\n  STEP 4 — Train Models")
        y_te_series = pd.Series(y_te.astype(np.float64))
        models, performance = trainer.train_all(
            X_tr_s, y_tr, X_v_s, y_v, X_te_s, y_te_series
        )

        # STEP 5a  Performance table
        logger.info("\n  STEP 5a — Performance Summary (yards, held-out test set)")
        _print_table(performance)
        best_name, best_p = min(
            ((n, p) for n, p in performance.items() if n != "ensemble"),
            key=lambda x: x[1]["rmse"],
        )
        logger.info(
            f"\n  Best single model: {best_name.upper()}  RMSE={best_p['rmse']:.3f} yd"
        )

        # STEP 5b  5-fold Cross-Validation (LR / RF / XGB)
        # NOTE: CV runs on train+val only (80% of data) to keep test set
        # completely held-out and avoid any data leakage into CV scores.
        logger.info("\n  STEP 5b — 5-Fold Cross-Validation (LR / RF / XGB, train+val only)")
        cv_scores = {}
        kf = KFold(n_splits=5, shuffle=True, random_state=cfg.RANDOM_STATE)
        X_cv = np.vstack([X_tr_s, X_v_s])
        y_cv = np.concatenate([y_tr, y_v])

        cv_candidates = {
            "linear_regression": models.get("linear_regression"),
            "random_forest": models.get("random_forest"),
            "xgboost": models.get("xgboost"),
        }
        for cv_name, m in cv_candidates.items():
            if m is None:
                continue
            rmse_folds = -cross_val_score(
                m, X_cv, y_cv, cv=kf, scoring="neg_root_mean_squared_error", n_jobs=-1
            )
            r2_folds = cross_val_score(m, X_cv, y_cv, cv=kf, scoring="r2", n_jobs=-1)
            mape_folds = -cross_val_score(
                m,
                X_cv,
                y_cv,
                cv=kf,
                scoring=make_scorer(_mape_scorer, greater_is_better=False),
                n_jobs=-1,
            )
            cv_scores[cv_name] = {
                "rmse_mean": round(float(rmse_folds.mean()), 3),
                "rmse_std": round(float(rmse_folds.std()), 3),
                "r2_mean": round(float(r2_folds.mean()), 4),
                "r2_std": round(float(r2_folds.std()), 4),
                "mape_mean": round(float(mape_folds.mean()), 2),
                "mape_std": round(float(mape_folds.std()), 2),
                "folds_rmse": [round(float(f), 3) for f in rmse_folds.tolist()],
            }
            logger.info(
                f"  {cv_name:<22}  RMSE={rmse_folds.mean():.3f}+/-{rmse_folds.std():.3f}  "
                f"R2={r2_folds.mean():.4f}  MAPE={mape_folds.mean():.2f}%"
            )

        # STEP 5c  Feature Importance
        logger.info("\n  STEP 5c — Feature Importance")
        feature_importance = {}
        for fi_name in ("xgboost", "random_forest"):
            m = models.get(fi_name)
            if m is not None and hasattr(m, "feature_importances_"):
                fi = m.feature_importances_.tolist()
                feature_importance[fi_name] = {
                    fn: round(float(v), 6) for fn, v in zip(cfg.FEATURES, fi)
                }
                top3 = sorted(zip(cfg.FEATURES, fi), key=lambda x: x[1], reverse=True)[
                    :3
                ]
                logger.info(
                    f"  {fi_name}: " + ", ".join(f"{n}={v:.4f}" for n, v in top3)
                )

        # STEP 6  Save
        logger.info("\n  STEP 6 — Save Models & Artefacts")
        for name, model in models.items():
            if name == "ensemble":
                saver.save_pkl(model, "ensemble_model.pkl")
            elif name == "lstm" and TENSORFLOW_AVAILABLE:
                saver.save_keras(model, "lstm_model.h5")
            else:
                saver.save_pkl(model, f"{name}_model.pkl")

        saver.save_pkl(preprocessor.scaler, "scaler.pkl")
        saver.save_pkl(preprocessor.label_encoders, "label_encoders.pkl")
        saver.save_metadata(
            performance,
            preprocessor,
            models["ensemble"]["weights"],
            trainer.history,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            cv_scores=cv_scores,
            feature_importance=feature_importance,
            bom_baseline=bom_baseline,
        )

        # STEP 7  Verify
        logger.info("\n  STEP 7 — Verify Saved Files")
        expected = [
            "xgboost_model.pkl",
            "random_forest_model.pkl",
            "linear_regression_model.pkl",
            "ensemble_model.pkl",
            "scaler.pkl",
            "label_encoders.pkl",
            "model_metadata.json",
        ]
        if TENSORFLOW_AVAILABLE:
            expected.append("lstm_model.h5")

        all_ok = True
        for fn in expected:
            fp = cfg.MODEL_PATH / fn
            if fp.exists():
                logger.info(f"  OK {fn:<38} ({fp.stat().st_size / 1024:>8.1f} KB)")
            else:
                logger.error(f"  MISSING: {fn}")
                all_ok = False

        if not all_ok:
            raise FileNotFoundError("One or more model files are missing after saving.")

        # Done
        logger.info(f"\n{bar}")
        logger.info("  TRAINING COMPLETED SUCCESSFULLY")
        logger.info(f"  Models saved to : {cfg.MODEL_PATH.absolute()}/")
        logger.info(
            f"  Best model      : {best_name.upper()}  RMSE={best_p['rmse']:.3f} yd"
        )
        logger.info("  Run the app     : streamlit run app.py")
        logger.info(bar)

        if not TENSORFLOW_AVAILABLE:
            logger.warning(
                "\n  LSTM was NOT trained (TensorFlow not installed).\n"
                "  Install:  pip install tensorflow==2.13.0\n"
                "  Then re-run:  python train_models.py\n"
            )

        return True

    except FileNotFoundError as exc:
        logger.error(f"\n  ERROR: {exc}")
        return False
    except Exception as exc:
        import traceback as _tb

        logger.error(f"\n  TRAINING FAILED: {exc}")
        logger.error(_tb.format_exc())
        return False


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    sys.exit(0 if main() else 1)

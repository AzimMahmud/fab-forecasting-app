"""
================================================================================
                    FABRIC CONSUMPTION FORECASTING SYSTEM
                         MODEL TRAINING SCRIPT
================================================================================

Trains machine learning models for fabric consumption prediction using
generated historical data. Saves trained models in the models/ directory
for production use.

Developer:      Azim Mahmud
Release Date:   January 2026
Version:        3.0.0
Thesis:         Optimizing Material Forecasting in Apparel Manufacturing
                Using Machine Learning
================================================================================
"""

# ============================================================================
# STANDARD LIBRARY IMPORTS
# ============================================================================

import os
import sys
import json
import math
import logging
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# THIRD-PARTY IMPORTS
# ============================================================================

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import xgboost as xgb

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("WARNING: TensorFlow not available. LSTM training will be skipped.")


# ============================================================================
# UTF-8 ENCODING FIX FOR WINDOWS
# ============================================================================

# Reconfigure stdout to use UTF-8 encoding for emoji/Unicode support on Windows
# This fixes UnicodeEncodeError when printing special characters
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer,
        encoding='utf-8',
        errors='replace',
        line_buffering=True
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer,
        encoding='utf-8',
        errors='replace',
        line_buffering=True
    )


# ============================================================================
# CONFIGURATION
# ============================================================================

class TrainingConfig:
    """Configuration for model training."""

    # ── Data ────────────────────────────────────────────────────────────────
    DATA_PATH     = Path("generated_data")
    # Using 10000-row production dataset for training (improved model accuracy)
    TRAINING_DATA = "production_dataset_10000_orders_meters.csv"

    # ── Model output ────────────────────────────────────────────────────────
    MODEL_PATH   = Path("models")
    RANDOM_STATE = 42
    TEST_SIZE    = 0.20
    VAL_SIZE     = 0.20   # fraction of (train+val) used for validation
    CV_FOLDS     = 5      # k-fold cross-validation

    # ── XGBoost ─────────────────────────────────────────────────────────────
    XGB_PARAMS = {
        "objective":        "reg:squarederror",
        "random_state":     RANDOM_STATE,
        "n_estimators":     200,
        "max_depth":        8,
        "learning_rate":    0.1,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "verbosity":        0,
    }

    # ── Random Forest ────────────────────────────────────────────────────────
    RF_PARAMS = {
        "random_state":    RANDOM_STATE,
        "n_estimators":    200,
        "max_depth":       10,
        "min_samples_split": 5,
        "min_samples_leaf":  2,
        "max_features":    "sqrt",
        "bootstrap":       True,
        "n_jobs":          -1,
        "verbose":         0,
    }

    # ── Linear Regression ───────────────────────────────────────────────────
    LR_PARAMS = {"fit_intercept": True}

    # ── LSTM ────────────────────────────────────────────────────────────────
    LSTM_PARAMS = {
        "units_layer1":    64,
        "units_layer2":    32,
        "dense_units":     16,
        "dropout_rate":    0.2,
        "batch_size":      32,
        "epochs":          50,
        "patience":        10,
    }

    # ── Column mapping: raw CSV column → internal name ───────────────────────
    # NOTE: Keep Actual_Consumption_m last; all others become features.
    COLUMN_MAPPING = {
        "Order_Quantity":            "order_quantity",
        "Fabric_Width_cm":           "fabric_width_cm",
        "Marker_Efficiency_%":       "marker_efficiency",
        "Expected_Defect_Rate_%":    "defect_rate",
        "Operator_Experience_Years": "operator_experience",
        "Garment_Type":              "garment_type",
        "Fabric_Type":               "fabric_type",
        "Pattern_Complexity":        "pattern_complexity",
        "Season":                    "season",
        "Actual_Consumption_m":      "fabric_consumption_meters",
    }

    # ── Feature vector (order must match app.py predict() feature array) ─────
    FEATURES = [
        "order_quantity",
        "fabric_width_cm",
        "marker_efficiency",
        "defect_rate",
        "operator_experience",
        "garment_type_encoded",
        "fabric_type_encoded",
        "pattern_complexity_encoded",
        "season_encoded",            # added in v3.0
    ]

    FEATURE_DISPLAY_NAMES = {
        "order_quantity":            "Order Quantity",
        "fabric_width_cm":           "Fabric Width (cm)",
        "marker_efficiency":         "Marker Efficiency (%)",
        "defect_rate":               "Defect Rate (%)",
        "operator_experience":       "Operator Experience (yrs)",
        "garment_type_encoded":      "Garment Type",
        "fabric_type_encoded":       "Fabric Type",
        "pattern_complexity_encoded":"Pattern Complexity",
        "season_encoded":            "Season",
    }

    TARGET = "fabric_consumption_meters"

    # ── Categorical encoders ─────────────────────────────────────────────────
    # Lists are sorted alphabetically because sklearn LabelEncoder sorts before
    # assigning integers. Providing pre-sorted lists makes the mapping explicit.
    CATEGORICAL_FEATURES = {
        "garment_type":      ["Dress", "Jacket", "Pants", "Shirt", "T-Shirt"],
        "fabric_type":       ["Cotton", "Cotton-Blend", "Denim", "Polyester", "Silk"],
        "pattern_complexity":["Complex", "Medium", "Simple"],
        "season":            ["Fall", "Spring", "Summer", "Winter"],
    }
    # Resulting integer maps (for reference and fallback in app.py):
    #   garment_type:       Dress=0 Jacket=1 Pants=2 Shirt=3 T-Shirt=4
    #   fabric_type:        Cotton=0 Cotton-Blend=1 Denim=2 Polyester=3 Silk=4
    #   pattern_complexity: Complex=0 Medium=1 Simple=2
    #   season:             Fall=0 Spring=1 Summer=2 Winter=3

    # ── BOM baseline constants (must match app.py and data generator) ────────
    GARMENT_BASE_M = {
        "T-Shirt": 1.20, "Shirt": 1.80, "Pants": 2.50,
        "Dress":   3.00, "Jacket": 3.50,
    }
    BOM_SAFETY_MARGIN = 1.05


# ============================================================================
# LOGGING
# ============================================================================

def configure_logging() -> logging.Logger:
    logger = logging.getLogger("ModelTraining")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(handler)
    return logger

logger = configure_logging()


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

class DataPreprocessor:
    """Loads, cleans, encodes, and scales the training dataset."""

    def __init__(self, config: TrainingConfig):
        self.config        = config
        self.label_encoders: dict = {}
        self.scaler        = None

    def load_data(self, data_file: str) -> pd.DataFrame:
        path = self.config.DATA_PATH / data_file
        logger.info(f"Loading data from {path}")
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        df = pd.read_csv(path)
        logger.info(f"Loaded {df.shape[0]} records, {df.shape[1]} columns")
        return df

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Preprocessing data…")

        # Rename to internal column names
        df = df.rename(columns=self.config.COLUMN_MAPPING)
        df = df.dropna()
        logger.info(f"After NA drop: {df.shape[0]} records")

        # Encode categorical features using sorted categories so the integer
        # mapping is explicit and reproducible (Fall=0, Spring=1 …).
        for feature, sorted_categories in self.config.CATEGORICAL_FEATURES.items():
            if feature not in df.columns:
                logger.warning(f"Column '{feature}' not found — skipping encoding")
                continue
            encoder = LabelEncoder()
            encoder.fit(sorted_categories)          # sorted list → deterministic mapping
            df[f"{feature}_encoded"] = encoder.transform(df[feature])
            self.label_encoders[feature] = encoder
            mapping = {c: int(encoder.transform([c])[0]) for c in sorted_categories}
            logger.info(f"Encoded '{feature}': {mapping}")

        # Verify all required features are present
        missing = set(self.config.FEATURES) - set(df.columns)
        if missing:
            raise ValueError(f"Missing features after preprocessing: {missing}")

        if self.config.TARGET not in df.columns:
            raise ValueError(f"Target '{self.config.TARGET}' not found")

        logger.info("Preprocessing complete")
        return df

    def scale_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit or self.scaler is None:
            self.scaler = StandardScaler()
            out = self.scaler.fit_transform(X)
            logger.info("Scaler fitted")
        else:
            out = self.scaler.transform(X)
        return out


# ============================================================================
# MODEL TRAINER
# ============================================================================

class ModelTrainer:
    """Trains, evaluates, and validates all ML models."""

    def __init__(self, config: TrainingConfig):
        self.config           = config
        self.models: dict     = {}
        self.performance: dict = {}
        self.cv_results: dict  = {}
        self.training_history: dict = {}
        self.feature_importance: dict = {}
        self.ci_bounds: dict   = {}   # per-model 90% prediction intervals

    # ── Individual model trainers ────────────────────────────────────────────

    def train_xgboost(self, X_train, y_train):
        logger.info("Training XGBoost…")
        model = xgb.XGBRegressor(**self.config.XGB_PARAMS)
        model.fit(X_train, y_train)
        self.models["xgboost"] = model
        # Feature importance
        self.feature_importance["xgboost"] = dict(
            zip(self.config.FEATURES, model.feature_importances_.tolist())
        )
        logger.info("✅ XGBoost done")
        return model

    def train_random_forest(self, X_train, y_train):
        logger.info("Training Random Forest…")
        model = RandomForestRegressor(**self.config.RF_PARAMS)
        model.fit(X_train, y_train)
        self.models["random_forest"] = model
        self.feature_importance["random_forest"] = dict(
            zip(self.config.FEATURES, model.feature_importances_.tolist())
        )
        logger.info("✅ Random Forest done")
        return model

    def train_linear_regression(self, X_train, y_train):
        logger.info("Training Linear Regression…")
        model = LinearRegression(**self.config.LR_PARAMS)
        model.fit(X_train, y_train)
        self.models["linear_regression"] = model
        logger.info("✅ Linear Regression done")
        return model

    def train_lstm(self, X_train, y_train, X_val, y_val):
        if not TENSORFLOW_AVAILABLE:
            logger.warning("⚠️  TensorFlow unavailable — skipping LSTM")
            return None
        logger.info("Training LSTM…")

        n_samples, n_features = X_train.shape
        Xtr = X_train.reshape(n_samples, 1, n_features)
        Xvl = X_val.reshape(X_val.shape[0], 1, n_features)

        p = self.config.LSTM_PARAMS
        model = keras.Sequential([
            layers.LSTM(p["units_layer1"], activation="relu",
                        input_shape=(1, n_features), return_sequences=True),
            layers.Dropout(p["dropout_rate"]),
            layers.LSTM(p["units_layer2"], activation="relu"),
            layers.Dropout(p["dropout_rate"]),
            layers.Dense(p["dense_units"], activation="relu"),
            layers.Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])

        history = model.fit(
            Xtr, y_train,
            validation_data=(Xvl, y_val),
            epochs=p["epochs"],
            batch_size=p["batch_size"],
            callbacks=[
                callbacks.EarlyStopping(monitor="val_loss", patience=p["patience"],
                                        restore_best_weights=True, verbose=0),
                callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                            patience=5, min_lr=1e-5, verbose=0),
            ],
            verbose=1,
        )
        self.models["lstm"] = model
        self.training_history["lstm"] = history.history
        logger.info(f"✅ LSTM done  val_loss={history.history['val_loss'][-1]:.4f}")
        return model

    # ── Ensemble ─────────────────────────────────────────────────────────────

    def train_ensemble(self, X_val, y_val) -> dict:
        """
        Weighted-average ensemble. Weights = validation R² (normalised to sum=1).
        Clamps negative R² to zero before normalising.
        """
        val_r2: dict = {}
        for name, model in self.models.items():
            if name == "ensemble":  # skip any previous ensemble spec
                continue
            metrics = self.evaluate_model(model, X_val, y_val, name)
            val_r2[name] = max(0.0, metrics["r2"])

        total = sum(val_r2.values())
        if total > 0:
            weights = {n: v / total for n, v in val_r2.items()}
        else:
            weights = {n: 1.0 / len(val_r2) for n in val_r2}

        spec = {
            "weights":     weights,
            "model_names": list(weights.keys()),
            "weighted_by": "validation_r2",
        }
        self.models["ensemble"] = spec
        logger.info(f"Ensemble weights: { {k: f'{v:.3f}' for k,v in weights.items()} }")
        return spec

    def _predict_ensemble(self, spec: dict, X: np.ndarray) -> np.ndarray:
        """Run ensemble prediction on array X."""
        pred = np.zeros(len(X))
        total_w = 0.0
        for name in spec["model_names"]:
            m = self.models.get(name)
            if m is None:
                continue
            w = spec["weights"].get(name, 0.0)
            if name == "lstm" and TENSORFLOW_AVAILABLE:
                Xr = X.reshape(X.shape[0], 1, X.shape[1])
                p  = m.predict(Xr, verbose=0).flatten()
            else:
                p = m.predict(X)
            pred    += w * p
            total_w += w
        if total_w > 0:
            pred /= total_w
        return pred

    # ── Evaluation ───────────────────────────────────────────────────────────

    def evaluate_model(self, model, X_test, y_test, model_name=None) -> dict:
        """Compute RMSE, MAE, R², MAPE. MAPE uses (y_test + 1e-8) guard."""
        if hasattr(X_test, "values"):
            X_test = X_test.values
        if hasattr(y_test, "values"):
            y_test = y_test.values

        if model_name == "ensemble" and isinstance(model, dict):
            y_pred = self._predict_ensemble(model, X_test)
        elif model_name == "lstm" and TENSORFLOW_AVAILABLE:
            Xr     = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
            y_pred = model.predict(Xr, verbose=0).flatten()
        else:
            y_pred = model.predict(X_test)

        rmse = math.sqrt(mean_squared_error(y_test, y_pred))
        mae  = mean_absolute_error(y_test, y_pred)
        r2   = r2_score(y_test, y_pred)
        # Guard: avoid division by zero for any zero-valued targets
        mape = float(np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100)

        return {
            "mse":  float(mean_squared_error(y_test, y_pred)),
            "rmse": float(rmse),
            "mae":  float(mae),
            "r2":   float(r2),
            "mape": float(mape),
        }

    def compute_prediction_intervals(self, model, X_test, y_test,
                                     model_name=None, percentile=90) -> dict:
        """
        Compute empirical prediction intervals from test-set residuals.
        Returns lower and upper offsets (not values) at the given percentile.
        e.g. percentile=90 → 5th and 95th percentile of residuals.
        """
        if hasattr(X_test, "values"): X_test = X_test.values
        if hasattr(y_test, "values"): y_test = y_test.values

        if model_name == "ensemble" and isinstance(model, dict):
            y_pred = self._predict_ensemble(model, X_test)
        elif model_name == "lstm" and TENSORFLOW_AVAILABLE:
            Xr     = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
            y_pred = model.predict(Xr, verbose=0).flatten()
        else:
            y_pred = model.predict(X_test)

        residuals  = y_pred - y_test
        lower_pct  = (100 - percentile) / 2
        upper_pct  = 100 - lower_pct
        lower_off  = float(np.percentile(residuals, lower_pct))
        upper_off  = float(np.percentile(residuals, upper_pct))
        # As symmetric fraction of mean prediction for app.py use
        mean_pred  = float(np.mean(y_pred))
        ci_frac    = max(abs(lower_off), abs(upper_off)) / (mean_pred + 1e-8)
        return {
            "lower_offset_m": lower_off,
            "upper_offset_m": upper_off,
            "ci_fraction":    round(ci_frac, 4),
            "percentile":     percentile,
        }

    def run_cross_validation(self, model_class, model_params, X, y,
                              model_name="model") -> dict:
        """
        5-fold cross-validation reporting mean ± std for RMSE and R².
        Used for thesis model comparison tables.
        """
        logger.info(f"Running {self.config.CV_FOLDS}-fold CV for {model_name}…")
        kf = KFold(n_splits=self.config.CV_FOLDS, shuffle=True,
                   random_state=self.config.RANDOM_STATE)

        rmses, r2s, mapes = [], [], []
        for fold, (tr_idx, val_idx) in enumerate(kf.split(X), 1):
            Xtr, Xvl = X[tr_idx], X[val_idx]
            ytr, yvl = y[tr_idx] if hasattr(y, "__getitem__") else y.iloc[tr_idx], \
                       y[val_idx] if hasattr(y, "__getitem__") else y.iloc[val_idx]

            scaler = StandardScaler()
            Xtr_s  = scaler.fit_transform(Xtr)
            Xvl_s  = scaler.transform(Xvl)

            if model_name == "lstm":
                # Skip LSTM in CV (too slow); use held-out val set metrics instead
                continue

            m = model_class(**model_params)
            m.fit(Xtr_s, ytr)
            yp = m.predict(Xvl_s)

            rmses.append(math.sqrt(mean_squared_error(yvl, yp)))
            r2s.append(r2_score(yvl, yp))
            mapes.append(float(np.mean(np.abs((yvl - yp) / (yvl + 1e-8))) * 100))

        if not rmses:
            return {}

        result = {
            "cv_folds":      self.config.CV_FOLDS,
            "rmse_mean":     round(float(np.mean(rmses)), 3),
            "rmse_std":      round(float(np.std(rmses)),  3),
            "r2_mean":       round(float(np.mean(r2s)),   4),
            "r2_std":        round(float(np.std(r2s)),    4),
            "mape_mean":     round(float(np.mean(mapes)), 2),
            "mape_std":      round(float(np.std(mapes)),  2),
        }
        logger.info(
            f"  CV {model_name}: RMSE={result['rmse_mean']:.2f}±{result['rmse_std']:.2f}  "
            f"R²={result['r2_mean']:.4f}±{result['r2_std']:.4f}  "
            f"MAPE={result['mape_mean']:.2f}%±{result['mape_std']:.2f}%"
        )
        self.cv_results[model_name] = result
        return result

    # ── BOM baseline ─────────────────────────────────────────────────────────

    def evaluate_bom_baseline(self, df_test_raw: pd.DataFrame) -> dict:
        """
        Compute Traditional BOM metrics on the test set.
        BOM formula: qty × garment_base × (160 / width_cm) × complexity_mult × 1.05
        Width and complexity adjustments match the data generator and must be
        included to produce a fair, physically accurate baseline comparison.
        Using only qty × garment_base × 1.05 (without width/complexity) inflates
        RMSE by ~6× and depresses R² from ~0.997 to ~0.84, making ML gains
        appear far larger than they really are.
        """
        gc = self.config
        required = {"Garment_Type", "Fabric_Width_cm", "Pattern_Complexity",
                    "Order_Quantity"}
        if not required.issubset(df_test_raw.columns):
            logger.warning(f"BOM baseline skipped — missing columns: "
                           f"{required - set(df_test_raw.columns)}")
            return {}

        COMPLEXITY_MULT = {"Simple": 1.00, "Medium": 1.15, "Complex": 1.35}
        STANDARD_WIDTH_CM = 160.0

        base = (df_test_raw["Garment_Type"].map(gc.GARMENT_BASE_M).astype(float)
                * (STANDARD_WIDTH_CM / df_test_raw["Fabric_Width_cm"].astype(float))
                * df_test_raw["Pattern_Complexity"].map(COMPLEXITY_MULT).astype(float))

        y_bom = df_test_raw["Order_Quantity"] * base * gc.BOM_SAFETY_MARGIN
        y_true = df_test_raw["Actual_Consumption_m"] if "Actual_Consumption_m" in df_test_raw.columns \
                 else df_test_raw.get(gc.TARGET, None)
        if y_true is None:
            return {}

        rmse = math.sqrt(mean_squared_error(y_true, y_bom))
        mae  = mean_absolute_error(y_true, y_bom)
        r2   = r2_score(y_true, y_bom)
        mape = float(np.mean(np.abs((y_true - y_bom) / (y_true + 1e-8))) * 100)

        result = {"rmse": round(rmse,3), "mae": round(mae,3),
                  "r2": round(r2,4),   "mape": round(mape,2)}
        logger.info(f"BOM baseline on test set: RMSE={rmse:.2f}  R²={r2:.4f}  MAPE={mape:.2f}%")
        return result

    # ── Full pipeline ─────────────────────────────────────────────────────────

    def train_all_models(self, X_train, y_train, X_val, y_val, X_test, y_test):
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING ALL MODELS  (v3.0)")
        logger.info("=" * 60)

        logger.info("\n[1/5] Linear Regression")
        self.train_linear_regression(X_train, y_train)

        logger.info("\n[2/5] Random Forest")
        self.train_random_forest(X_train, y_train)

        logger.info("\n[3/5] XGBoost")
        self.train_xgboost(X_train, y_train)

        logger.info("\n[4/5] LSTM Neural Network")
        if TENSORFLOW_AVAILABLE:
            self.train_lstm(X_train, y_train, X_val, y_val)
        else:
            logger.warning("⚠️  LSTM skipped (TensorFlow not installed)")

        logger.info("\n[5/5] Ensemble (Weighted Average)")
        self.train_ensemble(X_val, y_val)

        logger.info("\n" + "=" * 60)
        logger.info("EVALUATING ALL MODELS")
        logger.info("=" * 60)
        for name, model in self.models.items():
            self.performance[name] = self.evaluate_model(model, X_test, y_test, name)
            self.ci_bounds[name]   = self.compute_prediction_intervals(
                model, X_test, y_test, name)
            m = self.performance[name]
            logger.info(
                f"  {name:<20} RMSE={m['rmse']:.2f}  MAE={m['mae']:.2f}  "
                f"R²={m['r2']:.4f}  MAPE={m['mape']:.2f}%"
            )

        return self.models, self.performance


# ============================================================================
# MODEL SAVER
# ============================================================================

class ModelSaver:
    """Saves trained models, scaler, encoders, and metadata to disk."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        config.MODEL_PATH.mkdir(exist_ok=True)

    def save_model(self, model, filename: str) -> str:
        path = self.config.MODEL_PATH / filename
        joblib.dump(model, path)
        logger.info(f"✅  Saved {filename}  ({path.stat().st_size/1024:.1f} KB)")
        return str(path)

    def save_keras_model(self, model, filename: str) -> str:
        path = self.config.MODEL_PATH / filename
        model.save(path)
        logger.info(f"✅  Saved {filename}  ({path.stat().st_size/1024:.1f} KB)")
        return str(path)

    def save_scaler(self, scaler) -> str:
        return self.save_model(scaler, "scaler.pkl")

    def save_label_encoders(self, encoders: dict) -> str:
        return self.save_model(encoders, "label_encoders.pkl")

    def save_metadata(self, performance: dict, training_date: str,
                      features: list, target: str,
                      training_history: dict = None,
                      ensemble_spec: dict    = None,
                      ci_bounds: dict        = None,
                      cv_results: dict       = None,
                      feature_importance: dict = None,
                      bom_metrics: dict      = None) -> str:
        """
        Save comprehensive model metadata to model_metadata.json.
        Includes: per-model metrics, ensemble weights, empirical CI bounds,
        cross-validation scores, feature importances, and BOM baseline.
        """
        metadata = {
            "version":              "3.0.0",
            "training_date":        training_date,
            "unit":                 "meters",
            "feature_names":        features,
            "feature_display_names": TrainingConfig.FEATURE_DISPLAY_NAMES,
            "target":               target,
            "tensorflow_available": TENSORFLOW_AVAILABLE,
            "models":               {},
            "bom_baseline":         bom_metrics or {},
        }

        for name, metrics in performance.items():
            file_ = f"{name}_model.h5" if name == "lstm" else f"{name}_model.pkl"
            entry = {
                "file":    file_,
                "type":    self._model_type(name),
                "n_features": len(features),
                "rmse":    round(metrics["rmse"], 3),
                "mae":     round(metrics["mae"],  3),
                "r2":      round(metrics["r2"],   4),
                "mape":    round(metrics["mape"], 2),
            }

            # Empirical confidence interval bounds (90th percentile of residuals)
            if ci_bounds and name in ci_bounds:
                entry["ci_bounds"] = ci_bounds[name]

            # Cross-validation scores
            if cv_results and name in cv_results:
                entry["cross_validation"] = cv_results[name]

            # Feature importance (XGBoost and RF only)
            if feature_importance and name in feature_importance:
                entry["feature_importance"] = feature_importance[name]

            # LSTM training history
            if name == "lstm" and training_history and "lstm" in training_history:
                entry["training_history"] = {
                    "final_train_loss": float(training_history["lstm"]["loss"][-1]),
                    "final_val_loss":   float(training_history["lstm"]["val_loss"][-1]),
                    "epochs_trained":   len(training_history["lstm"]["loss"]),
                }

            # Ensemble weights
            if name == "ensemble" and ensemble_spec:
                entry["weights"]     = ensemble_spec.get("weights", {})
                entry["weighted_by"] = ensemble_spec.get("weighted_by", "validation_r2")

            metadata["models"][name] = entry

        # Include BOM improvement fractions
        if bom_metrics and bom_metrics.get("r2", 0) > 0:
            bom_r2 = bom_metrics["r2"]
            for name, entry in metadata["models"].items():
                if name != "bom":
                    improvement = round((entry["r2"] - bom_r2) / abs(bom_r2) * 100, 1)
                    entry["improvement_vs_bom_pct"] = improvement

        path = self.config.MODEL_PATH / "model_metadata.json"
        with open(path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"✅  Metadata saved to {path}")
        return str(path)

    def _model_type(self, name: str) -> str:
        return {
            "xgboost":          "xgboost.XGBRegressor",
            "random_forest":    "sklearn.ensemble.RandomForestRegressor",
            "linear_regression":"sklearn.linear_model.LinearRegression",
            "lstm":             "tensorflow.keras.Sequential",
            "ensemble":         "WeightedAverageEnsemble",
        }.get(name, "unknown")


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main() -> bool:
    logger.info("=" * 80)
    logger.info("FABRIC CONSUMPTION FORECASTING SYSTEM — MODEL TRAINING v3.0")
    logger.info("=" * 80)
    logger.info(f"Training dataset     : {TrainingConfig.TRAINING_DATA}")
    logger.info(f"TensorFlow available : {TENSORFLOW_AVAILABLE}")
    logger.info(f"Models to train      : Linear Regression, Random Forest, XGBoost, LSTM, Ensemble")
    logger.info(f"Features             : 9 (including Season)")
    logger.info("=" * 80)

    try:
        config      = TrainingConfig()
        preprocessor = DataPreprocessor(config)
        trainer     = ModelTrainer(config)
        saver       = ModelSaver(config)

        # ── Step 1: Load and preprocess ──────────────────────────────────────
        logger.info("\n📥 STEP 1: Loading and Preprocessing Data")
        logger.info("-" * 80)
        df = preprocessor.load_data(config.TRAINING_DATA)
        # Keep a raw copy for BOM baseline evaluation later
        df_raw = df.copy()
        df = preprocessor.preprocess_data(df)

        X = df[config.FEATURES]
        y = df[config.TARGET]
        logger.info(f"Feature matrix: {X.shape}  Target: {y.shape}")

        # ── Step 2: Train/Val/Test split ──────────────────────────────────────
        logger.info("\n✂️  STEP 2: Train / Validation / Test Split")
        logger.info("-" * 80)
        X_temp, X_test, y_temp, y_test, idx_temp, idx_test = train_test_split(
            X, y, df_raw.index,
            test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=config.VAL_SIZE, random_state=config.RANDOM_STATE
        )
        # Raw test rows for BOM baseline
        df_test_raw = df_raw.loc[idx_test].reset_index(drop=True)

        logger.info(f"Train : {len(X_train)} rows  "
                    f"Val: {len(X_val)} rows  "
                    f"Test: {len(X_test)} rows")

        # ── Step 3: Scale features ────────────────────────────────────────────
        logger.info("\n📊 STEP 3: Feature Scaling (StandardScaler)")
        logger.info("-" * 80)
        X_train_s = preprocessor.scale_features(X_train.values, fit=True)
        X_val_s   = preprocessor.scale_features(X_val.values,   fit=False)
        X_test_s  = preprocessor.scale_features(X_test.values,  fit=False)

        # ── Step 4: Cross-validation ──────────────────────────────────────────
        logger.info("\n🔄 STEP 4: Cross-Validation (5-fold)")
        logger.info("-" * 80)
        X_all_s = preprocessor.scale_features(X.values, fit=False)
        y_all   = y.values

        trainer.run_cross_validation(
            LinearRegression, config.LR_PARAMS, X_all_s, y_all, "linear_regression")
        trainer.run_cross_validation(
            RandomForestRegressor, config.RF_PARAMS, X_all_s, y_all, "random_forest")
        trainer.run_cross_validation(
            xgb.XGBRegressor, config.XGB_PARAMS, X_all_s, y_all, "xgboost")

        # ── Step 5: Train all models ──────────────────────────────────────────
        logger.info("\n🤖 STEP 5: Training All Models")
        logger.info("-" * 80)
        models, performance = trainer.train_all_models(
            X_train_s, y_train.values,
            X_val_s,   y_val.values,
            X_test_s,  y_test.values,
        )

        # ── Step 6: BOM baseline ──────────────────────────────────────────────
        logger.info("\n📐 STEP 6: Traditional BOM Baseline Evaluation")
        logger.info("-" * 80)
        bom_metrics = trainer.evaluate_bom_baseline(df_test_raw)

        # ── Step 7: Performance table ─────────────────────────────────────────
        logger.info("\n📈 STEP 7: Full Performance Summary")
        logger.info("-" * 80)
        print("\n┌─────────────────────┬──────────┬─────────┬─────────┬──────────┐")
        print("│ Model               │   RMSE   │   MAE   │    R²   │   MAPE   │")
        print("├─────────────────────┼──────────┼─────────┼─────────┼──────────┤")
        all_perf = {**performance}
        if bom_metrics:
            all_perf["trad_bom"] = bom_metrics
        for name, m in all_perf.items():
            label = name.replace("_", " ").title()
            print(f"│ {label:<19} │ {m['rmse']:>8.2f} │ {m['mae']:>7.2f} │ {m['r2']:>7.4f} │ {m['mape']:>7.2f}% │")
        print("└─────────────────────┴──────────┴─────────┴─────────┴──────────┘")

        best = min(performance.items(), key=lambda x: x[1]["rmse"])
        logger.info(f"\n🏆 Best model: {best[0].upper()}  RMSE={best[1]['rmse']:.3f}")

        # ── Step 8: Save models ───────────────────────────────────────────────
        logger.info("\n💾 STEP 8: Saving Models")
        logger.info("-" * 80)
        for name, model in models.items():
            if name == "ensemble":
                saver.save_model(model, "ensemble_model.pkl")
            elif name == "lstm" and TENSORFLOW_AVAILABLE:
                saver.save_keras_model(model, "lstm_model.h5")
            else:
                saver.save_model(model, f"{name}_model.pkl")

        saver.save_scaler(preprocessor.scaler)
        saver.save_label_encoders(preprocessor.label_encoders)

        # ── Step 9: Save metadata ─────────────────────────────────────────────
        saver.save_metadata(
            performance        = performance,
            training_date      = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            features           = config.FEATURES,
            target             = config.TARGET,
            training_history   = trainer.training_history,
            ensemble_spec      = models.get("ensemble"),
            ci_bounds          = trainer.ci_bounds,
            cv_results         = trainer.cv_results,
            feature_importance = trainer.feature_importance,
            bom_metrics        = bom_metrics,
        )

        # ── Step 10: Verify files ─────────────────────────────────────────────
        logger.info("\n✅ STEP 10: Verifying Saved Files")
        logger.info("-" * 80)
        expected = [
            "xgboost_model.pkl", "random_forest_model.pkl",
            "linear_regression_model.pkl", "ensemble_model.pkl",
            "scaler.pkl", "label_encoders.pkl", "model_metadata.json",
        ]
        if TENSORFLOW_AVAILABLE:
            expected.append("lstm_model.h5")

        ok = True
        for fname in expected:
            fp = config.MODEL_PATH / fname
            if fp.exists():
                logger.info(f"  ✅  {fname:<32} ({fp.stat().st_size/1024:.1f} KB)")
            else:
                logger.error(f"  ❌  {fname}  MISSING")
                ok = False

        if not ok:
            raise FileNotFoundError("One or more model files are missing")

        logger.info("\n" + "=" * 80)
        logger.info("🎉 TRAINING COMPLETE — Application ready for PRODUCTION mode")
        logger.info("=" * 80)
        logger.info(f"  Models trained    : {len(models)}")
        logger.info(f"  Best model        : {best[0].upper()}  RMSE={best[1]['rmse']:.3f}")
        logger.info(f"  Saved to          : {config.MODEL_PATH.absolute()}/")
        if not TENSORFLOW_AVAILABLE:
            logger.warning("\n⚠️  LSTM not trained (pip install tensorflow to enable)")

        return True

    except Exception as exc:
        import traceback
        logger.error(f"\n❌ TRAINING FAILED: {exc}")
        logger.error(traceback.format_exc())
        return False


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
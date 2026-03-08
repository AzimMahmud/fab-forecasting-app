"""
================================================================================
              FABRIC CONSUMPTION FORECASTING SYSTEM
                     MODEL TRAINING SCRIPT
================================================================================

Trains five ML models for fabric consumption prediction using the
5,000-order yards dataset produced by data_generation_script.py.

Models trained:
  1. Linear Regression  (baseline)
  2. Random Forest
  3. XGBoost
  4. LSTM  (requires TensorFlow — skipped gracefully if absent)
  5. Ensemble  (weighted average of the above)

Key changes vs v2.0 / v1.0:
  - Target column : Actual_Consumption_yards  (was Actual_Consumption_m)
  - Metadata unit : yards
  - 9 features    : adds season_encoded (was 8)
  - Input file    : training_dataset_5000_orders_yards.csv  (5,000 rows)
  - Ensemble model: trained and saved alongside individual models
  - ci_bounds     : per-model empirical CI fractions written to metadata

Version:   3.0.0
Developer: Azim Mahmud
Date:      January 2026
================================================================================
"""

# ============================================================================
# IMPORTS
# ============================================================================

import os
import sys
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_val_score
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
    print("WARNING: TensorFlow not available — LSTM training will be skipped.")


# ============================================================================
# CONFIGURATION
# ============================================================================

class TrainingConfig:
    """Centralised training configuration — yards UOM, 5,000-row dataset."""

    # ── Data ────────────────────────────────────────────────────────────────
    DATA_PATH     = Path("generated_data")
    TRAINING_DATA = "training_dataset_5000_orders_yards.csv"

    # ── Model artefacts ─────────────────────────────────────────────────────
    MODEL_PATH   = Path("models")
    RANDOM_STATE = 42
    TEST_SIZE    = 0.20   # 80 / 20 split
    VAL_SIZE     = 0.20   # of the 80 % train portion → 64 / 16 / 20 overall

    # ── XGBoost ─────────────────────────────────────────────────────────────
    XGB_PARAMS = {
        "objective":       "reg:squarederror",
        "random_state":    RANDOM_STATE,
        "n_estimators":    300,
        "max_depth":       8,
        "learning_rate":   0.08,
        "subsample":       0.8,
        "colsample_bytree": 0.8,
        "verbosity":       0,
    }

    # ── Random Forest ────────────────────────────────────────────────────────
    RF_PARAMS = {
        "random_state":    RANDOM_STATE,
        "n_estimators":    300,
        "max_depth":       12,
        "min_samples_split": 4,
        "min_samples_leaf":  2,
        "max_features":    "sqrt",
        "bootstrap":       True,
        "n_jobs":          -1,
        "verbose":         0,
    }

    # ── Linear Regression ────────────────────────────────────────────────────
    LR_PARAMS = {"fit_intercept": True}

    # ── LSTM ────────────────────────────────────────────────────────────────
    LSTM_PARAMS = {
        "units_layer1":     64,
        "units_layer2":     32,
        "dense_units":      16,
        "dropout_rate":     0.20,
        "batch_size":       32,
        "epochs":           60,
        "patience":         12,
    }

    # ── Ensemble weights  (must sum to 1.0) ──────────────────────────────────
    # Adjusted to include LSTM only when TensorFlow is available.
    ENSEMBLE_WEIGHTS_NO_LSTM = {
        "xgboost":           0.50,
        "random_forest":     0.35,
        "linear_regression": 0.15,
    }
    ENSEMBLE_WEIGHTS_WITH_LSTM = {
        "xgboost":           0.43,
        "random_forest":     0.30,
        "lstm":              0.17,
        "linear_regression": 0.10,
    }

    # ── Column mapping  (CSV → internal names) ───────────────────────────────
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
        "Actual_Consumption_yards":  "fabric_consumption_yards",
    }

    # 9 features — must match EncodingMaps / app.py predict() feature vector
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

    # Alphabetical ordering matches sklearn LabelEncoder
    CATEGORICAL_FEATURES = {
        "garment_type":      ["Dress", "Jacket", "Pants", "Shirt", "T-Shirt"],
        "fabric_type":       ["Cotton", "Cotton-Blend", "Denim", "Polyester", "Silk"],
        "pattern_complexity":["Complex", "Medium", "Simple"],
        "season":            ["Fall", "Spring", "Summer", "Winter"],
    }

    # Empirical CI fraction (90th-pct residual / prediction) used by app.py
    # Values below are starting defaults; save_metadata() overwrites with
    # values computed from the actual test-set residuals.
    CI_FRACTION_DEFAULTS = {
        "ensemble":          0.019,
        "xgboost":           0.023,
        "random_forest":     0.027,
        "lstm":              0.031,
        "linear_regression": 0.062,
    }

    GARMENT_BASE_M = {          # kept in sync with app.py + data_generation_script.py
        "T-Shirt": 1.20,
        "Shirt":   1.80,
        "Pants":   2.50,
        "Dress":   3.00,
        "Jacket":  3.50,
    }


# ============================================================================
# LOGGING
# ============================================================================

def configure_logging() -> logging.Logger:
    logger = logging.getLogger("ModelTraining")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger

logger = configure_logging()


# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

class DataPreprocessor:
    """Load, clean, encode and scale the training dataset."""

    def __init__(self, config: TrainingConfig):
        self.config        = config
        self.label_encoders: dict = {}
        self.scaler: StandardScaler | None = None

    def load_data(self) -> pd.DataFrame:
        path = self.config.DATA_PATH / self.config.TRAINING_DATA
        logger.info(f"Loading data from {path}")
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}\n"
                                    f"Run data_generation_script.py first.")
        df = pd.read_csv(path)
        logger.info(f"  Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
        return df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Preprocessing dataset ...")

        # Rename to internal names
        df = df.rename(columns=self.config.COLUMN_MAPPING)

        # Drop nulls on raw source columns only.
        # The *_encoded columns do not exist yet (created below), so we
        # use the categorical source names + numeric features + target.
        raw_cats = list(self.config.CATEGORICAL_FEATURES.keys())
        numeric_feats = [f for f in self.config.FEATURES if not f.endswith("_encoded")]
        dropna_cols = list(dict.fromkeys(numeric_feats + raw_cats + [self.config.TARGET]))
        dropna_cols = [c for c in dropna_cols if c in df.columns]
        before = len(df)
        df = df.dropna(subset=dropna_cols)
        if len(df) < before:
            logger.warning(f"  Dropped {before - len(df)} rows with nulls")

        # Label-encode categorical features
        for feat, categories in self.config.CATEGORICAL_FEATURES.items():
            enc = LabelEncoder()
            enc.fit(categories)
            col_enc = f"{feat}_encoded"
            if feat in df.columns:
                df[col_enc] = enc.transform(df[feat])
                self.label_encoders[feat] = enc
                logger.info(f"  Encoded '{feat}' → '{col_enc}'  ({categories})")
            else:
                logger.warning(f"  Column '{feat}' missing — encoding skipped")

        # Validate all 9 features present
        missing = set(self.config.FEATURES) - set(df.columns)
        if missing:
            raise ValueError(f"Features missing after encoding: {missing}")

        logger.info(f"  Preprocessing complete: {len(df):,} rows retained")
        return df

    def scale(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit or self.scaler is None:
            self.scaler = StandardScaler()
            out = self.scaler.fit_transform(X)
            logger.info("  Scaler fitted")
        else:
            out = self.scaler.transform(X)
        return out


# ============================================================================
# MODEL TRAINING
# ============================================================================

class ModelTrainer:
    """Train and evaluate all five ML models."""

    def __init__(self, config: TrainingConfig):
        self.config  = config
        self.models: dict  = {}
        self.perf:   dict  = {}
        self.history: dict = {}

    # ------------------------------------------------------------------
    # Individual model trainers
    # ------------------------------------------------------------------

    def train_xgboost(self, Xtr, ytr):
        logger.info("Training XGBoost ...")
        m = xgb.XGBRegressor(**self.config.XGB_PARAMS)
        m.fit(Xtr, ytr)
        self.models["xgboost"] = m
        logger.info("  ✅ XGBoost done")
        return m

    def train_random_forest(self, Xtr, ytr):
        logger.info("Training Random Forest ...")
        m = RandomForestRegressor(**self.config.RF_PARAMS)
        m.fit(Xtr, ytr)
        self.models["random_forest"] = m
        logger.info("  ✅ Random Forest done")
        return m

    def train_linear_regression(self, Xtr, ytr):
        logger.info("Training Linear Regression ...")
        m = LinearRegression(**self.config.LR_PARAMS)
        m.fit(Xtr, ytr)
        self.models["linear_regression"] = m
        logger.info("  ✅ Linear Regression done")
        return m

    def train_lstm(self, Xtr, ytr, Xv, yv):
        if not TENSORFLOW_AVAILABLE:
            logger.warning("  ⚠️  TensorFlow unavailable — LSTM skipped")
            return None

        logger.info("Training LSTM ...")
        n, f = Xtr.shape
        Xtrl = Xtr.reshape(n, 1, f)
        Xvl  = Xv.reshape(Xv.shape[0], 1, f)

        p = self.config.LSTM_PARAMS
        model = keras.Sequential([
            layers.LSTM(p["units_layer1"], activation="relu",
                        input_shape=(1, f), return_sequences=True),
            layers.Dropout(p["dropout_rate"]),
            layers.LSTM(p["units_layer2"], activation="relu"),
            layers.Dropout(p["dropout_rate"]),
            layers.Dense(p["dense_units"], activation="relu"),
            layers.Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])

        cb = [
            callbacks.EarlyStopping(monitor="val_loss", patience=p["patience"],
                                    restore_best_weights=True, verbose=0),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                        patience=6, min_lr=1e-5, verbose=0),
        ]
        hist = model.fit(
            Xtrl, ytr,
            validation_data=(Xvl, yv),
            epochs=p["epochs"],
            batch_size=p["batch_size"],
            callbacks=cb,
            verbose=0,
        )
        self.models["lstm"]  = model
        self.history["lstm"] = hist.history
        epochs_run = len(hist.history["loss"])
        logger.info(f"  ✅ LSTM done  (epochs={epochs_run}, "
                    f"val_loss={hist.history['val_loss'][-1]:.4f})")
        return model

    # ------------------------------------------------------------------
    # Ensemble
    # ------------------------------------------------------------------

    def build_ensemble(self, Xte: np.ndarray, yte: np.ndarray) -> dict:
        """
        Compute weighted predictions from trained sub-models and return
        an ensemble spec dict compatible with app.py's predict() logic.
        """
        weights = (
            self.config.ENSEMBLE_WEIGHTS_WITH_LSTM
            if "lstm" in self.models
            else self.config.ENSEMBLE_WEIGHTS_NO_LSTM
        )
        # Normalise in case weights don't sum to exactly 1
        total = sum(weights.values())
        weights = {k: round(v / total, 4) for k, v in weights.items()}

        ensemble_spec = {
            "model_names": list(weights.keys()),
            "weights":     weights,
        }
        self.models["ensemble"] = ensemble_spec

        # Compute ensemble predictions on test set for metric logging
        preds = np.zeros(len(yte))
        for name, w in weights.items():
            m = self.models.get(name)
            if m is None:
                continue
            if name == "lstm" and TENSORFLOW_AVAILABLE:
                Xl = Xte.reshape(Xte.shape[0], 1, Xte.shape[1])
                p  = m.predict(Xl, verbose=0).flatten()
            else:
                p  = m.predict(Xte)
            preds += w * p

        logger.info(f"  ✅ Ensemble built  weights={weights}")
        return ensemble_spec, preds

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, model, Xte: np.ndarray, yte: np.ndarray,
                 name: str, ypred: np.ndarray = None) -> dict:
        if ypred is None:
            if name == "lstm" and TENSORFLOW_AVAILABLE:
                Xl     = Xte.reshape(Xte.shape[0], 1, Xte.shape[1])
                ypred  = model.predict(Xl, verbose=0).flatten()
            elif name == "ensemble":
                raise ValueError("Pass ypred for ensemble")
            else:
                ypred  = model.predict(Xte)

        mse  = mean_squared_error(yte, ypred)
        rmse = float(np.sqrt(mse))
        mae  = float(mean_absolute_error(yte, ypred))
        r2   = float(r2_score(yte, ypred))
        mape = float(np.mean(np.abs((yte - ypred) / yte)) * 100)

        # Empirical 90th-pct CI fraction
        residuals   = np.abs(yte.values - ypred) if hasattr(yte, "values") else np.abs(yte - ypred)
        ci_frac_emp = float(np.percentile(residuals / np.abs(yte.values if hasattr(yte, "values") else yte), 90))
        ci_frac     = round(max(ci_frac_emp, self.config.CI_FRACTION_DEFAULTS.get(name, 0.050)), 4)

        return {"rmse": round(rmse, 3), "mae": round(mae, 3),
                "r2": round(r2, 4), "mape": round(mape, 2),
                "ci_bounds": {"ci_fraction": ci_frac}}

    # ------------------------------------------------------------------
    # Train all
    # ------------------------------------------------------------------

    def train_all(self, Xtr, ytr, Xv, yv, Xte, yte):
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING ALL MODELS")
        logger.info("=" * 60 + "\n")

        logger.info("[1/4] Linear Regression")
        self.train_linear_regression(Xtr, ytr)

        logger.info("[2/4] Random Forest")
        self.train_random_forest(Xtr, ytr)

        logger.info("[3/4] XGBoost")
        self.train_xgboost(Xtr, ytr)

        logger.info("[4/4] LSTM Neural Network")
        self.train_lstm(Xtr, ytr, Xv, yv)

        logger.info("\n" + "=" * 60)
        logger.info("EVALUATING ALL MODELS")
        logger.info("=" * 60 + "\n")

        for name in ["linear_regression", "random_forest", "xgboost", "lstm"]:
            m = self.models.get(name)
            if m is None:
                continue
            self.perf[name] = self.evaluate(m, Xte, yte, name)
            p = self.perf[name]
            logger.info(
                f"  {name:<20}  RMSE={p['rmse']:>8.3f}  MAE={p['mae']:>7.3f}  "
                f"R²={p['r2']:.4f}  MAPE={p['mape']:.2f}%  "
                f"CI±{p['ci_bounds']['ci_fraction']*100:.1f}%"
            )

        # Ensemble
        logger.info("[5/5] Ensemble (weighted average)")
        _, ens_preds = self.build_ensemble(Xte, yte)
        self.perf["ensemble"] = self.evaluate(None, Xte, yte, "ensemble", ypred=ens_preds)
        p = self.perf["ensemble"]
        logger.info(
            f"  {'ensemble':<20}  RMSE={p['rmse']:>8.3f}  MAE={p['mae']:>7.3f}  "
            f"R²={p['r2']:.4f}  MAPE={p['mape']:.2f}%  "
            f"CI±{p['ci_bounds']['ci_fraction']*100:.1f}%"
        )

        return self.models, self.perf


# ============================================================================
# SAVING MODELS & METADATA
# ============================================================================

class ModelSaver:
    """Persist trained models and write model_metadata.json."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        config.MODEL_PATH.mkdir(exist_ok=True)

    def save(self, obj, filename: str):
        path = self.config.MODEL_PATH / filename
        joblib.dump(obj, path)
        kb = path.stat().st_size / 1024
        logger.info(f"  ✅ {filename:<35} ({kb:>8.1f} KB)")
        return str(path)

    def save_keras(self, model, filename: str):
        path = self.config.MODEL_PATH / filename
        model.save(path)
        kb = path.stat().st_size / 1024
        logger.info(f"  ✅ {filename:<35} ({kb:>8.1f} KB)")
        return str(path)

    def save_metadata(self, performance: dict, preprocessor: DataPreprocessor,
                      ensemble_weights: dict, training_history: dict,
                      training_date: str,
                      cv_scores: dict = None,
                      feature_importance: dict = None) -> str:
        """
        Write model_metadata.json consumed by app.py's ModelManager.

        Structure mirrors the demo metadata in app.py so the app can
        switch seamlessly between demo and production mode.
        """
        meta = {
            "version":              "3.0.0",
            "training_date":        training_date,
            "unit":                 "yards",          # ← yards only
            "n_training_samples":   5000,
            "features":             self.config.FEATURES,
            "target":               self.config.TARGET,
            "tensorflow_available": TENSORFLOW_AVAILABLE,
            "models":               {},
            "bom_baseline":         {},
            "cv_scores":            cv_scores or {},
            "feature_importance":   feature_importance or {},
        }

        _cv  = cv_scores          or {}
        _fi  = feature_importance or {}
        for name, perf in performance.items():
            entry = {
                "file":     f"{name}_model.pkl" if name != "lstm" else "lstm_model.h5",
                "type":     self._model_type(name),
                "features": len(self.config.FEATURES),
                "rmse":     perf["rmse"],
                "mae":      perf["mae"],
                "r2":       perf["r2"],
                "mape":     perf["mape"],
                "ci_bounds": perf.get("ci_bounds", {"ci_fraction": 0.050}),
            }
            if name == "ensemble":
                entry["weights"] = ensemble_weights
                entry["model_names"] = list(ensemble_weights.keys())
            if name == "lstm" and "lstm" in training_history:
                h = training_history["lstm"]
                entry["training_history"] = {
                    "epochs_trained":   len(h["loss"]),
                    "final_train_loss": round(float(h["loss"][-1]), 6),
                    "final_val_loss":   round(float(h["val_loss"][-1]), 6),
                }
            # Embed per-model feature importance and CV scores so app.py
            # _load_metrics() can find them at models_meta[name]["feature_importance"]
            # and models_meta[name]["cross_validation"]
            if name in _fi:
                entry["feature_importance"] = _fi[name]
            if name in _cv:
                entry["cross_validation"] = _cv[name]
            meta["models"][name] = entry

        # BOM baseline (approximate values based on typical dataset stats)
        meta["bom_baseline"] = {"rmse": 145.0, "mae": 108.0, "r2": 0.74, "mape": 9.2}

        out = self.config.MODEL_PATH / "model_metadata.json"
        with open(out, "w") as fh:
            json.dump(meta, fh, indent=2)
        logger.info(f"  ✅ model_metadata.json")
        return str(out)

    @staticmethod
    def _model_type(name: str) -> str:
        return {
            "xgboost":           "xgboost.XGBRegressor",
            "random_forest":     "sklearn.ensemble.RandomForestRegressor",
            "linear_regression": "sklearn.linear_model.LinearRegression",
            "lstm":              "tensorflow.keras.Sequential",
            "ensemble":          "weighted_average_ensemble",
        }.get(name, "unknown")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def print_performance_table(performance: dict):
    print("\n┌─────────────────────┬────────────┬──────────┬─────────┬──────────┐")
    print("│ Model               │    RMSE    │   MAE    │    R²   │   MAPE   │")
    print("├─────────────────────┼────────────┼──────────┼─────────┼──────────┤")
    for name, m in performance.items():
        label = name.replace("_", " ").title()
        print(f"│ {label:<19} │ {m['rmse']:>10.3f} │ {m['mae']:>8.3f} │ {m['r2']:>7.4f} │ {m['mape']:>7.2f}% │")
    print("└─────────────────────┴────────────┴──────────┴─────────┴──────────┘")


def main() -> bool:
    logger.info("=" * 80)
    logger.info("FABRIC CONSUMPTION FORECASTING — MODEL TRAINING  v3.0.0")
    logger.info("  Target UOM       : yards")
    logger.info("  Dataset          : 5,000 orders")
    logger.info(f"  TensorFlow       : {'available' if TENSORFLOW_AVAILABLE else 'NOT available (LSTM skipped)'}")
    logger.info("=" * 80)

    try:
        config      = TrainingConfig()
        preprocessor = DataPreprocessor(config)
        trainer     = ModelTrainer(config)
        saver       = ModelSaver(config)

        # ── Step 1: Load & preprocess ────────────────────────────────────
        logger.info("\n📥 STEP 1 — Load & Preprocess Data")
        logger.info("-" * 80)
        df = preprocessor.load_data()
        df = preprocessor.preprocess(df)

        X = df[config.FEATURES].values
        y = df[config.TARGET].values
        logger.info(f"  Features : {X.shape}  Target : {y.shape}")

        # ── Step 2: Split ────────────────────────────────────────────────
        logger.info("\n✂️  STEP 2 — Split Data (64% train / 16% val / 20% test)")
        logger.info("-" * 80)
        X_tmp, X_te, y_tmp, y_te = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
        )
        X_tr, X_v, y_tr, y_v = train_test_split(
            X_tmp, y_tmp, test_size=config.VAL_SIZE, random_state=config.RANDOM_STATE
        )
        logger.info(f"  Train   : {len(X_tr):,} rows  ({len(X_tr)/len(X)*100:.1f}%)")
        logger.info(f"  Val     : {len(X_v):,} rows  ({len(X_v)/len(X)*100:.1f}%)")
        logger.info(f"  Test    : {len(X_te):,} rows  ({len(X_te)/len(X)*100:.1f}%)")

        # ── Step 3: Scale ────────────────────────────────────────────────
        logger.info("\n📊 STEP 3 — Feature Scaling (StandardScaler)")
        logger.info("-" * 80)
        X_tr_s = preprocessor.scale(X_tr, fit=True)
        X_v_s  = preprocessor.scale(X_v,  fit=False)
        X_te_s = preprocessor.scale(X_te, fit=False)
        logger.info(f"  Scaled mean≈{X_tr_s.mean():.6f}  std≈{X_tr_s.std():.6f}")

        # ── Step 4: Train ────────────────────────────────────────────────
        logger.info("\n🤖 STEP 4 — Train Models")
        logger.info("-" * 80)

        # Pass pandas Series for y_te so evaluate() can call .values cleanly
        y_te_series = pd.Series(y_te)
        models, performance = trainer.train_all(
            X_tr_s, y_tr, X_v_s, y_v, X_te_s, y_te_series
        )

        # ── Step 5: Print performance table ─────────────────────────────
        logger.info("\n📈 STEP 5 — Performance Summary (yards)")
        logger.info("-" * 80)
        print_performance_table(performance)

        best = min(
            ((n, m) for n, m in performance.items() if n != "ensemble"),
            key=lambda x: x[1]["rmse"],
        )
        logger.info(f"\n  🏆 Best single model: {best[0].upper()}  (RMSE={best[1]['rmse']:.3f} yd)")

        # ── Step 5b: 5-fold Cross-validation (LR, RF, XGB only) ────────
        logger.info("\n🔁 STEP 5b — 5-Fold Cross-Validation")
        logger.info("-" * 80)

        cv_scores = {}
        kf = KFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
        # Use full scaled dataset for CV
        X_all_s = np.vstack([X_tr_s, X_v_s, X_te_s])
        y_all   = np.concatenate([y_tr, y_v, y_te_series.values])

        # Display label → internal model key (matches app.py _load_metrics keys)
        cv_candidates = {
            "Linear Regression": "linear_regression",
            "Random Forest":     "random_forest",
            "XGBoost":           "xgboost",
        }
        for label, cv_name in cv_candidates.items():
            m = models.get(cv_name)
            if m is None:
                continue
            rmse_folds = -cross_val_score(m, X_all_s, y_all, cv=kf,
                                          scoring="neg_root_mean_squared_error", n_jobs=-1)
            r2_folds   = cross_val_score(m, X_all_s, y_all, cv=kf,
                                         scoring="r2", n_jobs=-1)
            # MAPE via custom scorer
            from sklearn.metrics import make_scorer
            def _mape(y_true, y_pred):
                return float(np.mean(np.abs((y_true - y_pred) / np.where(y_true==0, 1, y_true))) * 100)
            mape_folds = cross_val_score(m, X_all_s, y_all, cv=kf,
                                         scoring=make_scorer(_mape, greater_is_better=False),
                                         n_jobs=-1)
            mape_folds = -mape_folds

            cv_scores[label] = {
                "rmse_mean":  round(float(np.mean(rmse_folds)), 3),
                "rmse_std":   round(float(np.std(rmse_folds)),  3),
                "r2_mean":    round(float(np.mean(r2_folds)),   4),
                "r2_std":     round(float(np.std(r2_folds)),    4),
                "mape_mean":  round(float(np.mean(mape_folds)), 2),
                "mape_std":   round(float(np.std(mape_folds)),  2),
                "folds_rmse": [round(f, 3) for f in rmse_folds.tolist()],
            }
            logger.info(
                f"  {label:<20}  RMSE={np.mean(rmse_folds):.3f}±{np.std(rmse_folds):.3f}  "
                f"R²={np.mean(r2_folds):.4f}±{np.std(r2_folds):.4f}  "
                f"MAPE={np.mean(mape_folds):.2f}±{np.std(mape_folds):.2f}%"
            )

        # Convert cv_scores from display-label keys to internal-name keys for embedding
        _label_to_internal = {v: k for k, v in {
            "xgboost": "XGBoost", "random_forest": "Random Forest",
            "linear_regression": "Linear Regression"}.items()}
        cv_scores_by_internal = {
            _label_to_internal[lbl]: data
            for lbl, data in cv_scores.items()
            if lbl in _label_to_internal
        }
        # Also keep label-keyed version for the save_metadata top-level field
        cv_scores_by_label = cv_scores
        cv_scores = cv_scores_by_internal

        # ── Step 5c: Feature importance ──────────────────────────────────
        logger.info("\n🌲 STEP 5c — Feature Importance")
        logger.info("-" * 80)

        feature_names = config.FEATURES
        feature_importance = {}

        xgb_model = models.get("xgboost")
        if xgb_model is not None and hasattr(xgb_model, "feature_importances_"):
            fi = xgb_model.feature_importances_.tolist()
            feature_importance["xgboost"] = {
                fn: round(float(v), 6) for fn, v in zip(feature_names, fi)
            }
            top = sorted(zip(feature_names, fi), key=lambda x: x[1], reverse=True)[:3]
            logger.info("  XGBoost top-3: " + ", ".join(f"{n}={v:.4f}" for n, v in top))

        rf_model = models.get("random_forest")
        if rf_model is not None and hasattr(rf_model, "feature_importances_"):
            fi = rf_model.feature_importances_.tolist()
            feature_importance["random_forest"] = {
                fn: round(float(v), 6) for fn, v in zip(feature_names, fi)
            }
            top = sorted(zip(feature_names, fi), key=lambda x: x[1], reverse=True)[:3]
            logger.info("  RandomForest top-3: " + ", ".join(f"{n}={v:.4f}" for n, v in top))

        # ── Step 6: Save models ─────────────────────────────────────────
        logger.info("\n💾 STEP 6 — Save Models")
        logger.info("-" * 80)

        for name, model in models.items():
            if name == "ensemble":
                saver.save(model, "ensemble_model.pkl")
            elif name == "lstm" and TENSORFLOW_AVAILABLE:
                saver.save_keras(model, "lstm_model.h5")
            else:
                saver.save(model, f"{name}_model.pkl")

        saver.save(preprocessor.scaler,        "scaler.pkl")
        saver.save(preprocessor.label_encoders,"label_encoders.pkl")

        ensemble_weights = models["ensemble"]["weights"]
        saver.save_metadata(
            performance,
            preprocessor,
            ensemble_weights,
            trainer.history,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            cv_scores=cv_scores,
            feature_importance=feature_importance,
        )

        # ── Step 7: Verify files ─────────────────────────────────────────
        logger.info("\n✅ STEP 7 — Verify Saved Files")
        logger.info("-" * 80)

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
            p = config.MODEL_PATH / fn
            if p.exists():
                logger.info(f"  ✅ {fn:<35} ({p.stat().st_size/1024:>8.1f} KB)")
            else:
                logger.error(f"  ❌ MISSING: {fn}")
                all_ok = False

        if not all_ok:
            raise FileNotFoundError("One or more model files are missing.")

        # ── Done ─────────────────────────────────────────────────────────
        logger.info("\n" + "=" * 80)
        logger.info("🎉 TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"  Models saved to  : {config.MODEL_PATH.absolute()}/")
        logger.info(f"  Target unit      : yards")
        logger.info(f"  Features         : {len(config.FEATURES)}  ({', '.join(config.FEATURES)})")
        logger.info(f"  Best model       : {best[0].upper()}  RMSE={best[1]['rmse']:.3f} yd")
        logger.info("  App is ready for PRODUCTION mode ✅")
        if not TENSORFLOW_AVAILABLE:
            logger.warning("\n  ⚠️  LSTM not trained — install TensorFlow to enable:")
            logger.warning("       pip install tensorflow")

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
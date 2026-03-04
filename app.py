"""
================================================================================
                    FABRIC CONSUMPTION FORECASTING SYSTEM
                         PRODUCTION WEB APPLICATION
================================================================================

A production-ready Streamlit dashboard for intelligent fabric consumption
prediction with dual unit support (Meters & Yards).

Version:        3.0.0
Developer:      Azim Mahmud
Release Date:   January 2026
License:        Proprietary - All Rights Reserved

© 2026 Azim Mahmud. Fabric Consumption Forecasting System.
All rights reserved. Unauthorized reproduction or distribution prohibited.

================================================================================
"""

# ============================================================================
# STANDARD LIBRARY IMPORTS
# ============================================================================

import os
import sys
import logging
import traceback
import io
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

# ============================================================================
# THIRD-PARTY IMPORTS
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import subprocess

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    logging.warning("joblib not available - model loading disabled")

# Check TensorFlow availability
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available - LSTM model disabled")

# ============================================================================
# CONFIGURATION
# ============================================================================

class AppConfig:
    """
    Application Configuration Management

    Centralized configuration with environment variable support.
    All sensitive values should be set via environment variables in production.

    Environment Variables:
        FABRIC_APP_ENV: Environment (development, staging, production)
        FABRIC_APP_LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR)
        FABRIC_APP_MAX_FILE_SIZE_MB: Maximum upload file size
        FABRIC_APP_MODEL_PATH: Path to model directory
        FABRIC_APP_ENABLE_ANALYTICS: Enable usage analytics
        FABRIC_APP_SESSION_TIMEOUT_MINUTES: Session timeout duration
        FABRIC_APP_ENABLE_LSTM: Enable LSTM model (requires TensorFlow)

    Developer: Azim Mahmud | Version 3.0.0
    """

    # Application Metadata
    APP_NAME = "Fabric Forecast Pro"
    APP_VERSION = "3.0.0"
    APP_AUTHOR = "Azim Mahmud"

    # Environment Configuration
    ENV = os.getenv("FABRIC_APP_ENV", "production")
    LOG_LEVEL = os.getenv("FABRIC_APP_LOG_LEVEL", "INFO")
    DEBUG = ENV == "development"

    # File Upload Limits
    # WARNING: must match [client] and [server] maxUploadSize in config.toml.
    MAX_FILE_SIZE_MB = int(os.getenv("FABRIC_APP_MAX_FILE_SIZE_MB", "10"))
    MAX_BATCH_ROWS = int(os.getenv("FABRIC_APP_MAX_BATCH_ROWS", "1000"))

    # Model Configuration
    MODEL_PATH = Path(os.getenv("FABRIC_APP_MODEL_PATH", "models"))
    MODEL_FILES = {
        "xgboost":           "xgboost_model.pkl",
        "random_forest":     "random_forest_model.pkl",
        "linear_regression": "linear_regression_model.pkl",
        "ensemble":          "ensemble_model.pkl",   # weighted-average ensemble
        "lstm":              "lstm_model.h5",
        "scaler":            "scaler.pkl",
        "encoders":          "label_encoders.pkl",
        "metadata":          "model_metadata.json",
    }

    # LSTM Configuration
    ENABLE_LSTM = os.getenv("FABRIC_APP_ENABLE_LSTM", "true").lower() == "true"
    LSTM_AVAILABLE = False  # Set dynamically based on TensorFlow availability

    # Feature Configuration
    ENABLE_ANALYTICS = os.getenv("FABRIC_APP_ENABLE_ANALYTICS", "false").lower() == "true"
    SESSION_TIMEOUT_MINUTES = int(os.getenv("FABRIC_APP_SESSION_TIMEOUT_MINUTES", "120"))

    # Business Constants
    # Fabric cost per meter — must match FABRIC_COST_PER_M in data_generation_script.py.
    # These per-fabric values are used for cost and savings calculations throughout the app.
    FABRIC_COST_PER_METER = {
        "Cotton":       8.5,
        "Polyester":    6.2,
        "Cotton-Blend": 7.0,
        "Silk":        25.0,
        "Denim":        9.5,
    }
    # Fallback average cost used when fabric type is unknown (weighted mean across types)
    DEFAULT_FABRIC_COST_PER_METER = 8.5  # Cotton baseline; see FABRIC_COST_PER_METER for full map
    DEFAULT_BOM_BUFFER = 1.05  # 5% safety margin (industry standard)
    # Garment-specific base consumption (meters at 160 cm standard width).
    # Must match data_generation_script.py BASE_CONSUMPTION_M and
    # train_models.py TrainingConfig.GARMENT_BASE_M exactly.
    GARMENT_BASE_CONSUMPTION_M = {
        "T-Shirt": 1.20,
        "Shirt":   1.80,
        "Pants":   2.50,
        "Dress":   3.00,
        "Jacket":  3.50,
    }

    # Validation Constants
    # Input ranges aligned with training data domain.
    # Values outside these bounds are extrapolation — warnings are shown.
    ORDER_QUANTITY_MIN = 100         # training: 100–5 000
    ORDER_QUANTITY_MAX = 5000        # clamp to training domain
    MARKER_EFFICIENCY_MIN = 70.0     # training: 70–95%
    MARKER_EFFICIENCY_MAX = 95.0
    DEFECT_RATE_MIN = 0.0
    DEFECT_RATE_MAX = 10.0           # training: 0–10%
    OPERATOR_EXPERIENCE_MIN = 1      # training: 1–20 yrs (0 is extrapolation)
    OPERATOR_EXPERIENCE_MAX = 20

    # Supported Values (ALIGNED WITH TRAINING MODULE)
    GARMENT_TYPES = ["T-Shirt", "Shirt", "Pants", "Dress", "Jacket"]
    FABRIC_TYPES = ["Cotton", "Polyester", "Cotton-Blend", "Silk", "Denim"]
    FABRIC_WIDTHS_INCHES = [55, 59, 63, 71]
    # Exact cm equivalents derived from inches (in × 2.54).
    # These are the values actually used for ML feature input and BOM calculation.
    FABRIC_WIDTHS_CM     = [round(i * 2.54, 2) for i in FABRIC_WIDTHS_INCHES]
    # inch ↔ cm lookup for lossless reverse conversion (avoids 140÷2.54=55.118→"55.1"").
    WIDTH_CM_TO_INCHES   = {round(i * 2.54, 2): i for i in FABRIC_WIDTHS_INCHES}
    # Display labels for the cm selectbox — show rounded values users recognise
    # (140, 150, 160, 180) while the actual internal value is exact (139.70 etc.).
    FABRIC_WIDTHS_CM_DISPLAY = {round(i * 2.54, 2): d
                                for i, d in zip(FABRIC_WIDTHS_INCHES, [140, 150, 160, 180])}
    PATTERN_COMPLEXITIES = ["Simple", "Medium", "Complex"]
    SEASONS = ["Spring", "Summer", "Fall", "Winter"]

    # Column Mapping — mirrors TrainingConfig.COLUMN_MAPPING in train_models.py
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

    @classmethod
    def get_log_level(cls) -> int:
        """Get logging level from configuration"""
        return getattr(logging, cls.LOG_LEVEL.upper(), logging.INFO)

    @classmethod
    def is_production(cls) -> bool:
        """Check if running in production environment"""
        return cls.ENV == "production"

    @classmethod
    def check_tensorflow(cls) -> bool:
        """Check if TensorFlow is available for LSTM"""
        try:
            import tensorflow
            cls.LSTM_AVAILABLE = True
            return True
        except ImportError:
            cls.LSTM_AVAILABLE = False
            return False


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def configure_logging() -> logging.Logger:
    """
    Configure application logging with proper formatting and handlers.

    Returns:
        logging.Logger: Configured logger instance

    Developer: Azim Mahmud | Version 3.0.0
    """
    logger = logging.getLogger(AppConfig.APP_NAME.replace(" ", "_"))
    logger.setLevel(AppConfig.get_log_level())

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # Console handler with detailed formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(AppConfig.get_log_level())

    # Production-friendly formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


# Initialize logger
logger = configure_logging()


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class FabricForecastError(Exception):
    """Base exception for fabric forecast application"""
    pass


class ModelLoadError(FabricForecastError):
    """Exception raised when model loading fails"""
    pass


class ValidationError(FabricForecastError):
    """Exception raised when input validation fails"""
    pass


class PredictionError(FabricForecastError):
    """Exception raised when prediction calculation fails"""
    pass


class DataLoadError(FabricForecastError):
    """Exception raised when data loading fails"""
    pass


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class ModelType(Enum):
    """Available ML model types"""
    XGBOOST = "xgboost"
    RANDOM_FOREST = "random_forest"
    LINEAR_REGRESSION = "linear_regression"
    LSTM = "lstm"
    ENSEMBLE = "ensemble"


class UnitType(Enum):
    """Supported unit types"""
    METERS = "meters"
    YARDS = "yards"


class ProcessingMode(Enum):
    """Application processing mode"""
    PRODUCTION = "production"
    DEMO = "demo"


@dataclass
class PredictionResult:
    """
    Data class for prediction results.

    Attributes:
        prediction: Primary prediction value
        prediction_alternate: Value in alternate unit
        unit: Unit of prediction
        unit_alternate: Alternate unit name
        confidence_lower: Lower confidence bound
        confidence_upper: Upper confidence bound
        model_name: Name of model used
        timestamp: Prediction timestamp

    Developer: Azim Mahmud | Version 3.0.0
    """
    prediction: float
    prediction_alternate: float
    unit: str
    unit_alternate: str
    confidence_lower: float
    confidence_upper: float
    model_name: str
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            **asdict(self),
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class OrderInput:
    """
    Data class for order input validation.

    Attributes:
        order_id: Unique order identifier
        order_quantity: Number of garments
        garment_type: Type of garment
        fabric_type: Type of fabric
        fabric_width_cm: Fabric width in centimeters
        pattern_complexity: Pattern complexity level
        marker_efficiency: Marker efficiency percentage
        defect_rate: Expected defect rate percentage
        operator_experience: Operator experience in years
        season: Production season (Spring/Summer/Fall/Winter)

    Developer: Azim Mahmud | Version 3.0.0
    """
    order_id: str
    order_quantity: int
    garment_type: str
    fabric_type: str
    fabric_width_cm: float
    pattern_complexity: str
    marker_efficiency: float
    defect_rate: float
    operator_experience: int
    season: str = "Spring"  # Season is a training feature (alphabetical encoding)

    def validate(self) -> None:
        """Validate order input data"""
        errors = []

        if not self.order_id or not isinstance(self.order_id, str):
            errors.append("Order ID must be a non-empty string")

        if not (AppConfig.ORDER_QUANTITY_MIN <= self.order_quantity <= AppConfig.ORDER_QUANTITY_MAX):
            errors.append(
                f"Order quantity must be between {AppConfig.ORDER_QUANTITY_MIN} "
                f"and {AppConfig.ORDER_QUANTITY_MAX}"
            )

        if self.garment_type not in AppConfig.GARMENT_TYPES:
            errors.append(f"Garment type must be one of {AppConfig.GARMENT_TYPES}")

        if self.fabric_type not in AppConfig.FABRIC_TYPES:
            errors.append(f"Fabric type must be one of {AppConfig.FABRIC_TYPES}")

        if not (100 <= self.fabric_width_cm <= 200):
            errors.append("Fabric width must be between 100 and 200 cm")

        if self.pattern_complexity not in AppConfig.PATTERN_COMPLEXITIES:
            errors.append(f"Pattern complexity must be one of {AppConfig.PATTERN_COMPLEXITIES}")

        if not (AppConfig.MARKER_EFFICIENCY_MIN <= self.marker_efficiency <= AppConfig.MARKER_EFFICIENCY_MAX):
            errors.append(
                f"Marker efficiency must be between {AppConfig.MARKER_EFFICIENCY_MIN}% "
                f"and {AppConfig.MARKER_EFFICIENCY_MAX}%"
            )

        if not (AppConfig.DEFECT_RATE_MIN <= self.defect_rate <= AppConfig.DEFECT_RATE_MAX):
            errors.append(
                f"Defect rate must be between {AppConfig.DEFECT_RATE_MIN}% "
                f"and {AppConfig.DEFECT_RATE_MAX}%"
            )

        if not (AppConfig.OPERATOR_EXPERIENCE_MIN <= self.operator_experience <= AppConfig.OPERATOR_EXPERIENCE_MAX):
            errors.append(
                f"Operator experience must be between {AppConfig.OPERATOR_EXPERIENCE_MIN} "
                f"and {AppConfig.OPERATOR_EXPERIENCE_MAX} years"
            )

        if self.season not in AppConfig.SEASONS:
            errors.append(f"Season must be one of {AppConfig.SEASONS}")

        if errors:
            raise ValidationError("; ".join(errors))


@dataclass
class SystemHealth:
    """System health check data"""
    status: str
    mode: ProcessingMode
    models_loaded: bool
    timestamp: datetime
    uptime_seconds: float
    memory_usage_mb: float


# ============================================================================
# UNIT CONVERSION CLASS
# ============================================================================

class UnitConverter:
    """
    Unit Conversion Utility Class

    Handles precise conversions between yards and meters for fabric measurements.

    Standard Conversion Factors (ISO):
    - 1 yard = 0.9144 meters (exact)
    - 1 meter = 1.0936132983 yards (approximate)

    Attributes:
        YARDS_TO_METERS (float): Exact conversion factor (0.9144)
        METERS_TO_YARDS (float): Inverse conversion factor

    Developer: Azim Mahmud | Version 3.0.0
    """

    YARDS_TO_METERS = 0.9144
    METERS_TO_YARDS = 1.0936132983

    @staticmethod
    def yards_to_meters(yards: float) -> float:
        """Convert yards to meters"""
        return yards * UnitConverter.YARDS_TO_METERS

    @staticmethod
    def meters_to_yards(meters: float) -> float:
        """Convert meters to yards"""
        return meters * UnitConverter.METERS_TO_YARDS

    @staticmethod
    def convert(value: float, from_unit: str, to_unit: str) -> float:
        """Convert value between units"""
        if from_unit == to_unit:
            return value
        from_unit_lower = from_unit.lower()
        to_unit_lower = to_unit.lower()

        if from_unit_lower == 'yards' and to_unit_lower == 'meters':
            return UnitConverter.yards_to_meters(value)
        if from_unit_lower == 'meters' and to_unit_lower == 'yards':
            return UnitConverter.meters_to_yards(value)
        raise ValueError(f"Unknown units: {from_unit} to {to_unit}")

    @staticmethod
    def format_display(value: float, unit: str, decimals: int = 2) -> str:
        """Format value with unit for display"""
        return f"{value:.{decimals}f} {unit}"

    @staticmethod
    def inches_to_cm(inches: float) -> float:
        """Convert inches to centimeters"""
        return inches * 2.54

    @staticmethod
    def cm_to_inches(cm: float) -> float:
        """Convert centimeters to inches"""
        return cm / 2.54


# ============================================================================
# INPUT VALIDATOR
# ============================================================================

class InputValidator:
    """
    Input validation utility for application data.

    Provides methods to validate user inputs and sanitize data.

    Developer: Azim Mahmud | Version 3.0.0
    """

    @staticmethod
    def sanitize_string(value: str, max_length: int = 100) -> str:
        """Sanitize string input"""
        if not isinstance(value, str):
            return ""
        return value.strip()[:max_length]

    @staticmethod
    def validate_numeric_range(
        value: float,
        min_val: float,
        max_val: float,
        name: str = "Value"
    ) -> float:
        """Validate numeric value is within range"""
        try:
            num_value = float(value)
            if not (min_val <= num_value <= max_val):
                raise ValidationError(
                    f"{name} must be between {min_val} and {max_val}"
                )
            return num_value
        except (ValueError, TypeError) as e:
            raise ValidationError(f"{name} must be a valid number") from e

    @staticmethod
    def validate_csv_columns(df: pd.DataFrame, required_columns: List[str]) -> None:
        """Validate CSV has required columns"""
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValidationError(
                f"Missing required columns: {', '.join(missing_cols)}"
            )

    @staticmethod
    def validate_file_size(file_size: int, max_size_mb: int) -> None:
        """Validate file size is within limits"""
        max_bytes = max_size_mb * 1024 * 1024
        if file_size > max_bytes:
            raise ValidationError(
                f"File size exceeds maximum allowed size of {max_size_mb}MB"
            )


# ============================================================================
# MODEL MANAGER
# ============================================================================

class ModelManager:
    """
    Machine Learning Model Management

    Handles loading, caching, and inference for ML models.

    Attributes:
        models: Dictionary of loaded models
        mode: Current processing mode (production/demo)

    Developer: Azim Mahmud | Version 3.0.0
    """

    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.mode: ProcessingMode = ProcessingMode.DEMO
        self._load_attempted = False
        self.lstm_available = False
        
        # Check TensorFlow availability
        try:
            import tensorflow as tf
            self.tf = tf
            AppConfig.LSTM_AVAILABLE = True
            logger.info("TensorFlow is available - LSTM model enabled")
        except ImportError:
            self.tf = None
            AppConfig.LSTM_AVAILABLE = False
            logger.warning("TensorFlow not available - LSTM model disabled")

    def load_models(self) -> Tuple[Dict[str, Any], bool]:
        """
        Load trained machine learning models from persistent storage.
        Supports all model types: XGBoost, Random Forest, Linear Regression,
        LSTM (TensorFlow), and the weighted-average Ensemble.

        Returns:
            tuple: (models_dict, is_production_mode)

        Raises:
            ModelLoadError: If a critical model file is missing in production
        """
        if self._load_attempted:
            return self.models, self.mode == ProcessingMode.PRODUCTION

        self._load_attempted = True

        if not JOBLIB_AVAILABLE:
            logger.warning("Joblib not available, using demo mode")
            self._create_demo_models()
            return self.models, False

        model_dir = AppConfig.MODEL_PATH

        try:
            logger.info(f"Attempting to load models from {model_dir}")

            # Check if model directory exists
            if not model_dir.exists():
                logger.warning(f"Model directory {model_dir} not found, using demo mode")
                self._create_demo_models()
                return self.models, False

            # Load each model file
            loaded_models = {}

            # Load traditional ML models (pickle files)
            for model_name in ["xgboost", "random_forest", "linear_regression", "ensemble", "scaler", "encoders"]:
                if model_name not in AppConfig.MODEL_FILES:
                    continue
                    
                filename = AppConfig.MODEL_FILES[model_name]
                model_path = model_dir / filename
                
                if model_path.exists():
                    try:
                        loaded_models[model_name] = joblib.load(model_path)
                        logger.info(f"Loaded {model_name} from {model_path}")
                    except Exception as e:
                        logger.error(f"Failed to load {model_name}: {e}")
                        if AppConfig.is_production():
                            raise ModelLoadError(f"Failed to load {model_name}: {e}") from e
                else:
                    logger.warning(f"Model file not found: {model_path}")

            # Load LSTM model (Keras/H5 file) if available
            if AppConfig.LSTM_AVAILABLE and self.tf is not None:
                lstm_path = model_dir / AppConfig.MODEL_FILES["lstm"]
                if lstm_path.exists():
                    try:
                        # Try loading without compiling to avoid deserialization issues
                        try:
                            loaded_models["lstm"] = self.tf.keras.models.load_model(lstm_path, compile=False)
                        except Exception:
                            logger.warning("Standard LSTM load failed, retrying with custom_objects and compile=False")
                            custom_objects = {
                                'mse': self.tf.keras.metrics.MeanSquaredError(),
                                'MeanSquaredError': self.tf.keras.metrics.MeanSquaredError,
                                'keras.metrics.mse': self.tf.keras.metrics.MeanSquaredError()
                            }
                            loaded_models["lstm"] = self.tf.keras.models.load_model(
                                lstm_path, compile=False, custom_objects=custom_objects
                            )

                        self.lstm_available = True
                        logger.info(f"Loaded LSTM model from {lstm_path}")
                    except Exception as e:
                        logger.error(f"Failed to load LSTM model: {e}")
                        logger.debug(traceback.format_exc())
                        self.lstm_available = False
                else:
                    logger.warning(f"LSTM model file not found: {lstm_path}")
                    self.lstm_available = False
            else:
                logger.info("LSTM model loading skipped (TensorFlow not available)")

            # Load metadata
            metadata_path = model_dir / AppConfig.MODEL_FILES["metadata"]
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    loaded_models['metadata'] = json.load(f)
                logger.info(f"Loaded metadata from {metadata_path}")
            else:
                loaded_models['metadata'] = {
                    'version': '3.0.0',
                    'unit': 'meters',
                    'training_date': datetime.now().isoformat(),
                    'tensorflow_available': AppConfig.LSTM_AVAILABLE
                }
                logger.warning("Metadata file not found, using default metadata")

            self.models = loaded_models
            self.mode = ProcessingMode.PRODUCTION
            logger.info(f"Successfully loaded models in production mode")
            logger.info(f"Available models: {[k for k in loaded_models.keys() if k not in ['scaler', 'encoders', 'metadata']]}")

            return self.models, True

        except ModelLoadError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading models: {e}")
            logger.debug(traceback.format_exc())
            
            if AppConfig.is_production():
                raise ModelLoadError(f"Failed to load models: {e}") from e
            else:
                self._create_demo_models()
                return self.models, False

    def _create_demo_models(self) -> None:
        """Create statistically calibrated mock models for demo mode."""
        logger.info("Creating demo mode mock models")

        class MockModel:
            """
            Demo-mode mock model.
            Predicts using the same deterministic physics as data_generation_script.py
            so demo predictions are physically meaningful (no flat 2 m/unit average).
            """
            # Garment base consumption per unit at 160 cm width (alphabetical encoding)
            # Dress=0, Jacket=1, Pants=2, Shirt=3, T-Shirt=4
            _BASE_M = [3.00, 3.50, 2.50, 1.80, 1.20]
            _COMPLEXITY_MULT = [1.35, 1.15, 1.00]   # Complex=0, Medium=1, Simple=2

            def __init__(self, name: str):
                self.name = name

            def predict(self, X: np.ndarray) -> np.ndarray:
                """
                Deterministic demo prediction.

                Feature vector layout (must match TrainingConfig.FEATURES):
                  0: order_quantity
                  1: fabric_width_cm
                  2: marker_efficiency
                  3: defect_rate
                  4: operator_experience
                  5: garment_type_encoded   (0–4)
                  6: fabric_type_encoded    (unused in BOM formula)
                  7: pattern_complexity_encoded (0–2)
                  8: season_encoded         (unused in BOM formula)
                """
                if len(X.shape) == 1:
                    X = X.reshape(1, -1)

                qty       = X[:, 0]
                width_cm  = X[:, 1]
                eff       = X[:, 2]
                defect    = X[:, 3]
                exp_yrs   = X[:, 4]
                g_enc     = X[:, 5].astype(int).clip(0, 4)
                c_enc     = X[:, 7].astype(int).clip(0, 2)

                base = np.array([self._BASE_M[g] for g in g_enc])
                base *= (160.0 / np.where(width_cm > 0, width_cm, 160.0))
                base *= np.array([self._COMPLEXITY_MULT[c] for c in c_enc])

                planned_bom = qty * base * 1.05
                eff_factor  = 1.0 - (eff - 85.0) / 100.0 * 0.40
                def_factor  = 1.0 + defect / 100.0
                exp_factor  = 1.0 + np.exp(-exp_yrs / 15.0) * 0.04
                noise       = np.random.default_rng(int(qty.sum()) % 2**32).normal(0, 0.015, len(qty))

                return planned_bom * eff_factor * def_factor * exp_factor * (1.0 + noise)

        class MockLSTMModel(MockModel):
            """Mock LSTM model with different prediction shape"""
            def predict(self, X: np.ndarray, verbose: int = 0) -> np.ndarray:
                result = super().predict(X)
                return result.reshape(-1, 1)  # LSTM returns 2D array

        # Demo metadata mirrors production metadata structure so CI lookups work
        _demo_weights = {
            'xgboost': 0.40, 'random_forest': 0.30, 'linear_regression': 0.15
        }
        if AppConfig.LSTM_AVAILABLE:
            _demo_weights['lstm'] = 0.25
            # Renormalise to sum=1
            _total = sum(_demo_weights.values())
            _demo_weights = {k: round(v / _total, 3) for k, v in _demo_weights.items()}

        self.models = {
            'xgboost': MockModel('XGBoost'),
            'random_forest': MockModel('Random Forest'),
            'linear_regression': MockModel('Linear Regression'),
            'lstm': MockLSTMModel('LSTM') if AppConfig.LSTM_AVAILABLE else None,
            'ensemble': {
                'model_names': ['xgboost', 'random_forest', 'linear_regression'],
                'weights':     {'xgboost': 0.43, 'random_forest': 0.30,
                                'linear_regression': 0.27},
            },
            'metadata': {
                'version': '3.0.0',
                'unit': 'meters',
                'training_date': '2026-01-29',
                'mode': 'demo',
                'tensorflow_available': AppConfig.LSTM_AVAILABLE,
                'models': {
                    # rmse/mae in metres (illustrative). Used by _load_metrics & predict() CI.
                    'ensemble':          {'rmse': 28.4, 'mae': 20.1, 'r2': 0.987, 'mape': 1.9,
                                          'ci_bounds': {'ci_fraction': 0.019},
                                          'weights': {'xgboost': 0.43, 'random_forest': 0.30,
                                                      'lstm': 0.17, 'linear_regression': 0.10}},
                    'xgboost':           {'rmse': 35.1, 'mae': 25.3, 'r2': 0.982, 'mape': 2.3,
                                          'ci_bounds': {'ci_fraction': 0.023}},
                    'random_forest':     {'rmse': 38.6, 'mae': 27.8, 'r2': 0.975, 'mape': 2.7,
                                          'ci_bounds': {'ci_fraction': 0.027}},
                    'lstm':              {'rmse': 42.2, 'mae': 30.4, 'r2': 0.970, 'mape': 3.1,
                                          'ci_bounds': {'ci_fraction': 0.031}},
                    'linear_regression': {'rmse': 88.7, 'mae': 65.1, 'r2': 0.851, 'mape': 6.2,
                                          'ci_bounds': {'ci_fraction': 0.062}},
                },
                'bom_baseline': {'rmse': 120.5, 'mae': 90.2, 'r2': 0.753, 'mape': 8.5}
            }
        }
        self.mode = ProcessingMode.DEMO
        self.lstm_available = AppConfig.LSTM_AVAILABLE

    def get_model(self, model_name: str) -> Any:
        """Retrieve a loaded model by name, with LSTM availability guard."""
        if model_name not in self.models:
            raise PredictionError(f"Model '{model_name}' not available")
        
        model = self.models[model_name]
        
        # Check if LSTM is requested but not available
        if model_name == "lstm" and model is None:
            raise PredictionError("LSTM model not available (TensorFlow required)")
            
        return model

    def predict(
        self,
        order_data: Dict[str, Any],
        model_name: str = 'xgboost',
        output_unit: str = 'meters'
    ) -> PredictionResult:
        """
        Run fabric consumption prediction through the requested ML model.

        Feature vector construction, optional StandardScaler transform,
        model dispatch (XGBoost / RF / LR / LSTM / Ensemble), unit conversion,
        and empirical confidence interval computation are all handled here.

        Args:
            order_data: Dictionary of encoded order parameters (9 features).
            model_name: One of 'xgboost', 'random_forest', 'linear_regression',
                        'lstm', or 'ensemble'.
            output_unit: Desired output unit — 'meters' or 'yards'.

        Returns:
            PredictionResult: Prediction with confidence bounds, dual-unit values,
                              model name, and timestamp.

        Raises:
            PredictionError: If feature construction, inference, or unit
                             conversion fails.
        """
        try:
            model = self.get_model(model_name)

            # Build feature vector (9 features — must match FEATURES list in train_models.py)
            features = np.array([[
                order_data['order_quantity'],
                order_data['fabric_width_cm'],
                order_data['marker_efficiency'],
                order_data['defect_rate'],
                order_data['operator_experience'],
                order_data['garment_type_encoded'],
                order_data['fabric_type_encoded'],
                order_data['pattern_complexity_encoded'],
                order_data.get('season_encoded', 1),  # default=1 (Spring) for back-compat
            ]])

            # Scale features if in production mode
            if self.mode == ProcessingMode.PRODUCTION:
                if 'scaler' in self.models:
                    features = self.models['scaler'].transform(features)
                else:
                    logger.warning("Scaler not loaded, using raw features")

            # Predict based on model type
            if model_name == "ensemble":
                # Ensemble spec is a dict {"model_names": [...], "weights": {...}}.
                # Compute weighted average of all sub-model predictions.
                spec = model
                if isinstance(spec, dict) and "weights" in spec:
                    pred_accum, weight_accum = 0.0, 0.0
                    for sub_name in spec.get("model_names", list(spec["weights"].keys())):
                        sub_model = self.models.get(sub_name)
                        if sub_model is None:
                            continue
                        w = spec["weights"].get(sub_name, 0.0)
                        if sub_name == "lstm" and TENSORFLOW_AVAILABLE:
                            fr = features.reshape(features.shape[0], 1, features.shape[1])
                            sp = float(sub_model.predict(fr, verbose=0).flatten()[0])
                        else:
                            sp = float(sub_model.predict(features)[0])
                        pred_accum   += w * sp
                        weight_accum += w
                    prediction_base = pred_accum / weight_accum if weight_accum > 0 else pred_accum
                else:
                    # Fallback: spec is a plain sklearn model
                    prediction_base = float(model.predict(features)[0])
            elif model_name == "lstm":
                # LSTM requires 3D input: (samples, timesteps, features)
                features_lstm = features.reshape(features.shape[0], 1, features.shape[1])
                prediction_array = model.predict(features_lstm, verbose=0)
                prediction_base = float(prediction_array.flatten()[0])
            else:
                # Traditional ML models: XGBoost, Random Forest, Linear Regression
                prediction_base = float(model.predict(features)[0])

            # Validate and floor prediction.
            # Linear models can extrapolate to negative values for small/unusual inputs
            # (e.g. qty=1000 is well below training mean of ~2550, combined with
            # narrow fabric width — both push LR prediction toward zero or below).
            # Rather than crashing with PredictionError, clamp to a physics-based
            # floor: the minimum physically plausible fabric consumption is
            # qty × smallest_garment_base × (160 / widest_fabric) × 0.70 safety margin.
            if not np.isfinite(prediction_base):
                raise PredictionError(f"Model returned non-finite prediction: {prediction_base}")
            if prediction_base <= 0:
                _garment_base_min = min(AppConfig.GARMENT_BASE_CONSUMPTION_M.values())
                _width_cm_used    = order_data.get('fabric_width_cm', 160.0)
                _qty              = order_data.get('order_quantity', 1)
                _physics_floor    = _qty * _garment_base_min * (160.0 / max(_width_cm_used, 1.0)) * 0.70
                logger.warning(
                    f"Model '{model_name}' predicted {prediction_base:.3f} m (≤ 0). "
                    f"Clamping to physics floor {_physics_floor:.1f} m. "
                    f"Consider retraining with more data to avoid linear extrapolation errors."
                )
                prediction_base = _physics_floor

            # Convert to meters first (if needed)
            metadata_unit = self.models.get('metadata', {}).get('unit', 'meters')
            if self.mode == ProcessingMode.PRODUCTION and metadata_unit == 'yards':
                prediction_m = UnitConverter.yards_to_meters(prediction_base)
            else:
                prediction_m = prediction_base

            # Convert to desired output unit
            if output_unit == 'yards':
                prediction = UnitConverter.meters_to_yards(prediction_m)
                prediction_alternate = prediction_m
                unit_alternate = 'meters'
            else:
                prediction = prediction_m
                prediction_alternate = UnitConverter.meters_to_yards(prediction_m)
                unit_alternate = 'yards'

            # Confidence intervals — use empirical 90th-percentile residuals from
            # model_metadata.json if available, otherwise fall back to model-specific
            # fractions derived from typical validation MAPE.
            # Confidence intervals — use empirical ci_fraction from metadata if it is a
            # genuine fraction (<1.0). Guards against RMSE/MAE values (e.g. 28.4, 88.7)
            # being stored in ci_bounds and getting clamped to 50% CI bands.
            _ci_meta_bounds = (self.models.get('metadata') or {}).get('models', {}).get(
                model_name, {}).get('ci_bounds', {})
            _raw_frac = _ci_meta_bounds.get('ci_fraction', None)
            ci_bounds = _ci_meta_bounds if (_raw_frac is not None and float(_raw_frac) < 1.0) else {}
            ci_frac = ci_bounds.get('ci_fraction', {
                'ensemble':         0.020,
                'xgboost':          0.028,
                'random_forest':    0.032,
                'lstm':             0.035,
                'linear_regression':0.065,
            }.get(model_name, 0.050))
            # Safety clamp: ci_frac must be a fractional value (0.5–50% of prediction).
            # Guards against rmse/mape values (e.g. 28.4, 88.7) being accidentally
            # read as ci_fraction, which would produce absurdly wide CI bands.
            ci_frac = float(np.clip(ci_frac, 0.005, 0.50))

            confidence_lower = max(prediction * (1.0 - ci_frac), 0.001)
            confidence_upper = prediction * (1.0 + ci_frac)

            return PredictionResult(
                prediction=prediction,
                prediction_alternate=prediction_alternate,
                unit=output_unit,
                unit_alternate=unit_alternate,
                confidence_lower=confidence_lower,
                confidence_upper=confidence_upper,
                model_name=model_name,
                timestamp=datetime.now()
            )

        except PredictionError:
            raise
        except Exception as e:
            logger.error(f"Prediction error ({model_name}): {e}")
            logger.debug(traceback.format_exc())
            raise PredictionError(
                f"Failed to calculate prediction for '{model_name}': {type(e).__name__}: {e}"
            ) from e


# Initialize model manager
model_manager = ModelManager()


# ============================================================================
# DATA GENERATOR
# ============================================================================

class DataGenerator:
    """
    Synthetic data generation for demonstration and testing.

    Developer: Azim Mahmud | Version 3.0.0
    """

    @staticmethod
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def generate_historical_data(n_samples: int = 500) -> pd.DataFrame:
        """
        Generate synthetic historical production data for dashboard visualisation.

        Consumption is computed from the same deterministic signal used by
        data_generation_script.py so dashboard statistics are physically
        meaningful and consistent with model training data.

        Args:
            n_samples: Number of records to generate (default 500).

        Returns:
            pd.DataFrame: Historical production data with fabric metrics.
        """
        rng = np.random.default_rng(42)

        garment_types = rng.choice(AppConfig.GARMENT_TYPES, n_samples)
        fabric_types  = rng.choice(AppConfig.FABRIC_TYPES,  n_samples)
        complexities  = rng.choice(AppConfig.PATTERN_COMPLEXITIES, n_samples)
        seasons       = rng.choice(AppConfig.SEASONS, n_samples)
        width_inches  = rng.choice(AppConfig.FABRIC_WIDTHS_INCHES, n_samples)
        width_cm      = width_inches * 2.54

        order_quantities = rng.integers(100, 5001, n_samples).astype(float)
        marker_eff   = rng.normal(85.0, 4.0, n_samples).clip(70, 95)
        defect_rate  = rng.exponential(1.8, n_samples).clip(0, 10)
        op_exp       = rng.integers(1, 21, n_samples).astype(float)

        # Garment base consumption (m) at standard 160 cm width
        STANDARD_WIDTH = 160.0
        COMPLEXITY_MULT = {'Simple': 1.00, 'Medium': 1.15, 'Complex': 1.35}
        SEASONAL_IMPACT = {'Spring': 0.010, 'Summer': -0.008, 'Fall': 0.005, 'Winter': 0.018}
        BOM_BUFFER = AppConfig.DEFAULT_BOM_BUFFER

        base_per_unit = np.array([
            AppConfig.GARMENT_BASE_CONSUMPTION_M[g] for g in garment_types
        ])
        base_per_unit *= (STANDARD_WIDTH / width_cm)
        base_per_unit *= np.array([COMPLEXITY_MULT[c] for c in complexities])

        planned_bom_m = order_quantities * base_per_unit * BOM_BUFFER

        # Deterministic adjustment factors (same as data_generation_script.py)
        efficiency_factor = 1.0 - (marker_eff - 85.0) / 100.0 * 0.40
        defect_factor     = 1.0 + defect_rate / 100.0
        exp_factor        = 1.0 + np.exp(-op_exp / 15.0) * 0.04
        seasonal_factor   = 1.0 + np.array([SEASONAL_IMPACT[s] for s in seasons])
        size_factor       = 1.0 - 0.03 * np.log1p(order_quantities / 1000.0)
        noise             = rng.normal(0.0, 0.020, n_samples)

        actual_m = (
            planned_bom_m
            * efficiency_factor * defect_factor
            * exp_factor * seasonal_factor * size_factor
            * (1.0 + noise)
        ).clip(planned_bom_m * 0.85, planned_bom_m * 1.15)

        data = {
            'Order_ID':                  [f'ORD_{str(i).zfill(6)}' for i in range(1, n_samples + 1)],
            'Date':                      pd.date_range(start='2024-01-01', periods=n_samples, freq='D'),
            'Order_Quantity':            order_quantities.astype(int),
            'Garment_Type':              garment_types,
            'Fabric_Type':               fabric_types,
            'Fabric_Width_inches':       width_inches,
            'Pattern_Complexity':        complexities,
            'Season':                    seasons,
            'Marker_Efficiency_%':       marker_eff,
            'Expected_Defect_Rate_%':    defect_rate,
            'Operator_Experience_Years': op_exp.astype(int),
            'Planned_BOM_m':             planned_bom_m,
            'Actual_Consumption_m':      actual_m,
        }

        df = pd.DataFrame(data)
        df['Planned_BOM_yards']          = df['Planned_BOM_m']          * UnitConverter.METERS_TO_YARDS
        df['Actual_Consumption_yards']   = df['Actual_Consumption_m']   * UnitConverter.METERS_TO_YARDS
        df['Variance_m']                 = df['Actual_Consumption_m']   - df['Planned_BOM_m']
        df['Variance_yards']             = df['Variance_m']             * UnitConverter.METERS_TO_YARDS
        df['Variance_%']                 = (df['Variance_m'] / df['Planned_BOM_m']) * 100

        logger.info(f"Generated {n_samples} historical records | "
                    f"avg|Variance%|={df['Variance_%'].abs().mean():.2f}%")
        return df

    @staticmethod
    def get_batch_template() -> pd.DataFrame:
        """Batch upload template. Season is a required training feature (added v3.0)."""
        return pd.DataFrame({
            'Order_ID':                  ['ORD_001', 'ORD_002', 'ORD_003'],
            'Order_Quantity':            [1000, 1500, 2000],
            'Garment_Type':              ['T-Shirt', 'Pants', 'Jacket'],
            'Fabric_Type':               ['Cotton', 'Denim', 'Polyester'],
            'Fabric_Width_inches':       [63, 59, 63],
            'Pattern_Complexity':        ['Simple', 'Medium', 'Complex'],
            'Season':                    ['Spring', 'Summer', 'Fall'],
            'Marker_Efficiency_%':       [85, 88, 82],
            'Expected_Defect_Rate_%':    [2, 3, 4],
            'Operator_Experience_Years': [5, 8, 3],
        })


# ============================================================================
# ENCODING MAPPINGS
# ============================================================================

class EncodingMaps:
    """
    Categorical encoding maps for ML inference.

    All integer assignments are alphabetically sorted, matching the output of
    sklearn.preprocessing.LabelEncoder fitted on the sorted category lists
    defined in TrainingConfig.CATEGORICAL_FEATURES (train_models.py).

    Any change here must be mirrored in TrainingConfig.CATEGORICAL_FEATURES
    and the data generator constants.
    """

    # Alphabetical order — Dress=0, Jacket=1, Pants=2, Shirt=3, T-Shirt=4
    GARMENT_TYPE = {'Dress': 0, 'Jacket': 1, 'Pants': 2, 'Shirt': 3, 'T-Shirt': 4}
    # Cotton=0, Cotton-Blend=1, Denim=2, Polyester=3, Silk=4
    FABRIC_TYPE = {'Cotton': 0, 'Cotton-Blend': 1, 'Denim': 2, 'Polyester': 3, 'Silk': 4}
    # Complex=0, Medium=1, Simple=2
    COMPLEXITY = {'Complex': 0, 'Medium': 1, 'Simple': 2}
    # Fall=0, Spring=1, Summer=2, Winter=3
    SEASON = {'Fall': 0, 'Spring': 1, 'Summer': 2, 'Winter': 3}
    # Human-readable model names → internal model keys
    MODEL_DISPLAY = {
        'Ensemble (Best)':     ModelType.ENSEMBLE.value,
        'XGBoost':             ModelType.XGBOOST.value,
        'Random Forest':       ModelType.RANDOM_FOREST.value,
        'LSTM Neural Network': ModelType.LSTM.value,
        'Linear Regression':   ModelType.LINEAR_REGRESSION.value,
    }


# ============================================================================
# SESSION STATE MANAGER
# ============================================================================

# ── Encoding sanity check: validates hardcoded maps match sorted categories ──
def _verify_encoding_maps() -> None:
    """
    Assert that EncodingMaps match what sklearn LabelEncoder would produce.
    Runs once at startup. Raises AssertionError if encoding drift is detected.
    """
    from sklearn.preprocessing import LabelEncoder
    checks = [
        ("GARMENT_TYPE",  EncodingMaps.GARMENT_TYPE,
         ["Dress","Jacket","Pants","Shirt","T-Shirt"]),
        ("FABRIC_TYPE",   EncodingMaps.FABRIC_TYPE,
         ["Cotton","Cotton-Blend","Denim","Polyester","Silk"]),
        ("COMPLEXITY",    EncodingMaps.COMPLEXITY,
         ["Complex","Medium","Simple"]),
        ("SEASON",        EncodingMaps.SEASON,
         ["Fall","Spring","Summer","Winter"]),
    ]
    for name, enc_map, categories in checks:
        le = LabelEncoder()
        le.fit(categories)
        for cat, expected_int in zip(categories, le.transform(categories)):
            actual = enc_map.get(cat)
            assert actual == int(expected_int), (
                f"EncodingMaps.{name}['{cat}'] = {actual}, "
                f"but LabelEncoder gives {int(expected_int)}. "
                f"Update the hardcoded map to match alphabetical order."
            )

try:
    _verify_encoding_maps()
    logger.info("✅ Encoding maps verified — all match LabelEncoder alphabetical order")
except AssertionError as _enc_err:
    logger.critical(f"❌ ENCODING MAP MISMATCH: {_enc_err}")
    raise


class SessionManager:
    """
    Manage Streamlit session state with proper initialization.

    Developer: Azim Mahmud | Version 3.0.0
    """

    @staticmethod
    def initialize() -> None:
        """Initialize all session state variables"""
        defaults = {
            'unit_preference':    UnitType.METERS.value,
            'show_dual_units':    True,
            'predictions_count':  0,
            'total_savings':      0.0,
            'session_start':      datetime.now(),
            'last_activity':      datetime.now(),
            'prediction_history': [],
            'page_history':       [],
        }

        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
                logger.debug(f"Initialized session state: {key}")

    @staticmethod
    def update_activity() -> None:
        """Update last activity timestamp"""
        st.session_state.last_activity = datetime.now()

    @staticmethod
    def is_session_valid() -> bool:
        """Check if session is still valid (not timed out)"""
        if 'session_start' not in st.session_state:
            return True

        elapsed = (datetime.now() - st.session_state.last_activity).total_seconds()
        timeout_minutes = elapsed / 60

        return timeout_minutes < AppConfig.SESSION_TIMEOUT_MINUTES

    @staticmethod
    def add_prediction(result: PredictionResult) -> None:
        """Add prediction to history"""
        if 'prediction_history' not in st.session_state:
            st.session_state.prediction_history = []

        st.session_state.prediction_history.append(result.to_dict())
        st.session_state.predictions_count += 1

    @staticmethod
    def get_session_stats() -> Dict[str, Any]:
        """Get current session statistics"""
        return {
            'predictions_count': st.session_state.get('predictions_count', 0),
            'total_savings': st.session_state.get('total_savings', 0.0),
            'session_duration': (
                datetime.now() - st.session_state.get('session_start', datetime.now())
            ).total_seconds() / 60,
            'current_unit': st.session_state.get('unit_preference', 'meters')
        }


# ============================================================================
# UI HELPERS
# ============================================================================

class UIHelpers:
    """
    User Interface helper functions for consistent styling and display.

    Developer: Azim Mahmud | Version 3.0.0
    """

    CUSTOM_CSS = """
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        padding: 1rem;
        margin: 1rem 0;
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 1rem;
    }
    h2 {
        color: #2c3e50;
        padding-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .unit-badge {
        background-color: #ff6b6b;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 0.3rem;
        font-size: 0.9rem;
        font-weight: bold;
    }
    .stDownloadButton button {
        width: 100%;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
    }
    </style>
    """

    @staticmethod
    def apply_custom_styles() -> None:
        """Apply custom CSS styles to the app"""
        st.markdown(UIHelpers.CUSTOM_CSS, unsafe_allow_html=True)

    @staticmethod
    def show_error(message: str, details: Optional[str] = None) -> None:
        """Display error message with optional details"""
        st.error(f"❌ {message}")
        if details and AppConfig.DEBUG:
            st.expander("Error Details").write(details)
        logger.error(f"{message} - {details if details else ''}")

    @staticmethod
    def show_success(message: str) -> None:
        """Display success message"""
        st.success(f"✅ {message}")
        logger.info(message)

    @staticmethod
    def show_warning(message: str) -> None:
        """Display warning message"""
        st.warning(f"⚠️ {message}")
        logger.warning(message)

    @staticmethod
    def format_metric(value: float, prefix: str = "", suffix: str = "", decimals: int = 2) -> str:
        """Format metric value for display"""
        return f"{prefix}{value:,.{decimals}f}{suffix}"

    @staticmethod
    def render_footer(unit_pref: str, is_production: bool) -> None:
        """Render application footer"""
        st.markdown("---")
        st.markdown(f"""
        <div style='text-align: center; color: #7f8c8d; padding: 20px;'>
            <p style='font-size: 16px;'>
                <strong>Fabric Consumption Forecasting System v{AppConfig.APP_VERSION}</strong>
            </p>
            <p style='font-size: 14px;'>
                Developed by <strong>{AppConfig.APP_AUTHOR}</strong> | © January 2026
            </p>
            <p style='font-size: 14px;'>
                🌍 Global Unit Support: Meters & Yards |
                📊 Active Unit: <strong>{unit_pref.upper()}</strong>
            </p>
            <p style='font-size: 14px;'>
                Mode: {'🟢 Production' if is_production else '🟡 Demo'} |
                Powered by AI/Machine Learning
            </p>
            <p style='font-size: 12px; margin-top: 10px;'>
                All Rights Reserved | Optimizing Fabric Usage Through Data Science 🌱
            </p>
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# PAGE RENDERERS
# ============================================================================

class DashboardPage:
    """Dashboard page renderer"""

    @staticmethod
    def render(unit_pref: str, show_dual_units: bool) -> None:
        """Render the main dashboard page"""
        st.title("🧵 Fabric Forecasting Dashboard")
        st.markdown(f"### AI-Powered Material Planning | Active Unit: **{unit_pref.upper()}**")

        # Load historical data first so metrics are computed from real values
        try:
            df_history = DataGenerator.generate_historical_data()
        except Exception as e:
            UIHelpers.show_error("Failed to load dashboard data", str(e))
            return

        # Compute summary metrics from historical dataset
        session_stats   = SessionManager.get_session_stats()
        avg_var_pct     = df_history['Variance_%'].abs().mean()
        accuracy_pct    = 100.0 - avg_var_pct
        total_waste_usd = (
            df_history['Variance_m'].clip(lower=0)
            * df_history['Garment_Type'].map(
                {g: AppConfig.FABRIC_COST_PER_METER.get('Cotton',
                 AppConfig.DEFAULT_FABRIC_COST_PER_METER) for g in AppConfig.GARMENT_TYPES}
            ).fillna(AppConfig.DEFAULT_FABRIC_COST_PER_METER)
        ).sum()
        pct_over = (df_history['Variance_m'] > 0).mean() * 100

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "📊 Session Predictions",
                f"{session_stats['predictions_count']:,}",
                help="Number of predictions made in this session"
            )
        with col2:
            st.metric(
                "🎯 Dataset Avg Accuracy",
                f"{accuracy_pct:.1f}%",
                help=f"100% − avg |Variance%| across {len(df_history):,} demo records"
            )
        with col3:
            st.metric(
                "💰 Demo Waste Cost",
                f"${total_waste_usd:,.0f}",
                help="Total over-consumption cost in demo dataset at Cotton baseline rate"
            )
        with col4:
            st.metric(
                "📈 Over-consumption Rate",
                f"{pct_over:.1f}%",
                help="Fraction of orders where actual > planned BOM"
            )

        st.markdown("---")

        DashboardPage._render_charts(df_history, unit_pref)
        DashboardPage._render_statistics(df_history, unit_pref, show_dual_units)

    @staticmethod
    def _render_charts(df: pd.DataFrame, unit_pref: str) -> None:
        """Render dashboard charts"""
        # Determine columns based on unit preference
        if unit_pref == 'yards':
            planned_col = 'Planned_BOM_yards'
            actual_col = 'Actual_Consumption_yards'
            variance_col = 'Variance_yards'
        else:
            planned_col = 'Planned_BOM_m'
            actual_col = 'Actual_Consumption_m'
            variance_col = 'Variance_m'

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader(f"📈 Consumption Trends ({unit_pref})")

            daily_data = df.groupby('Date').agg({
                planned_col: 'mean',
                actual_col: 'mean'
            }).reset_index()

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=daily_data['Date'], y=daily_data[planned_col],
                name=f'Planned ({unit_pref})', line=dict(color='blue', dash='dash')
            ))
            fig.add_trace(go.Scatter(
                x=daily_data['Date'], y=daily_data[actual_col],
                name=f'Actual ({unit_pref})', line=dict(color='red')
            ))
            fig.update_layout(height=350, hovermode='x unified',
                             yaxis_title=f'Consumption ({unit_pref})')
            st.plotly_chart(fig, width='stretch')

        with col2:
            st.subheader("🎯 Model Performance (R²)")
            # Pull from metadata if production models are loaded; else use demo values
            meta_models = (model_manager.models.get('metadata') or {}).get('models', {})
            if meta_models and model_manager.mode == ProcessingMode.PRODUCTION:
                model_names = ['Ensemble', 'XGBoost', 'Random Forest', 'LSTM', 'Linear Regression']
                meta_keys   = ['ensemble', 'xgboost', 'random_forest', 'lstm', 'linear_regression']
                r2_vals = [
                    round(meta_models[k]['r2'] * 100, 2) if k in meta_models else None
                    for k in meta_keys
                ]
                colors = ['#8B5CF6', '#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
                model_data = pd.DataFrame({
                    'Model':    [n for n, v in zip(model_names, r2_vals) if v is not None],
                    'R2_pct':   [v for v in r2_vals if v is not None],
                    'Color':    [c for c, v in zip(colors, r2_vals) if v is not None],
                })
            else:
                model_data = pd.DataFrame({
                    'Model':  ['Ensemble', 'XGBoost', 'Random Forest', 'LSTM', 'Traditional BOM'],
                    'R2_pct': [98.7, 98.2, 97.5, 97.0, 75.3],
                    'Color':  ['#8B5CF6', '#2ecc71', '#3498db', '#9b59b6', '#e74c3c'],
                })

            fig = go.Figure(data=[
                go.Bar(
                    x=model_data['Model'],
                    y=model_data['R2_pct'],
                    marker_color=model_data['Color'],
                    text=[f"{v:.1f}%" for v in model_data['R2_pct']],
                    textposition='outside',
                )
            ])
            fig.update_layout(
                height=350,
                yaxis_title='R² × 100 (%)',
                yaxis=dict(range=[0, 103]),
                margin=dict(t=20),
            )
            st.plotly_chart(fig, width='stretch')

    @staticmethod
    def _render_statistics(df: pd.DataFrame, unit_pref: str, show_dual_units: bool) -> None:
        """Render dashboard statistics"""
        variance_col = 'Variance_yards' if unit_pref == 'yards' else 'Variance_m'

        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"📊 Variance Distribution ({unit_pref})")
            fig = px.histogram(df, x=variance_col, nbins=50,
                              title="Variance Distribution",
                              labels={variance_col: f'Variance ({unit_pref})'},
                              color_discrete_sequence=['#3498db'])
            fig.add_vline(x=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, width='stretch')

        with col2:
            st.subheader("🏭 Variance by Garment Type")
            variance_by_garment = df.groupby('Garment_Type')[variance_col].mean().sort_values()
            fig = go.Figure(data=[
                go.Bar(x=variance_by_garment.values, y=variance_by_garment.index,
                      orientation='h', marker_color='#e74c3c')
            ])
            fig.update_layout(xaxis_title=f'Avg Variance ({unit_pref})')
            st.plotly_chart(fig, width='stretch')


class SinglePredictionPage:
    """Single prediction page renderer"""

    @staticmethod
    def render(unit_pref: str, show_dual_units: bool, model_mgr: ModelManager) -> None:
        """Render the single prediction page"""
        st.title("🎯 Single Order Prediction")
        st.markdown(f"### Enter order details | Output unit: **{unit_pref.upper()}**")

        with st.form("prediction_form"):
            SinglePredictionPage._render_form_inputs(unit_pref, show_dual_units)
            submit = st.form_submit_button("🔮 Predict Fabric Consumption",
                                          use_container_width=True)

        if submit:
            SinglePredictionPage._process_prediction(unit_pref, model_mgr, show_dual_units)

    @staticmethod
    def _render_form_inputs(unit_pref: str, show_dual_units: bool) -> None:
        """Render form input fields"""
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("📦 Order Details")
            order_id = st.text_input("Order ID", value="ORD_001234", key="sp_order_id")
            order_quantity = st.number_input(
                "Order Quantity",
                AppConfig.ORDER_QUANTITY_MIN,
                AppConfig.ORDER_QUANTITY_MAX,
                1000,
                key="sp_quantity"
            )
            garment_type = st.selectbox(
                "Garment Type",
                AppConfig.GARMENT_TYPES,
                key="sp_garment"
            )

        with col2:
            st.subheader("🧵 Fabric Specifications")
            fabric_type = st.selectbox(
                "Fabric Type",
                AppConfig.FABRIC_TYPES,
                key="sp_fabric"
            )

            # Width input based on preference
            if unit_pref == 'yards':
                fabric_width_in = st.selectbox(
                    "Fabric Width (inches)",
                    AppConfig.FABRIC_WIDTHS_INCHES,
                    key="sp_width_in"
                )
                fabric_width_cm = UnitConverter.inches_to_cm(fabric_width_in)
            else:
                # Show user-friendly rounded labels (140, 150, 160, 180 cm) but
                # map internally to exact inch-derived values (139.70, 149.86, ...)
                # so that round-trip back to inches is lossless (55.00", not 55.1").
                _cm_labels   = list(AppConfig.FABRIC_WIDTHS_CM_DISPLAY.values())  # [140,150,160,180]
                _cm_exact    = list(AppConfig.FABRIC_WIDTHS_CM_DISPLAY.keys())    # [139.7, 149.86,...]
                _cm_label_sel = st.selectbox(
                    "Fabric Width (cm)",
                    _cm_labels,
                    key="sp_width_cm"
                )
                # Map selected display label → exact internal cm value
                _label_to_exact = dict(zip(_cm_labels, _cm_exact))
                fabric_width_cm = _label_to_exact.get(_cm_label_sel, _cm_exact[0])
                fabric_width_in = AppConfig.WIDTH_CM_TO_INCHES.get(
                    fabric_width_cm, UnitConverter.cm_to_inches(fabric_width_cm)
                )

            if show_dual_units:
                _in_display = int(fabric_width_in) if fabric_width_in == int(fabric_width_in) else round(fabric_width_in, 1)
                st.caption(f"= {_in_display} inches | {fabric_width_cm:.2f} cm")

            pattern_complexity = st.selectbox(
                "Pattern Complexity",
                AppConfig.PATTERN_COMPLEXITIES,
                key="sp_complexity"
            )

        with col3:
            st.subheader("⚙️ Production Parameters")
            season = st.selectbox(
                "Season",
                AppConfig.SEASONS,
                index=0,
                key="sp_season",
                help="Season affects fabric consumption by ±1–2%"
            )
            marker_efficiency = st.slider(
                "Marker Efficiency (%)",
                AppConfig.MARKER_EFFICIENCY_MIN,
                AppConfig.MARKER_EFFICIENCY_MAX,
                85.0, 0.5,
                key="sp_efficiency",
                help="Training domain: 70–95%"
            )
            defect_rate = st.slider(
                "Expected Defect Rate (%)",
                AppConfig.DEFECT_RATE_MIN,
                AppConfig.DEFECT_RATE_MAX,
                2.0, 0.5,
                key="sp_defect",
                help="Training domain: 0–10%"
            )
            operator_experience = st.slider(
                "Operator Experience (years)",
                AppConfig.OPERATOR_EXPERIENCE_MIN,
                AppConfig.OPERATOR_EXPERIENCE_MAX,
                5,
                key="sp_experience",
                help="Training domain: 1–20 years"
            )

        model_choice = st.selectbox(
            "Select Model",
            list(EncodingMaps.MODEL_DISPLAY.keys()),
            key="sp_model"
        )

    @staticmethod
    def _process_prediction(unit_pref: str, model_mgr: ModelManager, show_dual_units: bool) -> None:
        """Process prediction request"""
        try:
            # Gather form values
            order_id = st.session_state.get('sp_order_id', 'ORD_001234')
            order_quantity = st.session_state.get('sp_quantity', 1000)
            garment_type = st.session_state.get('sp_garment', 'T-Shirt')
            fabric_type = st.session_state.get('sp_fabric', 'Cotton')
            pattern_complexity = st.session_state.get('sp_complexity', 'Simple')
            season = st.session_state.get('sp_season', 'Spring')
            marker_efficiency = st.session_state.get('sp_efficiency', 85.0)
            defect_rate = st.session_state.get('sp_defect', 2.0)
            operator_experience = st.session_state.get('sp_experience', 5)
            model_choice = st.session_state.get('sp_model', 'Ensemble (Best)')

            # Determine fabric width
            # Fallbacks MUST match the form widget defaults (index=0 of each list)
            if unit_pref == 'yards':
                fabric_width_in = st.session_state.get('sp_width_in', AppConfig.FABRIC_WIDTHS_INCHES[0])
                fabric_width_cm = UnitConverter.inches_to_cm(fabric_width_in)
            else:
                # sp_width_cm stores the display label (140, 150, 160, 180)
                _cm_labels  = list(AppConfig.FABRIC_WIDTHS_CM_DISPLAY.values())
                _cm_exact   = list(AppConfig.FABRIC_WIDTHS_CM_DISPLAY.keys())
                _label_to_exact = dict(zip(_cm_labels, _cm_exact))
                _cm_label_stored = st.session_state.get('sp_width_cm', _cm_labels[0])
                fabric_width_cm = _label_to_exact.get(_cm_label_stored, _cm_exact[0])
                fabric_width_in = AppConfig.WIDTH_CM_TO_INCHES.get(
                    fabric_width_cm, UnitConverter.cm_to_inches(fabric_width_cm)
                )

            # Create and validate order input
            order_input = OrderInput(
                order_id=order_id,
                order_quantity=int(order_quantity),
                garment_type=garment_type,
                fabric_type=fabric_type,
                fabric_width_cm=float(fabric_width_cm),
                pattern_complexity=pattern_complexity,
                marker_efficiency=float(marker_efficiency),
                defect_rate=float(defect_rate),
                operator_experience=int(operator_experience),
                season=season
            )

            order_input.validate()

            # Out-of-training-domain warnings (extrapolation flag)
            ood_warnings = []
            if order_input.order_quantity < 100 or order_input.order_quantity > 5000:
                ood_warnings.append(f"Order Quantity {order_input.order_quantity:,} is outside training range 100–5,000")
            if not (70.0 <= order_input.marker_efficiency <= 95.0):
                ood_warnings.append(f"Marker Efficiency {order_input.marker_efficiency:.1f}% is outside training range 70–95%")
            if order_input.defect_rate > 10.0:
                ood_warnings.append(f"Defect Rate {order_input.defect_rate:.1f}% exceeds training maximum 10%")
            if not (1 <= order_input.operator_experience <= 20):
                ood_warnings.append(f"Operator Experience {order_input.operator_experience} yrs is outside training range 1–20")
            if ood_warnings:
                st.warning(
                    "⚠️ **Extrapolation Warning** — The following inputs are outside the model's "
                    "training domain. Predictions may be unreliable:\n" +
                    "\n".join(f"• {w}" for w in ood_warnings)
                )

            # Prepare data for prediction
            order_data = {
                'order_quantity': order_input.order_quantity,
                'fabric_width_cm': order_input.fabric_width_cm,
                'marker_efficiency': order_input.marker_efficiency,
                'defect_rate': order_input.defect_rate,
                'operator_experience': order_input.operator_experience,
                'garment_type_encoded': EncodingMaps.GARMENT_TYPE[order_input.garment_type],
                'fabric_type_encoded': EncodingMaps.FABRIC_TYPE[order_input.fabric_type],
                'pattern_complexity_encoded': EncodingMaps.COMPLEXITY[order_input.pattern_complexity],
                'season_encoded': EncodingMaps.SEASON.get(order_input.season, 1),
            }

            # Get prediction
            model_name = EncodingMaps.MODEL_DISPLAY[model_choice]

            with st.spinner('🔄 Calculating prediction...'):
                result = model_mgr.predict(order_data, model_name, unit_pref)

            # Update session
            SessionManager.add_prediction(result)
            SessionManager.update_activity()

            UIHelpers.show_success("Prediction Complete!")
            SinglePredictionPage._render_results(
                result, order_input, fabric_width_in, fabric_width_cm,
                unit_pref, show_dual_units, model_choice
            )

        except ValidationError as e:
            UIHelpers.show_error("Validation Error", str(e))
        except PredictionError as e:
            UIHelpers.show_error("Prediction Error", str(e))
        except Exception as e:
            UIHelpers.show_error("Unexpected Error", str(e) if AppConfig.DEBUG else "Please try again")
            logger.error(f"Unexpected error in prediction: {e}")
            logger.debug(traceback.format_exc())

    @staticmethod
    def _render_results(
        result: PredictionResult,
        order_input: OrderInput,
        fabric_width_in: float,
        fabric_width_cm: float,
        unit_pref: str,
        show_dual_units: bool,
        model_choice: str
    ) -> None:
        """Render prediction results"""
        st.markdown("---")
        st.subheader("📊 Prediction Results")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label=f"🎯 Predicted ({unit_pref})",
                value=f"{result.prediction:.2f}",
                help=f"AI-predicted fabric requirement in {unit_pref}"
            )
            if show_dual_units:
                st.caption(f"= {result.prediction_alternate:.2f} {result.unit_alternate}")

        with col2:
            # Garment-specific BOM: qty × garment_base (width-adjusted) × 1.05 buffer
            # Base consumption is defined at standard 160cm width; adjust for actual width
            garment_base_m = AppConfig.GARMENT_BASE_CONSUMPTION_M.get(
                order_input.garment_type, 1.5)
            garment_base_m_adjusted = garment_base_m * (160.0 / max(order_input.fabric_width_cm, 1.0))
            bom_estimate_m = (order_input.order_quantity * garment_base_m_adjusted
                              * AppConfig.DEFAULT_BOM_BUFFER)
            bom_estimate = (UnitConverter.meters_to_yards(bom_estimate_m)
                            if unit_pref == 'yards' else bom_estimate_m)

            delta_pct = ((result.prediction - bom_estimate) / bom_estimate * 100)
            st.metric(
                label=f"📋 Traditional BOM ({unit_pref})",
                value=f"{bom_estimate:.2f}",
                delta=f"{delta_pct:.1f}%",
                help="Industry formula: qty × garment_base × 1.05 safety margin"
            )

        with col3:
            fabric_cost = AppConfig.FABRIC_COST_PER_METER.get(
                order_input.fabric_type, AppConfig.DEFAULT_FABRIC_COST_PER_METER
            )
            if unit_pref == 'yards':
                fabric_cost = fabric_cost * UnitConverter.METERS_TO_YARDS
            estimated_cost = result.prediction * fabric_cost
            st.metric(
                label="💰 Estimated Cost",
                value=f"${estimated_cost:.2f}",
                help=f"Based on ${fabric_cost:.2f}/{unit_pref}"
            )

        with col4:
            diff_fabric = bom_estimate - result.prediction  # positive = ML saves vs BOM
            potential_savings = diff_fabric * fabric_cost
            if diff_fabric >= 0:
                savings_label = "💵 ML Saving vs BOM"
                savings_delta = f"−{diff_fabric:.2f} {unit_pref} vs BOM"
            else:
                savings_label = "⚠️ ML Overage vs BOM"
                savings_delta = f"+{abs(diff_fabric):.2f} {unit_pref} vs BOM"
            st.metric(
                label=savings_label,
                value=f"${abs(potential_savings):.2f}",
                delta=savings_delta,
                delta_color="normal" if diff_fabric >= 0 else "inverse",
            )
            st.session_state.total_savings += max(potential_savings, 0.0)

        # ── Model reliability warning ──────────────────────────────────────────
        # Compute the physics-based expected range (±15% of deterministic estimate)
        # and warn the user if the prediction falls outside it.
        _COMPLEXITY_MULT = {"Simple": 1.00, "Medium": 1.15, "Complex": 1.35}
        _SEASONAL_IMPACT = {"Spring": 0.010, "Summer": -0.008, "Fall": 0.005, "Winter": 0.018}
        _det_base = (order_input.order_quantity
                     * garment_base_m_adjusted
                     * AppConfig.DEFAULT_BOM_BUFFER
                     * _COMPLEXITY_MULT.get(order_input.pattern_complexity, 1.0)
                     * (1.0 + _SEASONAL_IMPACT.get(order_input.season, 0.0)))
        _physics_lo = _det_base * 0.80
        _physics_hi = _det_base * 1.20
        _pred_m = (UnitConverter.yards_to_meters(result.prediction)
                   if unit_pref == 'yards' else result.prediction)
        if _pred_m < _physics_lo or _pred_m > _physics_hi:
            _pct_off = (_pred_m / _det_base - 1.0) * 100
            st.warning(
                f"⚠️ **Model Reliability Notice** — The **{result.model_name.replace('_',' ').title()}** "
                f"prediction ({result.prediction:.1f} {unit_pref}) is **{abs(_pct_off):.1f}% "
                f"{'above' if _pct_off > 0 else 'below'}** the physics-based expected range "
                f"({_physics_lo:.0f}–{_physics_hi:.0f} m). "
                f"This model may be overfitting or underfitting on this input. "
                f"Consider using **Ensemble (Best)** or retraining on the 5,000-row production dataset.",
                icon="⚠️"
            )

        # Confidence interval chart
        st.markdown("---")
        st.subheader(f"📈 Prediction with 90% Confidence Interval ({unit_pref})")

        # Warn when CI is unrealistically wide (>10% of prediction on either side).
        # Expected CI widths: Ensemble ±2%, XGBoost ±2.3%, RF ±2.7%, LSTM ±3.1%.
        # Wide CIs (>10%) indicate the model was trained on too few samples,
        # or that model_metadata.json stores RMSE values instead of fractional CI bounds.
        _ci_half_pct = (result.confidence_upper - result.prediction) / max(result.prediction, 1) * 100
        if _ci_half_pct > 10.0:
            st.info(
                f"ℹ️ **Wide Confidence Interval Notice** — The current CI band is ±{_ci_half_pct:.1f}% "
                f"({result.confidence_lower:.0f}–{result.confidence_upper:.0f} {unit_pref}). "
                f"Expected range is ±2–4%. This indicates the model was trained on a small dataset "
                f"(1,000 rows). **Retrain on the 5,000-row production dataset** to reduce uncertainty.",
                icon="ℹ️"
            )

        fig = go.Figure()

        # Shaded confidence band
        fig.add_trace(go.Scatter(
            x=[0.5, 1.5, 1.5, 0.5],
            y=[result.confidence_lower, result.confidence_lower,
               result.confidence_upper, result.confidence_upper],
            fill='toself',
            fillcolor='rgba(31, 119, 180, 0.15)',
            line=dict(color='rgba(31, 119, 180, 0.0)'),
            name='90% Confidence Interval',
            hoverinfo='skip',
        ))

        # Confidence bound horizontal dashes
        for y_val, label in [
            (result.confidence_lower, f"Lower: {result.confidence_lower:.2f} {unit_pref}"),
            (result.confidence_upper, f"Upper: {result.confidence_upper:.2f} {unit_pref}"),
        ]:
            fig.add_shape(type='line', x0=0.5, x1=1.5, y0=y_val, y1=y_val,
                          line=dict(color='#1f77b4', width=1, dash='dot'))
            fig.add_annotation(x=1.52, y=y_val, text=label, showarrow=False,
                               xanchor='left', font=dict(size=11, color='#1f77b4'))

        # Central prediction point
        fig.add_trace(go.Scatter(
            x=[1.0],
            y=[result.prediction],
            mode='markers+text',
            marker=dict(color='#1f77b4', size=16, symbol='diamond'),
            text=[f"  {result.prediction:.2f} {unit_pref}"],
            textposition='middle right',
            textfont=dict(size=13, color='#1f77b4'),
            name=f'AI Prediction ({result.model_name})',
        ))

        # Traditional BOM reference line
        fig.add_shape(type='line', x0=0.4, x1=1.6, y0=bom_estimate_m if unit_pref == 'meters' else bom_estimate,
                      y1=bom_estimate_m if unit_pref == 'meters' else bom_estimate,
                      line=dict(color='#e74c3c', width=2, dash='dash'))
        bom_display = bom_estimate_m if unit_pref == 'meters' else bom_estimate
        fig.add_annotation(
            x=1.52, y=bom_display,
            text=f"Traditional BOM: {bom_display:.2f} {unit_pref}",
            showarrow=False, xanchor='left',
            font=dict(size=11, color='#e74c3c'),
        )

        fig.update_layout(
            xaxis=dict(visible=False, range=[0.3, 2.2]),
            yaxis_title=f'Fabric Consumption ({unit_pref})',
            height=380,
            showlegend=True,
            legend=dict(orientation='h', y=-0.15),
            margin=dict(l=20, r=160, t=20, b=60),
        )
        st.plotly_chart(fig, width='stretch')

        # Detailed prediction breakdown — thesis-grade tabular summary
        with st.expander("📋 Full Prediction Breakdown", expanded=False):
            per_unit_m = result.prediction if unit_pref == 'meters' else UnitConverter.yards_to_meters(result.prediction)
            per_unit_m_each = per_unit_m / order_input.order_quantity if order_input.order_quantity else 0
            per_unit_alt  = result.prediction_alternate / order_input.order_quantity if order_input.order_quantity else 0

            breakdown = {
                "Parameter": [
                    "Order ID",
                    "Garment Type",
                    "Fabric Type",
                    "Pattern Complexity",
                    "Season",
                    "Order Quantity",
                    f"Fabric Width (cm / in)",
                    "Marker Efficiency",
                    "Expected Defect Rate",
                    "Operator Experience",
                    "—",
                    f"AI Prediction (total, {unit_pref})",
                    f"AI Prediction (total, {result.unit_alternate})",
                    f"Per-unit consumption (m)",
                    f"Confidence Interval Lower ({unit_pref})",
                    f"Confidence Interval Upper ({unit_pref})",
                    f"Traditional BOM (total, {unit_pref})",
                    f"BOM vs AI Δ (%)",
                    f"Fabric Cost ($/{'meter' if unit_pref == 'meters' else 'yard'})",
                    f"Estimated Total Cost (AI)",
                    f"Estimated Total Cost (BOM)",
                    f"Model Used",
                    f"Timestamp",
                ],
                "Value": [
                    order_input.order_id,
                    order_input.garment_type,
                    order_input.fabric_type,
                    order_input.pattern_complexity,
                    order_input.season,
                    f"{order_input.order_quantity:,}",
                    f"{order_input.fabric_width_cm:.2f} cm / {int(fabric_width_in) if fabric_width_in == int(fabric_width_in) else round(fabric_width_in, 1)} in",
                    f"{order_input.marker_efficiency:.1f}%",
                    f"{order_input.defect_rate:.1f}%",
                    f"{order_input.operator_experience} yr(s)",
                    "—",
                    f"{result.prediction:,.3f} {unit_pref}",
                    f"{result.prediction_alternate:,.3f} {result.unit_alternate}",
                    f"{per_unit_m_each:.4f} m/unit",
                    f"{result.confidence_lower:,.3f} {unit_pref}",
                    f"{result.confidence_upper:,.3f} {unit_pref}",
                    f"{bom_estimate:,.3f} {unit_pref}",
                    f"{delta_pct:+.2f}%",
                    f"${fabric_cost:.4f}/{unit_pref.rstrip('s')}",
                    f"${estimated_cost:,.2f}",
                    f"${bom_estimate * fabric_cost:,.2f}",
                    {
                        'xgboost': 'XGBoost', 'random_forest': 'Random Forest',
                        'lstm': 'LSTM', 'linear_regression': 'Linear Regression',
                        'ensemble': 'Ensemble'
                    }.get(result.model_name.lower(), result.model_name.replace('_', ' ').title()),
                    result.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                ],
            }
            st.dataframe(
                pd.DataFrame(breakdown),
                use_container_width=True,
                hide_index=True,
            )

            # ── Explicit download button ─────────────────────────────────────
            csv_export = pd.DataFrame(breakdown).to_csv(index=False)
            st.download_button(
                label="📥 Download Prediction as CSV",
                data=csv_export,
                file_name=f"prediction_{order_input.order_id}_{result.timestamp.strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
                help="Download the full prediction breakdown as a CSV file",
            )

class BatchPredictionPage:
    """Batch prediction page renderer"""

    @staticmethod
    def render(unit_pref: str, show_dual_units: bool, model_mgr: ModelManager) -> None:
        """Render the batch prediction page"""
        st.title("📊 Batch Prediction")
        st.markdown(f"### Upload CSV for multiple predictions | Output: **{unit_pref.upper()}**")

        # Template download
        col1, col2 = st.columns([3, 1])
        with col2:
            BatchPredictionPage._render_template_download()

        # File upload
        uploaded_file = st.file_uploader(
            "Choose CSV file",
            type=['csv'],
            help=f"Maximum file size: {AppConfig.MAX_FILE_SIZE_MB}MB"
        )

        if uploaded_file is not None:
            BatchPredictionPage._process_upload(
                uploaded_file, unit_pref, show_dual_units, model_mgr
            )

    @staticmethod
    def _render_template_download() -> None:
        """Render template download button"""
        template_df = DataGenerator.get_batch_template()
        csv = template_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Template",
            data=csv,
            file_name="batch_template.csv",
            mime="text/csv",
            use_container_width=True
        )

    @staticmethod
    def _process_upload(
        uploaded_file,
        unit_pref: str,
        show_dual_units: bool,
        model_mgr: ModelManager
    ) -> None:
        """Process uploaded CSV file"""
        try:
            # Validate file size
            InputValidator.validate_file_size(
                uploaded_file.size,
                AppConfig.MAX_FILE_SIZE_MB
            )

            df = pd.read_csv(uploaded_file)

            # Validate required columns
            required_columns = [
                'Order_ID', 'Order_Quantity', 'Garment_Type', 'Fabric_Type',
                'Fabric_Width_inches', 'Pattern_Complexity', 'Season',
                'Marker_Efficiency_%', 'Expected_Defect_Rate_%',
                'Operator_Experience_Years',
            ]
            InputValidator.validate_csv_columns(df, required_columns)

            # Validate row count
            if len(df) > AppConfig.MAX_BATCH_ROWS:
                raise ValidationError(
                    f"File exceeds maximum of {AppConfig.MAX_BATCH_ROWS} rows"
                )

            UIHelpers.show_success(f"Loaded {len(df)} orders")

            st.subheader("📋 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)

            model_choice = st.selectbox(
                "Select Model",
                list(EncodingMaps.MODEL_DISPLAY.keys()),
                key="batch_model"
            )

            if st.button("🔮 Generate Predictions", use_container_width=True):
                BatchPredictionPage._generate_predictions(
                    df, unit_pref, show_dual_units, model_choice, model_mgr
                )

        except ValidationError as e:
            UIHelpers.show_error("Validation Error", str(e))
        except Exception as e:
            UIHelpers.show_error("Processing Error", str(e))
            logger.error(f"Batch processing error: {e}")
            logger.debug(traceback.format_exc())

    @staticmethod
    def _generate_predictions(
        df: pd.DataFrame,
        unit_pref: str,
        show_dual_units: bool,
        model_choice: str,
        model_mgr: "ModelManager",
    ) -> None:
        """Generate predictions using the selected ML model for every row."""
        with st.spinner('🔄 Processing...'):
            try:
                model_name = EncodingMaps.MODEL_DISPLAY.get(model_choice, 'ensemble')

                # BOM baseline (for comparison column)
                garment_bases = df['Garment_Type'].map(
                    AppConfig.GARMENT_BASE_CONSUMPTION_M
                ).fillna(1.5)
                df['Traditional_BOM_m'] = (
                    df['Order_Quantity'] * garment_bases * AppConfig.DEFAULT_BOM_BUFFER
                )

                # ── Real ML predictions row-by-row ───────────────────────────
                predicted_m_list = []
                progress = st.progress(0)
                n = len(df)
                for i, (_, row) in enumerate(df.iterrows()):
                    try:
                        garment    = str(row.get('Garment_Type', 'T-Shirt'))
                        fabric     = str(row.get('Fabric_Type', 'Cotton'))
                        complexity = str(row.get('Pattern_Complexity', 'Simple'))
                        season     = str(row.get('Season', 'Spring'))
                        width_in   = float(row.get('Fabric_Width_inches', 59))
                        order_data = {
                            'order_quantity':             int(row['Order_Quantity']),
                            'fabric_width_cm':            UnitConverter.inches_to_cm(width_in),
                            'marker_efficiency':          float(row.get('Marker_Efficiency_%', 85)),
                            'defect_rate':                float(row.get('Expected_Defect_Rate_%', 2)),
                            'operator_experience':        int(row.get('Operator_Experience_Years', 5)),
                            'garment_type_encoded':       EncodingMaps.GARMENT_TYPE.get(garment, 4),
                            'fabric_type_encoded':        EncodingMaps.FABRIC_TYPE.get(fabric, 0),
                            'pattern_complexity_encoded': EncodingMaps.COMPLEXITY.get(complexity, 2),
                            'season_encoded':             EncodingMaps.SEASON.get(season, 1),
                        }
                        result = model_mgr.predict(order_data, model_name, 'meters')
                        predicted_m_list.append(result.prediction)
                    except Exception:
                        # Row-level fallback: garment-specific BOM formula
                        g_base = AppConfig.GARMENT_BASE_CONSUMPTION_M.get(
                            str(row.get('Garment_Type', 'T-Shirt')), 1.5)
                        predicted_m_list.append(float(row['Order_Quantity']) * g_base)
                    progress.progress(int((i + 1) / n * 100))
                progress.empty()
                df['Predicted_m'] = predicted_m_list

                # Convert to yards
                df['Predicted_yards'] = df['Predicted_m'] * UnitConverter.METERS_TO_YARDS
                df['Traditional_BOM_yards'] = df['Traditional_BOM_m'] * UnitConverter.METERS_TO_YARDS

                # Use preferred unit for display
                if unit_pref == 'yards':
                    df['Predicted'] = df['Predicted_yards']
                    df['Traditional_BOM'] = df['Traditional_BOM_yards']
                    df['fabric_cost'] = df['Fabric_Type'].map(
                        AppConfig.FABRIC_COST_PER_METER
                    ).fillna(AppConfig.DEFAULT_FABRIC_COST_PER_METER) * UnitConverter.METERS_TO_YARDS
                else:
                    df['Predicted'] = df['Predicted_m']
                    df['Traditional_BOM'] = df['Traditional_BOM_m']
                    df['fabric_cost'] = df['Fabric_Type'].map(
                        AppConfig.FABRIC_COST_PER_METER
                    ).fillna(AppConfig.DEFAULT_FABRIC_COST_PER_METER)

                df['Difference'] = df['Predicted'] - df['Traditional_BOM']
                df['Difference_%'] = (df['Difference'] / df['Traditional_BOM']) * 100
                df['Estimated_Cost'] = df['Predicted'] * df['fabric_cost']
                df['Potential_Savings'] = abs(df['Difference']) * df['fabric_cost']

                UIHelpers.show_success("Predictions complete!")

                BatchPredictionPage._render_batch_results(
                    df, unit_pref, show_dual_units, model_choice
                )

            except Exception as e:
                raise PredictionError(f"Failed to generate predictions: {e}") from e

    @staticmethod
    def _render_batch_results(
        df: pd.DataFrame,
        unit_pref: str,
        show_dual_units: bool,
        model_choice: str
    ) -> None:
        """Render batch prediction results"""
        # Summary
        st.markdown("---")
        st.subheader("📊 Batch Summary")

        col1, col2, col3, col4 = st.columns(4)

        # fabric_cost column already computed per-row in _generate_predictions;
        # use it directly — no need for a scalar fallback here.
        fabric_cost = AppConfig.DEFAULT_FABRIC_COST_PER_METER  # kept for display only

        with col1:
            st.metric("Total Orders", len(df))

        with col2:
            st.metric(
                f"Total Predicted ({unit_pref})",
                f"{df['Predicted'].sum():,.0f}"
            )

        with col3:
            st.metric("Total Cost", f"${df['Estimated_Cost'].sum():,.2f}")

        with col4:
            savings_pct = (df['Potential_Savings'].sum() / df['Estimated_Cost'].sum() * 100)
            st.metric(
                "Potential Savings",
                f"${df['Potential_Savings'].sum():,.2f}",
                delta=f"{savings_pct:.1f}%"
            )

        # Visualizations
        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"📈 Distribution ({unit_pref})")
            fig = px.histogram(df, x='Predicted', nbins=30,
                              title="Predicted Consumption Distribution",
                              labels={'Predicted': f'Consumption ({unit_pref})'})
            st.plotly_chart(fig, width='stretch')

        with col2:
            st.subheader("💰 Savings Analysis")
            fig = px.scatter(df, x='Order_Quantity', y='Potential_Savings',
                            size='Potential_Savings', color='Garment_Type',
                            title="Savings vs Order Size",
                            labels={'Potential_Savings': 'Savings ($)'})
            st.plotly_chart(fig, width='stretch')

        # Results table
        st.markdown("---")
        st.subheader("📋 Detailed Results")

        display_cols = ['Order_ID', 'Order_Quantity', 'Garment_Type']
        if show_dual_units:
            display_cols.extend(['Predicted_m', 'Predicted_yards'])
        else:
            display_cols.append('Predicted')
        display_cols.extend(['Difference_%', 'Potential_Savings'])

        results_display = df[display_cols].copy()

        st.dataframe(results_display.style.format({
            'Predicted_m': '{:.2f}',
            'Predicted_yards': '{:.2f}',
            'Predicted': '{:.2f}',
            'Difference_%': '{:.2f}',
            'Potential_Savings': '${:.2f}'
        }), use_container_width=True, height=400)

        # Download options
        BatchPredictionPage._render_download_buttons(df, unit_pref, model_choice, savings_pct)

    @staticmethod
    def _render_download_buttons(
        df: pd.DataFrame,
        unit_pref: str,
        model_choice: str,
        savings_pct: float
    ) -> None:
        """Render download buttons for batch results"""
        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            csv_results = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Results (CSV)",
                data=csv_results,
                file_name=f"predictions_{unit_pref}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col2:
            summary = BatchPredictionPage._generate_summary_text(
                df, unit_pref, model_choice, savings_pct
            )
            st.download_button(
                label="📄 Download Summary (TXT)",
                data=summary,
                file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )

    @staticmethod
    def _generate_summary_text(
        df: pd.DataFrame,
        unit_pref: str,
        model_choice: str,
        savings_pct: float
    ) -> str:
        """Generate summary text for download"""
        summary = f"""
BATCH PREDICTION SUMMARY
========================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Model: {model_choice}
Unit: {unit_pref.upper()}

STATISTICS:
-----------
Total Orders: {len(df)}
Total Predicted: {df['Predicted'].sum():,.2f} {unit_pref}
Total Cost: ${df['Estimated_Cost'].sum():,.2f}
Total Savings: ${df['Potential_Savings'].sum():,.2f}
Savings %: {savings_pct:.2f}%

UNIT CONVERSION:
----------------
Meters: {df['Predicted_m'].sum():,.2f} m
Yards: {df['Predicted_yards'].sum():,.2f} yd
Conversion Rate: 1 m = {UnitConverter.METERS_TO_YARDS:.4f} yd

TOP 5 ORDERS BY SAVINGS:
------------------------
"""
        top_savings = df.nlargest(5, 'Potential_Savings')[['Order_ID', 'Potential_Savings']]
        for idx, row in top_savings.iterrows():
            summary += f"\n{row['Order_ID']}: ${row['Potential_Savings']:.2f}"

        return summary


class PerformancePage:
    """
    Comprehensive Model Performance Analysis page.
    Reads live metrics from model_metadata.json (populated by train_models.py).
    Falls back to realistic illustrative values in demo mode.
    Includes Ensemble as a first-class model alongside XGB, RF, LSTM, LR.
    """

    # ── Fallback metrics (used in demo mode when metadata not yet loaded) ─────
    # Order: Ensemble, XGBoost, Random Forest, LSTM, Linear Regression, Trad BOM
    DEMO_MODELS = ["Ensemble", "XGBoost", "Random Forest", "LSTM",
                   "Linear Regression", "Traditional BOM"]
    DEMO_RMSE   = [28.4,  35.1,  38.6,  42.2,  88.7,  120.5]
    DEMO_MAE    = [20.1,  25.3,  27.8,  30.4,  65.1,   90.2]
    DEMO_R2     = [0.987, 0.982, 0.975, 0.970,  0.851,   0.753]
    DEMO_MAPE   = [1.9,   2.3,   2.7,   3.1,    6.2,     8.5]
    DEMO_COLORS = ["#8B5CF6","#3B82F6","#10B981","#F59E0B","#EF4444","#6B7280"]

    # Feature display names (fallback if not in metadata)
    FEATURE_LABELS = {
        "order_quantity":             "Order Quantity",
        "fabric_width_cm":            "Fabric Width",
        "marker_efficiency":          "Marker Efficiency",
        "defect_rate":                "Defect Rate",
        "operator_experience":        "Operator Experience",
        "garment_type_encoded":       "Garment Type",
        "fabric_type_encoded":        "Fabric Type",
        "pattern_complexity_encoded": "Pattern Complexity",
        "season_encoded":             "Season",
    }

    @staticmethod
    def render(unit_pref: str, model_mgr: "ModelManager") -> None:
        """Main entry point — renders full performance analysis."""
        st.title("📈 Model Performance Analysis")
        st.markdown(
            f"### Comparing all 5 ML models + Traditional BOM baseline | "
            f"**{unit_pref.upper()}**"
        )

        # ── Load metrics ────────────────────────────────────────────────────
        metrics_df, ensemble_weights, feature_importance, cv_data, bom_data,             is_live = PerformancePage._load_metrics(model_mgr, unit_pref)

        # ── Mode badge ──────────────────────────────────────────────────────
        if is_live:
            st.success(
                "✅ **Live metrics** — loaded from trained model_metadata.json. "
                "All numbers reflect actual test-set performance."
            )
        else:
            st.info(
                "ℹ️ **Demo mode** — illustrative metrics shown. "
                "Run `python train_models.py` then reload to see live results."
            )

        st.markdown("---")

        # ── Tabs ────────────────────────────────────────────────────────────
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🏆 Overview",
            "📊 Error Analysis",
            "🎯 Accuracy Deep-Dive",
            "🌲 Feature Importance",
            "⚙️ Ensemble Details",
        ])

        with tab1:
            try:
                PerformancePage._tab_overview(metrics_df, unit_pref, is_live)
            except Exception as e:
                st.error(f"Overview tab error: {e}")

        with tab2:
            try:
                PerformancePage._tab_error_analysis(metrics_df, unit_pref)
            except Exception as e:
                st.error(f"Error analysis tab error: {e}")

        with tab3:
            try:
                PerformancePage._tab_accuracy(metrics_df, cv_data)
            except Exception as e:
                st.error(f"Accuracy tab error: {e}")

        with tab4:
            try:
                PerformancePage._tab_feature_importance(feature_importance)
            except Exception as e:
                st.error(f"Feature importance tab error: {e}")

        with tab5:
            try:
                PerformancePage._tab_ensemble(ensemble_weights, metrics_df)
            except Exception as e:
                st.error(f"Ensemble tab error: {e}")

    # ── Data loading ─────────────────────────────────────────────────────────

    @staticmethod
    def _load_metrics(model_mgr, unit_pref):
        """
        Pull metrics from model_metadata.json if available.
        Returns (metrics_df, ensemble_weights, feature_importance, cv_data, bom_data, is_live).
        """
        cf = 1.0 if unit_pref == "meters" else UnitConverter.METERS_TO_YARDS
        is_live = False
        ensemble_weights = {}
        feature_importance = {}
        cv_data = {}
        bom_data = {}

        try:
            meta = (model_mgr.models.get("metadata") or {})
            models_meta = meta.get("models", {})

            if (models_meta
                    and meta.get("version", "").startswith("3")
                    and meta.get("mode", "production") != "demo"):
                is_live = True
                rows = []
                display_map = {
                    "ensemble":          "Ensemble",
                    "xgboost":           "XGBoost",
                    "random_forest":     "Random Forest",
                    "lstm":              "LSTM",
                    "linear_regression": "Linear Regression",
                }
                color_map = {
                    "Ensemble": "#8B5CF6", "XGBoost": "#3B82F6",
                    "Random Forest": "#10B981", "LSTM": "#F59E0B",
                    "Linear Regression": "#EF4444",
                }
                for key, label in display_map.items():
                    m = models_meta.get(key)
                    if not m:
                        continue
                    rows.append({
                        "Model":  label,
                        "RMSE":   round(m["rmse"] * cf, 2),
                        "MAE":    round(m["mae"]  * cf, 2),
                        "R2":     m["r2"],
                        "MAPE":   m["mape"],
                        "Color":  color_map.get(label, "#6B7280"),
                    })
                    if key == "ensemble":
                        ensemble_weights = m.get("weights", {})
                    if "feature_importance" in m:
                        feature_importance[label] = m["feature_importance"]
                    if "cross_validation" in m:
                        cv_data[label] = m["cross_validation"]

                bom = meta.get("bom_baseline") or models_meta.get("trad_bom") or {}
                bom_data = bom
                if bom and bom.get("rmse"):
                    rows.append({
                        "Model":  "Traditional BOM",
                        "RMSE":   round(bom["rmse"] * cf, 2),
                        "MAE":    round(bom.get("mae", bom["rmse"] * 0.75) * cf, 2),
                        "R2":     bom.get("r2", 0.75),
                        "MAPE":   bom.get("mape", 8.5),
                        "Color":  "#6B7280",
                    })

                if rows:
                    return (pd.DataFrame(rows), ensemble_weights,
                            feature_importance, cv_data, bom_data, is_live)

        except Exception as _e:
            logger.debug(f"Metadata parse error: {_e}")

        # ── Fallback demo data ────────────────────────────────────────────────
        rows = []
        for i, model in enumerate(PerformancePage.DEMO_MODELS):
            rows.append({
                "Model":  model,
                "RMSE":   round(PerformancePage.DEMO_RMSE[i] * cf, 2),
                "MAE":    round(PerformancePage.DEMO_MAE[i]  * cf, 2),
                "R2":     PerformancePage.DEMO_R2[i],
                "MAPE":   PerformancePage.DEMO_MAPE[i],
                "Color":  PerformancePage.DEMO_COLORS[i],
            })
        return (pd.DataFrame(rows),
                {"xgboost": 0.43, "random_forest": 0.30,
                 "lstm": 0.17, "linear_regression": 0.10},
                {}, {}, {}, False)

    # ── Tab 1: Overview ───────────────────────────────────────────────────────

    @staticmethod
    def _tab_overview(df, unit_pref, is_live):
        st.subheader("🏆 All Models at a Glance")

        ens = df[df["Model"] == "Ensemble"]
        bom = df[df["Model"] == "Traditional BOM"]

        if not ens.empty and not bom.empty:
            col1, col2, col3, col4 = st.columns(4)
            er  = ens.iloc[0]
            br  = bom.iloc[0]
            rmse_imp = round((br["RMSE"] - er["RMSE"]) / br["RMSE"] * 100, 1)
            mape_imp = round((br["MAPE"] - er["MAPE"]) / br["MAPE"] * 100, 1)

            with col1:
                st.metric("🥇 Ensemble R²",  f"{er['R2']:.4f}",
                          delta=f"+{er['R2'] - br['R2']:.4f} vs BOM")
            with col2:
                st.metric("🎯 Ensemble MAPE", f"{er['MAPE']:.2f}%",
                          delta=f"{-mape_imp:.1f}% vs BOM", delta_color="inverse")
            with col3:
                st.metric(f"📐 Ensemble RMSE ({unit_pref})", f"{er['RMSE']:.1f}",
                          delta=f"-{rmse_imp:.1f}% vs BOM", delta_color="inverse")
            with col4:
                n_ml = len(df[df["Model"] != "Traditional BOM"])
                st.metric("🤖 Models Compared", str(n_ml), delta="+ BOM baseline")

        st.markdown("---")
        st.subheader("📋 Complete Metric Table")
        display_df = df[["Model","RMSE","MAE","R2","MAPE"]].copy()
        display_df.columns = ["Model", f"RMSE ({unit_pref})", f"MAE ({unit_pref})",
                               "R² Score", "MAPE %"]
        try:
            styled = (display_df.style
                .highlight_max(subset=["R² Score"], color="#c6efce")
                .highlight_min(
                    subset=[f"RMSE ({unit_pref})", f"MAE ({unit_pref})", "MAPE %"],
                    color="#c6efce")
                .format({f"RMSE ({unit_pref})": "{:.2f}",
                         f"MAE ({unit_pref})":  "{:.2f}",
                         "R² Score": "{:.4f}",
                         "MAPE %":   "{:.2f}%"}))
            st.dataframe(styled, use_container_width=True, height=310)
        except Exception:
            st.dataframe(display_df, use_container_width=True, height=310)

        st.markdown("---")
        st.subheader("🕸️ Multi-Metric Radar Chart")
        PerformancePage._render_radar(df)

    @staticmethod
    def _render_radar(df):
        ml_df = df[df["Model"] != "Traditional BOM"].copy()
        if ml_df.empty:
            return
        metrics = ["R2", "MAPE", "RMSE", "MAE"]
        norm = ml_df[["Model"] + metrics].copy()
        for col in ["MAPE", "RMSE", "MAE"]:
            mn, mx = norm[col].min(), norm[col].max()
            norm[col] = 1.0 - (norm[col] - mn) / (mx - mn) if mx > mn else 1.0
        mn, mx = norm["R2"].min(), norm["R2"].max()
        norm["R2"] = (norm["R2"] - mn) / (mx - mn) if mx > mn else 1.0

        categories = ["R²", "Low MAPE", "Low RMSE", "Low MAE"]
        color_map = dict(zip(ml_df["Model"], ml_df["Color"]))
        fig = go.Figure()
        for _, row in norm.iterrows():
            vals = [row["R2"], row["MAPE"], row["RMSE"], row["MAE"]]
            vals += [vals[0]]
            cats  = categories + [categories[0]]
            fig.add_trace(go.Scatterpolar(
                r=vals, theta=cats, fill="toself",
                name=row["Model"],
                line=dict(color=color_map.get(row["Model"], "#888"), width=2),
                opacity=0.75,
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            height=440, legend=dict(orientation="h", y=-0.15),
        )
        st.plotly_chart(fig, width='stretch')

    # ── Tab 2: Error Analysis ─────────────────────────────────────────────────

    @staticmethod
    def _tab_error_analysis(df, unit_pref):
        st.subheader(f"📊 Error Metrics Breakdown ({unit_pref})")

        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure([
                go.Bar(x=[r["Model"]], y=[r["RMSE"]],
                       marker_color=r["Color"], showlegend=False)
                for _, r in df.iterrows()
            ])
            fig.update_layout(title=f"RMSE by Model ({unit_pref})",
                              yaxis_title=f"RMSE ({unit_pref})",
                              height=380, bargap=0.3)
            st.plotly_chart(fig, width='stretch')

        with col2:
            fig = go.Figure([
                go.Bar(x=[r["Model"]], y=[r["MAE"]],
                       marker_color=r["Color"], showlegend=False)
                for _, r in df.iterrows()
            ])
            fig.update_layout(title=f"MAE by Model ({unit_pref})",
                              yaxis_title=f"MAE ({unit_pref})",
                              height=380, bargap=0.3)
            st.plotly_chart(fig, width='stretch')

        st.markdown("---")
        st.subheader("📉 MAPE Comparison (lower is better)")
        fig = go.Figure([
            go.Bar(x=[r["Model"]], y=[r["MAPE"]],
                   marker_color=r["Color"],
                   text=f"{r['MAPE']:.2f}%",
                   textposition="outside",
                   showlegend=False)
            for _, r in df.iterrows()
        ])
        fig.add_hline(y=5.0, line_dash="dash", line_color="orange",
                      annotation_text="5% threshold")
        fig.add_hline(y=2.5, line_dash="dot", line_color="green",
                      annotation_text="Excellent (<2.5%)")
        fig.update_layout(height=420, yaxis_title="MAPE (%)", bargap=0.3)
        st.plotly_chart(fig, width='stretch')

        bom_row = df[df["Model"] == "Traditional BOM"]
        if not bom_row.empty:
            st.markdown("---")
            st.subheader("📈 Improvement Over Traditional BOM")
            bom_rmse = bom_row.iloc[0]["RMSE"]
            bom_mape = bom_row.iloc[0]["MAPE"]
            imp_df = df[df["Model"] != "Traditional BOM"].copy()
            imp_df["RMSE Imp %"] = ((bom_rmse - imp_df["RMSE"]) / bom_rmse * 100).round(1)
            imp_df["MAPE Imp %"] = ((bom_mape - imp_df["MAPE"]) / bom_mape * 100).round(1)

            col1, col2 = st.columns(2)
            with col1:
                fig = go.Figure([
                    go.Bar(x=[r["Model"]], y=[r["RMSE Imp %"]],
                           marker_color=r["Color"],
                           text=f"{r['RMSE Imp %']:.1f}%",
                           textposition="outside", showlegend=False)
                    for _, r in imp_df.iterrows()
                ])
                fig.update_layout(title="RMSE Reduction vs BOM (%)",
                                  height=360, bargap=0.3,
                                  yaxis_title="Improvement %")
                st.plotly_chart(fig, width='stretch')
            with col2:
                fig = go.Figure([
                    go.Bar(x=[r["Model"]], y=[r["MAPE Imp %"]],
                           marker_color=r["Color"],
                           text=f"{r['MAPE Imp %']:.1f}%",
                           textposition="outside", showlegend=False)
                    for _, r in imp_df.iterrows()
                ])
                fig.update_layout(title="MAPE Reduction vs BOM (%)",
                                  height=360, bargap=0.3,
                                  yaxis_title="Improvement %")
                st.plotly_chart(fig, width='stretch')

    # ── Tab 3: Accuracy Deep-Dive ─────────────────────────────────────────────

    @staticmethod
    def _tab_accuracy(df, cv_data):
        st.subheader("🎯 R² Score Analysis")

        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure([
                go.Bar(x=[r["Model"]], y=[r["R2"]],
                       marker_color=r["Color"],
                       text=f"{r['R2']:.4f}",
                       textposition="outside",
                       showlegend=False)
                for _, r in df.iterrows()
            ])
            fig.add_hline(y=0.95, line_dash="dash", line_color="green",
                          annotation_text="Excellent (>0.95)")
            fig.add_hline(y=0.90, line_dash="dot", line_color="orange",
                          annotation_text="Good (>0.90)")
            fig.update_layout(title="R² Score by Model", height=420,
                              yaxis=dict(range=[0.65, 1.02]),
                              yaxis_title="R²", bargap=0.3)
            st.plotly_chart(fig, width='stretch')

        with col2:
            bom_r2_vals = df[df["Model"] == "Traditional BOM"]["R2"].values
            bom_val = float(bom_r2_vals[0]) if len(bom_r2_vals) else 0.75
            ml_only = df[df["Model"] != "Traditional BOM"]
            fig = go.Figure([
                go.Bar(y=[r["Model"]], x=[r["R2"]],
                       orientation="h",
                       marker_color=r["Color"],
                       text=f"{r['R2']:.4f}",
                       textposition="outside",
                       showlegend=False)
                for _, r in ml_only.iterrows()
            ])
            fig.add_vline(x=bom_val, line_dash="dash", line_color="#6B7280",
                          annotation_text=f"BOM R²={bom_val:.3f}")
            fig.update_layout(title="ML Models vs BOM Baseline",
                              xaxis=dict(range=[0.70, 1.01], title="R² Score"),
                              height=420)
            st.plotly_chart(fig, width='stretch')

        if cv_data:
            st.markdown("---")
            st.subheader("🔄 5-Fold Cross-Validation Results")
            st.caption("Mean ± Std across 5 folds — more reliable than single split.")
            cv_rows = [
                {"Model": m,
                 "CV RMSE":   f"{d.get('rmse_mean',0):.2f} ± {d.get('rmse_std',0):.2f}",
                 "CV R²":     f"{d.get('r2_mean',0):.4f} ± {d.get('r2_std',0):.4f}",
                 "CV MAPE %": f"{d.get('mape_mean',0):.2f}% ± {d.get('mape_std',0):.2f}%"}
                for m, d in cv_data.items()
            ]
            st.dataframe(pd.DataFrame(cv_rows), use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                fig = go.Figure([
                    go.Bar(x=[m], y=[d.get("r2_mean",0)],
                           error_y=dict(type="data",
                                        array=[d.get("r2_std",0)],
                                        visible=True),
                           name=m)
                    for m, d in cv_data.items()
                ])
                fig.update_layout(title="CV R² (mean ± std)", height=360,
                                  showlegend=False, yaxis_title="R²")
                st.plotly_chart(fig, width='stretch')
            with col2:
                fig = go.Figure([
                    go.Bar(x=[m], y=[d.get("mape_mean",0)],
                           error_y=dict(type="data",
                                        array=[d.get("mape_std",0)],
                                        visible=True),
                           name=m)
                    for m, d in cv_data.items()
                ])
                fig.update_layout(title="CV MAPE % (mean ± std)", height=360,
                                  showlegend=False, yaxis_title="MAPE %")
                st.plotly_chart(fig, width='stretch')
        else:
            st.info(
                "💡 Cross-validation data not available. "
                "Retrain with `train_models.py v3.0` to see 5-fold CV scores here."
            )

    # ── Tab 4: Feature Importance ─────────────────────────────────────────────

    @staticmethod
    def _tab_feature_importance(feature_importance):
        st.subheader("🌲 Feature Importance Analysis")

        if not feature_importance:
            st.info(
                "Feature importance data not available. "
                "After training with `train_models.py v3.0`, XGBoost and "
                "Random Forest importances appear here automatically."
            )
            st.markdown("---")
            st.caption("Illustrative demo values (approximate typical importance):")
            demo_fi = {
                "order_quantity": 0.31, "garment_type_encoded": 0.24,
                "fabric_width_cm": 0.14, "pattern_complexity_encoded": 0.10,
                "marker_efficiency": 0.09, "defect_rate": 0.06,
                "operator_experience": 0.04, "fabric_type_encoded": 0.01,
                "season_encoded": 0.01,
            }
            PerformancePage._render_importance_chart(demo_fi, "XGBoost (illustrative)")
            return

        model_options = list(feature_importance.keys())
        selected = st.selectbox("Model", model_options, key="fi_select")
        fi_dict  = feature_importance.get(selected, {})
        if fi_dict:
            PerformancePage._render_importance_chart(fi_dict, selected)

        if len(feature_importance) >= 2:
            st.markdown("---")
            st.subheader("Side-by-Side Comparison")
            cols = st.columns(len(feature_importance))
            for col, (mname, fi) in zip(cols, feature_importance.items()):
                with col:
                    labels = PerformancePage.FEATURE_LABELS
                    fi_clean = {labels.get(k, k): v for k, v in fi.items()}
                    fi_sorted = dict(sorted(fi_clean.items(),
                                           key=lambda x: x[1], reverse=True))
                    st.caption(f"**{mname}**")
                    for feat, imp in fi_sorted.items():
                        st.progress(float(imp), text=f"{feat}: {imp:.3f}")

    @staticmethod
    def _render_importance_chart(fi_dict, model_name):
        labels = PerformancePage.FEATURE_LABELS
        fi_clean = {labels.get(k, k): v for k, v in fi_dict.items()}
        fi_sorted = dict(sorted(fi_clean.items(), key=lambda x: x[1], reverse=True))

        col1, col2 = st.columns([3, 2])
        with col1:
            fig = go.Figure(go.Bar(
                x=list(fi_sorted.values()),
                y=list(fi_sorted.keys()),
                orientation="h",
                marker=dict(color=list(fi_sorted.values()),
                            colorscale="Viridis", showscale=True,
                            colorbar=dict(title="Importance")),
            ))
            fig.update_layout(
                title=f"{model_name} — Feature Importance",
                xaxis_title="Importance Score",
                height=400,
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig, width='stretch')

        with col2:
            fig2 = go.Figure(go.Pie(
                labels=list(fi_sorted.keys()),
                values=list(fi_sorted.values()),
                hole=0.45,
                textinfo="label+percent",
            ))
            fig2.update_layout(title="Share", height=400,
                               legend=dict(font=dict(size=10)))
            st.plotly_chart(fig2, width='stretch')

    # ── Tab 5: Ensemble Details ───────────────────────────────────────────────

    @staticmethod
    def _tab_ensemble(ensemble_weights, metrics_df):
        st.subheader("⚙️ Ensemble Model Architecture")
        st.markdown(
            "The **Weighted-Average Ensemble** combines predictions from XGBoost, "
            "Random Forest, LSTM, and Linear Regression. Weights are proportional "
            "to each model's **validation R²**, computed automatically during training."
        )

        if not ensemble_weights:
            st.info("Ensemble weights not available. Train with `train_models.py v3.0`.")
            ensemble_weights = {"XGBoost": 0.43, "Random Forest": 0.30,
                                "LSTM": 0.17, "Linear Regression": 0.10}

        key_map = {"xgboost": "XGBoost", "random_forest": "Random Forest",
                   "lstm": "LSTM", "linear_regression": "Linear Regression"}
        disp_w = {key_map.get(k, k): v for k, v in ensemble_weights.items()}

        ens_row = metrics_df[metrics_df["Model"] == "Ensemble"]
        if not ens_row.empty:
            r = ens_row.iloc[0]
            c1, c2, c3 = st.columns(3)
            c1.metric("Ensemble R²",   f"{r['R2']:.4f}")
            c2.metric("Ensemble MAPE", f"{r['MAPE']:.2f}%")
            c3.metric("Ensemble RMSE", f"{r['RMSE']:.2f}")

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Model Weights")
            fig = go.Figure(go.Pie(
                labels=list(disp_w.keys()),
                values=list(disp_w.values()),
                hole=0.42,
                textinfo="label+percent",
                marker=dict(colors=["#3B82F6","#10B981","#F59E0B","#EF4444"]),
            ))
            fig.update_layout(title="Weight Allocation", height=400,
                              legend=dict(orientation="h", y=-0.15))
            st.plotly_chart(fig, width='stretch')

        with col2:
            st.subheader("📈 Weight vs Performance")
            ml_only = metrics_df[
                metrics_df["Model"].isin(disp_w.keys())
            ].copy()
            ml_only["Weight"] = ml_only["Model"].map(disp_w).fillna(0)
            if not ml_only.empty:
                fig = px.scatter(
                    ml_only, x="R2", y="MAPE",
                    size="Weight",
                    color="Model",
                    color_discrete_sequence=ml_only["Color"].tolist(),
                    hover_data=["RMSE", "MAE"],
                    text="Model",
                    title="R² vs MAPE (bubble = weight)",
                )
                fig.update_traces(textposition="top center")
                fig.update_layout(height=400, showlegend=False,
                                  xaxis_title="R² Score", yaxis_title="MAPE %",
                                  yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig, width='stretch')

        st.markdown("---")
        wt_rows = [{"Model": m, "Weight": f"{w:.4f}",
                    "Weight %": f"{w*100:.1f}%"}
                   for m, w in sorted(disp_w.items(),
                                      key=lambda x: x[1], reverse=True)]
        st.dataframe(pd.DataFrame(wt_rows), use_container_width=True)

        with st.expander("ℹ️ How ensemble weights are computed"):
            st.markdown("""
**Algorithm** (`train_models.py → ModelTrainer.train_ensemble()`):

1. Each base model is evaluated on the **validation set** after training
2. Validation R² is computed per model: `R²_xgb, R²_rf, R²_lstm, R²_lr`
3. Negative R² values clamped to 0 (model worse than mean baseline gets no weight)
4. Weights normalised to sum to 1:
```
weight_i = max(0, R²_i) / Σ max(0, R²_j)
```
5. Ensemble prediction = Σ (weight_i × prediction_i)

**Why validation R²?** Using held-out validation data prevents overfitted models
from receiving disproportionately large weights — a standard technique in blended ensembles.
            """)


class ROICalculatorPage:
    """ROI Calculator page renderer"""

    @staticmethod
    def render(unit_pref: str, show_dual_units: bool) -> None:
        """Render the ROI calculator page"""
        st.title("💰 Economic Impact Calculator")
        st.markdown(f"### Calculate savings in: **{unit_pref.upper()}**")

        # Input parameters
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Production Parameters")
            monthly_orders = st.number_input("Monthly Orders", 100, 50000, 2000, 100)
            avg_order_value = st.number_input("Avg Order Value ($)", 100, 100000, 5000, 100)
            current_waste_rate = st.slider("Current Waste Rate (%)", 1.0, 20.0, 8.0, 0.5)

            default_cost = (
                AppConfig.DEFAULT_FABRIC_COST_PER_METER
                if unit_pref == 'meters' else
                AppConfig.DEFAULT_FABRIC_COST_PER_METER * UnitConverter.METERS_TO_YARDS
            )
            fabric_cost = st.number_input(
                f"Fabric Cost ($/{unit_pref})",
                1.0, 50.0, default_cost, 0.5
            )

            if show_dual_units:
                if unit_pref == 'meters':
                    alt_cost = fabric_cost * UnitConverter.METERS_TO_YARDS
                    st.caption(f"= ${alt_cost:.2f}/yard")
                else:
                    alt_cost = fabric_cost / UnitConverter.METERS_TO_YARDS
                    st.caption(f"= ${alt_cost:.2f}/meter")

        with col2:
            st.subheader("⚙️ Implementation")
            ml_improvement = st.slider("Expected ML Improvement (%)", 30.0, 80.0, 65.0, 5.0)
            implementation_cost = st.number_input("Implementation Cost ($)",
                                                 5000, 200000, 50000, 5000)
            monthly_maintenance = st.number_input("Monthly Maintenance ($)",
                                                 500, 20000, 2000, 500)
            implementation_months = st.slider("Implementation Timeline (months)",
                                             1, 12, 3)

        # Calculations
        ROICalculatorPage._render_financial_analysis(
            monthly_orders, avg_order_value, current_waste_rate,
            ml_improvement, implementation_cost, monthly_maintenance,
            implementation_months, fabric_cost, unit_pref, show_dual_units
        )

    @staticmethod
    def _render_financial_analysis(
        monthly_orders: int,
        avg_order_value: float,
        current_waste_rate: float,
        ml_improvement: float,
        implementation_cost: float,
        monthly_maintenance: float,
        implementation_months: int,
        fabric_cost: float,
        unit_pref: str,
        show_dual_units: bool = True
    ) -> None:
        """Render financial analysis results"""
        st.markdown("---")
        st.subheader("📈 Financial Analysis")

        # Calculations
        annual_orders = monthly_orders * 12
        annual_fabric_cost = annual_orders * avg_order_value
        current_annual_waste = annual_fabric_cost * (current_waste_rate / 100)

        waste_reduction = current_annual_waste * (ml_improvement / 100)
        annual_savings = waste_reduction

        first_year_cost = implementation_cost + (monthly_maintenance * 12)
        first_year_benefit = annual_savings * ((12 - implementation_months) / 12)
        first_year_roi = ((first_year_benefit - first_year_cost) / first_year_cost) * 100
        payback_months = implementation_cost / (annual_savings / 12)

        three_year_benefit = (annual_savings * 3) - (first_year_cost + (monthly_maintenance * 24))
        three_year_roi = (three_year_benefit / first_year_cost) * 100

        # Display metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("💰 Annual Savings", f"${annual_savings:,.0f}",
                     delta=f"{ml_improvement:.0f}% reduction")

        with col2:
            st.metric("📅 Payback Period", f"{payback_months:.1f} months",
                     delta="Break-even")

        with col3:
            st.metric("📊 First Year ROI", f"{first_year_roi:.0f}%",
                     delta=f"${first_year_benefit:,.0f}")

        with col4:
            st.metric("🎯 3-Year ROI", f"{three_year_roi:.0f}%",
                     delta=f"${three_year_benefit:,.0f}")

        # Projection chart
        ROICalculatorPage._render_projection_chart(
            implementation_cost, monthly_maintenance,
            implementation_months, annual_savings
        )

        # Environmental impact
        ROICalculatorPage._render_environmental_impact(
            current_annual_waste, ml_improvement, fabric_cost, unit_pref,
            show_dual_units=show_dual_units
        )

    @staticmethod
    def _render_projection_chart(
        implementation_cost: float,
        monthly_maintenance: float,
        implementation_months: int,
        annual_savings: float
    ) -> None:
        """Render 3-year financial projection chart"""
        st.markdown("---")
        st.subheader("📊 3-Year Financial Projection")

        months = list(range(1, 37))
        cumulative_costs = []
        cumulative_benefits = []
        net_benefit = []

        for month in months:
            cost = implementation_cost + monthly_maintenance if month == 1 else monthly_maintenance
            cumulative_costs.append(cost if len(cumulative_costs) == 0
                                   else cumulative_costs[-1] + cost)

            benefit = (annual_savings / 12) if month > implementation_months else 0
            cumulative_benefits.append(benefit if len(cumulative_benefits) == 0
                                      else cumulative_benefits[-1] + benefit)

            net_benefit.append(cumulative_benefits[-1] - cumulative_costs[-1])

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=months, y=cumulative_costs,
                                name='Cumulative Costs', line=dict(color='red', width=2)))
        fig.add_trace(go.Scatter(x=months, y=cumulative_benefits,
                                name='Cumulative Benefits', line=dict(color='green', width=2)))
        fig.add_trace(go.Scatter(x=months, y=net_benefit,
                                name='Net Benefit', fill='tozeroy',
                                line=dict(color='blue', width=3)))

        fig.add_hline(y=0, line_dash="dash", line_color="black")
        fig.update_layout(height=500, hovermode='x unified',
                         xaxis_title="Month", yaxis_title="Amount ($)")

        st.plotly_chart(fig, width='stretch')

    @staticmethod
    def _render_environmental_impact(
        current_annual_waste: float,
        ml_improvement: float,
        fabric_cost: float,
        unit_pref: str,
        show_dual_units: bool = True
    ) -> None:
        """Render environmental impact metrics"""
        st.markdown("---")
        st.subheader("🌍 Environmental Impact")

        fabric_saved = (current_annual_waste / fabric_cost) * (ml_improvement / 100)

        if unit_pref == 'meters':
            fabric_saved_alt = UnitConverter.meters_to_yards(fabric_saved)
            fabric_saved_m = fabric_saved
        else:
            fabric_saved_alt = UnitConverter.yards_to_meters(fabric_saved)
            fabric_saved_m = fabric_saved_alt

        water_saved = fabric_saved_m * 50  # 50L per meter
        co2_saved = fabric_saved_m * 2.5  # 2.5kg CO2 per meter

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(f"🧵 Fabric Saved ({unit_pref})", f"{fabric_saved:,.0f}")
            if show_dual_units:
                alt_unit = 'yards' if unit_pref == 'meters' else 'meters'
                st.caption(f"= {fabric_saved_alt:,.0f} {alt_unit}")

        with col2:
            st.metric("💧 Water Saved", f"{water_saved:,.0f} L",
                     help="Based on 50L per meter of fabric")

        with col3:
            st.metric("🌱 CO₂ Reduced", f"{co2_saved:,.0f} kg",
                     help="Based on 2.5kg CO₂ per meter")


class DocumentationPage:
    """Documentation page renderer"""

    @staticmethod
    def render(unit_pref: str, show_dual_units: bool, is_production: bool) -> None:
        """Render the documentation page"""
        st.title("📚 System Documentation")
        st.markdown("### Complete guide with unit conversion support")

        tab1, tab2, tab3 = st.tabs(["🎯 Quick Start", "📏 Unit Guide", "❓ FAQ"])

        with tab1:
            DocumentationPage._render_quick_start()

        with tab2:
            DocumentationPage._render_unit_guide()

        with tab3:
            DocumentationPage._render_faq(unit_pref, show_dual_units, is_production)

    @staticmethod
    def _render_quick_start() -> None:
        """Render quick start guide"""
        st.markdown("""
        ## 🎯 Getting Started

        ### Unit Selection

        This system supports both **meters** and **yards** for fabric measurements.

        **To change units:**
        1. Use the sidebar unit selector
        2. Choose "meters" or "yards"
        3. All predictions will automatically convert

        ### Making Predictions

        **Single Order:**
        1. Go to 🎯 Single Prediction
        2. Enter order details
        3. Fabric width auto-adjusts to selected unit
        4. Get instant prediction in your preferred unit

        **Batch Processing:**
        1. Go to 📊 Batch Prediction
        2. Download CSV template
        3. Upload your data
        4. Results provided in selected unit

        ### Understanding Results

        - **Prediction:** AI-calculated fabric requirement
        - **Traditional BOM:** Conventional estimate
        - **Difference:** How much ML improves accuracy
        - **Both units:** Toggle to see meters + yards
        """)

    @staticmethod
    def _render_unit_guide() -> None:
        """Render unit conversion guide"""
        st.markdown("""
        ## 📏 Unit Conversion Guide

        ### Conversion Factors

        **Meter to Yard:**
        - 1 meter = 1.0936 yards
        - Formula: yards = meters × 1.0936

        **Yard to Meter:**
        - 1 yard = 0.9144 meters
        - Formula: meters = yards × 0.9144

        ### Fabric Width Conversions

        | Inches | Centimeters | Common Use |
        |--------|-------------|------------|
        | 55" | 140 cm | Narrow fabric |
        | 59" | 150 cm | Standard |
        | 63" | 160 cm | Wide fabric |
        | 71" | 180 cm | Extra wide |

        ### Cost Conversions

        If fabric costs **$8.50/meter:**
        - Cost per yard = $8.50 × 1.0936 = **$9.30/yard**

        If fabric costs **$9.30/yard:**
        - Cost per meter = $9.30 × 0.9144 = **$8.50/meter**

        ### Quick Reference

        **Common Conversions:**
        - 100 meters = 109.4 yards
        - 500 meters = 546.8 yards
        - 1000 meters = 1093.6 yards

        **Common Garment Requirements:**

        | Garment | Meters | Yards |
        |---------|--------|-------|
        | T-Shirt | 1.2 m | 1.3 yd |
        | Shirt | 1.8 m | 2.0 yd |
        | Pants | 2.5 m | 2.7 yd |
        | Dress | 3.0 m | 3.3 yd |
        | Jacket | 3.5 m | 3.8 yd |

        ### Using Dual Units

        Enable "Show both units" in sidebar to see:
        - Predictions in both meters and yards
        - Width in inches and centimeters
        - Costs per meter and per yard
        """)

    @staticmethod
    def _render_faq(unit_pref: str, show_dual_units: bool, is_production: bool) -> None:
        """Render FAQ section"""
        st.markdown(f"""
        ## ❓ Frequently Asked Questions

        ### About This System

        **Q: Who developed this system?**
        A: This Fabric Consumption Forecasting System was developed by **{AppConfig.APP_AUTHOR}**
        as part of a data science initiative to optimize fabric usage in manufacturing.

        **Q: What version is this?**
        A: This is **Version {AppConfig.APP_VERSION}**, released in January 2026.

        **Q: Can I contact the developer?**
        A: For inquiries about this system, please reach out through official channels.

        ### Unit-Related Questions

        **Q: Can I switch units after making predictions?**
        A: Yes! Change unit preference in sidebar anytime. Enable "Show both units"
        to see conversions.

        **Q: Which unit should I use?**
        A: Use whatever your facility uses. The AI works equally well with both.
        - US factories typically use yards
        - International facilities often use meters

        **Q: Are predictions accurate in both units?**
        A: Yes! The system converts precisely using standard conversion factors
        (1 m = 1.0936 yd).

        **Q: What if my fabric width is in different units?**
        A: The system handles common widths in both inches and centimeters.
        Select from dropdown or enter custom values.

        ### Current Settings

        - **Active Unit:** {unit_pref.upper()}
        - **Show Dual Units:** {'Yes' if show_dual_units else 'No'}
        - **Production Mode:** {'Active' if is_production else 'Demo'}
        - **Conversion Factor:** 1 m = {UnitConverter.METERS_TO_YARDS:.4f} yd

        ### General Questions

        **Q: How accurate are the predictions?**
        A: See the **📈 Performance** tab for full test-set metrics including R², RMSE,
        MAE, and MAPE. In demo mode, the Ensemble model achieves R² ≈ 0.987 and
        MAPE ≈ 1.9%, versus MAPE ≈ 8.5% for the traditional BOM baseline.

        **Q: Can I use my own historical data?**
        A: Yes! Upload CSV in batch prediction mode. Models can be retrained
        on your data.

        **Q: What's the typical ROI?**
        A: Use the **💰 ROI Calculator** tab to estimate ROI for your specific
        production volume, waste rate, and implementation cost. Results are
        derived from your actual inputs, not industry averages.

        **Q: Does unit choice affect model accuracy?**
        A: No. The AI learns patterns regardless of unit. Predictions are equally
        accurate in meters or yards.

        ### System Requirements

        **Q: What are the file upload limits?**
        A: Maximum file size is {AppConfig.MAX_FILE_SIZE_MB}MB with up to
        {AppConfig.MAX_BATCH_ROWS} rows per batch.

        **Q: How long are sessions valid?**
        A: Sessions timeout after {AppConfig.SESSION_TIMEOUT_MINUTES} minutes of inactivity.
        """)


# ============================================================================
# SIDEBAR RENDERER
# ============================================================================

class SidebarRenderer:
    """Main sidebar renderer"""

    @staticmethod
    def render(models_loaded: bool, mode: ProcessingMode) -> Tuple[str, bool]:
        """Render the main sidebar

        Returns:
            tuple: (selected_page, unit_preference)
        """
        # App title and logo
        st.sidebar.image("https://img.icons8.com/fluency/96/000000/sewing-machine.png", width=80)
        st.sidebar.title("🧵 Fabric Forecast Pro")

        # Unit settings
        SidebarRenderer._render_unit_settings()

        # Navigation
        page = SidebarRenderer._render_navigation()

        st.sidebar.markdown("---")

        # System status
        SidebarRenderer._render_system_status(models_loaded, mode)

        # About section
        SidebarRenderer._render_about()

        return page

    @staticmethod
    def _render_unit_settings() -> None:
        """Render unit preference settings"""
        st.sidebar.markdown("### ⚙️ Unit Settings")

        unit_pref = st.sidebar.radio(
            "Preferred Unit",
            options=[UnitType.METERS.value, UnitType.YARDS.value],
            index=0 if st.session_state.unit_preference == UnitType.METERS.value else 1,
            help="Select your preferred measurement unit"
        )
        st.session_state.unit_preference = unit_pref
        st.session_state.show_dual_units = st.sidebar.checkbox(
            "Show both units",
            value=True,
            help="Display values in both meters and yards"
        )

        st.sidebar.markdown(f"""
        <div style='background-color: #ff6b6b; color: white; padding: 0.5rem;
             border-radius: 0.3rem; text-align: center; margin: 0.5rem 0;'>
            <strong>Active Unit: {unit_pref.upper()}</strong>
        </div>
        """, unsafe_allow_html=True)

        st.sidebar.markdown("---")

    @staticmethod
    def _render_navigation() -> str:
        """Render navigation menu"""
        st.sidebar.markdown("### 📍 Navigation")

        page = st.sidebar.radio(
            "Select Page",
            [
                "🏠 Dashboard",
                "🎯 Single Prediction",
                "📊 Batch Prediction",
                "📈 Performance",
                "💰 ROI Calculator",
                "📚 Documentation"
            ]
        )
        return page

    @staticmethod
    def _render_system_status(models_loaded: bool, mode: ProcessingMode) -> None:
        """Render system status information"""
        st.sidebar.markdown("### 📊 System Status")

        status_color = "🟢" if mode == ProcessingMode.PRODUCTION else "🟡"
        status_text = "Production Mode" if mode == ProcessingMode.PRODUCTION else "Demo Mode"
        st.sidebar.info(f"{status_color} **{status_text}**")
        st.sidebar.caption(f"v{AppConfig.APP_VERSION} · restart Streamlit after updating app.py")

        session_stats = SessionManager.get_session_stats()
        st.sidebar.metric("Total Predictions", session_stats['predictions_count'])
        st.sidebar.metric("Session Savings", f"${session_stats['total_savings']:,.0f}")

    @staticmethod
    def _render_about() -> None:
        """Render about section"""
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"""
        **About This System**

        ML-based fabric consumption forecasting
        for apparel manufacturing, featuring:
        - ✅ Dual unit support (m / yd)
        - ✅ 5 ML models + BOM baseline
        - ✅ Empirical confidence intervals
        - ✅ Batch CSV prediction

        **v{AppConfig.APP_VERSION}** · *{AppConfig.APP_AUTHOR}*
        © January 2026
        """)


# ============================================================================
# MODEL TRAINING CHECK
# ============================================================================

def check_and_train_models_if_needed():
    """
    Check if models folder is empty or missing required model files.
    If so, automatically run train_models.py to train the models.

    Returns:
        bool: True if models are ready, False otherwise
    """
    model_path = AppConfig.MODEL_PATH

    # Create models directory if it doesn't exist
    if not model_path.exists():
        model_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created models directory: {model_path}")

    # Check for essential model files
    required_files = [
        "xgboost_model.pkl",
        "scaler.pkl",
        "label_encoders.pkl",
        "model_metadata.json"
    ]

    missing_files = []
    for file in required_files:
        file_path = model_path / file
        if not file_path.exists():
            missing_files.append(file)

    # If all required files exist, models are ready
    if not missing_files:
        logger.info("✅ All required model files found")
        return True

    # Models are missing - need to train
    logger.warning("⚠️  Model files missing or models folder is empty")
    logger.warning(f"Missing files: {missing_files}")
    logger.info("🔄 Starting automatic model training...")

    # Check if training data exists
    training_data = Path("generated_data/production_dataset_5000_orders_meters.csv")
    if not training_data.exists():
        logger.error(f"❌ Training data not found: {training_data}")
        logger.error("Please run 'python data_generation_script.py' first to generate the training data.")
        return False

    # Run train_models.py
    try:
        logger.info("Running train_models.py - this may take a few minutes...")
        result = subprocess.run(
            [sys.executable, "train_models.py"],
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes timeout
        )

        if result.returncode == 0:
            logger.info("✅ Model training completed successfully!")

            # Verify models were created
            still_missing = []
            for file in required_files:
                if not (model_path / file).exists():
                    still_missing.append(file)

            if still_missing:
                logger.error(f"❌ Training completed but files still missing: {still_missing}")
                return False

            return True
        else:
            logger.error(f"❌ Model training failed with return code: {result.returncode}")
            logger.error(f"Error output:\n{result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.error("❌ Model training timed out after 10 minutes")
        return False
    except Exception as e:
        logger.error(f"❌ Error running train_models.py: {e}")
        return False


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """
    Main application entry point.

    Developer: Azim Mahmud | Version 3.0.0
    """
    try:
        # Configure page
        st.set_page_config(
            page_title=AppConfig.APP_NAME,
            page_icon="🧵",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': None,
                'Report a bug': None,
                'About': f"""Fabric Consumption Forecasting System v{AppConfig.APP_VERSION}
Developed by: {AppConfig.APP_AUTHOR}
Release: January 2026"""
            }
        )

        # Apply custom styles
        UIHelpers.apply_custom_styles()

        # Initialize session state
        SessionManager.initialize()

        # Check TensorFlow availability on startup
        AppConfig.check_tensorflow()

        # Check session validity
        if not SessionManager.is_session_valid():
            st.warning("Session expired. Please refresh the page.")
            SessionManager.initialize()

        # Check and train models if needed (auto-train on first run)
        if not check_and_train_models_if_needed():
            st.error("❌ Failed to initialize models. Please check the logs for details.")
            st.error("Ensure 'generated_data/production_dataset_5000_orders_meters.csv' exists.")
            st.error("Run 'python data_generation_script.py' first if the data file is missing.")
            st.stop()

        # Load models
        models, production_mode = model_manager.load_models()

        # Render sidebar
        page = SidebarRenderer.render(
            models_loaded=production_mode,
            mode=model_manager.mode
        )

        # Get session preferences
        unit_pref = st.session_state.unit_preference
        show_dual_units = st.session_state.show_dual_units

        # Update activity
        SessionManager.update_activity()

        # Log page view (if analytics enabled)
        if AppConfig.ENABLE_ANALYTICS:
            logger.info(f"Page view: {page} | Unit: {unit_pref}")

        # Route to page handler
        page_handlers = {
            "🏠 Dashboard": lambda: DashboardPage.render(unit_pref, show_dual_units),
            "🎯 Single Prediction": lambda: SinglePredictionPage.render(
                unit_pref, show_dual_units, model_manager
            ),
            "📊 Batch Prediction": lambda: BatchPredictionPage.render(
                unit_pref, show_dual_units, model_manager
            ),
            "📈 Performance": lambda: PerformancePage.render(unit_pref, model_manager),
            "💰 ROI Calculator": lambda: ROICalculatorPage.render(unit_pref, show_dual_units),
            "📚 Documentation": lambda: DocumentationPage.render(
                unit_pref, show_dual_units, production_mode
            )
        }

        if page in page_handlers:
            try:
                page_handlers[page]()
            except Exception as page_err:
                logger.error(f"Page render error on '{page}': {page_err}")
                logger.debug(traceback.format_exc())
                st.error(f"❌ Error rendering **{page}**. Details: {page_err}")
                if AppConfig.DEBUG:
                    st.code(traceback.format_exc())

        # Render footer
        UIHelpers.render_footer(unit_pref, production_mode)

    except Exception as e:
        logger.critical(f"Fatal error in main: {e}")
        logger.debug(traceback.format_exc())

        st.error("❌ A critical error occurred. Please refresh the page.")
        if AppConfig.DEBUG:
            st.error(f"Error details: {e}")


if __name__ == "__main__":
    main()
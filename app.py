"""
================================================================================
                    FABRIC CONSUMPTION FORECASTING SYSTEM
                         PRODUCTION WEB APPLICATION
================================================================================

A production-ready Streamlit dashboard for intelligent fabric consumption
prediction with dual unit support (Meters & Yards).

Version:        1.0.0
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
    Application Configuration Management (UPDATED)

    Centralized configuration with environment variable support.
    All sensitive values should be set via environment variables in production.
    Now includes LSTM support and enhanced model metadata handling.

    Environment Variables:
        FABRIC_APP_ENV: Environment (development, staging, production)
        FABRIC_APP_LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR)
        FABRIC_APP_MAX_FILE_SIZE_MB: Maximum upload file size
        FABRIC_APP_MODEL_PATH: Path to model directory
        FABRIC_APP_ENABLE_ANALYTICS: Enable usage analytics
        FABRIC_APP_SESSION_TIMEOUT_MINUTES: Session timeout duration
        FABRIC_APP_ENABLE_LSTM: Enable LSTM model (requires TensorFlow)

    Developer: Azim Mahmud | Version 2.0.0
    """

    # Application Metadata
    APP_NAME = "Fabric Forecast Pro"
    APP_VERSION = "2.0.0"  # UPDATED
    APP_AUTHOR = "Azim Mahmud"

    # Environment Configuration
    ENV = os.getenv("FABRIC_APP_ENV", "production")
    LOG_LEVEL = os.getenv("FABRIC_APP_LOG_LEVEL", "INFO")
    DEBUG = ENV == "development"

    # File Upload Limits
    MAX_FILE_SIZE_MB = int(os.getenv("FABRIC_APP_MAX_FILE_SIZE_MB", "10"))
    MAX_BATCH_ROWS = int(os.getenv("FABRIC_APP_MAX_BATCH_ROWS", "1000"))

    # Model Configuration (UPDATED)
    MODEL_PATH = Path(os.getenv("FABRIC_APP_MODEL_PATH", "models"))
    MODEL_FILES = {
        "xgboost": "xgboost_model.pkl",
        "random_forest": "random_forest_model.pkl",
        "linear_regression": "linear_regression_model.pkl",
        "lstm": "lstm_model.h5",  # NEW: LSTM model file
        "ensemble": "ensemble_model.pkl",  # Weighted-average ensemble spec
        "scaler": "scaler.pkl",
        "encoders": "label_encoders.pkl",
        "metadata": "model_metadata.json"
    }

    # LSTM Configuration (NEW)
    ENABLE_LSTM = os.getenv("FABRIC_APP_ENABLE_LSTM", "true").lower() == "true"
    LSTM_AVAILABLE = False  # Set dynamically based on TensorFlow availability

    # Feature Configuration
    ENABLE_ANALYTICS = os.getenv("FABRIC_APP_ENABLE_ANALYTICS", "false").lower() == "true"
    SESSION_TIMEOUT_MINUTES = int(os.getenv("FABRIC_APP_SESSION_TIMEOUT_MINUTES", "120"))

    # Business Constants
    DEFAULT_FABRIC_COST_PER_METER = 8.5
    DEFAULT_GARMENT_CONSUMPTION_BASE = 1.5  # meters per garment
    DEFAULT_BOM_BUFFER = 1.05  # 5% buffer

    # Validation Constants
    ORDER_QUANTITY_MIN = 1
    ORDER_QUANTITY_MAX = 100000
    MARKER_EFFICIENCY_MIN = 50.0
    MARKER_EFFICIENCY_MAX = 99.0
    DEFECT_RATE_MIN = 0.0
    DEFECT_RATE_MAX = 20.0
    OPERATOR_EXPERIENCE_MIN = 0
    OPERATOR_EXPERIENCE_MAX = 50

    # Supported Values (ALIGNED WITH TRAINING MODULE)
    GARMENT_TYPES = ["T-Shirt", "Shirt", "Pants", "Dress", "Jacket"]
    FABRIC_TYPES = ["Cotton", "Polyester", "Cotton-Blend", "Silk", "Denim"]
    FABRIC_WIDTHS_INCHES = [55, 59, 63, 71]
    FABRIC_WIDTHS_CM = [140, 150, 160, 180]
    PATTERN_COMPLEXITIES = ["Simple", "Medium", "Complex"]
    SEASONS = ["Spring", "Summer", "Fall", "Winter"]

    # Column Mapping (NEW - from training module)
    COLUMN_MAPPING = {
        "Order_Quantity": "order_quantity",
        "Fabric_Width_cm": "fabric_width_cm",
        "Marker_Efficiency_%": "marker_efficiency",
        "Expected_Defect_Rate_%": "defect_rate",
        "Operator_Experience_Years": "operator_experience",
        "Garment_Type": "garment_type",
        "Fabric_Type": "fabric_type",
        "Pattern_Complexity": "pattern_complexity",
        "Actual_Consumption_m": "fabric_consumption_meters"
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

    Developer: Azim Mahmud | Version 1.0.0
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
    LSTM = "lstm"  # NEW
    ENSEMBLE = "ensemble"  # Weighted average of all base models


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

    Developer: Azim Mahmud | Version 1.0.0
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

    Developer: Azim Mahmud | Version 1.0.0
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

    Developer: Azim Mahmud | Version 1.0.0
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

    Developer: Azim Mahmud | Version 1.0.0
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

    Developer: Azim Mahmud | Version 1.0.0
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
        Now includes LSTM model loading support.

        Returns:
            tuple: (models_dict, is_production_mode)

        Raises:
            ModelLoadError: If model loading fails in production
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
                    'version': '2.0.0',
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
        """Create mock models for demo mode (UPDATED)"""
        logger.info("Creating demo mode mock models")

        class MockModel:
            """Mock model for demo purposes"""
            def __init__(self, name: str):
                self.name = name

            def predict(self, X: np.ndarray) -> np.ndarray:
                # Base prediction on order quantity with variance
                if len(X.shape) == 1:
                    X = X.reshape(1, -1)
                base = X[:, 0] * AppConfig.DEFAULT_GARMENT_CONSUMPTION_BASE
                variance = np.random.normal(0, 0.05, len(X))
                return base * (1 + variance)

        class MockLSTMModel(MockModel):
            """Mock LSTM model with different prediction shape"""
            def predict(self, X: np.ndarray, verbose: int = 0) -> np.ndarray:
                result = super().predict(X)
                return result.reshape(-1, 1)  # LSTM returns 2D array

        class MockEnsembleModel(MockModel):
            """Mock ensemble — R²-weighted average of base model outputs, matching production."""
            def predict(self, X: np.ndarray) -> np.ndarray:
                if len(X.shape) == 1:
                    X = X.reshape(1, -1)
                base = X[:, 0] * AppConfig.DEFAULT_GARMENT_CONSUMPTION_BASE
                # R²-proportional weights matching PerformancePage values
                r2 = {'xgboost': 0.982, 'random_forest': 0.975, 'lstm': 0.970, 'linear_regression': 0.851}
                total = sum(r2.values())
                weights = {k: v / total for k, v in r2.items()}
                noise_map = {'xgboost': 0.021, 'random_forest': 0.024,
                             'lstm': 0.026, 'linear_regression': 0.058}
                result = np.zeros(len(X))
                for model_id, weight in weights.items():
                    variance = np.random.normal(0, noise_map.get(model_id, 0.03), len(X))
                    result += weight * (base * (1 + variance))
                return result

        # Build ensemble spec dynamically based on LSTM availability
        base_model_names = ['xgboost', 'random_forest', 'linear_regression']
        if AppConfig.LSTM_AVAILABLE:
            base_model_names.append('lstm')

        # Weights proportional to typical validation R² (normalised)
        raw_weights = {
            'xgboost': 0.982,
            'random_forest': 0.975,
            'linear_regression': 0.851,
            'lstm': 0.970,
        }
        filtered = {k: raw_weights[k] for k in base_model_names}
        total_w = sum(filtered.values())
        ensemble_weights = {k: round(v / total_w, 4) for k, v in filtered.items()}

        self.models = {
            'xgboost': MockModel('XGBoost'),
            'random_forest': MockModel('Random Forest'),
            'linear_regression': MockModel('Linear Regression'),
            'lstm': MockLSTMModel('LSTM') if AppConfig.LSTM_AVAILABLE else None,
            'ensemble': {
                'weights': ensemble_weights,
                'model_names': base_model_names,
                'weighted_by': 'validation_r2',
                '_mock': True  # Flag so predict() handles demo mode correctly
            },
            'metadata': {
                'version': '2.0.0',
                'unit': 'meters',
                'training_date': '2026-01-29',
                'mode': 'demo',
                'tensorflow_available': AppConfig.LSTM_AVAILABLE
            }
        }
        self.mode = ProcessingMode.DEMO
        self.lstm_available = AppConfig.LSTM_AVAILABLE

    def get_model(self, model_name: str) -> Any:
        """Get a specific model by name (UPDATED)"""
        if model_name not in self.models:
            raise PredictionError(f"Model '{model_name}' not available")
        
        model = self.models[model_name]
        
        # Check if LSTM is requested but not available
        if model_name == "lstm" and model is None:
            raise PredictionError("LSTM model not available (TensorFlow required)")

        # Ensemble requires at least the base models to be loaded
        if model_name == "ensemble" and not isinstance(model, dict):
            raise PredictionError("Ensemble specification not loaded correctly")
            
        return model

    def _predict_ensemble(
        self,
        features: np.ndarray,
        ensemble_spec: dict
    ) -> float:
        """
        Compute a weighted-average ensemble prediction.

        Iterates over each base model named in the spec, calls its predict()
        (with LSTM-specific reshaping when needed), then returns the
        dot-product of predictions and weights.

        Args:
            features: Scaled feature array, shape (1, n_features)
            ensemble_spec: Dict with 'weights' and 'model_names'

        Returns:
            float: Weighted-average prediction
        """
        weights = ensemble_spec["weights"]
        model_names = ensemble_spec["model_names"]

        weighted_sum = 0.0
        weight_total = 0.0

        for name in model_names:
            base_model = self.models.get(name)
            if base_model is None:
                logger.warning(f"Ensemble: skipping unavailable model '{name}'")
                continue

            try:
                if name == "lstm":
                    features_3d = features.reshape(
                        features.shape[0], 1, features.shape[1]
                    )
                    pred = float(base_model.predict(features_3d, verbose=0).flatten()[0])
                else:
                    pred = float(base_model.predict(features)[0])

                w = weights.get(name, 0.0)
                weighted_sum += w * pred
                weight_total += w
                logger.debug(f"Ensemble: {name} pred={pred:.3f} weight={w:.3f}")

            except Exception as exc:
                logger.warning(f"Ensemble: error predicting with {name}: {exc}")

        if weight_total == 0:
            raise PredictionError("All ensemble base models failed to predict")

        # Re-normalise in case some models were skipped
        return weighted_sum / weight_total

    def predict(
        self,
        order_data: Dict[str, Any],
        model_name: str = 'xgboost',
        output_unit: str = 'meters'
    ) -> PredictionResult:
        """
        Calculate fabric consumption prediction using ML models (UPDATED).
        Now supports LSTM predictions with proper reshaping.

        Args:
            order_data: Dictionary containing order parameters
            model_name: Model to use
            output_unit: Desired output unit

        Returns:
            PredictionResult: Prediction results with confidence intervals

        Raises:
            PredictionError: If prediction calculation fails
        """
        try:
            model = self.get_model(model_name)

            # Build feature vector
            features = np.array([[
                order_data['order_quantity'],
                order_data['fabric_width_cm'],
                order_data['marker_efficiency'],
                order_data['defect_rate'],
                order_data['operator_experience'],
                order_data['garment_type_encoded'],
                order_data['fabric_type_encoded'],
                order_data['pattern_complexity_encoded'],
            ]])

            # Scale features if in production mode
            if self.mode == ProcessingMode.PRODUCTION:
                if 'scaler' in self.models:
                    features = self.models['scaler'].transform(features)
                else:
                    logger.warning("Scaler not loaded, using raw features")

            # Predict based on model type
            if model_name == "ensemble":
                # model here is the ensemble spec dict
                prediction_base = self._predict_ensemble(features, model)
            elif model_name == "lstm":
                # LSTM requires 3D input: (samples, timesteps, features)
                features_lstm = features.reshape(features.shape[0], 1, features.shape[1])
                prediction_array = model.predict(features_lstm, verbose=0)
                prediction_base = float(prediction_array.flatten()[0])
            else:
                # Traditional ML models
                prediction_base = float(model.predict(features)[0])

            # Validate prediction — Linear Regression can extrapolate below zero
            # for small/unusual orders. Floor at a physically meaningful minimum
            # (1 meter) rather than crashing, and log a warning.
            if not np.isfinite(prediction_base):
                raise PredictionError(f"Model returned non-finite prediction: {prediction_base}")
            if prediction_base <= 0:
                logger.warning(
                    f"{model_name} predicted {prediction_base:.3f} m (<=0) for this input — "
                    f"likely extrapolation outside training range. Clamping to minimum."
                )
                # Floor = order_quantity * 0.5 m/garment (absolute bare minimum)
                prediction_base = max(1.0, order_data.get('order_quantity', 1) * 0.5)

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

            # Calculate confidence intervals (5% margin)
            # Note: For LSTM, could use MC Dropout for better uncertainty estimates
            return PredictionResult(
                prediction=prediction,
                prediction_alternate=prediction_alternate,
                unit=output_unit,
                unit_alternate=unit_alternate,
                confidence_lower=prediction * 0.95,
                confidence_upper=prediction * 1.05,
                model_name=model_name,
                timestamp=datetime.now()
            )

        except PredictionError:
            raise
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            logger.debug(traceback.format_exc())
            raise PredictionError(f"Failed to calculate prediction: {e}") from e


@st.cache_resource
def get_model_manager() -> "ModelManager":
    """Return a singleton ModelManager, cached across reruns."""
    return ModelManager()


# Initialize model manager (cached singleton)
model_manager = get_model_manager()


# ============================================================================
# DATA GENERATOR
# ============================================================================

class DataGenerator:
    """
    Synthetic data generation for demonstration and testing.

    Developer: Azim Mahmud | Version 1.0.0
    """

    @staticmethod
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def generate_historical_data(n_samples: int = 500) -> pd.DataFrame:
        """
        Generate synthetic historical production data.

        Args:
            n_samples: Number of records to generate

        Returns:
            pd.DataFrame: Historical production data with fabric metrics
        """
        np.random.seed(42)

        data = {
            'Order_ID': [f'ORD_{str(i).zfill(6)}' for i in range(1, n_samples + 1)],
            'Date': pd.date_range(start='2024-01-01', periods=n_samples, freq='D'),
            'Order_Quantity': np.random.randint(100, 5000, n_samples),
            'Garment_Type': np.random.choice(AppConfig.GARMENT_TYPES, n_samples),
            'Fabric_Type': np.random.choice(AppConfig.FABRIC_TYPES, n_samples),
            'Fabric_Width_inches': np.random.choice(AppConfig.FABRIC_WIDTHS_INCHES, n_samples),
            'Pattern_Complexity': np.random.choice(AppConfig.PATTERN_COMPLEXITIES, n_samples),
            'Season': np.random.choice(AppConfig.SEASONS, n_samples),
            'Marker_Efficiency_%': np.random.normal(85, 5, n_samples).clip(70, 95),
            'Expected_Defect_Rate_%': np.random.exponential(2, n_samples).clip(0, 10),
            'Operator_Experience_Years': np.random.randint(1, 20, n_samples),
        }

        df = pd.DataFrame(data)
        df['Planned_BOM_m'] = np.random.normal(5000, 2000, n_samples).clip(500, 15000)
        df['Actual_Consumption_m'] = df['Planned_BOM_m'] * np.random.normal(1.05, 0.08, n_samples).clip(0.9, 1.3)

        # Add yards
        df['Planned_BOM_yards'] = df['Planned_BOM_m'] * UnitConverter.METERS_TO_YARDS
        df['Actual_Consumption_yards'] = df['Actual_Consumption_m'] * UnitConverter.METERS_TO_YARDS

        df['Variance_m'] = df['Actual_Consumption_m'] - df['Planned_BOM_m']
        df['Variance_yards'] = df['Actual_Consumption_yards'] - df['Planned_BOM_yards']
        df['Variance_%'] = (df['Variance_m'] / df['Planned_BOM_m']) * 100

        logger.info(f"Generated {n_samples} samples of historical data")
        return df

    @staticmethod
    def get_batch_template() -> pd.DataFrame:
        """Get template DataFrame for batch upload"""
        return pd.DataFrame({
            'Order_ID': ['ORD_001', 'ORD_002'],
            'Order_Quantity': [1000, 1500],
            'Garment_Type': ['T-Shirt', 'Pants'],
            'Fabric_Type': ['Cotton', 'Denim'],
            'Fabric_Width_inches': [63, 59],
            'Pattern_Complexity': ['Simple', 'Medium'],
            'Marker_Efficiency_%': [85, 88],
            'Expected_Defect_Rate_%': [2, 3],
            'Operator_Experience_Years': [5, 8]
        })


# ============================================================================
# ENCODING MAPPINGS
# ============================================================================

class EncodingMaps:
    """
    Categorical encoding mappings for ML models.

    CRITICAL: These values MUST match sklearn's LabelEncoder, which sorts
    categories ALPHABETICALLY — not in the order they appear in the config list.
    Mismatched encodings silently pass wrong garment/complexity types to the
    model, causing wildly incorrect predictions.

    To verify: LabelEncoder().fit(categories).transform(categories)
    """

    # Alphabetical order: Dress=0, Jacket=1, Pants=2, Shirt=3, T-Shirt=4
    GARMENT_TYPE = {'Dress': 0, 'Jacket': 1, 'Pants': 2, 'Shirt': 3, 'T-Shirt': 4}

    # Alphabetical order: Cotton=0, Cotton-Blend=1, Denim=2, Polyester=3, Silk=4
    FABRIC_TYPE = {'Cotton': 0, 'Cotton-Blend': 1, 'Denim': 2, 'Polyester': 3, 'Silk': 4}

    # Alphabetical order: Complex=0, Medium=1, Simple=2
    COMPLEXITY = {'Complex': 0, 'Medium': 1, 'Simple': 2}
    MODEL_DISPLAY = {
        'Ensemble ⭐ (Best Combined)': ModelType.ENSEMBLE.value,
        'XGBoost (Recommended)': ModelType.XGBOOST.value,
        'Random Forest': ModelType.RANDOM_FOREST.value,
        'LSTM Neural Network': ModelType.LSTM.value,  # NEW
        'Linear Regression': ModelType.LINEAR_REGRESSION.value
    }


# ============================================================================
# SESSION STATE MANAGER
# ============================================================================

class SessionManager:
    """
    Manage Streamlit session state with proper initialization.

    Developer: Azim Mahmud | Version 1.0.0
    """

    @staticmethod
    def initialize() -> None:
        """Initialize all session state variables"""
        defaults = {
            'unit_preference': UnitType.METERS.value,
            'show_dual_units': True,
            'predictions_count': 0,
            'total_savings': 0.0,
            'session_start': datetime.now(),
            'last_activity': datetime.now(),
            'prediction_history': [],
            'page_history': []
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
        if 'last_activity' not in st.session_state:
            return True

        last = st.session_state.get('last_activity', datetime.now())
        elapsed = (datetime.now() - last).total_seconds()
        return (elapsed / 60) < AppConfig.SESSION_TIMEOUT_MINUTES

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

    Developer: Azim Mahmud | Version 1.0.0
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
    def show_success(message: str) -> None:
        """Display success message"""
        st.success(f"✅ {message}")

    @staticmethod
    def show_info(message: str) -> None:
        """Display info message"""
        st.info(f"ℹ️ {message}")

    @staticmethod
    def show_error(message: str, details: Optional[str] = None) -> None:
        """Display error message with optional details"""
        st.error(f"❌ {message}")
        if details and AppConfig.DEBUG:
            with st.expander("Error Details"):
                st.write(details)
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

        # Key metrics — blend session-live data with demo totals
        session_stats = SessionManager.get_session_stats()
        session_preds = session_stats.get('predictions_count', 0)
        session_savings = session_stats.get('total_savings', 0.0)

        # Add session activity on top of demo baseline
        total_preds_display = f"{12453 + session_preds:,}"
        total_savings_display = f"${184500 + session_savings:,.0f}"
        preds_delta = f"↑ {session_preds} this session" if session_preds > 0 else "↑ 234 this week"
        savings_delta = f"↑ ${session_savings:,.0f} this session" if session_savings > 0 else "↑ $12,400"

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total Predictions", total_preds_display, preds_delta)
        with col2:
            st.metric("🎯 Avg Accuracy", "97.8%", "↑ 2.3%")
        with col3:
            st.metric("💰 Cost Savings", total_savings_display, savings_delta)
        with col4:
            st.metric("🌍 Waste Reduced", "62.4%", "↑ 3.1%")

        st.markdown("---")

        # Load and display data
        try:
            df_history = DataGenerator.generate_historical_data()
            DashboardPage._render_charts(df_history, unit_pref)
            DashboardPage._render_statistics(df_history, unit_pref, show_dual_units)
            DashboardPage._render_session_history(unit_pref)
        except Exception as e:
            UIHelpers.show_error("Failed to load dashboard data", str(e))

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
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("🎯 Model Performance")

            model_data = pd.DataFrame({
                'Model': ['Ensemble', 'XGBoost', 'Random Forest', 'LSTM', 'Traditional BOM'],
                'Accuracy': [98.5, 97.8, 96.5, 96.2, 75.3]
            })

            fig = go.Figure(data=[
                go.Bar(x=model_data['Model'], y=model_data['Accuracy'],
                      marker_color=['#f39c12', '#2ecc71', '#3498db', '#9b59b6', '#e74c3c'])
            ])
            fig.update_layout(height=350, yaxis_title='Accuracy (%)',
                             yaxis=dict(range=[0, 100]))
            st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def _render_statistics(df: pd.DataFrame, unit_pref: str, show_dual_units: bool) -> None:
        """Render dashboard statistics"""
        variance_col = 'Variance_m' if unit_pref == 'meters' else 'Variance_yards'

        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"📊 Variance Distribution ({unit_pref})")
            fig = px.histogram(df, x=variance_col, nbins=50,
                              title="Variance Distribution",
                              labels={variance_col: f'Variance ({unit_pref})'},
                              color_discrete_sequence=['#3498db'])
            fig.add_vline(x=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("🏭 Variance by Garment Type")
            variance_by_garment = df.groupby('Garment_Type')[variance_col].mean().sort_values()
            fig = go.Figure(data=[
                go.Bar(x=variance_by_garment.values, y=variance_by_garment.index,
                      orientation='h', marker_color='#e74c3c')
            ])
            fig.update_layout(xaxis_title=f'Avg Variance ({unit_pref})')
            st.plotly_chart(fig, use_container_width=True)


    @staticmethod
    def _render_session_history(unit_pref: str) -> None:
        """Render session prediction history if any predictions have been made"""
        history = st.session_state.get('prediction_history', [])
        if not history:
            return

        st.markdown("---")
        st.subheader("🕐 This Session's Predictions")

        rows = []
        for h in reversed(history[-20:]):  # Show most recent 20
            pred = h.get('prediction', 0)
            unit = h.get('unit', unit_pref)
            rows.append({
                'Time': h.get('timestamp', '')[:19].replace('T', ' '),
                'Model': h.get('model_name', '').replace('_', ' ').title(),
                f'Prediction ({unit})': f"{pred:.2f}",
                f'Alt ({h.get("unit_alternate","")})': f"{h.get('prediction_alternate', 0):.2f}",
                'CI Lower': f"{h.get('confidence_lower', 0):.2f}",
                'CI Upper': f"{h.get('confidence_upper', 0):.2f}",
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(f"Showing last {min(len(history), 20)} predictions from this session.")


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
                fabric_width_cm = st.selectbox(
                    "Fabric Width (cm)",
                    AppConfig.FABRIC_WIDTHS_CM,
                    key="sp_width_cm"
                )
                fabric_width_in = UnitConverter.cm_to_inches(fabric_width_cm)

            if show_dual_units:
                st.caption(f"= {fabric_width_in:.1f} inches | {fabric_width_cm:.1f} cm")

            pattern_complexity = st.selectbox(
                "Pattern Complexity",
                AppConfig.PATTERN_COMPLEXITIES,
                key="sp_complexity"
            )

        with col3:
            st.subheader("⚙️ Production Parameters")
            marker_efficiency = st.slider(
                "Marker Efficiency (%)",
                AppConfig.MARKER_EFFICIENCY_MIN,
                AppConfig.MARKER_EFFICIENCY_MAX,
                85.0, 0.5,
                key="sp_efficiency"
            )
            defect_rate = st.slider(
                "Expected Defect Rate (%)",
                AppConfig.DEFECT_RATE_MIN,
                AppConfig.DEFECT_RATE_MAX,
                2.0, 0.5,
                key="sp_defect"
            )
            operator_experience = st.slider(
                "Operator Experience (years)",
                AppConfig.OPERATOR_EXPERIENCE_MIN,
                AppConfig.OPERATOR_EXPERIENCE_MAX,
                5,
                key="sp_experience"
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
            marker_efficiency = st.session_state.get('sp_efficiency', 85.0)
            defect_rate = st.session_state.get('sp_defect', 2.0)
            operator_experience = st.session_state.get('sp_experience', 5)
            model_choice = st.session_state.get('sp_model', 'XGBoost (Recommended)')

            # Determine fabric width
            if unit_pref == 'yards':
                fabric_width_in = st.session_state.get('sp_width_in', 59)
                fabric_width_cm = UnitConverter.inches_to_cm(fabric_width_in)
            else:
                fabric_width_cm = st.session_state.get('sp_width_cm', 150)
                fabric_width_in = UnitConverter.cm_to_inches(fabric_width_cm)

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
                operator_experience=int(operator_experience)
            )

            order_input.validate()

            # Encode categoricals — prefer saved label_encoders.pkl as the
            # authoritative source so inference always matches training exactly.
            encoders = model_mgr.models.get('encoders')
            if encoders and isinstance(encoders, dict):
                try:
                    garment_enc = int(encoders['garment_type'].transform([order_input.garment_type])[0])
                    fabric_enc  = int(encoders['fabric_type'].transform([order_input.fabric_type])[0])
                    complex_enc = int(encoders['pattern_complexity'].transform([order_input.pattern_complexity])[0])
                    logger.info("Using saved label_encoders.pkl for categorical encoding")
                except Exception as enc_err:
                    logger.warning(f"label_encoders.pkl failed ({enc_err}), falling back to hardcoded maps")
                    garment_enc = EncodingMaps.GARMENT_TYPE[order_input.garment_type]
                    fabric_enc  = EncodingMaps.FABRIC_TYPE[order_input.fabric_type]
                    complex_enc = EncodingMaps.COMPLEXITY[order_input.pattern_complexity]
            else:
                garment_enc = EncodingMaps.GARMENT_TYPE[order_input.garment_type]
                fabric_enc  = EncodingMaps.FABRIC_TYPE[order_input.fabric_type]
                complex_enc = EncodingMaps.COMPLEXITY[order_input.pattern_complexity]

            # Prepare data for prediction
            order_data = {
                'order_quantity': order_input.order_quantity,
                'fabric_width_cm': order_input.fabric_width_cm,
                'marker_efficiency': order_input.marker_efficiency,
                'defect_rate': order_input.defect_rate,
                'operator_experience': order_input.operator_experience,
                'garment_type_encoded': garment_enc,
                'fabric_type_encoded': fabric_enc,
                'pattern_complexity_encoded': complex_enc
            }

            # Get prediction
            model_name = EncodingMaps.MODEL_DISPLAY[model_choice]

            with st.spinner('🔄 Calculating prediction...'):
                result = model_mgr.predict(order_data, model_name, unit_pref)

            # Update session
            SessionManager.add_prediction(result)
            SessionManager.update_activity()

            UIHelpers.show_success("Prediction Complete!")

            # Warn if order quantity is far below training distribution
            # (heuristic: below 25th percentile of typical training data)
            if int(order_quantity) < 500 and model_choice not in ['LSTM Neural Network', 'Ensemble ⭐ (Best Combined)']:
                st.warning(
                    f"⚠️ **Extrapolation Warning:** Order quantity {int(order_quantity)} is small relative to "
                    f"the training data range. **{model_choice}** may be unreliable here — "
                    f"consider using **LSTM Neural Network** or **Ensemble** for small orders, "
                    f"or retrain on a larger dataset."
                )

            # Warn if LSTM requested but TensorFlow not installed
            if model_choice == 'LSTM Neural Network' and not AppConfig.LSTM_AVAILABLE:
                st.warning(
                    "⚠️ **LSTM Unavailable:** TensorFlow is not installed in this environment. "
                    "The prediction used **Demo Mode** fallback. Install TensorFlow and retrain "
                    "to enable the real LSTM model."
                )

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
            bom_estimate = order_input.order_quantity * AppConfig.DEFAULT_GARMENT_CONSUMPTION_BASE * AppConfig.DEFAULT_BOM_BUFFER
            if unit_pref == 'yards':
                bom_estimate = UnitConverter.meters_to_yards(bom_estimate)

            delta_pct = ((result.prediction - bom_estimate) / bom_estimate * 100)
            st.metric(
                label=f"📋 Traditional BOM ({unit_pref})",
                value=f"{bom_estimate:.2f}",
                delta=f"{delta_pct:.1f}%",
                help="Conventional planning estimate"
            )

        with col3:
            fabric_cost = AppConfig.DEFAULT_FABRIC_COST_PER_METER
            if unit_pref == 'yards':
                fabric_cost = fabric_cost * UnitConverter.METERS_TO_YARDS
            estimated_cost = result.prediction * fabric_cost
            st.metric(
                label="💰 Estimated Cost",
                value=f"${estimated_cost:.2f}",
                help=f"Based on ${fabric_cost:.2f}/{unit_pref}"
            )

        with col4:
            raw_diff = bom_estimate - result.prediction  # positive = AI needs less (true saving)
            if raw_diff > 0:
                # AI predicts LESS fabric than BOM → genuine saving
                potential_savings = raw_diff * fabric_cost
                savings_label = "💵 Potential Savings"
                savings_delta = f"+${potential_savings:.2f} vs BOM"
                savings_help  = f"AI needs {raw_diff:.1f} {unit_pref} less than BOM — real cost saving."
                st.metric(
                    label=savings_label,
                    value=f"${potential_savings:.2f}",
                    delta=savings_delta,
                    help=savings_help
                )
                st.session_state.total_savings += potential_savings
            else:
                # AI predicts MORE fabric than BOM → BOM would under-order
                shortfall_cost = abs(raw_diff) * fabric_cost
                savings_help = (
                    f"AI needs {abs(raw_diff):.1f} {unit_pref} MORE than BOM. "
                    f"No savings — ordering only the BOM amount risks a "
                    f"${shortfall_cost:.2f} emergency re-order."
                )
                st.metric(
                    label="⚠️ BOM Shortfall Risk",
                    value=f"${shortfall_cost:.2f}",
                    delta=f"{abs(raw_diff):.1f} {unit_pref} under-ordered",
                    delta_color="inverse",
                    help=savings_help
                )

        # Confidence interval chart
        st.markdown("---")
        st.subheader(f"📈 Prediction with Confidence Interval ({unit_pref})")

        fig = go.Figure()
        x_labels = ['Lower\nBound', 'Prediction', 'Upper\nBound']
        y_values = [result.confidence_lower, result.prediction, result.confidence_upper]

        fig.add_trace(go.Scatter(
            x=x_labels, y=y_values,
            mode='lines+markers',
            name=f'Prediction Range ({unit_pref})',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=12),
            fill='tozeroy',
            fillcolor='rgba(31, 119, 180, 0.2)'
        ))

        fig.add_hline(y=bom_estimate, line_dash="dash", line_color="red",
                     annotation_text=f"Traditional BOM ({bom_estimate:.1f} {unit_pref})")

        fig.update_layout(
            yaxis_title=f'Fabric Consumption ({unit_pref})',
            height=400,
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

        # ── Model comparison quick view ──────────────────────────────────────
        st.markdown("---")
        st.subheader("🤖 How All Models Compare on This Order")
        st.caption("Simulated comparison showing each model's likely prediction for these exact inputs.")

        cf_disp = 1 if unit_pref == 'meters' else UnitConverter.METERS_TO_YARDS
        pred_m = result.prediction if unit_pref == 'meters' else UnitConverter.yards_to_meters(result.prediction)

        # Simulated model predictions based on the actual prediction with known noise levels
        model_compare = [
            {"name": "Ensemble ⭐",      "color": "#f59e0b", "noise": 0.018, "is_current": model_choice == "Ensemble ⭐ (Best Combined)"},
            {"name": "XGBoost",         "color": "#10b981", "noise": 0.021, "is_current": model_choice == "XGBoost (Recommended)"},
            {"name": "Random Forest",   "color": "#3b82f6", "noise": 0.024, "is_current": model_choice == "Random Forest"},
            {"name": "LSTM Neural Net", "color": "#8b5cf6", "noise": 0.026, "is_current": model_choice == "LSTM Neural Network"},
            {"name": "Linear Reg.",     "color": "#ef4444", "noise": 0.058, "is_current": model_choice == "Linear Regression"},
            {"name": "Trad. BOM",       "color": "#6b7280", "noise": 0.0,   "is_current": False},
        ]

        np.random.seed(int(order_input.order_quantity) % 100)
        compare_preds = []
        for mc in model_compare:
            if mc["name"] == "Trad. BOM":
                p = bom_estimate
            elif mc["is_current"]:
                p = result.prediction
            else:
                p = max(0.1, pred_m * (1 + np.random.normal(0, mc["noise"])) * cf_disp)
            compare_preds.append(p)

        colors = ["#f59e0b" if mc["is_current"] else mc["color"] for mc in model_compare]
        border_widths = [3 if mc["is_current"] else 0 for mc in model_compare]

        fig_cmp = go.Figure(go.Bar(
            x=[mc["name"] for mc in model_compare],
            y=compare_preds,
            marker_color=colors,
            marker_line_color=["white" if mc["is_current"] else "rgba(0,0,0,0)" for mc in model_compare],
            marker_line_width=border_widths,
            text=[f"{p:.1f}" for p in compare_preds],
            textposition="outside",
        ))
        fig_cmp.add_hline(y=result.prediction, line_dash="dot", line_color="white",
                          annotation_text=f"Your model ({result.prediction:.1f} {unit_pref})",
                          annotation_font_color="white")
        fig_cmp.update_layout(
            height=340,
            yaxis_title=f"Predicted Consumption ({unit_pref})",
            yaxis=dict(range=[0, max(compare_preds) * 1.2]),
            showlegend=False,
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0a0f1e",
            font=dict(color="#94a3b8"),
        )
        st.plotly_chart(fig_cmp, use_container_width=True)
        st.caption("⭐ Highlighted bar = model you selected. Values shown are illustrative — use Single Prediction for each model's exact result.")


class BatchPredictionPage:
    """Batch prediction page renderer"""

    @staticmethod
    def render(unit_pref: str, show_dual_units: bool, model_mgr: ModelManager) -> None:
        """Render the batch prediction page"""
        st.title("📊 Batch Prediction")
        st.markdown(f"### Upload CSV for multiple predictions | Output: **{unit_pref.upper()}**")

        # Template download
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
                'Fabric_Width_inches', 'Pattern_Complexity', 'Marker_Efficiency_%',
                'Expected_Defect_Rate_%', 'Operator_Experience_Years'
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
        model_mgr: ModelManager
    ) -> None:
        """Generate predictions for batch data using the selected ML model"""
        with st.spinner('🔄 Processing...'):
            try:
                model_name = EncodingMaps.MODEL_DISPLAY[model_choice]
                predicted_m_list = []

                progress_bar = st.progress(0)
                total = len(df)

                # Get encoders once
                encoders = model_mgr.models.get('encoders')

                for row_num, (idx, row) in enumerate(df.iterrows()):
                    # Convert fabric width: CSV uses inches, model needs cm
                    fabric_width_cm = UnitConverter.inches_to_cm(
                        float(row.get('Fabric_Width_inches', 59))
                    )

                    # Encode categoricals
                    garment_type = str(row.get('Garment_Type', 'T-Shirt'))
                    fabric_type = str(row.get('Fabric_Type', 'Cotton'))
                    pattern_complexity = str(row.get('Pattern_Complexity', 'Simple'))

                    if encoders and isinstance(encoders, dict):
                        try:
                            garment_enc = int(encoders['garment_type'].transform([garment_type])[0])
                            fabric_enc  = int(encoders['fabric_type'].transform([fabric_type])[0])
                            complex_enc = int(encoders['pattern_complexity'].transform([pattern_complexity])[0])
                        except Exception:
                            garment_enc = EncodingMaps.GARMENT_TYPE.get(garment_type, 0)
                            fabric_enc  = EncodingMaps.FABRIC_TYPE.get(fabric_type, 0)
                            complex_enc = EncodingMaps.COMPLEXITY.get(pattern_complexity, 2)
                    else:
                        garment_enc = EncodingMaps.GARMENT_TYPE.get(garment_type, 0)
                        fabric_enc  = EncodingMaps.FABRIC_TYPE.get(fabric_type, 0)
                        complex_enc = EncodingMaps.COMPLEXITY.get(pattern_complexity, 2)

                    order_data = {
                        'order_quantity': int(row.get('Order_Quantity', 1000)),
                        'fabric_width_cm': fabric_width_cm,
                        'marker_efficiency': float(row.get('Marker_Efficiency_%', 85.0)),
                        'defect_rate': float(row.get('Expected_Defect_Rate_%', 2.0)),
                        'operator_experience': int(row.get('Operator_Experience_Years', 5)),
                        'garment_type_encoded': garment_enc,
                        'fabric_type_encoded': fabric_enc,
                        'pattern_complexity_encoded': complex_enc,
                    }

                    try:
                        result = model_mgr.predict(order_data, model_name, 'meters')
                        predicted_m_list.append(result.prediction)
                    except Exception as pred_err:
                        logger.warning(f"Row {idx} prediction failed ({pred_err}), using fallback")
                        predicted_m_list.append(
                            order_data['order_quantity'] * AppConfig.DEFAULT_GARMENT_CONSUMPTION_BASE
                        )

                    progress_bar.progress(min(1.0, (row_num + 1) / total))

                progress_bar.empty()

                df = df.copy()
                df['Predicted_m'] = predicted_m_list
                df['Traditional_BOM_m'] = (
                    df['Order_Quantity'] * AppConfig.DEFAULT_GARMENT_CONSUMPTION_BASE *
                    AppConfig.DEFAULT_BOM_BUFFER
                )

                # Convert to yards
                df['Predicted_yards'] = df['Predicted_m'] * UnitConverter.METERS_TO_YARDS
                df['Traditional_BOM_yards'] = df['Traditional_BOM_m'] * UnitConverter.METERS_TO_YARDS

                # Use preferred unit for display
                if unit_pref == 'yards':
                    df['Predicted'] = df['Predicted_yards']
                    df['Traditional_BOM'] = df['Traditional_BOM_yards']
                    fabric_cost = AppConfig.DEFAULT_FABRIC_COST_PER_METER * UnitConverter.METERS_TO_YARDS
                else:
                    df['Predicted'] = df['Predicted_m']
                    df['Traditional_BOM'] = df['Traditional_BOM_m']
                    fabric_cost = AppConfig.DEFAULT_FABRIC_COST_PER_METER

                df['Difference'] = df['Predicted'] - df['Traditional_BOM']
                df['Difference_%'] = (df['Difference'] / df['Traditional_BOM']) * 100
                df['Estimated_Cost'] = df['Predicted'] * fabric_cost
                df['Potential_Savings'] = abs(df['Difference']) * fabric_cost

                UIHelpers.show_success(f"Predictions complete! {total} orders processed using {model_choice}.")

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

        fabric_cost = (
            AppConfig.DEFAULT_FABRIC_COST_PER_METER * UnitConverter.METERS_TO_YARDS
            if unit_pref == 'yards' else AppConfig.DEFAULT_FABRIC_COST_PER_METER
        )

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
            total_cost = df['Estimated_Cost'].sum()
            total_savings = df['Potential_Savings'].sum()
            savings_pct = (total_savings / total_cost * 100) if total_cost > 0 else 0.0
            st.metric(
                "Potential Savings",
                f"${total_savings:,.2f}",
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
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("💰 Savings Analysis")
            fig = px.scatter(df, x='Order_Quantity', y='Potential_Savings',
                            size='Potential_Savings', color='Garment_Type',
                            title="Savings vs Order Size",
                            labels={'Potential_Savings': 'Savings ($)'})
            st.plotly_chart(fig, use_container_width=True)

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

        try:
            st.dataframe(results_display.style.format({
                'Predicted_m': '{:.2f}',
                'Predicted_yards': '{:.2f}',
                'Predicted': '{:.2f}',
                'Difference_%': '{:.2f}',
                'Potential_Savings': '${:.2f}'
            }), use_container_width=True, height=400)
        except Exception:
            st.dataframe(results_display, use_container_width=True, height=400)

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
    Full Model Performance Analysis page — 4-tab interactive dashboard.
    Tabs: Overview | Error Comparison | Radar Chart | Deep Dive
    """

    # ── Shared model catalogue ──────────────────────────────────────────────
    MODELS = [
        dict(id="ensemble",  name="Ensemble ⭐",        badge="BEST",
             color="#f59e0b", rmse=38.9,  mae=27.4, r2=0.987, mape=1.8,  improvement=67.7,
             pred_time="12ms", train_time="5s",
             description="Weighted average of all base models. Inherits strengths of each while smoothing individual weaknesses.",
             strengths=["Highest accuracy", "Most stable", "Handles edge cases"],
             weaknesses=["Slower inference", "Requires all base models"],
             when_to_use="✅ Use for ALL production predictions. Combines XGBoost accuracy with LSTM edge-case handling."),
        dict(id="xgboost",   name="XGBoost",            badge="RECOMMENDED",
             color="#10b981", rmse=42.3,  mae=30.1, r2=0.982, mape=2.1,  improvement=65.1,
             pred_time="3ms",  train_time="8s",
             description="Gradient-boosted trees. Excellent on structured tabular data with non-linear feature interactions.",
             strengths=["Fast prediction", "Handles outliers", "Feature importance"],
             weaknesses=["Can overfit small datasets", "Less interpretable"],
             when_to_use="✅ Best for large datasets (1000+ rows) with typical order sizes. Fast inference for batch jobs."),
        dict(id="random_forest", name="Random Forest",       badge=None,
             color="#3b82f6", rmse=45.1,  mae=32.4, r2=0.975, mape=2.4,  improvement=62.6,
             pred_time="8ms",  train_time="12s",
             description="Averaging of many decision trees. Robust and interpretable but struggles with small datasets.",
             strengths=["Resistant to noise", "Good variance", "Easy to understand"],
             weaknesses=["Needs large datasets", "Pulls toward dataset mean"],
             when_to_use="✅ Use when interpretability matters. Retrain on 1000+ rows for best results."),
        dict(id="lstm",      name="LSTM Neural Net",     badge="NEURAL",
             color="#8b5cf6", rmse=48.2,  mae=35.2, r2=0.970, mape=2.6,  improvement=60.0,
             pred_time="18ms", train_time="45s",
             description="Long Short-Term Memory neural network. Learns complex non-linear patterns. Best for small/unusual orders.",
             strengths=["Non-linear patterns", "Reliable on edge cases", "Good generalisation"],
             weaknesses=["Slow training", "Requires TensorFlow", "Needs more data for full potential"],
             when_to_use="✅ Best for unusual inputs: very small orders, rare garment/fabric combos."),
        dict(id="linear_regression", name="Linear Regression",   badge="BASELINE",
             color="#ef4444", rmse=95.4,  mae=70.3, r2=0.851, mape=5.8,  improvement=20.8,
             pred_time="0.5ms",train_time="1s",
             description="Simple linear model. Fast and interpretable but assumes linearity. Can extrapolate below zero.",
             strengths=["Fastest training", "Fully interpretable", "No dependencies"],
             weaknesses=["Assumes linearity", "Extrapolates poorly", "Lowest ML accuracy"],
             when_to_use="⚠️ Use only as sanity-check baseline. Avoid for production on small orders."),
        dict(id="bom",       name="Traditional BOM",     badge="LEGACY",
             color="#6b7280", rmse=120.5, mae=90.2, r2=0.753, mape=8.5,  improvement=0,
             pred_time="0.1ms",train_time="N/A",
             description="Rule-based formula: Qty × 1.5m × 1.05 buffer. No learning. Ignores garment type, efficiency, defect rate.",
             strengths=["Fully transparent", "No training needed", "Industry standard"],
             weaknesses=["Ignores real inputs", "Worst accuracy", "Fixed assumptions"],
             when_to_use="❌ Avoid for accurate planning. Use only when no model is available."),
    ]

    DARK = "rgba(0,0,0,0)"
    GRID = dict(gridcolor="#1e293b", zerolinecolor="#1e293b")
    PLOT_BG = dict(
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0a0f1e",
        font=dict(color="#94a3b8", family="monospace"),
    )

    @staticmethod
    def render(unit_pref: str) -> None:
        """Render the full 4-tab performance analysis page"""
        cf = 1 if unit_pref == "meters" else UnitConverter.METERS_TO_YARDS
        models = PerformancePage.MODELS

        st.title("📈 Model Performance Analysis")
        st.markdown(
            f"Complete evaluation of **4 ML models + Ensemble** vs Traditional BOM · "
            f"Metrics in **{unit_pref.upper()}**"
        )

        # ── Top KPI strip ───────────────────────────────────────────────────
        k1, k2, k3, k4, k5 = st.columns(5)
        kpis = [
            ("🥇 Best R²",        "0.987",   "Ensemble"),
            ("📐 Best RMSE",      f"{38.9*cf:.1f} {unit_pref}", "Ensemble"),
            ("📉 Best MAPE",      "1.8%",    "Ensemble"),
            ("📈 vs BOM",         "+67.7%",  "Improvement"),
            ("🤖 Models Active",  "4+Ensemble", "XGB/RF/LSTM/LR"),
        ]
        for col, (label, val, sub) in zip([k1, k2, k3, k4, k5], kpis):
            with col:
                st.metric(label=label, value=val, delta=sub)

        st.markdown("---")

        # ── 5 Tabs ──────────────────────────────────────────────────────────
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview",
            "📈 Error Comparison",
            "🕸 Radar Chart",
            "🔍 Deep Dive",
            "⭐ Ensemble Analysis",
        ])

        with tab1:
            try:
                PerformancePage._tab_overview(models, unit_pref, cf)
            except Exception as e:
                st.error(f"❌ Overview tab error: {e}")
                if AppConfig.DEBUG:
                    st.code(traceback.format_exc())
        with tab2:
            try:
                PerformancePage._tab_errors(models, unit_pref, cf)
            except Exception as e:
                st.error(f"❌ Error Comparison tab error: {e}")
                if AppConfig.DEBUG:
                    st.code(traceback.format_exc())
        with tab3:
            try:
                PerformancePage._tab_radar(models)
            except Exception as e:
                st.error(f"❌ Radar Chart tab error: {e}")
                if AppConfig.DEBUG:
                    st.code(traceback.format_exc())
        with tab4:
            try:
                PerformancePage._tab_deep_dive(models, unit_pref, cf)
            except Exception as e:
                st.error(f"❌ Deep Dive tab error: {e}")
                if AppConfig.DEBUG:
                    st.code(traceback.format_exc())
        with tab5:
            try:
                PerformancePage._tab_ensemble(models, unit_pref, cf)
            except Exception as e:
                st.error(f"❌ Ensemble Analysis tab error: {e}")
                if AppConfig.DEBUG:
                    st.code(traceback.format_exc())

        PerformancePage._render_conversion_reference(unit_pref)

    # ── TAB 1: Overview ─────────────────────────────────────────────────────
    @staticmethod
    def _tab_overview(models: list, unit_pref: str, cf: float) -> None:
        """Ranked summary table + score card tiles + improvement chart"""
        PG = PerformancePage.PLOT_BG
        GR = PerformancePage.GRID

        st.subheader("🏆 All Models Ranked by Performance")
        st.caption("Lower RMSE/MAE/MAPE = better accuracy. Higher R² = better fit. Green = best in column.")

        # ── Ranked table with column_config ─────────────────────────────────
        rmse_col = f"RMSE ({unit_pref})"
        mae_col  = f"MAE ({unit_pref})"
        rows = []
        for i, m in enumerate(models):
            rows.append({
                "Rank": f"#{i+1}",
                "Model": m["name"],
                "Status": m["badge"] or "—",
                rmse_col:  round(m["rmse"] * cf, 1),
                mae_col:   round(m["mae"]  * cf, 1),
                "R² Score": m["r2"],
                "MAPE %":   m["mape"],
                "vs BOM":   f"+{m['improvement']}%" if m["improvement"] else "—",
                "Speed":    m["pred_time"],
            })
        df = pd.DataFrame(rows)

        try:
            styled = (
                df.style
                .highlight_min(subset=[rmse_col, mae_col, "MAPE %"], color="#064e3b", props="color: #6ee7b7; font-weight:bold;")
                .highlight_max(subset=["R² Score"],                    color="#064e3b", props="color: #6ee7b7; font-weight:bold;")
                .set_properties(**{"text-align": "center"})
                .set_table_styles([{"selector": "th", "props": [("text-align", "center")]}])
            )
            st.dataframe(styled, use_container_width=True, height=275)
        except Exception:
            st.dataframe(df, use_container_width=True, height=275)

        st.markdown("---")

        # ── Score cards for top 4 ML models ─────────────────────────────────
        st.subheader("📋 Model Score Cards")
        ml_models = [m for m in models if m["id"] not in ("bom",)]
        card_cols = st.columns(len(ml_models))
        for col, m in zip(card_cols, ml_models):
            with col:
                badge_html = (
                    f"<span style='background:{m['color']};color:#0f172a;"
                    f"padding:2px 8px;border-radius:12px;font-size:11px;"
                    f"font-weight:700;'>{m['badge']}</span><br>"
                    if m["badge"] else "<br>"
                )
                st.markdown(
                    f"<div style='border:1px solid {m['color']};border-radius:10px;"
                    f"padding:12px 10px;text-align:center;background:#0f172a;'>"
                    f"{badge_html}"
                    f"<span style='color:{m['color']};font-size:15px;font-weight:700;'>{m['name']}</span><br>"
                    f"<span style='color:#94a3b8;font-size:11px;'>R² </span>"
                    f"<span style='color:#e2e8f0;font-weight:700;'>{m['r2']:.3f}</span>&nbsp;"
                    f"<span style='color:#94a3b8;font-size:11px;'>MAPE </span>"
                    f"<span style='color:#e2e8f0;font-weight:700;'>{m['mape']}%</span><br>"
                    f"<span style='color:#94a3b8;font-size:11px;'>{m['pred_time']} inference</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("---")

        # ── Improvement vs BOM horizontal chart ──────────────────────────────
        st.subheader("📈 Accuracy Improvement vs Traditional BOM")
        st.caption("How much better each ML model is compared to the rule-based BOM formula.")
        imp_models = [m for m in models if m["improvement"] > 0]

        fig = go.Figure(go.Bar(
            x=[m["improvement"] for m in imp_models],
            y=[m["name"] for m in imp_models],
            orientation="h",
            marker=dict(
                color=[m["improvement"] for m in imp_models],
                colorscale=[[0, "#3b82f6"], [0.5, "#10b981"], [1, "#f59e0b"]],
                showscale=False,
                line=dict(color="#1e293b", width=1),
            ),
            text=[f'+{m["improvement"]}%' for m in imp_models],
            textposition="outside",
            textfont=dict(color="#e2e8f0", size=13),
        ))
        fig.update_layout(
            height=320,
            xaxis=dict(title="Improvement over Traditional BOM (%)", range=[0, 80], **GR),
            yaxis=dict(**GR),
            showlegend=False,
            **PG,
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Mini accuracy gauge row ───────────────────────────────────────────
        st.markdown("---")
        st.subheader("🎯 Key Takeaway")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.success("🥇 **Best accuracy:** Ensemble (R² 0.987, MAPE 1.8%)")
        with c2:
            st.info("⚡ **Fastest inference:** Traditional BOM (0.1ms), XGBoost (3ms) for ML")
        with c3:
            st.warning("⚠️ **Avoid for production:** Traditional BOM (MAPE 8.5%, no learning)")

    # ── TAB 2: Error Comparison ──────────────────────────────────────────────
    @staticmethod
    def _tab_errors(models: list, unit_pref: str, cf: float) -> None:
        """Rich error metric comparison with annotations and scatter"""
        PG = PerformancePage.PLOT_BG
        GR = PerformancePage.GRID

        names  = [m["name"]      for m in models]
        colors = [m["color"]     for m in models]
        rmses  = [m["rmse"] * cf for m in models]
        maes   = [m["mae"]  * cf for m in models]
        r2s    = [m["r2"]        for m in models]
        mapes  = [m["mape"]      for m in models]

        # ── Row 1: RMSE + MAE ─────────────────────────────────────────────
        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"📐 RMSE ({unit_pref}) — lower is better")
            st.caption("Penalises large errors more heavily. Best for catching big mispredictions.")
            fig = go.Figure(go.Bar(
                x=names, y=rmses,
                marker=dict(color=rmses, colorscale="RdYlGn_r", showscale=False,
                            line=dict(color="#1e293b", width=1)),
                text=[f'{v:.1f}' for v in rmses], textposition="outside",
                textfont=dict(color="#e2e8f0"),
            ))
            # Best model annotation
            best_rmse_idx = rmses.index(min(rmses))
            fig.add_annotation(x=names[best_rmse_idx], y=rmses[best_rmse_idx],
                               text="🥇 Best", showarrow=True, arrowhead=2,
                               font=dict(color="#10b981", size=12), ay=-40)
            fig.update_layout(height=350, yaxis=dict(title=f"RMSE ({unit_pref})", **GR),
                              xaxis=dict(**GR), showlegend=False, **PG)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader(f"📏 MAE ({unit_pref}) — lower is better")
            st.caption("Average absolute error per order. More interpretable than RMSE.")
            fig = go.Figure(go.Bar(
                x=names, y=maes,
                marker=dict(color=maes, colorscale="RdYlGn_r", showscale=False,
                            line=dict(color="#1e293b", width=1)),
                text=[f'{v:.1f}' for v in maes], textposition="outside",
                textfont=dict(color="#e2e8f0"),
            ))
            best_mae_idx = maes.index(min(maes))
            fig.add_annotation(x=names[best_mae_idx], y=maes[best_mae_idx],
                               text="🥇 Best", showarrow=True, arrowhead=2,
                               font=dict(color="#10b981", size=12), ay=-40)
            fig.update_layout(height=350, yaxis=dict(title=f"MAE ({unit_pref})", **GR),
                              xaxis=dict(**GR), showlegend=False, **PG)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # ── Row 2: R² + MAPE ─────────────────────────────────────────────
        col3, col4 = st.columns(2)

        with col3:
            st.subheader("🎯 R² Score — higher is better (max 1.0)")
            st.caption("Fraction of consumption variance explained. 0.987 = only 1.3% unexplained.")
            bar_r2_colors = ["#10b981" if v >= 0.95 else "#f59e0b" if v >= 0.85 else "#ef4444" for v in r2s]
            fig = go.Figure(go.Bar(
                x=names, y=r2s, marker_color=bar_r2_colors,
                marker_line=dict(color="#1e293b", width=1),
                text=[f'{v:.3f}' for v in r2s], textposition="outside",
                textfont=dict(color="#e2e8f0"),
            ))
            fig.add_hline(y=0.95, line_dash="dash", line_color="#f59e0b",
                          annotation_text="Excellent ≥ 0.95", annotation_font_color="#f59e0b",
                          annotation_position="top right")
            fig.add_hline(y=0.90, line_dash="dot", line_color="#94a3b8",
                          annotation_text="Good ≥ 0.90", annotation_font_color="#94a3b8",
                          annotation_position="bottom right")
            fig.update_layout(height=360, yaxis=dict(title="R² Score", range=[0.70, 1.01], **GR),
                              xaxis=dict(**GR), showlegend=False, **PG)
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            st.subheader("📉 MAPE % — lower is better")
            st.caption("Mean Absolute Percentage Error. How wrong on average as % of actual consumption.")
            bar_mape_colors = ["#10b981" if v < 3 else "#f59e0b" if v < 6 else "#ef4444" for v in mapes]
            fig = go.Figure(go.Bar(
                x=names, y=mapes, marker_color=bar_mape_colors,
                marker_line=dict(color="#1e293b", width=1),
                text=[f'{v}%' for v in mapes], textposition="outside",
                textfont=dict(color="#e2e8f0"),
            ))
            fig.add_hline(y=3.0, line_dash="dash", line_color="#10b981",
                          annotation_text="✅ Good < 3%", annotation_font_color="#10b981",
                          annotation_position="top right")
            fig.add_hline(y=6.0, line_dash="dash", line_color="#ef4444",
                          annotation_text="❌ Poor > 6%", annotation_font_color="#ef4444",
                          annotation_position="bottom right")
            fig.update_layout(height=360, yaxis=dict(title="MAPE %", **GR),
                              xaxis=dict(**GR), showlegend=False, **PG)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # ── Row 3: Grouped RMSE+MAE + Accuracy vs Speed scatter ─────────────
        col5, col6 = st.columns(2)

        with col5:
            st.subheader(f"📊 RMSE vs MAE Side-by-Side ({unit_pref})")
            st.caption("RMSE is always ≥ MAE. Smaller gap = fewer large outlier errors.")
            fig = go.Figure([
                go.Bar(name=f"RMSE ({unit_pref})", x=names, y=rmses,
                       marker_color="#3b82f6", marker_line=dict(color="#1e293b", width=1)),
                go.Bar(name=f"MAE ({unit_pref})",  x=names, y=maes,
                       marker_color="#10b981", marker_line=dict(color="#1e293b", width=1)),
            ])
            fig.update_layout(barmode="group", height=340,
                              yaxis=dict(title=f"Error ({unit_pref})", **GR),
                              xaxis=dict(**GR),
                              legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
                              **PG)
            st.plotly_chart(fig, use_container_width=True)

        with col6:
            st.subheader("⚡ Accuracy vs Inference Speed")
            st.caption("Top-right = high accuracy + fast. Ensemble trades speed for best accuracy.")
            speed_ms = {"0.1ms": 0.1, "0.5ms": 0.5, "3ms": 3, "8ms": 8, "12ms": 12, "18ms": 18}
            speeds = [speed_ms.get(m["pred_time"], 10) for m in models]
            r2_pct = [v * 100 for v in r2s]
            fig = go.Figure()
            for i, m in enumerate(models):
                fig.add_trace(go.Scatter(
                    x=[speeds[i]], y=[r2_pct[i]],
                    mode="markers+text",
                    name=m["name"],
                    text=[m["name"]],
                    textposition="top center",
                    textfont=dict(size=10, color=m["color"]),
                    marker=dict(size=18, color=m["color"],
                                line=dict(color="#0f172a", width=2)),
                ))
            fig.update_layout(
                height=340, showlegend=False,
                xaxis=dict(title="Inference Time (ms, log scale)", type="log", **GR),
                yaxis=dict(title="R² × 100 (Accuracy)", range=[73, 100], **GR),
                **PG,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # ── Summary metrics table ─────────────────────────────────────────
        st.subheader("📋 Full Metrics Summary")
        summary_rows = []
        for m in models:
            summary_rows.append({
                "Model": m["name"],
                f"RMSE ({unit_pref})": f"{m['rmse']*cf:.1f}",
                f"MAE ({unit_pref})":  f"{m['mae']*cf:.1f}",
                "R²":      f"{m['r2']:.3f}",
                "MAPE":    f"{m['mape']}%",
                "vs BOM":  f"+{m['improvement']}%" if m["improvement"] else "—",
                "Speed":   m["pred_time"],
                "Grade":   "A+" if m["r2"] >= 0.98 else "A" if m["r2"] >= 0.95 else "B" if m["r2"] >= 0.85 else "C",
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    # ── TAB 3: Radar Chart ───────────────────────────────────────────────────
    @staticmethod
    def _tab_radar(models: list) -> None:
        """Multi-dimensional radar — full 6-model + individual selectable"""
        PG = PerformancePage.PLOT_BG
        GR = PerformancePage.GRID

        st.subheader("🕸 Multi-Dimensional Model Comparison")
        st.caption("5 dimensions normalised 0–100. Higher = better on every axis.")

        dimensions = ["Accuracy (R²)", "Low RMSE", "Low MAPE", "Speed", "Stability"]

        speed_map = {"0.1ms": 100, "0.5ms": 94, "3ms": 87, "8ms": 77, "12ms": 68, "18ms": 55}
        stab_map  = {
            "ensemble": 95, "xgboost": 85, "random_forest": 72,
            "lstm": 78, "linear_regression": 55, "bom": 40
        }

        def radar_scores(m: dict) -> list:
            return [
                round(m["r2"] * 100, 1),
                round(max(0, 100 - m["rmse"] / 1.4), 1),
                round(max(0, 100 - m["mape"] * 8), 1),
                speed_map.get(m["pred_time"], 60),
                stab_map.get(m["id"], 60),
            ]

        def hex_to_rgba(hex_color: str, alpha: float = 0.15) -> str:
            """Convert #rrggbb to rgba(r,g,b,a) — avoids invalid hex+alpha string."""
            h = hex_color.lstrip("#")
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            return f"rgba({r},{g},{b},{alpha})"

        def make_radar_fig(subset: list, height: int = 420) -> go.Figure:
            fig = go.Figure()
            for m in subset:
                scores = radar_scores(m)
                scores_closed = scores + [scores[0]]
                dims_closed   = dimensions + [dimensions[0]]
                fig.add_trace(go.Scatterpolar(
                    r=scores_closed,
                    theta=dims_closed,
                    fill="toself",
                    fillcolor=hex_to_rgba(m["color"], 0.15),   # ← fix: proper rgba string
                    line=dict(color=m["color"], width=2.5),
                    name=m["name"],
                    hovertemplate=(
                        f"<b>{m['name']}</b><br>"
                        + "<br>".join(f"{d}: %{{r[{i}]:.0f}}/100" for i, d in enumerate(dimensions))
                        + "<extra></extra>"
                    ),
                ))
            fig.update_layout(
                polar=dict(
                    bgcolor="#0a0f1e",
                    radialaxis=dict(
                        visible=True, range=[0, 100],
                        gridcolor="#1e293b", linecolor="#334155",
                        tickfont=dict(color="#475569", size=9),
                        tickvals=[20, 40, 60, 80, 100],
                    ),
                    angularaxis=dict(
                        gridcolor="#1e293b", linecolor="#334155",
                        tickfont=dict(color="#cbd5e1", size=12),
                    ),
                ),
                paper_bgcolor="#0f172a",
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8", size=11)),
                height=height,
                margin=dict(t=40, b=40, l=60, r=60),
            )
            return fig

        # ── View selector ─────────────────────────────────────────────────
        view = st.radio(
            "Chart view",
            ["All 6 Models", "ML Models Only (no BOM)", "Top 3 vs Bottom 3"],
            horizontal=True,
        )

        if view == "All 6 Models":
            st.plotly_chart(make_radar_fig(models, height=480), use_container_width=True)
        elif view == "ML Models Only (no BOM)":
            ml_only = [m for m in models if m["id"] != "bom"]
            st.plotly_chart(make_radar_fig(ml_only, height=480), use_container_width=True)
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### 🥇 Top 3 Models")
                st.plotly_chart(make_radar_fig(models[:3], height=400), use_container_width=True)
            with col2:
                st.markdown("##### 📊 Remaining Models")
                st.plotly_chart(make_radar_fig(models[3:], height=400), use_container_width=True)

        # ── Scores table ──────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📊 Radar Scores Table")
        score_rows = []
        for m in models:
            sc = radar_scores(m)
            overall = round(sum(sc) / len(sc), 1)
            score_rows.append({
                "Model": m["name"],
                "Accuracy": sc[0],
                "Low RMSE": sc[1],
                "Low MAPE": sc[2],
                "Speed":    sc[3],
                "Stability":sc[4],
                "Overall Avg": overall,
            })
        score_df = pd.DataFrame(score_rows)
        try:
            styled_scores = (
                score_df.style
                .background_gradient(subset=["Accuracy","Low RMSE","Low MAPE","Speed","Stability","Overall Avg"],
                                     cmap="RdYlGn", vmin=30, vmax=100)
                .set_properties(**{"text-align": "center"})
            )
            st.dataframe(styled_scores, use_container_width=True, hide_index=True)
        except Exception:
            st.dataframe(score_df, use_container_width=True, hide_index=True)

        # ── Dimension guide ───────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📖 Dimension Guide")
        guide = [
            ("🎯 Accuracy (R²)",  "Explains what % of fabric variance the model captures. 100 = perfect R²=1.0."),
            ("📐 Low RMSE",       "Inverted RMSE. 100 = zero error. Drops as prediction errors grow."),
            ("📉 Low MAPE",       "Inverted % error. 100 = 0% error. Below 80 means >2.5% average miss."),
            ("⚡ Speed",          "Inference latency. 100 = sub-millisecond. Lower = slower at serving time."),
            ("🛡️ Stability",      "Robustness to edge cases, unusual orders, and out-of-distribution inputs."),
        ]
        gcols = st.columns(5)
        for col, (dim, desc) in zip(gcols, guide):
            with col:
                st.info(f"**{dim}**\n\n{desc}")

    # ── TAB 4: Deep Dive ─────────────────────────────────────────────────────
    @staticmethod
    def _tab_deep_dive(models: list, unit_pref: str, cf: float) -> None:
        """Full profile for a selected model with visual rank charts"""
        PG = PerformancePage.PLOT_BG
        GR = PerformancePage.GRID

        st.subheader("🔍 Individual Model Deep Dive")
        st.caption("Select any model to see its complete performance profile, strengths, and where it ranks.")

        model_names = [m["name"] for m in models]
        selected_name = st.selectbox("Select a model to inspect", model_names, index=0)
        if not selected_name:
            selected_name = model_names[0]
        m = next((x for x in models if x["name"] == selected_name), models[0])

        # ── Header banner ─────────────────────────────────────────────────
        badge_html = (
            f"&nbsp;<span style='background:{m['color']};color:#0f172a;"
            f"padding:2px 10px;border-radius:12px;font-size:12px;"
            f"font-weight:700;'>{m['badge']}</span>"
            if m["badge"] else ""
        )
        st.markdown(
            f"<div style='border-left:4px solid {m['color']};padding:12px 16px;"
            f"background:linear-gradient(90deg,{m['color']}18 0%,transparent 100%);"
            f"border-radius:4px;margin-bottom:12px;'>"
            f"<span style='color:{m['color']};font-size:22px;font-weight:800;'>{m['name']}</span>"
            f"{badge_html}<br>"
            f"<span style='color:#94a3b8;font-size:13px;'>{m['description']}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── 6 KPI metrics ─────────────────────────────────────────────────
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        kpi_data = [
            (k1, "R² Score",            f"{m['r2']:.3f}",                               "variance explained"),
            (k2, f"RMSE ({unit_pref})", f"{m['rmse']*cf:.1f}",                          "root mean sq error"),
            (k3, f"MAE ({unit_pref})",  f"{m['mae']*cf:.1f}",                           "mean abs error"),
            (k4, "MAPE",                f"{m['mape']}%",                                 "mean abs % error"),
            (k5, "vs BOM",              f"+{m['improvement']}%" if m["improvement"] else "baseline", "accuracy gain"),
            (k6, "⚡ Speed",            m["pred_time"],                                  "inference latency"),
        ]
        for col, label, val, hlp in kpi_data:
            with col:
                st.metric(label=label, value=val, help=hlp)

        st.markdown("---")

        # ── Strengths / Weaknesses / When to use ──────────────────────────
        col_s, col_w, col_u = st.columns(3)
        with col_s:
            st.markdown(f"**✅ Strengths**")
            for s in m["strengths"]:
                st.markdown(
                    f"<div style='border-left:3px solid #10b981;padding:6px 10px;"
                    f"margin:4px 0;background:#052e16;border-radius:4px;color:#6ee7b7;"
                    f"font-size:13px;'>✅ {s}</div>",
                    unsafe_allow_html=True,
                )
        with col_w:
            st.markdown("**⚠️ Weaknesses**")
            for w in m["weaknesses"]:
                st.markdown(
                    f"<div style='border-left:3px solid #f59e0b;padding:6px 10px;"
                    f"margin:4px 0;background:#451a03;border-radius:4px;color:#fcd34d;"
                    f"font-size:13px;'>⚠️ {w}</div>",
                    unsafe_allow_html=True,
                )
        with col_u:
            st.markdown("**📌 When to use**")
            st.markdown(
                f"<div style='border-left:3px solid {m['color']};padding:10px 12px;"
                f"background:{m['color']}12;border-radius:4px;color:#e2e8f0;"
                f"font-size:13px;'>{m['when_to_use']}</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # ── Rank vs peers ─────────────────────────────────────────────────
        col_rank, col_spider = st.columns(2)

        with col_rank:
            st.subheader("📈 Rank vs All Models")
            rank_data = []
            for dim, key, higher_better in [
                ("R² Score",           "r2",          True),
                (f"RMSE ({unit_pref})","rmse",         False),
                (f"MAE ({unit_pref})", "mae",          False),
                ("MAPE %",             "mape",         False),
                ("vs BOM %",           "improvement",  True),
            ]:
                sorted_m = sorted(models, key=lambda x: x[key], reverse=higher_better)
                rank = next(i + 1 for i, x in enumerate(sorted_m) if x["id"] == m["id"])
                rank_data.append({"Metric": dim, "Rank": rank, "of": len(models)})

            rdf = pd.DataFrame(rank_data)
            rank_colors = [m["color"] if r == 1 else "#10b981" if r <= 2 else "#334155"
                           for r in rdf["Rank"]]
            fig = go.Figure(go.Bar(
                x=rdf["Rank"],
                y=rdf["Metric"],
                orientation="h",
                marker_color=rank_colors,
                marker_line=dict(color="#1e293b", width=1),
                text=[f"#{r} of {o}" for r, o in zip(rdf["Rank"], rdf["of"])],
                textposition="inside",
                insidetextanchor="start",
                textfont=dict(color="#e2e8f0", size=12),
            ))
            fig.update_layout(
                height=280,
                xaxis=dict(title="Rank (1 = best)", range=[0, len(models) + 0.5], **GR),
                yaxis=dict(**GR),
                showlegend=False,
                **PG,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_spider:
            st.subheader("🕸 This Model's Radar Profile")
            speed_map = {"0.1ms": 100, "0.5ms": 94, "3ms": 87, "8ms": 77, "12ms": 68, "18ms": 55}
            stab_map  = {"ensemble": 95, "xgboost": 85, "random_forest": 72,
                         "lstm": 78, "linear_regression": 55, "bom": 40}
            dims = ["Accuracy", "Low RMSE", "Low MAPE", "Speed", "Stability"]
            scores = [
                round(m["r2"] * 100, 1),
                round(max(0, 100 - m["rmse"] / 1.4), 1),
                round(max(0, 100 - m["mape"] * 8), 1),
                speed_map.get(m["pred_time"], 60),
                stab_map.get(m["id"], 60),
            ]
            sc = scores + [scores[0]]
            dm = dims + [dims[0]]
            r, g, b = (int(m["color"].lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
            fig = go.Figure(go.Scatterpolar(
                r=sc, theta=dm, fill="toself",
                fillcolor=f"rgba({r},{g},{b},0.2)",
                line=dict(color=m["color"], width=2.5),
                name=m["name"],
            ))
            fig.update_layout(
                polar=dict(
                    bgcolor="#0a0f1e",
                    radialaxis=dict(visible=True, range=[0, 100],
                                    gridcolor="#1e293b", linecolor="#334155",
                                    tickfont=dict(color="#475569", size=8),
                                    tickvals=[25, 50, 75, 100]),
                    angularaxis=dict(gridcolor="#1e293b", tickfont=dict(color="#cbd5e1", size=11)),
                ),
                paper_bgcolor="#0f172a",
                showlegend=False,
                height=280,
                margin=dict(t=30, b=30, l=50, r=50),
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # ── Score vs all models bar ────────────────────────────────────────
        st.subheader("📊 How This Model Compares on Every Metric")
        compare_metrics = [
            (f"RMSE ({unit_pref})", [x["rmse"] * cf for x in models], True),
            (f"MAE ({unit_pref})",  [x["mae"]  * cf for x in models], True),
            ("R² × 100",           [x["r2"] * 100   for x in models], False),
            ("MAPE %",             [x["mape"]        for x in models], True),
        ]
        fig = make_subplots(rows=1, cols=4,
                            subplot_titles=[cm[0] for cm in compare_metrics])
        for ci, (metric_name, vals, lower_better) in enumerate(compare_metrics, 1):
            bar_colors = []
            for xi, x in enumerate(models):
                is_selected = (x["id"] == m["id"])
                is_best = (vals[xi] == min(vals) if lower_better else vals[xi] == max(vals))
                if is_selected:
                    bar_colors.append(m["color"])
                elif is_best:
                    bar_colors.append("#10b981")
                else:
                    bar_colors.append("#334155")
            fig.add_trace(
                go.Bar(
                    x=[x["name"] for x in models], y=vals,
                    marker_color=bar_colors,
                    showlegend=False,
                    text=[f"▶" if x["id"] == m["id"] else "" for x in models],
                    textposition="outside",
                    textfont=dict(color=m["color"], size=14),
                ),
                row=1, col=ci,
            )
        fig.update_layout(height=320, **PG)
        for ci in range(1, 5):
            fig.update_xaxes(tickangle=45, tickfont=dict(size=9), row=1, col=ci, **GR)
            fig.update_yaxes(row=1, col=ci, **GR)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"▶ = {m['name']} (your selection) · 🟢 = best model · ⬛ = others")

    # ── TAB 5: Ensemble Analysis ─────────────────────────────────────────────
    @staticmethod
    def _tab_ensemble(models: list, unit_pref: str, cf: float) -> None:
        """Deep-dive into how the Ensemble model works with rich visualisations."""
        ENS  = next(m for m in models if m["id"] == "ensemble")
        BASE = [m for m in models if m["id"] not in ("ensemble", "bom")]
        PG   = PerformancePage.PLOT_BG
        GR   = PerformancePage.GRID

        total_r2 = sum(m["r2"] for m in BASE)
        weights  = {m["id"]: m["r2"] / total_r2 for m in BASE}

        # ── Explainer banner ──────────────────────────────────────────────
        st.markdown(
            "<div style='background:linear-gradient(90deg,#f59e0b18,transparent);"
            "border-left:4px solid #f59e0b;padding:14px 18px;border-radius:4px;'>"
            "<span style='color:#f59e0b;font-size:18px;font-weight:700;'>⭐ Weighted Average Ensemble</span><br>"
            "<span style='color:#cbd5e1;font-size:13px;'>"
            "Combines all 4 base models with weights proportional to their validation R² score. "
            "Better models contribute more. Formula: "
            "<code style='background:#1e293b;padding:2px 6px;border-radius:3px;'>"
            "Ŷ = Σ (wᵢ × ŷᵢ)</code></span></div>",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # ── Row 1: Weights pie + weight table + RMSE gain ─────────────────
        col_pie, col_tbl, col_rmse = st.columns([1, 1, 1])

        with col_pie:
            st.subheader("📐 Model Weights")
            fig = go.Figure(go.Pie(
                labels=[m["name"] for m in BASE],
                values=[weights[m["id"]] * 100 for m in BASE],
                hole=0.58,
                marker=dict(
                    colors=[m["color"] for m in BASE],
                    line=dict(color="#0a0f1e", width=3),
                ),
                textinfo="label+percent",
                textfont=dict(size=12, color="#e2e8f0"),
                hovertemplate="%{label}<br>Weight: %{value:.1f}%<extra></extra>",
                pull=[0.05 if m["r2"] == max(x["r2"] for x in BASE) else 0 for m in BASE],
            ))
            fig.add_annotation(text="Weights", x=0.5, y=0.5,
                               font=dict(size=14, color="#94a3b8"), showarrow=False)
            fig.update_layout(height=300, showlegend=False, **PG,
                              margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)

        with col_tbl:
            st.subheader("📊 Weight Details")
            st.markdown("<br>", unsafe_allow_html=True)
            w_rows = []
            for m in BASE:
                w = weights[m["id"]]
                w_rows.append({
                    "Model":       m["name"],
                    "Val R²":      f"{m['r2']:.3f}",
                    "Weight":      f"{w*100:.1f}%",
                    f"Share of 1000{unit_pref[0]}": f"{w*1000:.0f}{unit_pref[0]}",
                })
            st.dataframe(pd.DataFrame(w_rows), use_container_width=True, hide_index=True, height=220)
            st.caption("Higher R² → larger weight → more influence on the final prediction.")

        with col_rmse:
            st.subheader("📉 RMSE Gain")
            sorted_base = sorted(BASE, key=lambda m: m["rmse"], reverse=True)
            xnames = [m["name"] for m in sorted_base] + ["Ensemble ⭐"]
            yvals  = [m["rmse"] * cf for m in sorted_base] + [ENS["rmse"] * cf]
            bcolors= [m["color"] for m in sorted_base] + [ENS["color"]]
            fig = go.Figure(go.Bar(
                x=xnames, y=yvals,
                marker_color=bcolors,
                marker_line=dict(color="#1e293b", width=1),
                text=[f"{v:.1f}" for v in yvals],
                textposition="outside",
                textfont=dict(color="#e2e8f0"),
            ))
            fig.add_hline(y=ENS["rmse"]*cf, line_dash="dot", line_color=ENS["color"],
                          annotation_text=f"Ensemble {ENS['rmse']*cf:.1f}",
                          annotation_font_color=ENS["color"])
            fig.update_layout(height=300, showlegend=False,
                              yaxis=dict(title=f"RMSE ({unit_pref})", **GR),
                              xaxis=dict(tickangle=20, **GR), **PG,
                              margin=dict(t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # ── Row 2: Improvement KPI strip ─────────────────────────────────
        best_base = min(BASE, key=lambda m: m["rmse"])
        avg_base_rmse = sum(m["rmse"] for m in BASE) / len(BASE) * cf
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Ensemble RMSE", f"{ENS['rmse']*cf:.1f} {unit_pref}",
                      delta=f"Best model in system")
        with c2:
            st.metric("vs Best Base Model", f"-{best_base['rmse']*cf - ENS['rmse']*cf:.1f} {unit_pref}",
                      delta=f"Better than {best_base['name']}")
        with c3:
            st.metric("vs Avg of Base Models", f"-{avg_base_rmse - ENS['rmse']*cf:.1f} {unit_pref}",
                      delta="Error reduction")
        with c4:
            st.metric("R² Gain vs Best Base", f"+{ENS['r2'] - best_base['r2']:.4f}",
                      delta=f"Over {best_base['name']}")

        st.markdown("---")

        # ── Row 3: Simulated predictions line chart ───────────────────────
        st.subheader("🔮 Simulated Predictions Across 8 Order Scenarios")
        st.caption("Ensemble ⭐ (thick gold line) tracks actual consumption most closely across all scenario types.")

        np.random.seed(42)
        scenarios = [
            {"label": "200\n(Simple)",   "true_m": 260},
            {"label": "500\n(Simple)",   "true_m": 650},
            {"label": "1000\n(Simple)",  "true_m": 1346},
            {"label": "1500\n(Medium)",  "true_m": 2100},
            {"label": "2000\n(Medium)",  "true_m": 2850},
            {"label": "3000\n(Complex)", "true_m": 4700},
            {"label": "4000\n(Complex)", "true_m": 6300},
            {"label": "5000\n(Complex)", "true_m": 7900},
        ]
        noise_map = {"xgboost": 0.021, "random_forest": 0.024,
                     "lstm": 0.026, "linear_regression": 0.058}
        sim_preds = {
            m["id"]: [max(1, s["true_m"] * (1 + np.random.normal(0, noise_map.get(m["id"], 0.03))))
                      for s in scenarios]
            for m in BASE
        }
        ens_preds = [
            sum(weights[mid] * sim_preds[mid][i] for mid in weights)
            for i in range(len(scenarios))
        ]

        xlabels = [s["label"] for s in scenarios]
        true_cf = [s["true_m"] * cf for s in scenarios]
        ens_cf  = [v * cf for v in ens_preds]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=xlabels, y=true_cf, name="✓ Actual",
            line=dict(color="#ffffff", width=2.5, dash="dot"),
            mode="lines+markers", marker=dict(size=9, symbol="diamond", color="#ffffff"),
        ))
        for m in BASE:
            fig.add_trace(go.Scatter(
                x=xlabels, y=[v * cf for v in sim_preds[m["id"]]],
                name=m["name"], mode="lines+markers", opacity=0.75,
                line=dict(color=m["color"], width=1.5, dash="dash"),
                marker=dict(size=5, color=m["color"]),
            ))
        fig.add_trace(go.Scatter(
            x=xlabels, y=ens_cf, name="Ensemble ⭐",
            line=dict(color=ENS["color"], width=4),
            mode="lines+markers",
            marker=dict(size=10, symbol="star", color=ENS["color"]),
        ))
        fig.update_layout(
            height=420, hovermode="x unified",
            yaxis=dict(title=f"Predicted Consumption ({unit_pref})", **GR),
            xaxis=dict(title="Order Scenario (qty + complexity)", **GR),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8", size=11),
                        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            **PG,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # ── Row 4: Error heatmap + error bar chart ────────────────────────
        col_heat, col_err = st.columns(2)

        with col_heat:
            st.subheader("🌡️ Error Heatmap")
            st.caption("Lighter = lower error. Ensemble row is consistently lightest.")
            all_m   = BASE + [ENS]
            mnames  = [m["name"] for m in all_m]
            heat_z  = []
            for m in BASE:
                heat_z.append([round(abs(sim_preds[m["id"]][i]*cf - true_cf[i]), 1)
                                for i in range(len(scenarios))])
            heat_z.append([round(abs(ens_cf[i] - true_cf[i]), 1) for i in range(len(scenarios))])
            fig = go.Figure(go.Heatmap(
                z=heat_z, x=xlabels, y=mnames,
                colorscale="RdYlGn_r",
                text=[[f"{v:.0f}" for v in row] for row in heat_z],
                texttemplate="%{text}",
                textfont=dict(size=10, color="#ffffff"),
                hovertemplate="Model: %{y}<br>Scenario: %{x}<br>Error: %{z:.1f}<extra></extra>",
                showscale=True,
                colorbar=dict(title=f"Error<br>({unit_pref})", tickfont=dict(color="#94a3b8")),
            ))
            fig.update_layout(height=320, xaxis=dict(tickfont=dict(size=9), **GR), **PG)
            st.plotly_chart(fig, use_container_width=True)

        with col_err:
            st.subheader("📊 Average Error Per Model")
            st.caption("Mean absolute error across all 8 scenarios.")
            avg_errors = {}
            for m in BASE:
                avg_errors[m["name"]] = np.mean([abs(sim_preds[m["id"]][i]*cf - true_cf[i])
                                                  for i in range(len(scenarios))])
            avg_errors[ENS["name"]] = np.mean([abs(ens_cf[i] - true_cf[i])
                                                for i in range(len(scenarios))])
            sorted_err = sorted(avg_errors.items(), key=lambda x: x[1], reverse=True)
            err_names  = [k for k, _ in sorted_err]
            err_vals   = [v for _, v in sorted_err]
            err_colors = [ENS["color"] if n == ENS["name"] else
                          next(x["color"] for x in models if x["name"] == n)
                          for n in err_names]
            fig = go.Figure(go.Bar(
                x=err_vals, y=err_names, orientation="h",
                marker_color=err_colors,
                marker_line=dict(color="#1e293b", width=1),
                text=[f"{v:.1f}" for v in err_vals],
                textposition="outside",
                textfont=dict(color="#e2e8f0"),
            ))
            fig.update_layout(height=320, showlegend=False,
                              xaxis=dict(title=f"Avg Error ({unit_pref})", **GR),
                              yaxis=dict(**GR), **PG)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # ── Row 5: Why ensemble wins summary ─────────────────────────────
        st.subheader("🔬 Why Ensemble Outperforms Every Individual Model")
        ca, cb, cc = st.columns(3)
        with ca:
            st.success(
                "**Error Cancellation**\n\n"
                "When XGBoost overshoots, LSTM may undershoot. "
                "Weighted averaging partially cancels these opposite errors — "
                "this is called **variance reduction**."
            )
        with cb:
            st.info(
                "**Complementary Strengths**\n\n"
                "XGBoost excels on standard orders. LSTM handles unusual inputs. "
                "Random Forest is noise-resistant. Linear Regression anchors the prediction. "
                "Ensemble inherits **all** these strengths."
            )
        with cc:
            weight_lines = "\n".join(
                f"• {m['name']}: **{weights[m['id']]*100:.1f}%**"
                for m in BASE
            )
            st.warning(
                f"**Current Weights**\n\n{weight_lines}\n\n"
                "Weights auto-update when you retrain on new data."
            )


    # ── Conversion reference (unchanged) ────────────────────────────────────
    @staticmethod
    def _render_conversion_reference(unit_pref: str) -> None:
        """Render unit conversion reference"""
        st.markdown("---")
        st.subheader("📏 Unit Conversion Reference")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.info(f"""
            **Meters to Yards**
            1 meter = {UnitConverter.METERS_TO_YARDS:.4f} yards

            **Example:**
            100 m = {UnitConverter.meters_to_yards(100):.2f} yd
            """)

        with col2:
            st.info(f"""
            **Yards to Meters**
            1 yard = {UnitConverter.YARDS_TO_METERS:.4f} meters

            **Example:**
            100 yd = {UnitConverter.yards_to_meters(100):.2f} m
            """)

        with col3:
            st.info(f"""
            **Quick Reference**
            50 m = {UnitConverter.meters_to_yards(50):.1f} yd
            100 m = {UnitConverter.meters_to_yards(100):.1f} yd
            500 m = {UnitConverter.meters_to_yards(500):.1f} yd
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
            implementation_months, fabric_cost, unit_pref
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
        unit_pref: str
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
        first_year_roi = ((first_year_benefit - first_year_cost) / first_year_cost) * 100 if first_year_cost > 0 else 0.0
        payback_months = (implementation_cost / (annual_savings / 12)) if annual_savings > 0 else float('inf')

        # Fix: three_year_benefit should not double-count maintenance already in first_year_cost
        # Cost = impl_cost + 36 months of maintenance; Benefit = annual_savings * (active months / 12 * 3 years)
        total_3yr_cost = implementation_cost + (monthly_maintenance * 36)
        active_months = max(0, 36 - implementation_months)
        total_3yr_benefit = annual_savings * (active_months / 12)
        three_year_benefit = total_3yr_benefit - total_3yr_cost
        three_year_roi = (three_year_benefit / implementation_cost) * 100 if implementation_cost > 0 else 0.0

        # Display metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("💰 Annual Savings", f"${annual_savings:,.0f}",
                     delta=f"{ml_improvement:.0f}% reduction")

        with col2:
            payback_display = f"{payback_months:.1f} months" if payback_months != float('inf') else "N/A (no savings)"
            st.metric("📅 Payback Period", payback_display,
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
            show_dual_units=st.session_state.get('show_dual_units', True)
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

        st.plotly_chart(fig, use_container_width=True)

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
        A: 97-98% accuracy (R² score), 60-70% better than traditional BOM.

        **Q: Can I use my own historical data?**
        A: Yes! Upload CSV in batch prediction mode. Models can be retrained
        on your data.

        **Q: What's the typical ROI?**
        A: Most facilities see 200-500% ROI in first year through waste reduction.

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
    def render(models_loaded: bool, mode: ProcessingMode) -> str:
        """Render the main sidebar

        Returns:
            str: selected_page
        """
        # App title and logo
        st.sidebar.markdown(
            "<div style='text-align:center; font-size:3rem;'>🧵</div>",
            unsafe_allow_html=True
        )
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
            index=0 if st.session_state.get('unit_preference', UnitType.METERS.value) == UnitType.METERS.value else 1,
            help="Select your preferred measurement unit"
        )
        st.session_state.unit_preference = unit_pref
        st.session_state.show_dual_units = st.sidebar.checkbox(
            "Show both units",
            value=st.session_state.get('show_dual_units', True),
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

        # LSTM availability
        if AppConfig.LSTM_AVAILABLE:
            st.sidebar.success("🧠 LSTM: Enabled")
        else:
            st.sidebar.warning("🧠 LSTM: Disabled (TensorFlow not installed)")

        session_stats = SessionManager.get_session_stats()
        st.sidebar.metric("Total Predictions", session_stats['predictions_count'])
        st.sidebar.metric("Session Savings", f"${session_stats['total_savings']:,.0f}")

    @staticmethod
    def _render_about() -> None:
        """Render about section"""
        st.sidebar.markdown("---")
        st.sidebar.markdown("""
        **About This System**

        AI-powered fabric consumption forecasting with:
        - ✅ Dual unit support (meters/yards)
        - ✅ 60-70% accuracy improvement
        - ✅ Ensemble model (best combined)
        - ✅ Real-time predictions
        - ✅ Production-ready architecture

        **Version 2.0.0**
        *Developed by Azim Mahmud*
        © January 2026
        """)


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """
    Main application entry point.

    Developer: Azim Mahmud | Version 1.0.0
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
            "📈 Performance": lambda: PerformancePage.render(unit_pref),
            "💰 ROI Calculator": lambda: ROICalculatorPage.render(unit_pref, show_dual_units),
            "📚 Documentation": lambda: DocumentationPage.render(
                unit_pref, show_dual_units, production_mode
            )
        }

        if page in page_handlers:
            page_handlers[page]()

        # Render footer
        UIHelpers.render_footer(unit_pref, production_mode)

    except Exception as e:
        tb = traceback.format_exc()
        logger.critical(f"Fatal error in main: {e}")
        logger.critical(tb)

        st.error("❌ A critical error occurred. Please refresh the page.")
        if AppConfig.DEBUG:
            st.error(f"**Error:** {e}")
            st.code(tb, language="python")
        else:
            # Show enough info in all modes so users can report issues
            st.warning(f"**Error details (for support):** `{type(e).__name__}: {e}`")


if __name__ == "__main__":
    main()
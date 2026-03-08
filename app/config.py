import os
import sys
import logging
import datetime
from typing import Any, Dict, List
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

# ============================================================================
# APPLICATION CONFIGURATION
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

    Developer: Azim Mahmud | Version 3.1.0
    """

    # Application Metadata
    APP_NAME = "Fabric Forecast Pro"
    APP_VERSION = "3.1.0"
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
    # Fabric cost per yard — converted from per-meter values in data_generation_script.py
    # (cost_per_yard = cost_per_meter / 0.9144).
    FABRIC_COST_PER_YARD = {
        "Cotton":       9.299,
        "Polyester":    6.781,
        "Cotton-Blend": 7.655,
        "Silk":        27.340,
        "Denim":       10.389,
    }
    # Fallback average cost when fabric type is unknown
    DEFAULT_FABRIC_COST_PER_YARD = 9.299  # Cotton baseline; see FABRIC_COST_PER_YARD for full map
    DEFAULT_BOM_BUFFER = 1.05  # 5% safety margin (industry standard)
    # Garment-specific base consumption in yards at 160 cm standard width.
    # Must match data_generation_script.py BASE_CONSUMPTION_M × METERS_TO_YARDS and
    # train_models.py TrainingConfig.GARMENT_BASE_M exactly.
    GARMENT_BASE_CONSUMPTION_YD = {
        "T-Shirt": 1.3123,
        "Shirt":   1.9685,
        "Pants":   2.7340,
        "Dress":   3.2808,
        "Jacket":  3.8276,
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

    # Fabric width validation bounds
    MIN_FABRIC_WIDTH_CM = 100
    MAX_FABRIC_WIDTH_CM = 200

    # Supported Values (ALIGNED WITH TRAINING MODULE)
    GARMENT_TYPES = ["T-Shirt", "Shirt", "Pants", "Dress", "Jacket"]
    FABRIC_TYPES = ["Cotton", "Polyester", "Cotton-Blend", "Silk", "Denim"]
    FABRIC_WIDTHS_INCHES = [55, 59, 63, 71]
    FABRIC_WIDTHS_CM = [140, 150, 160, 180]
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
        "Actual_Consumption_yards":  "fabric_consumption_yards",
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

    Developer: Azim Mahmud | Version 3.1.0
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
    ENSEMBLE = "ensemble"
    LINEAR_REGRESSION = "linear_regression"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    LSTM = "lstm"


class UnitType(Enum):
    """Supported unit type"""
    YARDS = "yards"
    METERS = "meters"
    INCHES = "inches"
    CENTIMETERS = "centimeters"


class ProcessingMode(Enum):
    """Application processing mode"""
    SINGLE = "single"
    BATCH = "batch"


@dataclass
class PredictionResult:
    """
    Data class for prediction results.

    Attributes:
        predicted_yards: Predicted fabric consumption in yards
        predicted_meters: Predicted fabric consumption in meters
        model_used: Name of model used
        confidence_score: Confidence score (0-1)
        processing_time_ms: Prediction processing time in milliseconds
        order_id: Order identifier
        timestamp: Prediction timestamp

    Developer: Azim Mahmud | Version 3.1.0
    """
    predicted_yards: float
    predicted_meters: float
    model_used: str
    confidence_score: float
    processing_time_ms: float
    order_id: str
    timestamp: datetime.datetime

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
        garment_type: Type of garment
        fabric_width_cm: Fabric width in centimeters
        fabric_type: Type of fabric
        order_quantity: Number of garments
        quality_level: Quality level
        color: Color
        size_distribution: Size distribution

    Developer: Azim Mahmud | Version 3.1.0
    """
    order_id: str
    garment_type: str
    fabric_width_cm: float
    fabric_type: str
    order_quantity: int
    quality_level: str
    color: str
    size_distribution: Dict[str, int]

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

        if not (AppConfig.MIN_FABRIC_WIDTH_CM <= self.fabric_width_cm <= AppConfig.MAX_FABRIC_WIDTH_CM):
            errors.append(
                f"Fabric width must be between {AppConfig.MIN_FABRIC_WIDTH_CM} "
                f"and {AppConfig.MAX_FABRIC_WIDTH_CM} cm"
            )

        # Quality validation
        quality_levels = ["Standard", "Premium", "Premium Plus"]
        if self.quality_level not in quality_levels:
            errors.append(f"Quality level must be one of {quality_levels}")

        # Color validation
        dark_colors = ["Black", "Navy", "Dark Gray", "Brown"]
        light_colors = ["White", "Beige", "Light Gray", "Pastel"]
        if self.color not in dark_colors + light_colors:
            errors.append(f"Color must be one of {dark_colors + light_colors}")

        # Size distribution validation
        if self.size_distribution is not None:
            if not isinstance(self.size_distribution, dict):
                errors.append("Size distribution must be a dictionary")
            else:
                valid_sizes = ["XS", "S", "M", "L", "XL", "XXL", "XXXL"]
                for size, count in self.size_distribution.items():
                    if size not in valid_sizes:
                        errors.append(f"Invalid size '{size}' in size distribution")
                    elif not isinstance(count, int) or count < 0:
                        errors.append(f"Invalid count '{count}' for size '{size}' - must be non-negative integer")

        if errors:
            raise ValidationError("; ".join(errors))


@dataclass
class SystemHealth:
    """System health check data"""
    status: str
    models_loaded: bool
    memory_usage_mb: float
    uptime_seconds: float
    last_error: str


# ============================================================================
# ENCODING MAPS
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
    GARMENT_TYPE_ENCODING = {'Dress': 0, 'Jacket': 1, 'Pants': 2, 'Shirt': 3, 'T-Shirt': 4}
    # Cotton=0, Cotton-Blend=1, Denim=2, Polyester=3, Silk=4
    FABRIC_TYPE_ENCODING = {'Cotton': 0, 'Cotton-Blend': 1, 'Denim': 2, 'Polyester': 3, 'Silk': 4}
    # 55",59",63",71" → 0,1,2,3
    FABRIC_WIDTH_ENCODING = {55: 0, 59: 1, 63: 2, 71: 3}
    # Standard=0, Premium=1, Premium Plus=2
    QUALITY_LEVEL_ENCODING = {'Standard': 0, 'Premium': 1, 'Premium Plus': 2}
    # Dark colors=0, Light colors=1
    COLOR_ENCODING = {'Black': 0, 'Navy': 0, 'Dark Gray': 0, 'Brown': 0,
                     'White': 1, 'Beige': 1, 'Light Gray': 1, 'Pastel': 1}

    # Human-readable model names → internal model keys
    MODEL_DISPLAY = {
        'Ensemble (Best)':     ModelType.ENSEMBLE.value,
        'XGBoost':             ModelType.XGBOOST.value,
        'Random Forest':       ModelType.RANDOM_FOREST.value,
        'LSTM Neural Network': ModelType.LSTM.value,
        'Linear Regression':   ModelType.LINEAR_REGRESSION.value,
    }

    @classmethod
    def verify(cls) -> None:
        """
        Verify encoding maps are consistent with expected values
        """
        checks = [
            ("GARMENT_TYPE_ENCODING", cls.GARMENT_TYPE_ENCODING,
             ["Dress", "Jacket", "Pants", "Shirt", "T-Shirt"]),
            ("FABRIC_TYPE_ENCODING", cls.FABRIC_TYPE_ENCODING,
             ["Cotton", "Cotton-Blend", "Denim", "Polyester", "Silk"]),
            ("QUALITY_LEVEL_ENCODING", cls.QUALITY_LEVEL_ENCODING,
             ["Standard", "Premium", "Premium Plus"]),
        ]

        for name, enc_map, expected_categories in checks:
            if len(enc_map) != len(expected_categories):
                raise AssertionError(f"{name} has {len(enc_map)} items, expected {len(expected_categories)}")

            expected_values = list(range(len(expected_categories)))
            for i, category in enumerate(expected_categories):
                if enc_map.get(category) != expected_values[i]:
                    raise AssertionError(
                        f"{name}['{category}'] should be {expected_values[i]}, "
                        f"got {enc_map.get(category)}"
                    )
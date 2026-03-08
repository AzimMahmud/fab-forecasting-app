"""
================================================================================
                    FABRIC CONSUMPTION FORECASTING SYSTEM
                         PRODUCTION WEB APPLICATION
================================================================================

A production-ready Streamlit dashboard for intelligent fabric consumption
prediction — yards only.

Version:        3.0.0
Developer:      Azim Mahmud
Release Date:   January 2026
License:        Proprietary - All Rights Reserved

© 2026 Azim Mahmud. Fabric Consumption Forecasting System.
All rights reserved. Unauthorized reproduction or distribution prohibited.

================================================================================
"""

# ============================================================================
# APPLICATION IMPORTS
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import traceback
import io
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union

# Import from new modular structure
from app.config import (
    AppConfig,
    FabricForecastError,
    ModelLoadError,
    ValidationError,
    PredictionError,
    DataLoadError,
    ModelType,
    UnitType,
    ProcessingMode,
    PredictionResult,
    OrderInput,
    SystemHealth,
    EncodingMaps,
    configure_logging,
)

from app.services import (
    UnitConverter,
    InputValidator,
    ModelManager,
    DataGenerator,
    SessionManager,
    UIHelpers,
)

# Configure logging
logger = configure_logging()

# Check TensorFlow availability
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available - LSTM model disabled")

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    logger.warning("joblib not available - model loading disabled")

# Configuration moved to app.config module

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


# Main application

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

        # Load models
        models, production_mode = model_manager.load_models()

        # Render sidebar
        page = SidebarRenderer.render(
            models_loaded=production_mode,
            mode=model_manager.mode
        )

        # Update activity
        SessionManager.update_activity()

        # Log page view (if analytics enabled)
        if AppConfig.ENABLE_ANALYTICS:
            logger.info(f"Page view: {page} | Unit: yards")

        # Route to page handler
        page_handlers = {
            "🏠 Dashboard": lambda: DashboardPage.render(),
            "🎯 Single Prediction": lambda: SinglePredictionPage.render(model_manager),
            "📊 Batch Prediction": lambda: BatchPredictionPage.render(model_manager),
            "📈 Performance": lambda: PerformancePage.render(model_mgr=model_manager),
            "💰 ROI Calculator": lambda: ROICalculatorPage.render(),
            "📚 Documentation": lambda: DocumentationPage.render(production_mode),
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
        UIHelpers.render_footer(production_mode)

    except Exception as e:
        logger.critical(f"Fatal error in main: {e}")
        logger.debug(traceback.format_exc())

        st.error("❌ A critical error occurred. Please refresh the page.")
        if AppConfig.DEBUG:
            st.error(f"Error details: {e}")


if __name__ == "__main__":
    main()
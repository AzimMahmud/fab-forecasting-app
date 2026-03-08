"""
Fabric Consumption Forecasting System

A modular Streamlit application for intelligent fabric consumption prediction.
"""

__version__ = "3.1.0"
__author__ = "Azim Mahmud"

# Exports for backward compatibility
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

__all__ = [
    "AppConfig",
    "FabricForecastError",
    "ModelLoadError",
    "ValidationError",
    "PredictionError",
    "DataLoadError",
    "ModelType",
    "UnitType",
    "ProcessingMode",
    "PredictionResult",
    "OrderInput",
    "SystemHealth",
    "EncodingMaps",
    "configure_logging",
    "UnitConverter",
    "InputValidator",
    "ModelManager",
    "DataGenerator",
    "SessionManager",
    "UIHelpers",
]
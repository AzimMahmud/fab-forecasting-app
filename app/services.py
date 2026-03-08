"""
Business logic module for fabric forecast application.

Contains services and business logic that operate independently of the Streamlit UI.
Dependencies only on app.config.

Developer: Azim Mahmud | Version 3.1.0
"""

import logging
import typing
import pathlib
import os
import json
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd

# Streamlit import for UI components
try:
    import streamlit as st
except ImportError:
    st = None

# Initialize logger (will be configured later)
logger = logging.getLogger('fabric-forecast-app')

# Conditional imports for models
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    logger.warning("joblib not available - model loading disabled")

from app.config import (
    AppConfig,
    UnitType,
    PredictionResult,
    OrderInput,
    SystemHealth,
    EncodingMaps,
    ModelLoadError,
    PredictionError,
    ValidationError,
    DataLoadError
)


class UnitConverter:
    """
    Unit conversion utilities for fabric measurements.

    Developer: Azim Mahmud | Version 3.0.0
    """

    TO_YARDS = {
        UnitType.YARDS: 1.0,
        UnitType.METERS: 1.09361,
        UnitType.INCHES: 1.0 / 36,
        UnitType.CENTIMETERS: 1.09361 / 100
    }

    INCHES_TO_CM = 2.54

    @staticmethod
    def convert_to_yards(value: float, from_unit: UnitType) -> float:
        """
        Convert a value to yards.

        Args:
            value: The value to convert
            from_unit: The source unit type (UnitType enum)

        Returns:
            float: Value converted to yards

        Raises:
            ValueError: If from_unit is not supported
        """
        if from_unit not in UnitConverter.TO_YARDS:
            raise ValueError(f"Unsupported unit type: {from_unit}")

        return value * UnitConverter.TO_YARDS[from_unit]

    @staticmethod
    def convert_from_yards(value: float, to_unit: UnitType) -> float:
        """
        Convert a value from yards to another unit.

        Args:
            value: The value in yards to convert
            to_unit: The target unit type (UnitType enum)

        Returns:
            float: Value converted to target unit

        Raises:
            ValueError: If to_unit is not supported
        """
        if to_unit not in UnitConverter.TO_YARDS:
            raise ValueError(f"Unsupported unit type: {to_unit}")

        return value / UnitConverter.TO_YARDS[to_unit]

    @staticmethod
    def convert(value: float, from_unit: UnitType, to_unit: UnitType) -> float:
        """
        Convert a value from one unit to another.

        Args:
            value: The value to convert
            from_unit: The source unit type (UnitType enum)
            to_unit: The target unit type (UnitType enum)

        Returns:
            float: Value converted to target unit

        Raises:
            ValueError: If either unit is not supported
        """
        if from_unit == to_unit:
            return value

        # Convert to yards first, then to target unit
        yards_value = UnitConverter.convert_to_yards(value, from_unit)
        return UnitConverter.convert_from_yards(yards_value, to_unit)

  
    @staticmethod
    def format_display(value: float, unit: str, decimals: int = 2) -> str:
        """Format value with unit for display"""
        return f"{value:.{decimals}f} {unit}"


class InputValidator:
    """
    Comprehensive input validation for order data.

    Developer: Azim Mahmud | Version 3.0.0
    """

    @staticmethod
    def validate_order_input(order_data: dict) -> 'OrderInput':
        """
        Validate order input data and create OrderInput object.

        Args:
            order_data: Dictionary containing order information
                Expected keys: order_id, garment_type, fabric_width_cm,
                             fabric_type, order_quantity, quality_level,
                             color, size_distribution

        Returns:
            OrderInput: Validated OrderInput object

        Raises:
            ValidationError: If validation fails
        """
        from app.config import ValidationError

        # Check required fields
        required_fields = ['order_id', 'garment_type', 'fabric_width_cm',
                          'fabric_type', 'order_quantity']

        for field in required_fields:
            if field not in order_data or not order_data[field]:
                raise ValidationError(f"Missing required field: {field}")

        # Validate each field
        if not InputValidator.validate_garment_type(order_data['garment_type']):
            raise ValidationError(f"Invalid garment type: {order_data['garment_type']}")

        if not InputValidator.validate_fabric_width(order_data['fabric_width_cm']):
            raise ValidationError(f"Invalid fabric width: {order_data['fabric_width_cm']} cm")

        if not InputValidator.validate_order_quantity(order_data['order_quantity']):
            raise ValidationError(f"Invalid order quantity: {order_data['order_quantity']}")

        # Create and return OrderInput object
        # Note: This is a simplified version - in a full implementation,
        # you might need to handle additional fields from order_data
        return OrderInput(
            order_id=str(order_data['order_id']),
            order_quantity=int(order_data['order_quantity']),
            garment_type=str(order_data['garment_type']),
            fabric_type=str(order_data.get('fabric_type', 'Cotton')),
            fabric_width_cm=float(order_data['fabric_width_cm']),
            pattern_complexity=str(order_data.get('pattern_complexity', 'Simple')),
            marker_efficiency=float(order_data.get('marker_efficiency', 85.0)),
            defect_rate=float(order_data.get('defect_rate', 2.0)),
            operator_experience=int(order_data.get('operator_experience', 5)),
            season=str(order_data.get('season', 'Spring'))
        )

    @staticmethod
    def validate_fabric_width(width_cm: float) -> bool:
        """
        Validate fabric width against supported widths.

        Args:
            width_cm: Fabric width in centimeters

        Returns:
            bool: True if width is valid, False otherwise
        """
        return float(width_cm) in AppConfig.FABRIC_WIDTHS_CM

    @staticmethod
    def validate_garment_type(garment_type: str) -> bool:
        """
        Validate garment type against supported types.

        Args:
            garment_type: Type of garment

        Returns:
            bool: True if garment type is valid, False otherwise
        """
        return str(garment_type) in AppConfig.GARMENT_TYPES

    @staticmethod
    def validate_order_quantity(quantity: int) -> bool:
        """
        Validate order quantity is within acceptable range.

        Args:
            quantity: Order quantity

        Returns:
            bool: True if quantity is valid, False otherwise
        """
        try:
            qty = int(quantity)
            return 0 < qty <= 1000000
        except (ValueError, TypeError):
            return False


class ModelManager:
    """
    ML model loading and inference management.

    Developer: Azim Mahmud | Version 3.0.0
    """

    def __init__(self):
        """Initialize model manager with empty model storage and check TensorFlow availability."""
        self.models: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Any] = {}
        self._load_attempted = False
        self.lstm_available = False
        self.mode = "ensemble"  # Default processing mode

        # Check TensorFlow availability conditionally
        if AppConfig.ENABLE_LSTM:
            self._load_tensorflow()

    def _load_tensorflow(self):
        """Lazy load TensorFlow only if ENABLE_LSTM is true."""
        try:
            import tensorflow as tf
            self.tf = tf
            AppConfig.LSTM_AVAILABLE = True
            self.lstm_available = True
            logger.info("TensorFlow loaded successfully - LSTM model enabled")
        except ImportError:
            self.tf = None
            AppConfig.LSTM_AVAILABLE = False
            self.lstm_available = False
            logger.warning("TensorFlow not available - LSTM model disabled")

    def load_models(self) -> Tuple[Dict[str, Any], bool]:
        """
        Load all available ML models from disk.

        Returns:
            tuple: (models_dict, is_production_mode)

        Raises:
            ModelLoadError: If a critical model file is missing in production
        """
        if self._load_attempted:
            return self.models, len(self.models) > 0

        self._load_attempted = True

        model_dir = AppConfig.MODEL_PATH

        try:
            logger.info(f"Attempting to load models from {model_dir}")

            # Check if model directory exists
            if not model_dir.exists():
                logger.warning(f"Model directory {model_dir} not found, using demo mode")
                return self.models, False

            # Load each model file
            loaded_models = {}

            # Load traditional ML models
            for model_name in ["xgboost", "random_forest", "linear_regression", "ensemble", "scaler", "encoders"]:
                if model_name not in AppConfig.MODEL_FILES:
                    continue

                filename = AppConfig.MODEL_FILES[model_name]
                model_path = model_dir / filename

                if model_path.exists():
                    try:
                        # This would require joblib import
                        # loaded_models[model_name] = joblib.load(model_path)
                        logger.info(f"Mock loading {model_name} from {model_path}")
                        loaded_models[model_name] = f"mock_{model_name}_model"
                    except Exception as e:
                        logger.error(f"Failed to load {model_name}: {e}")
                        if AppConfig.is_production():
                            raise ModelLoadError(f"Failed to load {model_name}: {e}") from e
                else:
                    logger.warning(f"Model file not found: {model_path}")

            # Load LSTM model if available
            if self.lstm_available and self.tf is not None:
                lstm_path = model_dir / AppConfig.MODEL_FILES["lstm"]
                if lstm_path.exists():
                    try:
                        # Mock LSTM model
                        loaded_models["lstm"] = "mock_lstm_model"
                        logger.info(f"Mock loaded LSTM model from {lstm_path}")
                    except Exception as e:
                        logger.error(f"Failed to load LSTM model: {e}")
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
                    'unit': 'yards',
                    'training_date': datetime.now().isoformat(),
                    'tensorflow_available': self.lstm_available
                }
                logger.warning("Metadata file not found, using default metadata")

            self.models = loaded_models
            self.model_metadata = loaded_models.get('metadata', {})
            logger.info(f"Successfully loaded models")
            logger.info(f"Available models: {[k for k in loaded_models.keys() if k not in ['scaler', 'encoders', 'metadata']]}")

            return self.models, True

        except ModelLoadError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading models: {e}")

            if AppConfig.is_production():
                raise ModelLoadError(f"Failed to load models: {e}") from e
            else:
                return self.models, False

    def predict(self, order_input: Dict[str, Any], model_type: str = 'xgboost') -> 'PredictionResult':
        """
        Generate fabric consumption prediction.

        Args:
            order_input: Dictionary containing order information
            model_type: Type of model to use ('xgboost', 'random_forest', 'linear_regression', 'lstm', 'ensemble')

        Returns:
            PredictionResult: Prediction with confidence bounds
        """
        try:
            # Prepare features for prediction
            features = self._prepare_features(order_input)

            # Generate prediction based on model type
            if model_type == "ensemble":
                prediction_yd = self._ensemble_predict(features)
            else:
                prediction_yd = self._single_model_predict(features, model_type)

            # Calculate confidence
            confidence = self._calculate_confidence(order_input, model_type)

            return PredictionResult(
                prediction=prediction_yd,
                unit='yards',
                confidence_lower=prediction_yd * (1 - confidence),
                confidence_upper=prediction_yd * (1 + confidence),
                model_name=model_type,
                timestamp=datetime.now()
            )

        except PredictionError:
            raise
        except Exception as e:
            logger.error(f"Prediction error ({model_type}): {e}")
            raise PredictionError(f"Failed to calculate prediction for '{model_type}': {type(e).__name__}: {e}") from e

    def _prepare_features(self, order_input: Dict[str, Any]) -> np.ndarray:
        """
        Prepare feature array for model prediction.

        Args:
            order_input: Dictionary containing order information

        Returns:
            np.ndarray: Feature array ready for model prediction
        """
        features = np.array([[
            order_input['order_quantity'],
            order_input['fabric_width_cm'],
            order_input['marker_efficiency'],
            order_input['defect_rate'],
            order_input['operator_experience'],
            order_input['garment_type_encoded'],
            order_input['fabric_type_encoded'],
            order_input['pattern_complexity_encoded'],
            order_input.get('season_encoded', 1),  # default=1 (Spring) for back-compat
        ]])

        return features

    def _ensemble_predict(self, features: np.ndarray) -> float:
        """
        Generate prediction using ensemble of available models.

        Args:
            features: Feature array for prediction

        Returns:
            float: Ensemble prediction
        """
        # Mock ensemble prediction
        # In real implementation, this would weight predictions from multiple models
        return self._single_model_predict(features, 'xgboost') * 0.43 + \
               self._single_model_predict(features, 'random_forest') * 0.30 + \
               self._single_model_predict(features, 'linear_regression') * 0.27

    def _single_model_predict(self, features: np.ndarray, model_type: str) -> float:
        """
        Generate prediction using single model.

        Args:
            features: Feature array for prediction
            model_type: Type of model to use

        Returns:
            float: Model prediction
        """
        # Mock model prediction based on model type
        if model_type == "lstm":
            # LSTM requires 3D input: (samples, timesteps, features)
            features_3d = features.reshape(features.shape[0], 1, features.shape[1])
            # Mock LSTM prediction
            return 2.5  # Mock value
        else:
            # Traditional ML models
            # Mock predictions based on garment type and other factors
            garment_type = features[0][5]
            base_consumption = [3.00, 3.50, 2.50, 1.80, 1.20][int(garment_type)]  # Dress, Jacket, Pants, Shirt, T-Shirt
            width_factor = 160.0 / features[0][1]  # Width adjustment
            return base_consumption * width_factor

    def _fallback_prediction(self, features: np.ndarray) -> float:
        """
        Generate fallback prediction using BOM calculation.

        Args:
            features: Feature array for prediction

        Returns:
            float: BOM-based prediction
        """
        # Basic BOM calculation as fallback
        order_quantity = features[0][0]
        garment_type = int(features[0][5])
        width_cm = features[0][1]

        # Base consumption at 160cm width
        base_consumption_yd = [3.2808, 3.8276, 2.7340, 1.9685, 1.3123][garment_type]  # Dress, Jacket, Pants, Shirt, T-Shirt

        # Adjust for actual width
        adjusted_consumption = base_consumption_yd * (160.0 / width_cm)

        # Apply buffer for defects and other factors
        return order_quantity * adjusted_consumption * 1.05

    def _calculate_confidence(self, order_input: Dict[str, Any], model_type: str) -> float:
        """
        Calculate prediction confidence score.

        Args:
            order_input: Dictionary containing order information
            model_type: Type of model used for prediction

        Returns:
            float: Confidence score (0-1)
        """
        # Model-specific confidence fractions
        confidence_fractions = {
            'ensemble': 0.019,
            'xgboost': 0.023,
            'random_forest': 0.027,
            'lstm': 0.031,
            'linear_regression': 0.062,
        }

        # Get confidence for model type
        ci_frac = confidence_fractions.get(model_type, 0.050)

        # Safety clamp: ci_frac must be a fractional value (0.5-50% of prediction)
        return float(np.clip(ci_frac, 0.005, 0.50))

    def get_model_status(self) -> Dict[str, Any]:
        """
        Get status of all models.

        Returns:
            Dict[str, Any]: Status information for all models
        """
        status = {}

        for model_name in ["xgboost", "random_forest", "linear_regression", "lstm", "ensemble"]:
            if model_name in self.models:
                status[model_name] = {
                    'loaded': True,
                    'type': 'traditional' if model_name != 'lstm' else 'lstm',
                    'available': True
                }
            else:
                status[model_name] = {
                    'loaded': False,
                    'type': 'traditional' if model_name != 'lstm' else 'lstm',
                    'available': False
                }

        return status

    def is_model_loaded(self, model_name: str) -> bool:
        """
        Check if specific model is loaded.

        Args:
            model_name: Name of the model to check

        Returns:
            bool: True if model is loaded, False otherwise
        """
        return model_name in self.models and self.models[model_name] is not None

    def get_model_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for all loaded models.

        Returns:
            Dictionary with model names as keys and metrics as values
        """
        metrics = {}
        for model_name, model in self.models.items():
            if hasattr(model, 'score_'):
                metrics[model_name] = {
                    'r2_score': getattr(model, 'score_', None),
                    'loaded': True
                }
            else:
                metrics[model_name] = {
                    'r2_score': None,
                    'loaded': True
                }
        return metrics


class DataGenerator:
    """
    Generate synthetic training data for model development.

    Developer: Azim Mahmud | Version 3.0.0
    """

    @staticmethod
    def generate_synthetic_data(n_samples=1000) -> pd.DataFrame:
        """
        Generate synthetic fabric consumption data.

        Args:
            n_samples: Number of samples to generate (default: 1000)

        Returns:
            pd.DataFrame: Synthetic data with fabric consumption metrics
        """
        rng = np.random.default_rng(42)

        # Generate synthetic data
        garment_types = rng.choice(AppConfig.GARMENT_TYPES, n_samples)
        fabric_types = rng.choice(AppConfig.FABRIC_TYPES, n_samples)
        order_quantities = rng.integers(100, 5001, n_samples).astype(float)
        fabric_widths = rng.choice(AppConfig.FABRIC_WIDTHS_CM, n_samples)
        marker_efficiency = rng.normal(85.0, 5.0, n_samples).clip(70, 95)
        defect_rates = rng.exponential(2.0, n_samples).clip(0, 10)

        # Generate consumption data
        data = []
        for i in range(n_samples):
            row = {
                'Order_ID': f'ORD_{str(i+1).zfill(6)}',
                'Garment_Type': garment_types[i],
                'Fabric_Type': fabric_types[i],
                'Order_Quantity': int(order_quantities[i]),
                'Fabric_Width_cm': fabric_widths[i],
                'Marker_Efficiency_%': marker_efficiency[i],
                'Defect_Rate_%': defect_rates[i],
                'Fabric_Consumption_yd': DataGenerator._calculate_consumption({
                    'Garment_Type': garment_types[i],
                    'Order_Quantity': order_quantities[i],
                    'Fabric_Width_cm': fabric_widths[i],
                    'Marker_Efficiency_%': marker_efficiency[i],
                    'Defect_Rate_%': defect_rates[i]
                })
            }
            data.append(row)

        df = pd.DataFrame(data)
        logger.info(f"Generated {n_samples} synthetic records")
        return df

    @staticmethod
    def _calculate_consumption(row) -> float:
        """
        Calculate fabric consumption based on BOM.

        Args:
            row: Dictionary containing order parameters

        Returns:
            float: Fabric consumption in yards
        """
        # Base consumption values for different garment types (at 160cm width)
        base_consumption = {
            'Dress': 3.5,
            'Jacket': 3.2,
            'Pants': 2.8,
            'Shirt': 2.2,
            'T-Shirt': 1.5
        }

        # Get base consumption for garment type
        garment_type = row['Garment_Type']
        consumption = base_consumption.get(garment_type, 2.0)

        # Adjust for fabric width
        width_factor = 160.0 / row['Fabric_Width_cm']
        consumption *= width_factor

        # Adjust for order quantity (economies of scale)
        quantity_factor = 1.0 - (0.02 * np.log(row['Order_Quantity'] / 100))
        consumption *= quantity_factor

        # Adjust for marker efficiency
        efficiency_factor = 1.0 + ((85.0 - row['Marker_Efficiency_%']) / 100.0) * 0.05
        consumption *= efficiency_factor

        # Adjust for defect rate
        defect_factor = 1.0 + (row['Defect_Rate_%'] / 100.0) * 0.1
        consumption *= defect_factor

        # Add buffer
        buffer_factor = AppConfig.DEFAULT_BOM_BUFFER
        consumption *= buffer_factor

        return max(consumption, 0.1)  # Minimum 0.1 yards


class SessionManager:
    """
    Manage Streamlit session state.

    Developer: Azim Mahmud | Version 3.0.0
    """

    @staticmethod
    def initialize_session() -> None:
        """
        Initialize session state variables.

        Args:
            None
        """
        defaults = {
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
    def initialize():
        """Initialize session state variables (alias for initialize_session)."""
        SessionManager.initialize_session()

    @staticmethod
    def update_activity() -> None:
        """
        Update last activity timestamp.

        Args:
            None
        """
        st.session_state.last_activity = datetime.now()

    @staticmethod
    def is_session_valid() -> bool:
        """
        Check if session is still valid (not timed out).

        Returns:
            bool: True if session is valid, False if timed out
        """
        if 'session_start' not in st.session_state:
            return True

        elapsed = (datetime.now() - st.session_state.last_activity).total_seconds()
        elapsed_minutes = elapsed / 60

        return elapsed_minutes < AppConfig.SESSION_TIMEOUT_MINUTES

    @staticmethod
    def increment_prediction_count() -> int:
        """
        Increment and return prediction count.

        Returns:
            int: Updated prediction count
        """
        if 'predictions_count' not in st.session_state:
            st.session_state['predictions_count'] = 0

        st.session_state['predictions_count'] += 1
        return st.session_state['predictions_count']

    @staticmethod
    def get_session_stats() -> Dict[str, Any]:
        """
        Get session statistics.

        Returns:
            Dict[str, Any]: Session statistics dictionary
        """
        return {
            'predictions_count': st.session_state.get('predictions_count', 0),
            'total_savings': st.session_state.get('total_savings', 0.0),
            'session_duration': (
                datetime.now() - st.session_state.get('session_start', datetime.now())
            ).total_seconds() / 60,
            'current_unit': 'yards'
        }


class UIHelpers:
    """
    Reusable UI helper functions.

    Developer: Azim Mahmud | Version 3.0.0
    """

    @staticmethod
    def display_metric_card(title: str, value: float, delta: Optional[float] = None) -> None:
        """
        Display a metric card.

        Args:
            title: Title of the metric
            value: Value to display
            delta: Change value (optional)
        """
        if delta is not None:
            st.metric(title, f"{value:,.2f}", delta=f"{delta:+.2f}")
        else:
            st.metric(title, f"{value:,.2f}")

    @staticmethod
    def format_consumption(yards: float, meters: Optional[float] = None) -> str:
        """
        Format consumption value for display.

        Args:
            yards: Consumption value in yards
            meters: Optional consumption value in meters

        Returns:
            str: Formatted consumption string
        """
        if meters is not None:
            return f"{yards:.2f} yd / {meters:.2f} m"
        else:
            return f"{yards:.2f} yd"

    @staticmethod
    def create_comparison_chart(predictions: pd.DataFrame, title: str) -> None:
        """
        Create a comparison chart.

        Args:
            predictions: DataFrame containing prediction data
            title: Chart title
        """
        fig = {
            'data': [
                {
                    'x': predictions.index,
                    'y': predictions['actual'],
                    'name': 'Actual',
                    'type': 'scatter',
                    'mode': 'lines+markers'
                },
                {
                    'x': predictions.index,
                    'y': predictions['predicted'],
                    'name': 'Predicted',
                    'type': 'scatter',
                    'mode': 'lines+markers'
                }
            ],
            'layout': {
                'title': title,
                'xaxis': {'title': 'Order'},
                'yaxis': {'title': 'Fabric Consumption (yards)'}
            }
        }
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def display_system_health(health: Dict[str, Any]) -> None:
        """
        Display system health status.

        Args:
            health: Dictionary containing health metrics
        """
        status = health.get('status', 'unknown')
        message = health.get('message', '')

        if status == 'healthy':
            st.success(f"✅ System Health: {message}")
        elif status == 'warning':
            st.warning(f"⚠️ System Health: {message}")
        elif status == 'error':
            st.error(f"❌ System Health: {message}")
        else:
            st.info(f"ℹ️ System Health: {message}")

    @staticmethod
    def apply_custom_styles():
        """Apply custom CSS styles to the Streamlit application."""
        import streamlit as st

        custom_css = """
        <style>
            /* Your custom CSS styles here */
            .main-header {
                font-size: 2rem;
                font-weight: bold;
                color: #1f77b4;
            }
        </style>
        """

        st.markdown(custom_css, unsafe_allow_html=True)

    @staticmethod
    def render_footer(model_manager=None):
        """Render application footer with copyright and version info."""
        import streamlit as st
        from app.config import AppConfig

        st.markdown("---")
        st.markdown(
            f"""
            <div style='text-align: center; color: gray; font-size: 0.8rem;'>
                © 2026 {AppConfig.APP_AUTHOR}. {AppConfig.APP_NAME} v{AppConfig.APP_VERSION}
            </div>
            """,
            unsafe_allow_html=True
        )
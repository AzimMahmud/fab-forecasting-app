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
from typing import Optional, Dict, Any, Tuple, List

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

    def load_models(self) -> None:
        """
        Load all available ML models from disk.

        Raises:
            ModelLoadError: If model loading fails
        """
        model_path = AppConfig.MODEL_PATH

        if not model_path.exists():
            raise ModelLoadError(f"Model directory not found: {model_path}")

        # Load scikit-learn models
        sklearn_models = {
            "xgboost": "xgboost_model.pkl",
            "random_forest": "random_forest_model.pkl",
            "linear_regression": "linear_regression_model.pkl",
            "ensemble": "ensemble_model.pkl"
        }

        for model_name, filename in sklearn_models.items():
            model_file = model_path / filename

            if model_file.exists():
                try:
                    if JOBLIB_AVAILABLE:
                        import joblib
                        self.models[model_name] = joblib.load(model_file)
                        self.model_metadata[model_name] = {
                            "loaded": True,
                            "type": "sklearn",
                            "path": str(model_file),
                            "file_size": model_file.stat().st_size
                        }
                        logger.info(f"Loaded {model_name} model from {filename}")
                    else:
                        logger.error("joblib not available - cannot load sklearn models")
                        self.model_metadata[model_name] = {
                            "loaded": False,
                            "error": "joblib not available"
                        }
                except Exception as e:
                    logger.error(f"Failed to load {model_name}: {e}")
                    self.model_metadata[model_name] = {
                        "loaded": False,
                        "error": str(e)
                    }
            else:
                logger.warning(f"Model file not found: {model_file}")
                self.model_metadata[model_name] = {"loaded": False}

        # Load auxiliary models
        auxiliary_models = {
            "scaler": "scaler.pkl",
            "encoders": "label_encoders.pkl"
        }

        for model_name, filename in auxiliary_models.items():
            model_file = model_path / filename

            if model_file.exists():
                try:
                    if JOBLIB_AVAILABLE:
                        import joblib
                        self.models[model_name] = joblib.load(model_file)
                        self.model_metadata[model_name] = {
                            "loaded": True,
                            "type": "auxiliary",
                            "path": str(model_file),
                            "file_size": model_file.stat().st_size
                        }
                        logger.info(f"Loaded {model_name} from {filename}")
                    else:
                        logger.error("joblib not available - cannot load auxiliary models")
                        self.model_metadata[model_name] = {
                            "loaded": False,
                            "error": "joblib not available"
                        }
                except Exception as e:
                    logger.error(f"Failed to load {model_name}: {e}")
                    self.model_metadata[model_name] = {
                        "loaded": False,
                        "error": str(e)
                    }
            else:
                logger.warning(f"Model file not found: {model_file}")
                self.model_metadata[model_name] = {"loaded": False}

        # Load LSTM model if enabled
        if AppConfig.ENABLE_LSTM and self.tf is not None:
            lstm_path = model_path / "lstm_model.h5"

            if lstm_path.exists():
                try:
                    self.models["lstm"] = self.tf.keras.models.load_model(
                        lstm_path, compile=False
                    )
                    self.model_metadata["lstm"] = {
                        "loaded": True,
                        "type": "tensorflow",
                        "path": str(lstm_path),
                        "file_size": lstm_path.stat().st_size
                    }
                    logger.info("Loaded LSTM model")
                except Exception as e:
                    logger.error(f"Failed to load LSTM: {e}")
                    self.model_metadata["lstm"] = {
                        "loaded": False,
                        "error": str(e)
                    }
            else:
                logger.warning(f"LSTM model file not found: {lstm_path}")
                self.model_metadata["lstm"] = {"loaded": False}
        else:
            logger.info("LSTM model loading skipped (TensorFlow not available or disabled)")

        # Load metadata
        metadata_path = model_path / "model_metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    self.models['metadata'] = json.load(f)
                    self.model_metadata['metadata'] = self.models['metadata']
                logger.info(f"Loaded metadata from {metadata_path}")
            except Exception as e:
                logger.error(f"Failed to load metadata: {e}")
                self.model_metadata['metadata'] = {
                    'version': '3.0.0',
                    'unit': 'yards',
                    'training_date': datetime.now().isoformat(),
                    'tensorflow_available': self.lstm_available
                }
        else:
            self.model_metadata['metadata'] = {
                'version': '3.0.0',
                'unit': 'yards',
                'training_date': datetime.now().isoformat(),
                'tensorflow_available': self.lstm_available
            }
            logger.warning("Metadata file not found, using default metadata")

        logger.info(f"Model loading complete: {sum(1 for m in self.model_metadata.values() if m.get('loaded'))}/{len(self.model_metadata)} models loaded")

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
        # Check if model is loaded
        if model_type not in self.models or not self.model_metadata[model_type].get('loaded', False):
            logger.warning(f"Model {model_type} not loaded, using fallback")
            return self._fallback_prediction(features)

        try:
            if model_type == "lstm":
                # LSTM requires 3D input: (samples, timesteps, features)
                features_3d = features.reshape(features.shape[0], 1, features.shape[1])
                prediction = self.models[model_type].predict(features_3d)[0][0]
            else:
                # Traditional ML models
                prediction = self.models[model_type].predict(features)[0]

            logger.debug(f"Prediction from {model_type}: {prediction}")
            return float(prediction)

        except Exception as e:
            logger.error(f"Prediction failed for {model_type}: {e}")
            # Fallback to BOM calculation
            return self._fallback_prediction(features)

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

        # Check main prediction models
        for model_name in ["xgboost", "random_forest", "linear_regression", "lstm", "ensemble"]:
            metadata = self.model_metadata.get(model_name, {})
            status[model_name] = {
                'loaded': metadata.get('loaded', False),
                'type': metadata.get('type', 'traditional' if model_name != 'lstm' else 'lstm'),
                'available': model_name in self.models and metadata.get('loaded', False),
                'error': metadata.get('error', None)
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

    def predict_batch(self, df: pd.DataFrame) -> List['PredictionResult']:
        """
        Generate predictions for multiple orders.

        Args:
            df: DataFrame with order data

        Returns:
            List of PredictionResult objects
        """
        results = []
        for idx, row in df.iterrows():
            try:
                # Create OrderInput from row
                order_input = self._create_order_input_from_row(row)

                # Generate prediction
                result = self.predict(order_input.to_dict(), self.mode)
                results.append(result)
            except Exception as e:
                logger.error(f"Prediction failed for row {idx}: {e}")
                # Use fallback calculation
                results.append(self._fallback_prediction_from_row(row))

        return results

    def _create_order_input_from_row(self, row: pd.Series) -> 'OrderInput':
        """Create OrderInput from DataFrame row."""
        return OrderInput(
            order_id=str(row.get('Order_ID', f'BATCH_{len(row)}')),
            garment_type=str(row.get('Garment_Type', 'T-Shirt')),
            fabric_width_cm=float(row.get('Fabric_Width_CM', 150)),
            fabric_type=str(row.get('Fabric_Type', 'Single Jersey')),
            order_quantity=int(row.get('Order_Quantity', 0)),
            quality_level=str(row.get('Quality_Level', 'Standard')),
            color=str(row.get('Color', 'White'))
        )

    def _fallback_prediction_from_row(self, row: pd.Series) -> 'PredictionResult':
        """Generate fallback prediction from DataFrame row."""
        garment_type = str(row.get('Garment_Type', 'T-Shirt'))
        quantity = int(row.get('Order_Quantity', 0))

        base_consumption = AppConfig.GARMENT_BASE_CONSUMPTION_YD.get(
            garment_type, 1.641
        )
        predicted_yd = float(quantity) * base_consumption
        predicted_m = UnitConverter.convert_from_yards(predicted_yd, UnitType.METERS)

        return PredictionResult(
            predicted_yards=predicted_yd,
            predicted_meters=predicted_m,
            model_used='fallback_bom',
            confidence_score=0.7,
            processing_time_ms=0,
            order_id=str(row.get('Order_ID', f'BATCH_{len(row)}')),
            timestamp=datetime.now()
        )


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


class ROICalculator:
    """
    ROI Calculator for fabric cost optimization.

    Developer: Azim Mahmud | Version 4.0.0
    """

    @staticmethod
    def calculate_roi(
        current_consumption: float,
        predicted_consumption: float,
        fabric_cost_per_yard: float = 2.50,
        labor_cost_per_garment: float = 0.50
    ) -> Dict[str, Any]:
        """
        Calculate return on investment for optimization.

        Args:
            current_consumption: Current fabric consumption in yards
            predicted_consumption: Predicted/optimized consumption in yards
            fabric_cost_per_yard: Cost per yard of fabric
            labor_cost_per_garment: Labor cost per garment

        Returns:
            Dictionary with ROI metrics
        """
        savings_yards = current_consumption - predicted_consumption
        savings_fabric_cost = savings_yards * fabric_cost_per_yard

        # Calculate production efficiency
        efficiency = (predicted_consumption / current_consumption) * 100 if current_consumption > 0 else 100

        return {
            "current_consumption": current_consumption,
            "predicted_consumption": predicted_consumption,
            "savings_yards": savings_yards,
            "savings_percentage": ((savings_yards / current_consumption) * 100) if current_consumption > 0 else 0,
            "fabric_cost_savings": savings_fabric_cost,
            "efficiency_improvement": 100 - efficiency,
            "projected_annual_savings": savings_fabric_cost * 12,  # Assuming monthly
            "payback_period_months": 0  # Will be calculated based on implementation cost
        }

    @staticmethod
    def generate_roi_report(roi_data: Dict[str, Any]) -> str:
        """
        Generate human-readable ROI report.

        Args:
            roi_data: ROI calculation results

        Returns:
            Formatted report string
        """
        return f"""
ROI Analysis Report
===================

Current Consumption: {roi_data['current_consumption']:.2f} yards
Predicted Consumption: {roi_data['predicted_consumption']:.2f} yards

Savings:
- Fabric Saved: {roi_data['savings_yards']:.2f} yards ({roi_data['savings_percentage']:.1f}%)
- Cost Savings: ${roi_data['fabric_cost_savings']:.2f}
- Projected Annual Savings: ${roi_data['projected_annual_savings']:.2f}

Efficiency Improvement: {roi_data['efficiency_improvement']:.1f}%
"""


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
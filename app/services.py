"""
Business logic module for fabric forecast application.

Contains services and business logic that operate independently of the Streamlit UI.
Dependencies only on app.config.

Developer: Azim Mahmud | Version 3.1.0
"""

import logging
import typing
import datetime
import pathlib
from typing import Optional

import numpy as np
import pandas as pd

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

# Initialize logger
logger = logging.getLogger(AppConfig.APP_NAME)


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
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

    YARDS_TO_METERS = 0.9144
    METERS_TO_YARDS = 1.0936132983

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
    def yards_to_meters(yards: float) -> float:
        """Convert yards to meters"""
        return yards * UnitConverter.YARDS_TO_METERS

    @staticmethod
    def meters_to_yards(meters: float) -> float:
        """Convert meters to yards"""
        return meters * UnitConverter.METERS_TO_YARDS

    @staticmethod
    def inches_to_cm(inches: float) -> float:
        """Convert inches to centimeters"""
        return inches * 2.54

    @staticmethod
    def cm_to_inches(cm: float) -> float:
        """Convert centimeters to inches"""
        return cm / 2.54

    @staticmethod
    def format_display(value: float, unit: str, decimals: int = 2) -> str:
        """Format value with unit for display"""
        return f"{value:.{decimals}f} {unit}"
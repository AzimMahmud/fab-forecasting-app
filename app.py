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

import os
import logging
from pathlib import Path

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

# Import theme early for proper styling
from app.ui_theme import apply_custom_styles as apply_theme_styles

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


# UI Components

class SidebarRenderer:
    @staticmethod
    def render(models_loaded: bool, mode: str) -> str:
        """Render sidebar navigation"""
        with st.sidebar:
            st.title("🧵 Fabric Forecast Pro")

            # Model status
            if models_loaded:
                st.success("✅ Models loaded")
            else:
                st.warning("⚠️ Demo mode")

            # Navigation
            page = st.selectbox(
                "Navigation",
                ["🏠 Dashboard", "🎯 Single Prediction", "📊 Batch Prediction", "📈 Performance", "💰 ROI Calculator", "📚 Documentation"],
                key="page_select"
            )

            # Mode indicator
            st.caption(f"Mode: {mode}")

            return page

class DashboardPage:
    @staticmethod
    def render():
        """Render dashboard page"""
        st.header("🏠 Dashboard")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.metric("Active Models", "4", "2 from last week")
            st.metric("Accuracy", "94.2%", "+0.3%")

        with col2:
            st.metric("Predictions Today", "156", "+12")
            st.metric("Avg Processing Time", "0.45s", "-0.05s")

        # Quick actions
        st.subheader("Quick Actions")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🎯 Single Prediction", use_container_width=True):
                st.session_state.page = "🎯 Single Prediction"

        with col2:
            if st.button("📊 Batch Prediction", use_container_width=True):
                st.session_state.page = "📊 Batch Prediction"

        with col3:
            if st.button("📈 Performance", use_container_width=True):
                st.session_state.page = "📈 Performance"

class SinglePredictionPage:
    @staticmethod
    def render(model_manager: ModelManager):
        """Render single prediction page"""
        st.header("🎯 Single Prediction")

        # Input form
        with st.form("single_prediction_form"):
            col1, col2 = st.columns([1, 1])

            with col1:
                garment_type = st.selectbox("Garment Type", AppConfig.GARMENT_TYPES)
                fabric_type = st.selectbox("Fabric Type", AppConfig.FABRIC_TYPES)
                order_quantity = st.number_input("Order Quantity",
                                               min_value=AppConfig.ORDER_QUANTITY_MIN,
                                               max_value=AppConfig.ORDER_QUANTITY_MAX,
                                               value=1000)

            with col2:
                fabric_width = st.selectbox("Fabric Width (cm)", AppConfig.FABRIC_WIDTHS_CM)
                marker_efficiency = st.slider("Marker Efficiency (%)",
                                            min_value=AppConfig.MARKER_EFFICIENCY_MIN,
                                            max_value=AppConfig.MARKER_EFFICIENCY_MAX,
                                            value=85.0)
                defect_rate = st.slider("Defect Rate (%)",
                                      min_value=AppConfig.DEFECT_RATE_MIN,
                                      max_value=AppConfig.DEFECT_RATE_MAX,
                                      value=2.0)
                operator_experience = st.slider("Operator Experience (years)",
                                            min_value=AppConfig.OPERATOR_EXPERIENCE_MIN,
                                            max_value=AppConfig.OPERATOR_EXPERIENCE_MAX,
                                            value=5)

            pattern_complexity = st.selectbox("Pattern Complexity", AppConfig.PATTERN_COMPLEXITIES)
            season = st.selectbox("Season", AppConfig.SEASONS)

            submitted = st.form_submit_button("Predict")

            if submitted:
                # Prepare input data
                input_data = {
                    "Order_Quantity": order_quantity,
                    "Fabric_Width_cm": fabric_width,
                    "Marker_Efficiency_%": marker_efficiency,
                    "Expected_Defect_Rate_%": defect_rate,
                    "Operator_Experience_Years": operator_experience,
                    "Garment_Type": garment_type,
                    "Fabric_Type": fabric_type,
                    "Pattern_Complexity": pattern_complexity,
                    "Season": season
                }

                # Make prediction
                try:
                    result = model_manager.predict_single(input_data)

                    # Display results
                    st.success("✅ Prediction Successful!")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Predicted Consumption", f"{result['predicted_consumption']:.2f} yards")

                    with col2:
                        st.metric("Confidence", f"{result['confidence']:.1%}")

                    # Cost estimation
                    fabric_cost = AppConfig.FABRIC_COST_PER_YARD.get(fabric_type, AppConfig.DEFAULT_FABRIC_COST_PER_YARD)
                    total_cost = result['predicted_consumption'] * fabric_cost
                    st.metric("Estimated Fabric Cost", f"${total_cost:.2f}")

                except Exception as e:
                    st.error(f"❌ Prediction failed: {e}")

class BatchPredictionPage:
    @staticmethod
    def render(model_manager: ModelManager):
        """Render batch prediction page."""
        import streamlit as st
        from app.ui_notifications import show_success, show_error, show_warning

        st.header("📦 Batch Prediction")
        st.markdown("Upload a CSV file with multiple orders to generate predictions.")

        # File upload
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            help="Upload a CSV file with columns: Order_ID, Garment_Type, Fabric_Width_CM, Fabric_Type, Order_Quantity, Quality_Level, Color"
        )

        if uploaded_file is not None:
            try:
                # Read CSV
                df = pd.read_csv(uploaded_file)

                # Validate columns
                required_columns = ['Order_ID', 'Garment_Type', 'Fabric_Width_CM',
                                  'Fabric_Type', 'Order_Quantity', 'Quality_Level', 'Color']
                missing_columns = [col for col in required_columns if col not in df.columns]

                if missing_columns:
                    show_error(f"Missing required columns: {', '.join(missing_columns)}")
                    return

                # Show data preview
                st.subheader("Data Preview")
                st.dataframe(df.head(10))

                # Show file info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Rows", len(df))
                with col2:
                    st.metric("Columns", len(df.columns))
                with col3:
                    st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")

                # Generate predictions button
                if st.button("Generate Predictions", type="primary"):
                    with st.spinner("Processing predictions..."):
                        try:
                            # Generate predictions
                            results = model_manager.predict_batch(df)

                            # Add predictions to dataframe
                            df['Predicted_Yards'] = [r.predicted_yards for r in results]
                            df['Predicted_Meters'] = [r.predicted_meters for r in results]
                            df['Model_Used'] = [r.model_used for r in results]
                            df['Confidence'] = [r.confidence_score for r in results]

                            # Show results
                            st.subheader("Prediction Results")
                            st.dataframe(df)

                            # Summary statistics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Orders", len(df))
                            with col2:
                                st.metric("Total Yards", f"{df['Predicted_Yards'].sum():.2f}")
                            with col3:
                                st.metric("Avg Confidence", f"{df['Confidence'].mean():.1%}")

                            # Export button
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results as CSV",
                                data=csv,
                                file_name="predictions.csv",
                                mime="text/csv"
                            )
                            show_success(f"Successfully processed {len(df)} orders!")

                        except Exception as e:
                            show_error(f"Prediction failed: {str(e)}")
                            logger.error(f"Batch prediction error: {e}", exc_info=True)

            except Exception as e:
                show_error(f"Error reading file: {str(e)}")
                logger.error(f"File read error: {e}", exc_info=True)

class PerformancePage:
    @staticmethod
    def render(model_mgr: ModelManager):
        """Render performance page"""
        st.header("📈 Performance")

        # Model performance metrics
        metrics = model_mgr.get_model_metrics()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("XGBoost Accuracy", f"{metrics['xgboost_accuracy']:.1%}")

        with col2:
            st.metric("Random Forest Accuracy", f"{metrics['random_forest_accuracy']:.1%}")

        with col3:
            st.metric("Ensemble Accuracy", f"{metrics['ensemble_accuracy']:.1%}")

        # Feature importance
        st.subheader("Feature Importance")
        feature_impact = model_mgr.get_feature_importance()

        # Simple bar chart
        fig = px.bar(x=feature_impact.values, y=feature_impact.index, orientation='h')
        fig.update_layout(height=400)
        st.plotly_chart(fig)

class ROICalculatorPage:
    @staticmethod
    def render():
        """Render ROI calculator page"""
        st.header("💰 ROI Calculator")

        col1, col2 = st.columns(2)

        with col1:
            order_size = st.number_input("Order Size (units)", value=1000)
            unit_price = st.number_input("Unit Price ($)", value=25.0)
            fabric_cost_pct = st.slider("Fabric Cost % of Total", 20, 50, value=35)

        with col2:
            current_consumption = st.number_input("Current Consumption (yards/unit)", value=2.5)
            predicted_consumption = st.number_input("Predicted Consumption (yards/unit)", value=2.3)
            waste_reduction_pct = st.slider("Additional Waste Reduction (%)", 0, 10, value=5)

        # Calculate savings
        fabric_savings_per_unit = (current_consumption - predicted_consumption) * (unit_price * fabric_cost_pct / 100)
        total_savings = fabric_savings_per_unit * order_size
        additional_savings = total_savings * (waste_reduction_pct / 100)
        total_roi = total_savings + additional_savings

        # Display results
        st.subheader("ROI Summary")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Fabric Savings", f"${total_savings:,.0f}")

        with col2:
            st.metric("Additional Savings", f"${additional_savings:,.0f}")

        with col3:
            st.metric("Total ROI", f"${total_roi:,.0f}")

        # Breakdown
        st.subheader("Cost Breakdown")

        breakdown_data = {
            "Current Fabric Cost": current_consumption * (unit_price * fabric_cost_pct / 100) * order_size,
            "Predicted Fabric Cost": predicted_consumption * (unit_price * fabric_cost_pct / 100) * order_size,
            "Total Revenue": unit_price * order_size,
            "Profit Improvement": total_roi
        }

        breakdown_df = pd.DataFrame(list(breakdown_data.items()), columns=["Item", "Amount"])
        st.dataframe(breakdown_df.style.format({"Amount": "${:,.0f}"}))

class DocumentationPage:
    @staticmethod
    def render(production_mode: bool):
        """Render documentation page"""
        st.header("📚 Documentation")

        st.markdown("""
        ## Fabric Forecast Pro - User Guide

        ### Overview
        Fabric Forecast Pro is an intelligent system for predicting fabric consumption in garment manufacturing.

        ### Features
        - **Single Prediction**: Predict fabric consumption for individual orders
        - **Batch Prediction**: Process multiple orders simultaneously
        - **Performance Analytics**: View model accuracy and feature importance
        - **ROI Calculator**: Calculate potential savings from optimization

        ### Supported Models
        - XGBoost
        - Random Forest
        - Linear Regression
        - Ensemble (weighted average)
        - LSTM (if TensorFlow available)

        ### Input Parameters
        - Order Quantity: 100-5000 units
        - Fabric Width: 140-180 cm
        - Marker Efficiency: 70-95%
        - Defect Rate: 0-10%
        - Operator Experience: 1-20 years
        - Garment Type: T-Shirt, Shirt, Pants, Dress, Jacket
        - Fabric Type: Cotton, Polyester, Cotton-Blend, Silk, Denim
        - Pattern Complexity: Simple, Medium, Complex
        - Season: Spring, Summer, Fall, Winter
        """)

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

        # Apply professional theme first (takes precedence)
        apply_theme_styles()

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

        # Initialize model manager
        model_manager = ModelManager()

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
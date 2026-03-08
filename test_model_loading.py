#!/usr/bin/env python3
"""Test script for model loading functionality"""

import sys
import os
import logging

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_loading():
    """Test if models can be loaded correctly"""
    try:
        from app.services import ModelManager
        from app.config import AppConfig

        logger.info("Starting model loading test...")

        # Create model manager
        m = ModelManager()

        # Load models
        m.load_models()

        # Print results
        print("\n=== Model Loading Test Results ===")
        print(f"Models loaded: {list(m.models.keys())}")
        print("\nModel metadata:")
        for name, meta in m.model_metadata.items():
            if name != 'metadata':
                status = "✅ LOADED" if meta.get('loaded', False) else "❌ FAILED"
                print(f"  {name}: {status}")
                if meta.get('error'):
                    print(f"    Error: {meta['error']}")

        # Test prediction (optional)
        print("\n=== Testing Prediction ===")
        try:
            test_order = {
                'order_quantity': 1000,
                'fabric_width_cm': 150,
                'marker_efficiency': 85.0,
                'defect_rate': 2.0,
                'operator_experience': 5,
                'garment_type_encoded': 0,
                'fabric_type_encoded': 0,
                'pattern_complexity_encoded': 0,
                'season_encoded': 1,
            }

            # Test XGBoost prediction
            if 'xgboost' in m.models and m.model_metadata['xgboost'].get('loaded'):
                result = m.predict(test_order, 'xgboost')
                print(f"XGBoost prediction: {result.prediction:.2f} yards")

            # Test ensemble prediction
            if 'ensemble' in m.models and m.model_metadata['ensemble'].get('loaded'):
                result = m.predict(test_order, 'ensemble')
                print(f"Ensemble prediction: {result.prediction:.2f} yards")

        except Exception as e:
            logger.error(f"Prediction test failed: {e}")

        return True

    except Exception as e:
        logger.error(f"Model loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_model_loading()
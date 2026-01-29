"""
================================================================================
                    FABRIC CONSUMPTION FORECASTING SYSTEM
                         MODEL TRAINING SCRIPT
================================================================================

Trains machine learning models for fabric consumption prediction using
generated historical data. Saves trained models in the models/ directory
for production use.

Key Features:
- Trains XGBoost, Random Forest, and Linear Regression models
- Handles categorical feature encoding
- Feature scaling for numerical features
- Evaluates model performance
- Saves trained models as .pkl files
- Updates model metadata with training information

Version:        1.0.0
Developer:      Azim Mahmud
Release Date:   January 2026

================================================================================
"""

# ============================================================================
# STANDARD LIBRARY IMPORTS
# ============================================================================

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path

# ============================================================================
# THIRD-PARTY IMPORTS
# ============================================================================

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

# ============================================================================
# CONFIGURATION
# ============================================================================

class TrainingConfig:
    """Configuration for model training"""
    
    # Data Configuration
    DATA_PATH = Path("generated_data")
    TRAINING_DATA = "training_dataset_1000_orders_meters.csv"
    PRODUCTION_DATA = "production_dataset_5000_orders_meters.csv"
    
    # Model Configuration
    MODEL_PATH = Path("models")
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    
    # Model Parameters
    XGB_PARAMS = {
        "objective": "reg:squarederror",
        "random_state": RANDOM_STATE,
        "n_estimators": 200,
        "max_depth": 8,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8
    }
    
    RF_PARAMS = {
        "random_state": RANDOM_STATE,
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "bootstrap": True,
        "n_jobs": -1
    }
    
    LR_PARAMS = {
        "fit_intercept": True
    }
    
    # Dataset column mapping
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
    
    FEATURES = [
        "order_quantity",
        "fabric_width_cm",
        "marker_efficiency",
        "defect_rate",
        "operator_experience",
        "garment_type_encoded",
        "fabric_type_encoded",
        "pattern_complexity_encoded"
    ]
    
    TARGET = "fabric_consumption_meters"
    
    CATEGORICAL_FEATURES = {
        "garment_type": ["T-Shirt", "Shirt", "Pants", "Dress", "Jacket"],
        "fabric_type": ["Cotton", "Polyester", "Cotton-Blend", "Silk", "Denim"],
        "pattern_complexity": ["Simple", "Medium", "Complex"]
    }


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def configure_logging():
    """Configure logging for training script"""
    logger = logging.getLogger("ModelTraining")
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    
    return logger

logger = configure_logging()


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

class DataPreprocessor:
    """Handles data loading, preprocessing, and feature engineering"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.label_encoders = {}
        self.scaler = None
    
    def load_data(self, data_file: str) -> pd.DataFrame:
        """Load dataset from CSV file"""
        try:
            data_path = self.config.DATA_PATH / data_file
            logger.info(f"Loading data from {data_path}")
            
            if not data_path.exists():
                raise FileNotFoundError(f"Data file not found: {data_path}")
                
            df = pd.read_csv(data_path)
            logger.info(f"Data loaded successfully: {df.shape[0]} records, {df.shape[1]} columns")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess and clean the dataset"""
        logger.info("Starting data preprocessing")
        
        # Rename columns to match expected format
        df = df.rename(columns=self.config.COLUMN_MAPPING)
        logger.info("Columns renamed to standard format")
        
        # Handle missing values
        df = df.dropna()
        logger.info(f"Data after dropping NA: {df.shape[0]} records")
        
        # Encode categorical features
        for feature, categories in self.config.CATEGORICAL_FEATURES.items():
            encoder = LabelEncoder()
            encoder.fit(categories)
            encoded_feature = f"{feature}_encoded"
            
            if feature in df.columns:
                df[encoded_feature] = encoder.transform(df[feature])
                self.label_encoders[feature] = encoder
                logger.info(f"Encoded {feature} into {encoded_feature}")
        
        # Select only required features
        available_features = [f for f in self.config.FEATURES if f in df.columns]
        if len(available_features) != len(self.config.FEATURES):
            missing = set(self.config.FEATURES) - set(available_features)
            logger.warning(f"Missing features: {missing}")
        
        # Ensure target variable exists
        if self.config.TARGET not in df.columns:
            raise ValueError(f"Target variable '{self.config.TARGET}' not found in dataset")
        
        logger.info("Data preprocessing completed successfully")
        return df
    
    def scale_features(self, X: pd.DataFrame) -> np.ndarray:
        """Scale numerical features using StandardScaler"""
        if self.scaler is None:
            self.scaler = StandardScaler()
            scaled_features = self.scaler.fit_transform(X)
        else:
            scaled_features = self.scaler.transform(X)
            
        return scaled_features


# ============================================================================
# MODEL TRAINING
# ============================================================================

class ModelTrainer:
    """Trains and evaluates machine learning models"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.models = {}
        self.performance = {}
    
    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray) -> xgb.XGBRegressor:
        """Train XGBoost regression model"""
        logger.info("Training XGBoost model...")
        
        model = xgb.XGBRegressor(**self.config.XGB_PARAMS)
        model.fit(X_train, y_train)
        
        self.models["xgboost"] = model
        logger.info("XGBoost training completed")
        
        return model
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray) -> RandomForestRegressor:
        """Train Random Forest regression model"""
        logger.info("Training Random Forest model...")
        
        model = RandomForestRegressor(**self.config.RF_PARAMS)
        model.fit(X_train, y_train)
        
        self.models["random_forest"] = model
        logger.info("Random Forest training completed")
        
        return model
    
    def train_linear_regression(self, X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
        """Train Linear Regression model"""
        logger.info("Training Linear Regression model...")
        
        model = LinearRegression(**self.config.LR_PARAMS)
        model.fit(X_train, y_train)
        
        self.models["linear_regression"] = model
        logger.info("Linear Regression training completed")
        
        return model
    
    def evaluate_model(self, model: object, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate model performance"""
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        return {
            "mse": mse,
            "rmse": rmse,
            "r2": r2
        }
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                        X_test: np.ndarray, y_test: np.ndarray):
        """Train all models and evaluate performance"""
        
        # Train each model
        self.train_xgboost(X_train, y_train)
        self.train_random_forest(X_train, y_train)
        self.train_linear_regression(X_train, y_train)
        
        # Evaluate each model
        for name, model in self.models.items():
            self.performance[name] = self.evaluate_model(model, X_test, y_test)
            logger.info(f"{name} performance: R² = {self.performance[name]['r2']:.3f}")
        
        return self.models, self.performance


# ============================================================================
# MODEL SAVING AND METADATA
# ============================================================================

class ModelSaver:
    """Saves trained models and metadata"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        
        # Create model directory if it doesn't exist
        self.config.MODEL_PATH.mkdir(exist_ok=True)
    
    def save_model(self, model: object, filename: str) -> str:
        """Save model using joblib"""
        try:
            file_path = self.config.MODEL_PATH / filename
            joblib.dump(model, file_path)
            logger.info(f"Model saved to {file_path}")
            return str(file_path)
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
    
    def save_scaler(self, scaler: object) -> str:
        """Save feature scaler"""
        return self.save_model(scaler, "scaler.pkl")
    
    def save_label_encoders(self, encoders: dict) -> str:
        """Save label encoders"""
        return self.save_model(encoders, "label_encoders.pkl")
    
    def save_metadata(self, performance: dict, training_date: str, 
                     features: list, target: str) -> str:
        """Save model metadata"""
        metadata = {
            "version": "1.0.0",
            "training_date": training_date,
            "unit": "meters",
            "models": {},
            "feature_names": features,
            "target": target
        }
        
        for name, metrics in performance.items():
            metadata["models"][name] = {
                "file": f"{name}_model.pkl",
                "type": self._get_model_type(name),
                "features": len(features),
                "accuracy_r2": round(metrics["r2"], 3)
            }
        
        metadata_path = self.config.MODEL_PATH / "model_metadata.json"
        
        try:
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Metadata saved to {metadata_path}")
            return str(metadata_path)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
            raise
    
    def _get_model_type(self, model_name: str) -> str:
        """Get model type for metadata"""
        type_map = {
            "xgboost": "xgboost.XGBRegressor",
            "random_forest": "sklearn.ensemble.RandomForestRegressor",
            "linear_regression": "sklearn.linear_model.LinearRegression"
        }
        return type_map.get(model_name, "unknown")


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main():
    """Main training pipeline"""
    logger.info("=" * 60)
    logger.info("FABRIC CONSUMPTION FORECASTING SYSTEM - MODEL TRAINING")
    logger.info("=" * 60)
    
    try:
        # Initialize configuration
        config = TrainingConfig()
        
        # Initialize components
        preprocessor = DataPreprocessor(config)
        trainer = ModelTrainer(config)
        saver = ModelSaver(config)
        
        # Load and preprocess data
        logger.info("\n1. Loading and Preprocessing Data")
        logger.info("-" * 40)
        
        df = preprocessor.load_data(config.TRAINING_DATA)
        df = preprocessor.preprocess_data(df)
        
        logger.info(f"Available features: {config.FEATURES}")
        logger.info(f"Target variable: {config.TARGET}")
        
        # Split into features and target
        X = df[config.FEATURES]
        y = df[config.TARGET]
        
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Target shape: {y.shape}")
        
        # Split into training and testing sets
        logger.info("\n2. Splitting Data")
        logger.info("-" * 40)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE
        )
        
        logger.info(f"Training samples: {X_train.shape[0]}")
        logger.info(f"Testing samples: {X_test.shape[0]}")
        
        # Scale features
        logger.info("\n3. Scaling Features")
        logger.info("-" * 40)
        
        X_train_scaled = preprocessor.scale_features(X_train)
        X_test_scaled = preprocessor.scale_features(X_test)
        
        logger.info(f"Training data scaled to mean: {X_train_scaled.mean(axis=0).round(4)}")
        logger.info(f"Training data scaled to std: {X_train_scaled.std(axis=0).round(4)}")
        
        # Train models
        logger.info("\n4. Training Models")
        logger.info("-" * 40)
        
        models, performance = trainer.train_all_models(
            X_train_scaled, y_train,
            X_test_scaled, y_test
        )
        
        # Evaluate models
        logger.info("\n5. Model Performance")
        logger.info("-" * 40)
        
        for name, metrics in performance.items():
            logger.info(f"{name}:")
            logger.info(f"  R² Score: {metrics['r2']:.3f}")
            logger.info(f"  RMSE: {metrics['rmse']:.3f}")
            logger.info(f"  MSE: {metrics['mse']:.3f}")
            logger.info("")
        
        # Save models
        logger.info("6. Saving Models")
        logger.info("-" * 40)
        
        # Save each model
        for name, model in models.items():
            saver.save_model(model, f"{name}_model.pkl")
        
        # Save scaler and label encoders
        saver.save_scaler(preprocessor.scaler)
        saver.save_label_encoders(preprocessor.label_encoders)
        
        # Save metadata
        training_date = datetime.now().strftime("%Y-%m-%d")
        saver.save_metadata(
            performance,
            training_date,
            config.FEATURES,
            config.TARGET
        )
        
        # Verify all files are created
        logger.info("\n7. Verifying Model Files")
        logger.info("-" * 40)
        
        expected_files = [
            "xgboost_model.pkl",
            "random_forest_model.pkl", 
            "linear_regression_model.pkl",
            "scaler.pkl",
            "label_encoders.pkl",
            "model_metadata.json"
        ]
        
        missing_files = []
        for filename in expected_files:
            file_path = config.MODEL_PATH / filename
            if file_path.exists():
                logger.info(f"✅ {filename}")
            else:
                logger.error(f"❌ {filename}")
                missing_files.append(filename)
        
        if missing_files:
            raise FileNotFoundError(f"Missing model files: {', '.join(missing_files)}")
        
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info("\nModels saved to: models/ directory")
        logger.info("Application is now ready for production mode")
        
        return True
        
    except Exception as e:
        logger.error(f"\n❌ Training failed: {e}")
        logger.error(f"Stack trace: {sys.exc_info()[2]}")
        return False


# ============================================================================
# RUN TRAINING
# ============================================================================

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

"""
================================================================================
                    FABRIC CONSUMPTION FORECASTING SYSTEM
                         MODEL TRAINING SCRIPT
================================================================================

Trains machine learning models for fabric consumption prediction using
generated historical data. Saves trained models in the models/ directory
for production use.

Key Features:
- Trains XGBoost, Random Forest, LSTM and Linear Regression models
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
import warnings
warnings.filterwarnings('ignore')

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
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# XGBoost
import xgboost as xgb

# TensorFlow/Keras for LSTM
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("WARNING: TensorFlow not available. LSTM training will be skipped.")

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
    VALIDATION_SIZE = 0.2  # For LSTM validation
    
    # XGBoost Parameters
    XGB_PARAMS = {
        "objective": "reg:squarederror",
        "random_state": RANDOM_STATE,
        "n_estimators": 200,
        "max_depth": 8,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbosity": 0
    }
    
    # Random Forest Parameters
    RF_PARAMS = {
        "random_state": RANDOM_STATE,
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "bootstrap": True,
        "n_jobs": -1,
        "verbose": 0
    }
    
    # Linear Regression Parameters
    LR_PARAMS = {
        "fit_intercept": True
    }
    
    # LSTM Parameters
    LSTM_PARAMS = {
        "units_layer1": 64,
        "units_layer2": 32,
        "dense_units": 16,
        "dropout_rate": 0.2,
        "batch_size": 32,
        "epochs": 50,
        "validation_split": 0.2,
        "patience": 10  # Early stopping patience
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
    
    def scale_features(self, X: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """Scale numerical features using StandardScaler"""
        if fit or self.scaler is None:
            self.scaler = StandardScaler()
            scaled_features = self.scaler.fit_transform(X)
            logger.info("Scaler fitted and features scaled")
        else:
            scaled_features = self.scaler.transform(X)
            logger.info("Features scaled using existing scaler")
            
        return scaled_features


# ============================================================================
# MODEL TRAINING
# ============================================================================

class ModelTrainer:
    """Trains and evaluates all machine learning models"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.models = {}
        self.performance = {}
        self.training_history = {}
    
    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray) -> xgb.XGBRegressor:
        """Train XGBoost regression model"""
        logger.info("Training XGBoost model...")
        
        model = xgb.XGBRegressor(**self.config.XGB_PARAMS)
        model.fit(X_train, y_train)
        
        self.models["xgboost"] = model
        logger.info("✅ XGBoost training completed")
        
        return model
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray) -> RandomForestRegressor:
        """Train Random Forest regression model"""
        logger.info("Training Random Forest model...")
        
        model = RandomForestRegressor(**self.config.RF_PARAMS)
        model.fit(X_train, y_train)
        
        self.models["random_forest"] = model
        logger.info("✅ Random Forest training completed")
        
        return model
    
    def train_linear_regression(self, X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
        """Train Linear Regression model"""
        logger.info("Training Linear Regression model...")
        
        model = LinearRegression(**self.config.LR_PARAMS)
        model.fit(X_train, y_train)
        
        self.models["linear_regression"] = model
        logger.info("✅ Linear Regression training completed")
        
        return model
    
    def train_lstm(self, X_train: np.ndarray, y_train: np.ndarray, 
                   X_val: np.ndarray, y_val: np.ndarray) -> keras.Model:
        """Train LSTM neural network model"""
        
        if not TENSORFLOW_AVAILABLE:
            logger.warning("⚠️ TensorFlow not available. Skipping LSTM training.")
            return None
        
        logger.info("Training LSTM model...")
        
        # Reshape data for LSTM (samples, timesteps, features)
        # We treat each sample as a single timestep
        n_samples, n_features = X_train.shape
        X_train_lstm = X_train.reshape(n_samples, 1, n_features)
        X_val_lstm = X_val.reshape(X_val.shape[0], 1, n_features)
        
        logger.info(f"LSTM input shape: {X_train_lstm.shape}")
        
        # Build LSTM model
        model = keras.Sequential([
            # First LSTM layer
            layers.LSTM(
                self.config.LSTM_PARAMS["units_layer1"],
                activation='relu',
                input_shape=(1, n_features),
                return_sequences=True,
                name='lstm_layer_1'
            ),
            layers.Dropout(self.config.LSTM_PARAMS["dropout_rate"], name='dropout_1'),
            
            # Second LSTM layer
            layers.LSTM(
                self.config.LSTM_PARAMS["units_layer2"],
                activation='relu',
                name='lstm_layer_2'
            ),
            layers.Dropout(self.config.LSTM_PARAMS["dropout_rate"], name='dropout_2'),
            
            # Dense layers
            layers.Dense(
                self.config.LSTM_PARAMS["dense_units"],
                activation='relu',
                name='dense_1'
            ),
            layers.Dense(1, name='output')  # Single output for regression
        ])
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        logger.info("LSTM Model Architecture:")
        model.summary(print_fn=logger.info)
        
        # Callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.config.LSTM_PARAMS["patience"],
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )
        
        # Train model
        logger.info("Starting LSTM training...")
        history = model.fit(
            X_train_lstm, y_train,
            validation_data=(X_val_lstm, y_val),
            epochs=self.config.LSTM_PARAMS["epochs"],
            batch_size=self.config.LSTM_PARAMS["batch_size"],
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        self.models["lstm"] = model
        self.training_history["lstm"] = history.history
        
        logger.info(f"✅ LSTM training completed")
        logger.info(f"   Final training loss: {history.history['loss'][-1]:.4f}")
        logger.info(f"   Final validation loss: {history.history['val_loss'][-1]:.4f}")
        
        return model
    
    def evaluate_model(self, model: object, X_test: np.ndarray, y_test: np.ndarray, 
                      model_name: str = None) -> dict:
        """Evaluate model performance"""
        
        # Handle LSTM prediction differently (needs reshaping)
        if model_name == "lstm" and TENSORFLOW_AVAILABLE:
            X_test_reshaped = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
            y_pred = model.predict(X_test_reshaped, verbose=0).flatten()
        else:
            y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        return {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "mape": float(mape)
        }
    
    def train_ensemble(self, X_val: np.ndarray, y_val: np.ndarray) -> dict:
        """
        Build a weighted-average ensemble from all trained base models.

        Weights are computed as softmax-normalised R² scores on the validation
        set, so better models contribute more to the final prediction.
        LSTM is included only when TensorFlow is available.

        Returns:
            dict: {'weights': {model_name: weight}, 'model_names': [...]}
                  Saved as ensemble_model.pkl so ModelManager can apply it at
                  inference time without storing model references in the file.
        """
        if not self.models:
            raise ValueError("No base models trained yet. Call train_all_models first.")

        logger.info("Training Ensemble (weighted average of base models)...")

        val_r2 = {}
        for name, model in self.models.items():
            metrics = self.evaluate_model(model, X_val, y_val, name)
            # Clamp R² to [0, 1] so a poor model gets zero weight
            val_r2[name] = max(0.0, metrics["r2"])
            logger.info(f"  Validation R² for {name}: {val_r2[name]:.4f}")

        total_r2 = sum(val_r2.values())
        if total_r2 == 0:
            # Fallback: equal weights
            weights = {name: 1.0 / len(val_r2) for name in val_r2}
        else:
            weights = {name: r2 / total_r2 for name, r2 in val_r2.items()}

        ensemble_spec = {
            "weights": weights,
            "model_names": list(weights.keys()),
            "weighted_by": "validation_r2"
        }

        self.models["ensemble"] = ensemble_spec
        logger.info("Ensemble weights:")
        for name, w in weights.items():
            logger.info(f"  {name}: {w:.4f} ({w*100:.1f}%)")
        logger.info("✅ Ensemble specification created")
        return ensemble_spec

    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                        X_val: np.ndarray, y_val: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray):
        """Train all models and evaluate performance"""
        
        logger.info("\n" + "="*60)
        logger.info("TRAINING ALL MODELS")
        logger.info("="*60)
        
        # Train Linear Regression
        logger.info("\n[1/5] Linear Regression")
        logger.info("-"*40)
        self.train_linear_regression(X_train, y_train)
        
        # Train Random Forest
        logger.info("\n[2/5] Random Forest")
        logger.info("-"*40)
        self.train_random_forest(X_train, y_train)
        
        # Train XGBoost
        logger.info("\n[3/5] XGBoost")
        logger.info("-"*40)
        self.train_xgboost(X_train, y_train)
        
        # Train LSTM (if TensorFlow available)
        logger.info("\n[4/5] LSTM Neural Network")
        logger.info("-"*40)
        if TENSORFLOW_AVAILABLE:
            self.train_lstm(X_train, y_train, X_val, y_val)
        else:
            logger.warning("⚠️ LSTM training skipped (TensorFlow not available)")

        # Build Ensemble from base models
        logger.info("\n[5/5] Ensemble (Weighted Average)")
        logger.info("-"*40)
        self.train_ensemble(X_val, y_val)

        # Evaluate all non-ensemble models first, then ensemble
        logger.info("\n" + "="*60)
        logger.info("EVALUATING ALL MODELS")
        logger.info("="*60)
        
        base_model_names = [n for n in self.models if n != "ensemble"]
        for name in base_model_names:
            model = self.models[name]
            logger.info(f"\nEvaluating {name}...")
            self.performance[name] = self.evaluate_model(model, X_test, y_test, name)
            
            metrics = self.performance[name]
            logger.info(f"  RMSE: {metrics['rmse']:.3f}")
            logger.info(f"  MAE:  {metrics['mae']:.3f}")
            logger.info(f"  R²:   {metrics['r2']:.4f}")
            logger.info(f"  MAPE: {metrics['mape']:.2f}%")

        # Evaluate ensemble by computing weighted prediction on test set
        logger.info("\nEvaluating ensemble...")
        ensemble_spec = self.models["ensemble"]
        y_pred_ensemble = np.zeros(len(y_test))
        for name in ensemble_spec["model_names"]:
            model = self.models[name]
            if name == "lstm" and TENSORFLOW_AVAILABLE:
                X_reshaped = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
                preds = model.predict(X_reshaped, verbose=0).flatten()
            else:
                preds = model.predict(X_test)
            y_pred_ensemble += ensemble_spec["weights"][name] * preds

        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        mse = mean_squared_error(y_test, y_pred_ensemble)
        self.performance["ensemble"] = {
            "mse": float(mse),
            "rmse": float(np.sqrt(mse)),
            "mae": float(mean_absolute_error(y_test, y_pred_ensemble)),
            "r2": float(r2_score(y_test, y_pred_ensemble)),
            "mape": float(np.mean(np.abs((y_test - y_pred_ensemble) / y_test)) * 100)
        }
        m = self.performance["ensemble"]
        logger.info(f"  RMSE: {m['rmse']:.3f}")
        logger.info(f"  MAE:  {m['mae']:.3f}")
        logger.info(f"  R²:   {m['r2']:.4f}")
        logger.info(f"  MAPE: {m['mape']:.2f}%")
        
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
            logger.info(f"✅ Model saved to {file_path}")
            return str(file_path)
        except Exception as e:
            logger.error(f"❌ Failed to save model: {e}")
            raise
    
    def save_keras_model(self, model: keras.Model, filename: str) -> str:
        """Save Keras/LSTM model"""
        try:
            file_path = self.config.MODEL_PATH / filename
            model.save(file_path)
            logger.info(f"✅ Keras model saved to {file_path}")
            return str(file_path)
        except Exception as e:
            logger.error(f"❌ Failed to save Keras model: {e}")
            raise
    
    def save_scaler(self, scaler: object) -> str:
        """Save feature scaler"""
        return self.save_model(scaler, "scaler.pkl")
    
    def save_label_encoders(self, encoders: dict) -> str:
        """Save label encoders"""
        return self.save_model(encoders, "label_encoders.pkl")
    
    def save_metadata(self, performance: dict, training_date: str, 
                     features: list, target: str, training_history: dict = None,
                     ensemble_spec: dict = None) -> str:
        """Save model metadata"""
        metadata = {
            "version": "2.0.0",
            "training_date": training_date,
            "unit": "meters",
            "models": {},
            "feature_names": features,
            "target": target,
            "tensorflow_available": TENSORFLOW_AVAILABLE
        }
        
        for name, metrics in performance.items():
            metadata["models"][name] = {
                "file": (f"{name}_model.h5" if name == "lstm"
                         else f"{name}_model.pkl"),
                "type": self._get_model_type(name),
                "features": len(features),
                "rmse": round(metrics["rmse"], 3),
                "mae": round(metrics["mae"], 3),
                "r2": round(metrics["r2"], 4),
                "mape": round(metrics["mape"], 2)
            }
            
            # Add LSTM training history if available
            if name == "lstm" and training_history and "lstm" in training_history:
                metadata["models"][name]["training_history"] = {
                    "final_train_loss": float(training_history["lstm"]["loss"][-1]),
                    "final_val_loss": float(training_history["lstm"]["val_loss"][-1]),
                    "epochs_trained": len(training_history["lstm"]["loss"])
                }

            # Embed ensemble weights in metadata for transparency
            if name == "ensemble" and ensemble_spec:
                metadata["models"][name]["weights"] = ensemble_spec.get("weights", {})
                metadata["models"][name]["weighted_by"] = ensemble_spec.get("weighted_by", "")
        
        metadata_path = self.config.MODEL_PATH / "model_metadata.json"
        
        try:
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"✅ Metadata saved to {metadata_path}")
            return str(metadata_path)
        except Exception as e:
            logger.error(f"❌ Failed to save metadata: {e}")
            raise
    
    def _get_model_type(self, model_name: str) -> str:
        """Get model type for metadata"""
        type_map = {
            "xgboost": "xgboost.XGBRegressor",
            "random_forest": "sklearn.ensemble.RandomForestRegressor",
            "linear_regression": "sklearn.linear_model.LinearRegression",
            "lstm": "tensorflow.keras.Sequential",
            "ensemble": "WeightedAverageEnsemble"
        }
        return type_map.get(model_name, "unknown")


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main():
    """Main training pipeline"""
    logger.info("=" * 80)
    logger.info("FABRIC CONSUMPTION FORECASTING SYSTEM - COMPLETE MODEL TRAINING")
    logger.info("=" * 80)
    logger.info(f"TensorFlow Available: {TENSORFLOW_AVAILABLE}")
    logger.info(f"Training 5 models: Linear Regression, Random Forest, XGBoost, LSTM, Ensemble")
    logger.info("=" * 80)
    
    try:
        # Initialize configuration
        config = TrainingConfig()
        
        # Initialize components
        preprocessor = DataPreprocessor(config)
        trainer = ModelTrainer(config)
        saver = ModelSaver(config)
        
        # Load and preprocess data
        logger.info("\n📥 STEP 1: Loading and Preprocessing Data")
        logger.info("-" * 80)
        
        df = preprocessor.load_data(config.TRAINING_DATA)
        df = preprocessor.preprocess_data(df)
        
        logger.info(f"Available features: {config.FEATURES}")
        logger.info(f"Target variable: {config.TARGET}")
        
        # Split into features and target
        X = df[config.FEATURES]
        y = df[config.TARGET]
        
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Target shape: {y.shape}")
        
        # Split into training, validation, and testing sets
        logger.info("\n✂️ STEP 2: Splitting Data (Train/Val/Test)")
        logger.info("-" * 80)
        
        # First split: Train+Val vs Test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE
        )
        
        # Second split: Train vs Val
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=config.VALIDATION_SIZE,
            random_state=config.RANDOM_STATE
        )
        
        logger.info(f"Training samples: {X_train.shape[0]} ({X_train.shape[0]/len(X)*100:.1f}%)")
        logger.info(f"Validation samples: {X_val.shape[0]} ({X_val.shape[0]/len(X)*100:.1f}%)")
        logger.info(f"Testing samples: {X_test.shape[0]} ({X_test.shape[0]/len(X)*100:.1f}%)")
        
        # Scale features
        logger.info("\n📊 STEP 3: Scaling Features")
        logger.info("-" * 80)
        
        X_train_scaled = preprocessor.scale_features(X_train, fit=True)
        X_val_scaled = preprocessor.scale_features(X_val, fit=False)
        X_test_scaled = preprocessor.scale_features(X_test, fit=False)
        
        logger.info(f"Feature scaling completed")
        logger.info(f"Scaled data mean: ~{X_train_scaled.mean():.6f}")
        logger.info(f"Scaled data std: ~{X_train_scaled.std():.6f}")
        
        # Train models
        logger.info("\n🤖 STEP 4: Training All Models")
        logger.info("-" * 80)
        
        models, performance = trainer.train_all_models(
            X_train_scaled, y_train,
            X_val_scaled, y_val,
            X_test_scaled, y_test
        )
        
        # Display performance summary
        logger.info("\n📈 STEP 5: Performance Summary")
        logger.info("-" * 80)
        
        # Create comparison table
        print("\n┌─────────────────────┬──────────┬─────────┬─────────┬──────────┐")
        print("│ Model               │   RMSE   │   MAE   │    R²   │   MAPE   │")
        print("├─────────────────────┼──────────┼─────────┼─────────┼──────────┤")
        
        for name, metrics in performance.items():
            model_display = name.replace("_", " ").title()
            print(f"│ {model_display:<19} │ {metrics['rmse']:>8.2f} │ {metrics['mae']:>7.2f} │ {metrics['r2']:>7.4f} │ {metrics['mape']:>7.2f}% │")
        
        print("└─────────────────────┴──────────┴─────────┴─────────┴──────────┘")
        
        # Find best model
        best_model = min(performance.items(), key=lambda x: x[1]['rmse'])
        logger.info(f"\n🏆 Best Model: {best_model[0].upper()} (RMSE: {best_model[1]['rmse']:.3f})")
        
        # Save models
        logger.info("\n💾 STEP 6: Saving Models")
        logger.info("-" * 80)
        
        # Save each model
        for name, model in models.items():
            if name == "lstm" and TENSORFLOW_AVAILABLE:
                saver.save_keras_model(model, f"{name}_model.h5")
            elif name == "ensemble":
                # Ensemble spec is a plain dict — save with joblib
                saver.save_model(model, "ensemble_model.pkl")
            else:
                saver.save_model(model, f"{name}_model.pkl")
        
        # Save scaler and label encoders
        saver.save_scaler(preprocessor.scaler)
        saver.save_label_encoders(preprocessor.label_encoders)
        
        # Save metadata (include ensemble_spec for weight transparency)
        ensemble_spec = models.get("ensemble")
        training_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        saver.save_metadata(
            performance,
            training_date,
            config.FEATURES,
            config.TARGET,
            trainer.training_history,
            ensemble_spec=ensemble_spec
        )
        
        # Verify all files are created
        logger.info("\n✅ STEP 7: Verifying Model Files")
        logger.info("-" * 80)
        
        expected_files = [
            "xgboost_model.pkl",
            "random_forest_model.pkl", 
            "linear_regression_model.pkl",
            "ensemble_model.pkl",
            "scaler.pkl",
            "label_encoders.pkl",
            "model_metadata.json"
        ]
        
        if TENSORFLOW_AVAILABLE:
            expected_files.append("lstm_model.h5")
        
        all_files_exist = True
        for filename in expected_files:
            file_path = config.MODEL_PATH / filename
            if file_path.exists():
                file_size = file_path.stat().st_size / 1024  # KB
                logger.info(f"✅ {filename:<30} ({file_size:>8.1f} KB)")
            else:
                logger.error(f"❌ {filename}")
                all_files_exist = False
        
        if not all_files_exist:
            raise FileNotFoundError("Some model files are missing!")
        
        # Success message
        logger.info("\n" + "=" * 80)
        logger.info("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"\nModels trained: {len(models)} (including Ensemble)")
        logger.info(f"Models saved to: {config.MODEL_PATH.absolute()}/")
        logger.info(f"Best model: {best_model[0].upper()} (RMSE: {best_model[1]['rmse']:.3f})")
        logger.info("\n✅ Application is now ready for PRODUCTION mode")
        logger.info("✅ All models available for predictions")
        
        if not TENSORFLOW_AVAILABLE:
            logger.warning("\n⚠️ Note: LSTM model not trained (TensorFlow not installed)")
            logger.warning("   Install TensorFlow to enable LSTM: pip install tensorflow")
        
        return True
        
    except Exception as e:
        logger.error(f"\n❌ TRAINING FAILED: {e}")
        import traceback
        logger.error(f"Stack trace:\n{traceback.format_exc()}")
        return False


# ============================================================================
# RUN TRAINING
# ============================================================================

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from datetime import datetime
import warnings
from pathlib import Path
import json

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class FootballFoulPredictor:
    """Advanced ML model for predicting football fouls and cards."""
    
    def __init__(self, model_type: str = 'xgboost'):
        self.model_type = model_type
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.training_history = {}
        self.model_metadata = {}
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize different model types."""
        
        # XGBoost models (primary)
        self.models['xgboost'] = {
            'fouls': xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            ),
            'yellow_cards': xgb.XGBRegressor(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            ),
            'red_cards': xgb.XGBRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.15,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                n_jobs=-1
            )
        }
        
        # Random Forest models (ensemble)
        self.models['random_forest'] = {
            'fouls': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'yellow_cards': RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'red_cards': RandomForestRegressor(
                n_estimators=80,
                max_depth=6,
                min_samples_split=10,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1
            )
        }
        
        # Gradient Boosting models
        self.models['gradient_boosting'] = {
            'fouls': GradientBoostingRegressor(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            ),
            'yellow_cards': GradientBoostingRegressor(
                n_estimators=120,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            ),
            'red_cards': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.15,
                subsample=0.9,
                random_state=42
            )
        }
        
        # Linear models (baseline)
        self.models['linear'] = {
            'fouls': Ridge(alpha=1.0),
            'yellow_cards': Ridge(alpha=1.5),
            'red_cards': Lasso(alpha=0.1)
        }
    
    def train_model(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series, 
        target: str,
        model_type: str = None,
        hyperparameter_tuning: bool = False
    ) -> Dict[str, Any]:
        """Train a model for specific target."""
        
        if model_type is None:
            model_type = self.model_type
        
        logger.info(f"Training {model_type} model for {target}...")
        
        # Get model
        if model_type not in self.models or target not in self.models[model_type]:
            raise ValueError(f"Model {model_type} for target {target} not available")
        
        model = self.models[model_type][target]
        
        # Hyperparameter tuning if requested
        if hyperparameter_tuning:
            model = self._tune_hyperparameters(model, X_train, y_train, model_type, target)
        
        # Train model
        start_time = datetime.now()
        model.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Store trained model
        if model_type not in self.models:
            self.models[model_type] = {}
        self.models[model_type][target] = model
        
        # Calculate feature importance
        if hasattr(model, 'feature_importances_'):
            importance_dict = dict(zip(X_train.columns, model.feature_importances_))
            self.feature_importance[f"{model_type}_{target}"] = importance_dict
        
        # Cross-validation
        cv_scores = cross_val_score(
            model, X_train, y_train, 
            cv=5, scoring='neg_mean_squared_error', n_jobs=-1
        )
        
        # Training metrics
        y_pred_train = model.predict(X_train)
        train_metrics = self._calculate_metrics(y_train, y_pred_train)
        
        # Store training history
        training_info = {
            'model_type': model_type,
            'target': target,
            'training_time': training_time,
            'cv_scores': cv_scores.tolist(),
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'train_metrics': train_metrics,
            'feature_count': len(X_train.columns),
            'sample_count': len(X_train),
            'timestamp': datetime.now().isoformat()
        }
        
        if model_type not in self.training_history:
            self.training_history[model_type] = {}
        self.training_history[model_type][target] = training_info
        
        logger.info(
            f"Model trained successfully. CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})"
        )
        
        return training_info
    
    def _tune_hyperparameters(
        self, 
        model, 
        X_train: pd.DataFrame, 
        y_train: pd.Series, 
        model_type: str, 
        target: str
    ):
        """Perform hyperparameter tuning."""
        
        logger.info(f"Tuning hyperparameters for {model_type} {target} model...")
        
        # Define parameter grids
        param_grids = {
            'xgboost': {
                'n_estimators': [100, 200, 300],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1, 0.15],
                'subsample': [0.8, 0.9],
                'colsample_bytree': [0.8, 0.9]
            },
            'random_forest': {
                'n_estimators': [50, 100, 150],
                'max_depth': [6, 8, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'gradient_boosting': {
                'n_estimators': [100, 150, 200],
                'max_depth': [3, 4, 5],
                'learning_rate': [0.05, 0.1, 0.15],
                'subsample': [0.8, 0.9]
            }
        }
        
        if model_type in param_grids:
            # Use TimeSeriesSplit for temporal data
            cv = TimeSeriesSplit(n_splits=3)
            
            grid_search = GridSearchCV(
                model,
                param_grids[model_type],
                cv=cv,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            
            logger.info(f"Best parameters: {grid_search.best_params_}")
            logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
            
            return grid_search.best_estimator_
        
        return model
    
    def predict(
        self, 
        X: pd.DataFrame, 
        target: str, 
        model_type: str = None,
        return_confidence: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Make predictions for a specific target."""
        
        if model_type is None:
            model_type = self.model_type
        
        if model_type not in self.models or target not in self.models[model_type]:
            raise ValueError(f"Model {model_type} for target {target} not trained")
        
        model = self.models[model_type][target]
        predictions = model.predict(X)
        
        # Ensure non-negative predictions for counts
        predictions = np.maximum(predictions, 0)
        
        if return_confidence and hasattr(model, 'predict_proba'):
            # For models that support confidence intervals
            confidence = model.predict_proba(X)
            return predictions, confidence
        
        return predictions
    
    def predict_all_targets(
        self, 
        X: pd.DataFrame, 
        model_type: str = None
    ) -> Dict[str, np.ndarray]:
        """Predict all targets (fouls, yellow_cards, red_cards)."""
        
        if model_type is None:
            model_type = self.model_type
        
        predictions = {}
        targets = ['fouls', 'yellow_cards', 'red_cards']
        
        for target in targets:
            try:
                predictions[target] = self.predict(X, target, model_type)
            except ValueError as e:
                logger.warning(f"Could not predict {target}: {e}")
                predictions[target] = np.zeros(len(X))
        
        # Calculate derived metrics
        predictions['total_cards'] = predictions['yellow_cards'] + predictions['red_cards']
        predictions['disciplinary_points'] = predictions['yellow_cards'] + predictions['red_cards'] * 2
        
        return predictions
    
    def evaluate_model(
        self, 
        X_test: pd.DataFrame, 
        y_test: pd.Series, 
        target: str,
        model_type: str = None
    ) -> Dict[str, float]:
        """Evaluate model performance."""
        
        if model_type is None:
            model_type = self.model_type
        
        # Make predictions
        y_pred = self.predict(X_test, target, model_type)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred)
        
        # Add model-specific metrics
        metrics['model_type'] = model_type
        metrics['target'] = target
        metrics['test_samples'] = len(X_test)
        
        logger.info(f"Model evaluation for {model_type} {target}:")
        logger.info(f"  RMSE: {metrics['rmse']:.4f}")
        logger.info(f"  MAE: {metrics['mae']:.4f}")
        logger.info(f"  R²: {metrics['r2']:.4f}")
        
        return metrics
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics."""
        
        return {
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'r2': float(r2_score(y_true, y_pred)),
            'mse': float(mean_squared_error(y_true, y_pred)),
            'mean_actual': float(y_true.mean()),
            'mean_predicted': float(y_pred.mean()),
            'std_actual': float(y_true.std()),
            'std_predicted': float(y_pred.std())
        }
    
    def get_feature_importance(
        self, 
        target: str, 
        model_type: str = None, 
        top_k: int = 20
    ) -> Dict[str, float]:
        """Get feature importance for a specific model."""
        
        if model_type is None:
            model_type = self.model_type
        
        key = f"{model_type}_{target}"
        
        if key not in self.feature_importance:
            return {}
        
        # Sort by importance and return top k
        importance = self.feature_importance[key]
        sorted_importance = dict(
            sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_k]
        )
        
        return sorted_importance
    
    def ensemble_predict(
        self, 
        X: pd.DataFrame, 
        target: str, 
        models: List[str] = None,
        weights: List[float] = None
    ) -> np.ndarray:
        """Make ensemble predictions using multiple models."""
        
        if models is None:
            models = ['xgboost', 'random_forest', 'gradient_boosting']
        
        if weights is None:
            weights = [0.5, 0.3, 0.2]  # XGBoost gets highest weight
        
        if len(models) != len(weights):
            raise ValueError("Number of models and weights must match")
        
        predictions = []
        actual_weights = []
        
        for model_type, weight in zip(models, weights):
            try:
                pred = self.predict(X, target, model_type)
                predictions.append(pred)
                actual_weights.append(weight)
            except ValueError:
                logger.warning(f"Model {model_type} not available for ensemble")
        
        if not predictions:
            raise ValueError("No models available for ensemble prediction")
        
        # Normalize weights
        actual_weights = np.array(actual_weights)
        actual_weights = actual_weights / actual_weights.sum()
        
        # Weighted average
        ensemble_pred = np.average(predictions, axis=0, weights=actual_weights)
        
        return ensemble_pred
    
    def save_models(self, directory: str):
        """Save all trained models and metadata."""
        
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        # Save models
        for model_type, targets in self.models.items():
            for target, model in targets.items():
                if hasattr(model, 'predict'):  # Check if model is trained
                    model_path = directory / f"{model_type}_{target}_model.joblib"
                    joblib.dump(model, model_path)
        
        # Save scalers
        if self.scalers:
            scalers_path = directory / "scalers.joblib"
            joblib.dump(self.scalers, scalers_path)
        
        # Save metadata
        metadata = {
            'feature_importance': self.feature_importance,
            'training_history': self.training_history,
            'model_metadata': self.model_metadata,
            'model_type': self.model_type
        }
        
        metadata_path = directory / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Models saved to {directory}")
    
    def load_models(self, directory: str):
        """Load trained models and metadata."""
        
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Model directory {directory} not found")
        
        # Load models
        for model_file in directory.glob("*_model.joblib"):
            parts = model_file.stem.split('_')
            if len(parts) >= 3:
                model_type = parts[0]
                target = '_'.join(parts[1:-1])  # Handle multi-word targets
                
                if model_type not in self.models:
                    self.models[model_type] = {}
                
                self.models[model_type][target] = joblib.load(model_file)
        
        # Load scalers
        scalers_path = directory / "scalers.joblib"
        if scalers_path.exists():
            self.scalers = joblib.load(scalers_path)
        
        # Load metadata
        metadata_path = directory / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.feature_importance = metadata.get('feature_importance', {})
            self.training_history = metadata.get('training_history', {})
            self.model_metadata = metadata.get('model_metadata', {})
            self.model_type = metadata.get('model_type', 'xgboost')
        
        logger.info(f"Models loaded from {directory}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of all trained models."""
        
        summary = {
            'available_models': {},
            'training_history': self.training_history,
            'feature_importance_available': list(self.feature_importance.keys())
        }
        
        for model_type, targets in self.models.items():
            summary['available_models'][model_type] = []
            for target, model in targets.items():
                if hasattr(model, 'predict'):
                    summary['available_models'][model_type].append(target)
        
        return summary

class FootballRiskAssessment:
    """Risk assessment for player disciplinary actions."""
    
    def __init__(self, predictor: FootballFoulPredictor):
        self.predictor = predictor
    
    def assess_player_risk(
        self, 
        player_data: pd.DataFrame,
        risk_thresholds: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """Assess disciplinary risk for a player."""
        
        if risk_thresholds is None:
            risk_thresholds = {
                'fouls': {'low': 1.0, 'medium': 2.0, 'high': 3.0},
                'yellow_cards': {'low': 0.2, 'medium': 0.5, 'high': 1.0},
                'red_cards': {'low': 0.05, 'medium': 0.1, 'high': 0.2}
            }
        
        # Get predictions
        predictions = self.predictor.predict_all_targets(player_data)
        
        # Calculate risk levels
        risk_assessment = {}
        
        for target in ['fouls', 'yellow_cards', 'red_cards']:
            pred_value = predictions[target][0] if len(predictions[target]) > 0 else 0
            thresholds = risk_thresholds[target]
            
            if pred_value <= thresholds['low']:
                risk_level = 'low'
            elif pred_value <= thresholds['medium']:
                risk_level = 'medium'
            elif pred_value <= thresholds['high']:
                risk_level = 'high'
            else:
                risk_level = 'very_high'
            
            risk_assessment[target] = {
                'predicted_value': float(pred_value),
                'risk_level': risk_level,
                'percentile': self._calculate_percentile(pred_value, target)
            }
        
        # Overall risk score
        overall_score = (
            predictions['fouls'][0] * 0.4 +
            predictions['yellow_cards'][0] * 0.4 +
            predictions['red_cards'][0] * 0.2
        )
        
        risk_assessment['overall'] = {
            'risk_score': float(overall_score),
            'risk_level': self._get_overall_risk_level(overall_score)
        }
        
        return risk_assessment
    
    def _calculate_percentile(self, value: float, target: str) -> float:
        """Calculate percentile ranking for a prediction value."""
        # This would ideally use historical data distribution
        # For now, using rough estimates based on typical football statistics
        
        percentile_maps = {
            'fouls': {0.5: 25, 1.0: 50, 1.5: 75, 2.0: 85, 3.0: 95},
            'yellow_cards': {0.1: 25, 0.3: 50, 0.5: 75, 0.8: 85, 1.2: 95},
            'red_cards': {0.02: 25, 0.05: 50, 0.08: 75, 0.12: 85, 0.2: 95}
        }
        
        if target not in percentile_maps:
            return 50.0
        
        percentile_map = percentile_maps[target]
        
        for threshold, percentile in sorted(percentile_map.items()):
            if value <= threshold:
                return float(percentile)
        
        return 99.0
    
    def _get_overall_risk_level(self, score: float) -> str:
        """Determine overall risk level from composite score."""
        
        if score <= 1.0:
            return 'low'
        elif score <= 2.0:
            return 'medium'
        elif score <= 3.0:
            return 'high'
        else:
            return 'very_high'
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json
import joblib
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

from .feature_engineering import FeatureEngineer
from .models import FootballFoulPredictor, FootballRiskAssessment
from .clustering import PlayerProfileClustering, AnomalyDetection
from ..database.connection import get_db_connection
from ..core.config import settings
from .mlflow_config import get_mlflow_manager

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class MLPipeline:
    """Complete ML pipeline for football fouls analytics."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        
        # Initialize components
        self.feature_engineer = FeatureEngineer()
        self.predictor = FootballFoulPredictor()
        self.clustering = PlayerProfileClustering()
        self.anomaly_detector = AnomalyDetection()
        self.risk_assessor = FootballRiskAssessment(self.predictor)
        
        # Pipeline state
        self.is_trained = False
        self.training_history = {}
        self.model_performance = {}
        self.feature_importance = {}
        
        # Data cache
        self.cached_data = {}
        self.last_update = None
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default pipeline configuration."""
        
        return {
            'data': {
                'min_games_per_player': 5,
                'test_size': 0.2,
                'validation_size': 0.1,
                'time_split': True
            },
            'features': {
                'lookback_games': 10,
                'include_opponent_features': True,
                'include_temporal_features': True,
                'include_team_features': True
            },
            'models': {
                'primary_algorithm': 'xgboost',
                'ensemble_models': ['xgboost', 'random_forest', 'gradient_boosting'],
                'hyperparameter_tuning': False,
                'cross_validation_folds': 5
            },
            'clustering': {
                'n_clusters': 5,
                'algorithm': 'kmeans',
                'find_optimal_clusters': True
            },
            'anomaly_detection': {
                'contamination': 0.1,
                'algorithm': 'isolation_forest'
            },
            'output': {
                'save_models': True,
                'save_results': True,
                'generate_reports': True
            }
        }
    
    def load_data(self, force_refresh: bool = False) -> pd.DataFrame:
        """Load and cache data from database."""
        
        # Check if we need to refresh data
        if (not force_refresh and 
            'raw_data' in self.cached_data and 
            self.last_update and 
            (datetime.now() - self.last_update).hours < 1):
            
            logger.info("Using cached data")
            return self.cached_data['raw_data']
        
        logger.info("Loading data from database...")
        
        try:
            with get_db_connection() as conn:
                # Load player statistics
                query = """
                SELECT 
                    ps.*,
                    p.name as player_name,
                    p.position,
                    p.nationality,
                    t.name as team_name,
                    m.date,
                    m.home_team_id,
                    m.away_team_id,
                    m.competition
                FROM player_stats ps
                JOIN players p ON ps.player_id = p.id
                JOIN teams t ON ps.team_id = t.id
                JOIN matches m ON ps.match_id = m.id
                WHERE ps.minutes_played > 0
                ORDER BY ps.player_id, m.date
                """
                
                df = pd.read_sql_query(query, conn)
                
                # Convert date column
                df['date'] = pd.to_datetime(df['date'])
                
                # Filter players with minimum games
                min_games = self.config['data']['min_games_per_player']
                player_game_counts = df.groupby('player_id').size()
                valid_players = player_game_counts[player_game_counts >= min_games].index
                df = df[df['player_id'].isin(valid_players)]
                
                logger.info(f"Loaded {len(df)} records for {len(valid_players)} players")
                
                # Cache data
                self.cached_data['raw_data'] = df
                self.last_update = datetime.now()
                
                return df
                
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare features for training and prediction."""
        
        logger.info("Preparing features...")
        
        # Engineer features
        features_df = self.feature_engineer.create_features(
            df,
            lookback_games=self.config['features']['lookback_games'],
            include_opponent_features=self.config['features']['include_opponent_features'],
            include_temporal_features=self.config['features']['include_temporal_features'],
            include_team_features=self.config['features']['include_team_features']
        )
        
        # Prepare target variables
        targets_df = df[['player_id', 'match_id', 'fouls', 'yellow_cards', 'red_cards']].copy()
        
        # Align features and targets
        common_indices = features_df.index.intersection(targets_df.index)
        features_df = features_df.loc[common_indices]
        targets_df = targets_df.loc[common_indices]
        
        logger.info(f"Prepared {len(features_df.columns)} features for {len(features_df)} samples")
        
        # Cache prepared data
        self.cached_data['features'] = features_df
        self.cached_data['targets'] = targets_df
        
        return features_df, targets_df
    
    def split_data(
        self, 
        features_df: pd.DataFrame, 
        targets_df: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """Split data into train, validation, and test sets."""
        
        logger.info("Splitting data...")
        
        if self.config['data']['time_split']:
            # Time-based split to avoid data leakage
            df_combined = features_df.join(targets_df[['date']], how='inner')
            df_combined = df_combined.sort_values('date')
            
            n_samples = len(df_combined)
            test_size = self.config['data']['test_size']
            val_size = self.config['data']['validation_size']
            
            # Calculate split indices
            test_start = int(n_samples * (1 - test_size))
            val_start = int(n_samples * (1 - test_size - val_size))
            
            # Split indices
            train_indices = df_combined.index[:val_start]
            val_indices = df_combined.index[val_start:test_start]
            test_indices = df_combined.index[test_start:]
            
        else:
            # Random split
            train_val_indices, test_indices = train_test_split(
                features_df.index,
                test_size=self.config['data']['test_size'],
                random_state=42
            )
            
            train_indices, val_indices = train_test_split(
                train_val_indices,
                test_size=self.config['data']['validation_size'] / (1 - self.config['data']['test_size']),
                random_state=42
            )
        
        # Create splits
        splits = {
            'X_train': features_df.loc[train_indices],
            'X_val': features_df.loc[val_indices],
            'X_test': features_df.loc[test_indices],
            'y_train': targets_df.loc[train_indices],
            'y_val': targets_df.loc[val_indices],
            'y_test': targets_df.loc[test_indices]
        }
        
        logger.info(
            f"Data split - Train: {len(train_indices)}, "
            f"Val: {len(val_indices)}, Test: {len(test_indices)}"
        )
        
        return splits
    
    def train_models(self, splits: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Train all prediction models."""
        
        logger.info("Training prediction models...")
        
        X_train = splits['X_train']
        X_val = splits['X_val']
        y_train = splits['y_train']
        y_val = splits['y_val']
        
        targets = ['fouls', 'yellow_cards', 'red_cards']
        algorithms = self.config['models']['ensemble_models']
        
        training_results = {}
        
        for target in targets:
            logger.info(f"Training models for {target}...")
            
            target_results = {}
            
            for algorithm in algorithms:
                try:
                    # Train model
                    result = self.predictor.train_model(
                        X_train, 
                        y_train[target], 
                        target,
                        model_type=algorithm,
                        hyperparameter_tuning=self.config['models']['hyperparameter_tuning']
                    )
                    
                    # Validate model
                    val_metrics = self.predictor.evaluate_model(
                        X_val, y_val[target], target, algorithm
                    )
                    
                    result['validation_metrics'] = val_metrics
                    target_results[algorithm] = result
                    
                    # Log model to MLflow
                    try:
                        mlflow_manager = get_mlflow_manager()
                        
                        # Get feature importance
                        feature_importance = self.predictor.get_feature_importance(
                            target, algorithm, top_k=20
                        )
                        
                        # Log model training
                        mlflow_run_id = mlflow_manager.log_model_training(
                            model=self.predictor.models.get(f"{target}_{algorithm}"),
                            model_name=algorithm,
                            target=target,
                            X_train=X_train,
                            y_train=y_train[target],
                            X_val=X_val,
                            y_val=y_val[target],
                            training_metrics=result.get('training_metrics', {}),
                            validation_metrics=val_metrics,
                            feature_importance=feature_importance,
                            model_params=result.get('model_params', {}),
                            tags={
                                'target_variable': target,
                                'model_type': algorithm,
                                'training_date': datetime.now().strftime('%Y-%m-%d'),
                                'hyperparameter_tuning': self.config['models']['hyperparameter_tuning']
                            }
                        )
                        
                        result['mlflow_run_id'] = mlflow_run_id
                        logger.info(f"Model {algorithm} for {target} logged to MLflow: {mlflow_run_id}")
                        
                    except Exception as mlflow_error:
                        logger.warning(f"Failed to log {algorithm} model to MLflow: {mlflow_error}")
                    
                    logger.info(
                        f"{algorithm} {target} - Val RMSE: {val_metrics['rmse']:.4f}, "
                        f"R²: {val_metrics['r2']:.4f}"
                    )
                    
                except Exception as e:
                    logger.error(f"Error training {algorithm} for {target}: {e}")
                    continue
            
            training_results[target] = target_results
        
        self.training_history = training_results
        self.is_trained = True
        
        return training_results
    
    def evaluate_models(self, splits: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Evaluate all trained models on test set."""
        
        if not self.is_trained:
            raise ValueError("Models must be trained before evaluation")
        
        logger.info("Evaluating models on test set...")
        
        X_test = splits['X_test']
        y_test = splits['y_test']
        
        targets = ['fouls', 'yellow_cards', 'red_cards']
        algorithms = self.config['models']['ensemble_models']
        
        evaluation_results = {}
        
        for target in targets:
            target_results = {}
            
            for algorithm in algorithms:
                try:
                    metrics = self.predictor.evaluate_model(
                        X_test, y_test[target], target, algorithm
                    )
                    target_results[algorithm] = metrics
                    
                except Exception as e:
                    logger.error(f"Error evaluating {algorithm} for {target}: {e}")
                    continue
            
            # Ensemble evaluation
            try:
                ensemble_pred = self.predictor.ensemble_predict(
                    X_test, target, algorithms
                )
                ensemble_metrics = self.predictor._calculate_metrics(
                    y_test[target], ensemble_pred
                )
                ensemble_metrics['model_type'] = 'ensemble'
                ensemble_metrics['target'] = target
                target_results['ensemble'] = ensemble_metrics
                
            except Exception as e:
                logger.error(f"Error evaluating ensemble for {target}: {e}")
            
            evaluation_results[target] = target_results
        
        self.model_performance = evaluation_results
        
        # Log best performing models
        for target in targets:
            if target in evaluation_results:
                best_model = min(
                    evaluation_results[target].items(),
                    key=lambda x: x[1].get('rmse', float('inf'))
                )
                logger.info(
                    f"Best model for {target}: {best_model[0]} "
                    f"(RMSE: {best_model[1]['rmse']:.4f})"
                )
        
        return evaluation_results
    
    def train_clustering(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Train clustering models for player profiling."""
        
        logger.info("Training clustering models...")
        
        # Prepare clustering features
        clustering_features = self.clustering.prepare_features(
            self.cached_data['raw_data']
        )
        
        clustering_results = {}
        
        # Find optimal number of clusters if requested
        if self.config['clustering']['find_optimal_clusters']:
            optimal_clusters = self.clustering.find_optimal_clusters(
                clustering_features, max_clusters=10
            )
            optimal_k = optimal_clusters['consensus']['optimal_k']
            clustering_results['optimal_clusters'] = optimal_clusters
        else:
            optimal_k = self.config['clustering']['n_clusters']
        
        # Train clustering
        clustering_result = self.clustering.fit_clustering(
            clustering_features,
            algorithm=self.config['clustering']['algorithm'],
            n_clusters=optimal_k
        )
        
        clustering_results['clustering'] = clustering_result
        
        # Log clustering to MLflow
        try:
            mlflow_manager = get_mlflow_manager()
            
            if 'model' in clustering_result and 'labels' in clustering_result:
                mlflow_run_id = mlflow_manager.log_clustering_results(
                    clustering_model=clustering_result['model'],
                    algorithm=self.config['clustering']['algorithm'],
                    features=clustering_features,
                    labels=clustering_result['labels'],
                    metrics=clustering_result.get('metrics', {}),
                    cluster_profiles=clustering_result.get('cluster_profiles', {}),
                    tags={
                        'algorithm': self.config['clustering']['algorithm'],
                        'n_clusters': optimal_k,
                        'n_features': len(clustering_features.columns),
                        'clustering_date': datetime.now().strftime('%Y-%m-%d')
                    }
                )
                
                clustering_result['mlflow_run_id'] = mlflow_run_id
                logger.info(f"Clustering {self.config['clustering']['algorithm']} logged to MLflow: {mlflow_run_id}")
                
        except Exception as mlflow_error:
            logger.warning(f"Failed to log clustering {self.config['clustering']['algorithm']} to MLflow: {mlflow_error}")
        
        # Generate cluster summary
        cluster_summary = self.clustering.get_cluster_summary(
            self.config['clustering']['algorithm']
        )
        clustering_results['summary'] = cluster_summary
        
        logger.info(
            f"Clustering completed with {optimal_k} clusters. "
            f"Silhouette score: {clustering_result['metrics'].get('silhouette_score', 'N/A')}"
        )
        
        return clustering_results
    
    def train_anomaly_detection(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Train anomaly detection models."""
        
        logger.info("Training anomaly detection...")
        
        # Use clustering features for anomaly detection
        clustering_features = self.clustering.prepare_features(
            self.cached_data['raw_data']
        )
        
        # Detect anomalies
        anomaly_results = self.anomaly_detector.detect_anomalies(
            clustering_features,
            algorithm=self.config['anomaly_detection']['algorithm']
        )
        
        # Generate summary
        anomaly_summary = self.anomaly_detector.get_anomaly_summary(
            self.config['anomaly_detection']['algorithm']
        )
        
        results = {
            'detection': anomaly_results,
            'summary': anomaly_summary
        }
        
        # Log anomaly detection to MLflow
        try:
            mlflow_manager = get_mlflow_manager()
            
            if 'model' in anomaly_results and 'anomaly_scores' in anomaly_results:
                mlflow_run_id = mlflow_manager.log_anomaly_detection(
                    anomaly_model=anomaly_results['model'],
                    algorithm=self.config['anomaly_detection']['algorithm'],
                    features=clustering_features,
                    anomaly_scores=anomaly_results['anomaly_scores'],
                    anomaly_labels=anomaly_results.get('anomaly_labels', []),
                    metrics=anomaly_results.get('metrics', {}),
                    anomaly_summary=anomaly_summary,
                    tags={
                        'algorithm': self.config['anomaly_detection']['algorithm'],
                        'contamination': self.config['anomaly_detection']['contamination'],
                        'n_features': len(clustering_features.columns),
                        'anomaly_detection_date': datetime.now().strftime('%Y-%m-%d')
                    }
                )
                
                anomaly_results['mlflow_run_id'] = mlflow_run_id
                logger.info(f"Anomaly detection {self.config['anomaly_detection']['algorithm']} logged to MLflow: {mlflow_run_id}")
                
        except Exception as mlflow_error:
            logger.warning(f"Failed to log anomaly detection {self.config['anomaly_detection']['algorithm']} to MLflow: {mlflow_error}")
        
        logger.info(
            f"Anomaly detection completed. Found {anomaly_results['n_anomalies']} anomalies "
            f"({anomaly_results['anomaly_rate']*100:.1f}%)"
        )
        
        return results
    
    def run_full_pipeline(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Run the complete ML pipeline."""
        
        logger.info("Starting full ML pipeline...")
        start_time = datetime.now()
        
        # Initialize MLflow manager
        mlflow_manager = get_mlflow_manager()
        
        pipeline_results = {
            'start_time': start_time.isoformat(),
            'config': self.config
        }
        
        try:
            # 1. Load data
            raw_data = self.load_data(force_refresh)
            pipeline_results['data_info'] = {
                'n_records': len(raw_data),
                'n_players': raw_data['player_id'].nunique(),
                'n_teams': raw_data['team_id'].nunique(),
                'date_range': {
                    'start': raw_data['date'].min().isoformat(),
                    'end': raw_data['date'].max().isoformat()
                }
            }
            
            # 2. Prepare features
            features_df, targets_df = self.prepare_features(raw_data)
            
            # 3. Split data
            splits = self.split_data(features_df, targets_df)
            
            # 4. Train prediction models
            training_results = self.train_models(splits)
            pipeline_results['training'] = training_results
            
            # 5. Evaluate models
            evaluation_results = self.evaluate_models(splits)
            pipeline_results['evaluation'] = evaluation_results
            
            # 6. Train clustering
            clustering_results = self.train_clustering(features_df)
            pipeline_results['clustering'] = clustering_results
            
            # 7. Train anomaly detection
            anomaly_results = self.train_anomaly_detection(features_df)
            pipeline_results['anomaly_detection'] = anomaly_results
            
            # 8. Generate feature importance
            self._generate_feature_importance()
            pipeline_results['feature_importance'] = self.feature_importance
            
            # 9. Save results if configured
            if self.config['output']['save_models']:
                self.save_pipeline()
            
            # Pipeline completion
            end_time = datetime.now()
            pipeline_results['end_time'] = end_time.isoformat()
            pipeline_results['duration_minutes'] = (end_time - start_time).total_seconds() / 60
            pipeline_results['status'] = 'completed'
            
            # Log pipeline run to MLflow
            try:
                logger.info("Logging pipeline run to MLflow...")
                mlflow_run_id = mlflow_manager.log_pipeline_run(
                    pipeline=self,
                    pipeline_results=pipeline_results,
                    tags={
                        'pipeline_version': '1.0',
                        'environment': 'production'
                    }
                )
                pipeline_results['mlflow_run_id'] = mlflow_run_id
                logger.info(f"Pipeline run logged to MLflow with ID: {mlflow_run_id}")
            except Exception as mlflow_error:
                logger.warning(f"Failed to log pipeline run to MLflow: {mlflow_error}")
            
            logger.info(
                f"Pipeline completed successfully in {pipeline_results['duration_minutes']:.1f} minutes"
            )
            
        except Exception as e:
            end_time = datetime.now()
            pipeline_results['end_time'] = end_time.isoformat()
            pipeline_results['duration_minutes'] = (end_time - start_time).total_seconds() / 60
            pipeline_results['status'] = 'failed'
            pipeline_results['error'] = str(e)
            
            # Log error to MLflow
            try:
                mlflow_run_id = mlflow_manager.log_pipeline_run(
                    pipeline=self,
                    pipeline_results=pipeline_results,
                    tags={
                        'pipeline_version': '1.0',
                        'environment': 'production',
                        'status': 'failed'
                    }
                )
                pipeline_results['mlflow_run_id'] = mlflow_run_id
            except Exception:
                pass  # Don't fail if MLflow logging fails
            
            logger.error(f"Pipeline failed: {e}")
            raise
        
        return pipeline_results
    
    def _generate_feature_importance(self):
        """Generate consolidated feature importance across all models."""
        
        logger.info("Generating feature importance analysis...")
        
        targets = ['fouls', 'yellow_cards', 'red_cards']
        algorithms = self.config['models']['ensemble_models']
        
        self.feature_importance = {}
        
        for target in targets:
            target_importance = {}
            
            for algorithm in algorithms:
                importance = self.predictor.get_feature_importance(
                    target, algorithm, top_k=20
                )
                if importance:
                    target_importance[algorithm] = importance
            
            # Calculate average importance across algorithms
            if target_importance:
                all_features = set()
                for imp_dict in target_importance.values():
                    all_features.update(imp_dict.keys())
                
                avg_importance = {}
                for feature in all_features:
                    importances = [
                        imp_dict.get(feature, 0) 
                        for imp_dict in target_importance.values()
                    ]
                    avg_importance[feature] = np.mean(importances)
                
                # Sort by average importance
                avg_importance = dict(
                    sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
                )
                
                target_importance['average'] = avg_importance
            
            self.feature_importance[target] = target_importance
    
    def predict_player_risk(
        self, 
        player_id: int, 
        upcoming_matches: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Predict risk for a specific player."""
        
        if not self.is_trained:
            raise ValueError("Pipeline must be trained before making predictions")
        
        logger.info(f"Predicting risk for player {player_id}...")
        
        # Get player's recent data
        raw_data = self.cached_data.get('raw_data')
        if raw_data is None:
            raw_data = self.load_data()
        
        player_data = raw_data[raw_data['player_id'] == player_id]
        
        if len(player_data) == 0:
            raise ValueError(f"No data found for player {player_id}")
        
        # Prepare features for the player's most recent games
        recent_data = player_data.tail(self.config['features']['lookback_games'])
        
        # Engineer features
        player_features = self.feature_engineer.create_features(
            recent_data,
            lookback_games=self.config['features']['lookback_games']
        )
        
        if len(player_features) == 0:
            raise ValueError(f"Could not generate features for player {player_id}")
        
        # Use the most recent feature vector
        latest_features = player_features.tail(1)
        
        # Risk assessment
        risk_assessment = self.risk_assessor.assess_player_risk(latest_features)
        
        # Add player context
        player_info = {
            'player_id': player_id,
            'player_name': player_data['player_name'].iloc[-1],
            'position': player_data['position'].iloc[-1],
            'team': player_data['team_name'].iloc[-1],
            'recent_games': len(recent_data),
            'last_game_date': player_data['date'].max().isoformat()
        }
        
        # Cluster assignment
        if hasattr(self.clustering, 'cluster_labels'):
            clustering_features = self.clustering.prepare_features(player_data)
            if len(clustering_features) > 0:
                cluster_pred = self.clustering.predict_cluster(
                    clustering_features.tail(1),
                    self.config['clustering']['algorithm']
                )
                player_info['cluster'] = int(cluster_pred[0])
        
        result = {
            'player_info': player_info,
            'risk_assessment': risk_assessment,
            'prediction_timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and metrics."""
        
        status = {
            'is_trained': self.is_trained,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'config': self.config,
            'cached_data_available': list(self.cached_data.keys()),
            'model_performance': self.model_performance,
            'feature_importance_available': bool(self.feature_importance)
        }
        
        if self.is_trained:
            # Add model summaries
            status['model_summary'] = self.predictor.get_model_summary()
            
            if hasattr(self.clustering, 'cluster_profiles'):
                status['clustering_summary'] = {
                    'n_clusters': len(self.clustering.cluster_profiles.get(
                        self.config['clustering']['algorithm'], {}
                    )),
                    'algorithm': self.config['clustering']['algorithm']
                }
        
        return status
    
    def save_pipeline(self, directory: str = None):
        """Save the entire pipeline."""
        
        if directory is None:
            directory = Path(settings.MODEL_STORAGE_PATH) / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            directory = Path(directory)
        
        directory.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving pipeline to {directory}...")
        
        # Save individual components
        self.predictor.save_models(directory / "prediction_models")
        self.clustering.save_clustering_results(directory / "clustering")
        
        # Save pipeline metadata
        metadata = {
            'config': self.config,
            'is_trained': self.is_trained,
            'training_history': self.training_history,
            'model_performance': self.model_performance,
            'feature_importance': self.feature_importance,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'save_timestamp': datetime.now().isoformat()
        }
        
        with open(directory / "pipeline_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save feature engineer
        joblib.dump(self.feature_engineer, directory / "feature_engineer.joblib")
        
        logger.info(f"Pipeline saved successfully to {directory}")
    
    def load_pipeline(self, directory: str):
        """Load a saved pipeline."""
        
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Pipeline directory {directory} not found")
        
        logger.info(f"Loading pipeline from {directory}...")
        
        # Load metadata
        metadata_path = directory / "pipeline_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.config = metadata.get('config', self._get_default_config())
            self.is_trained = metadata.get('is_trained', False)
            self.training_history = metadata.get('training_history', {})
            self.model_performance = metadata.get('model_performance', {})
            self.feature_importance = metadata.get('feature_importance', {})
            
            if metadata.get('last_update'):
                self.last_update = datetime.fromisoformat(metadata['last_update'])
        
        # Load components
        prediction_models_dir = directory / "prediction_models"
        if prediction_models_dir.exists():
            self.predictor.load_models(prediction_models_dir)
        
        clustering_dir = directory / "clustering"
        if clustering_dir.exists():
            self.clustering.load_clustering_results(clustering_dir)
        
        # Load feature engineer
        feature_engineer_path = directory / "feature_engineer.joblib"
        if feature_engineer_path.exists():
            self.feature_engineer = joblib.load(feature_engineer_path)
        
        # Reinitialize risk assessor with loaded predictor
        self.risk_assessor = FootballRiskAssessment(self.predictor)
        
        logger.info(f"Pipeline loaded successfully from {directory}")

# Global pipeline instance
_pipeline_instance = None

def get_pipeline() -> MLPipeline:
    """Get or create global pipeline instance."""
    global _pipeline_instance
    
    if _pipeline_instance is None:
        _pipeline_instance = MLPipeline()
    
    return _pipeline_instance

def initialize_pipeline(config: Dict[str, Any] = None) -> MLPipeline:
    """Initialize pipeline with custom configuration."""
    global _pipeline_instance
    
    _pipeline_instance = MLPipeline(config)
    return _pipeline_instance
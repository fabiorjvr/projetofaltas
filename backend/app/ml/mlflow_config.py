import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
from mlflow.types.schema import Schema, ColSpec
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from datetime import datetime
from pathlib import Path
import json
import joblib
import os
from urllib.parse import urlparse

from ..core.config import settings
from .models import FootballFoulPredictor
from .clustering import PlayerProfileClustering, AnomalyDetection
from .pipeline import MLPipeline

logger = logging.getLogger(__name__)

class MLflowManager:
    """MLflow integration for model versioning and serving."""
    
    def __init__(self, tracking_uri: str = None, experiment_name: str = "football-fouls-analytics"):
        self.tracking_uri = tracking_uri or settings.MLFLOW_TRACKING_URI
        self.experiment_name = experiment_name
        self.client = None
        self.experiment_id = None
        
        # Initialize MLflow
        self._setup_mlflow()
    
    def _setup_mlflow(self):
        """Setup MLflow tracking and experiment."""
        
        # Set tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)
        logger.info(f"MLflow tracking URI set to: {self.tracking_uri}")
        
        # Initialize client
        self.client = MlflowClient()
        
        # Create or get experiment
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                self.experiment_id = mlflow.create_experiment(
                    self.experiment_name,
                    tags={
                        "project": "football-fouls-analytics",
                        "created_by": "ml_pipeline",
                        "created_at": datetime.now().isoformat()
                    }
                )
                logger.info(f"Created new experiment: {self.experiment_name}")
            else:
                self.experiment_id = experiment.experiment_id
                logger.info(f"Using existing experiment: {self.experiment_name}")
            
            mlflow.set_experiment(self.experiment_name)
            
        except Exception as e:
            logger.error(f"Error setting up MLflow experiment: {e}")
            raise
    
    def log_model_training(
        self,
        model,
        model_name: str,
        target: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        training_metrics: Dict[str, Any],
        validation_metrics: Dict[str, Any],
        feature_importance: Dict[str, float] = None,
        model_params: Dict[str, Any] = None,
        tags: Dict[str, str] = None
    ) -> str:
        """Log model training run to MLflow."""
        
        with mlflow.start_run(run_name=f"{model_name}_{target}") as run:
            # Log parameters
            if model_params:
                mlflow.log_params(model_params)
            
            # Log model hyperparameters
            if hasattr(model, 'get_params'):
                mlflow.log_params(model.get_params())
            
            # Log training metrics
            for metric_name, value in training_metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"train_{metric_name}", value)
            
            # Log validation metrics
            for metric_name, value in validation_metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"val_{metric_name}", value)
            
            # Log feature importance
            if feature_importance:
                # Log top features as metrics
                top_features = sorted(
                    feature_importance.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:10]
                
                for i, (feature, importance) in enumerate(top_features):
                    mlflow.log_metric(f"feature_importance_{i+1}", importance)
                    mlflow.log_param(f"top_feature_{i+1}", feature)
                
                # Save full feature importance as artifact
                importance_df = pd.DataFrame(
                    list(feature_importance.items()),
                    columns=['feature', 'importance']
                ).sort_values('importance', ascending=False)
                
                importance_path = "feature_importance.csv"
                importance_df.to_csv(importance_path, index=False)
                mlflow.log_artifact(importance_path)
                os.remove(importance_path)
            
            # Log model signature
            signature = infer_signature(X_train, y_train)
            
            # Log model based on type
            if hasattr(model, 'booster'):  # XGBoost
                mlflow.xgboost.log_model(
                    model,
                    f"model_{target}",
                    signature=signature,
                    registered_model_name=f"{model_name}_{target}"
                )
            else:  # Sklearn models
                mlflow.sklearn.log_model(
                    model,
                    f"model_{target}",
                    signature=signature,
                    registered_model_name=f"{model_name}_{target}"
                )
            
            # Log additional metadata
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("target", target)
            mlflow.log_param("n_features", len(X_train.columns))
            mlflow.log_param("n_train_samples", len(X_train))
            mlflow.log_param("n_val_samples", len(X_val))
            
            # Log tags
            default_tags = {
                "model_category": "prediction",
                "target_variable": target,
                "algorithm": model_name,
                "training_date": datetime.now().strftime("%Y-%m-%d")
            }
            
            if tags:
                default_tags.update(tags)
            
            mlflow.set_tags(default_tags)
            
            run_id = run.info.run_id
            logger.info(f"Logged {model_name} model for {target} with run_id: {run_id}")
            
            return run_id
    
    def log_clustering_results(
        self,
        clustering_model,
        algorithm: str,
        features: pd.DataFrame,
        labels: np.ndarray,
        metrics: Dict[str, Any],
        cluster_profiles: Dict[int, Dict[str, Any]],
        tags: Dict[str, str] = None
    ) -> str:
        """Log clustering results to MLflow."""
        
        with mlflow.start_run(run_name=f"clustering_{algorithm}") as run:
            # Log clustering parameters
            if hasattr(clustering_model, 'get_params'):
                mlflow.log_params(clustering_model.get_params())
            
            # Log clustering metrics
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(metric_name, value)
            
            # Log cluster information
            n_clusters = len(np.unique(labels)) - (1 if -1 in labels else 0)
            mlflow.log_param("n_clusters", n_clusters)
            mlflow.log_param("n_samples", len(features))
            mlflow.log_param("n_features", len(features.columns))
            
            # Save cluster profiles as artifact
            profiles_path = "cluster_profiles.json"
            with open(profiles_path, 'w') as f:
                json.dump(cluster_profiles, f, indent=2)
            mlflow.log_artifact(profiles_path)
            os.remove(profiles_path)
            
            # Save cluster labels
            labels_df = pd.DataFrame({
                'sample_index': features.index,
                'cluster_label': labels
            })
            labels_path = "cluster_labels.csv"
            labels_df.to_csv(labels_path, index=False)
            mlflow.log_artifact(labels_path)
            os.remove(labels_path)
            
            # Log model
            mlflow.sklearn.log_model(
                clustering_model,
                "clustering_model",
                registered_model_name=f"clustering_{algorithm}"
            )
            
            # Log tags
            default_tags = {
                "model_category": "clustering",
                "algorithm": algorithm,
                "training_date": datetime.now().strftime("%Y-%m-%d")
            }
            
            if tags:
                default_tags.update(tags)
            
            mlflow.set_tags(default_tags)
            
            run_id = run.info.run_id
            logger.info(f"Logged clustering model ({algorithm}) with run_id: {run_id}")
            
            return run_id
    
    def log_anomaly_detection(
        self,
        anomaly_model,
        algorithm: str,
        features: pd.DataFrame,
        labels: np.ndarray,
        scores: np.ndarray,
        anomaly_details: List[Dict[str, Any]],
        tags: Dict[str, str] = None
    ) -> str:
        """Log anomaly detection results to MLflow."""
        
        with mlflow.start_run(run_name=f"anomaly_detection_{algorithm}") as run:
            # Log model parameters
            if hasattr(anomaly_model, 'get_params'):
                mlflow.log_params(anomaly_model.get_params())
            
            # Log anomaly metrics
            n_anomalies = np.sum(labels == -1)
            anomaly_rate = n_anomalies / len(labels)
            
            mlflow.log_metric("n_anomalies", n_anomalies)
            mlflow.log_metric("anomaly_rate", anomaly_rate)
            mlflow.log_metric("n_samples", len(features))
            
            # Log score statistics
            mlflow.log_metric("score_mean", float(scores.mean()))
            mlflow.log_metric("score_std", float(scores.std()))
            mlflow.log_metric("score_min", float(scores.min()))
            mlflow.log_metric("score_max", float(scores.max()))
            
            # Save anomaly details
            details_path = "anomaly_details.json"
            with open(details_path, 'w') as f:
                json.dump(anomaly_details, f, indent=2)
            mlflow.log_artifact(details_path)
            os.remove(details_path)
            
            # Save anomaly scores
            scores_df = pd.DataFrame({
                'sample_index': features.index,
                'anomaly_label': labels,
                'anomaly_score': scores
            })
            scores_path = "anomaly_scores.csv"
            scores_df.to_csv(scores_path, index=False)
            mlflow.log_artifact(scores_path)
            os.remove(scores_path)
            
            # Log model (for algorithms that support it)
            if algorithm != 'local_outlier_factor':
                mlflow.sklearn.log_model(
                    anomaly_model,
                    "anomaly_model",
                    registered_model_name=f"anomaly_detection_{algorithm}"
                )
            
            # Log tags
            default_tags = {
                "model_category": "anomaly_detection",
                "algorithm": algorithm,
                "training_date": datetime.now().strftime("%Y-%m-%d")
            }
            
            if tags:
                default_tags.update(tags)
            
            mlflow.set_tags(default_tags)
            
            run_id = run.info.run_id
            logger.info(f"Logged anomaly detection model ({algorithm}) with run_id: {run_id}")
            
            return run_id
    
    def log_pipeline_run(
        self,
        pipeline: MLPipeline,
        pipeline_results: Dict[str, Any],
        tags: Dict[str, str] = None
    ) -> str:
        """Log complete pipeline run to MLflow."""
        
        with mlflow.start_run(run_name="pipeline_run") as run:
            # Log pipeline configuration
            mlflow.log_params({
                f"config_{key}": str(value)
                for key, value in pipeline.config.items()
                if isinstance(value, (str, int, float, bool))
            })
            
            # Log data information
            if 'data_info' in pipeline_results:
                data_info = pipeline_results['data_info']
                for key, value in data_info.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(f"data_{key}", value)
                    elif isinstance(value, str):
                        mlflow.log_param(f"data_{key}", value)
            
            # Log best model performance for each target
            if 'evaluation' in pipeline_results:
                evaluation = pipeline_results['evaluation']
                for target, models in evaluation.items():
                    if isinstance(models, dict):
                        # Find best model for this target
                        best_model = min(
                            models.items(),
                            key=lambda x: x[1].get('rmse', float('inf'))
                        )
                        
                        model_name, metrics = best_model
                        for metric_name, value in metrics.items():
                            if isinstance(value, (int, float)):
                                mlflow.log_metric(f"best_{target}_{metric_name}", value)
                        
                        mlflow.log_param(f"best_model_{target}", model_name)
            
            # Log clustering metrics
            if 'clustering' in pipeline_results:
                clustering = pipeline_results['clustering']
                if 'clustering' in clustering and 'metrics' in clustering['clustering']:
                    metrics = clustering['clustering']['metrics']
                    for metric_name, value in metrics.items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(f"clustering_{metric_name}", value)
            
            # Log anomaly detection metrics
            if 'anomaly_detection' in pipeline_results:
                anomaly = pipeline_results['anomaly_detection']
                if 'detection' in anomaly:
                    detection = anomaly['detection']
                    mlflow.log_metric("anomaly_count", detection.get('n_anomalies', 0))
                    mlflow.log_metric("anomaly_rate", detection.get('anomaly_rate', 0))
            
            # Log pipeline duration
            if 'duration_minutes' in pipeline_results:
                mlflow.log_metric("pipeline_duration_minutes", pipeline_results['duration_minutes'])
            
            # Save complete pipeline results
            results_path = "pipeline_results.json"
            with open(results_path, 'w') as f:
                json.dump(pipeline_results, f, indent=2, default=str)
            mlflow.log_artifact(results_path)
            os.remove(results_path)
            
            # Log tags
            default_tags = {
                "model_category": "pipeline",
                "pipeline_status": pipeline_results.get('status', 'unknown'),
                "training_date": datetime.now().strftime("%Y-%m-%d")
            }
            
            if tags:
                default_tags.update(tags)
            
            mlflow.set_tags(default_tags)
            
            run_id = run.info.run_id
            logger.info(f"Logged pipeline run with run_id: {run_id}")
            
            return run_id
    
    def get_best_model(
        self,
        model_name: str,
        target: str,
        metric: str = "val_rmse",
        stage: str = "Production"
    ) -> Any:
        """Get the best model from MLflow model registry."""
        
        try:
            registered_model_name = f"{model_name}_{target}"
            
            # Get model versions
            model_versions = self.client.search_model_versions(
                f"name='{registered_model_name}'"
            )
            
            if not model_versions:
                logger.warning(f"No model versions found for {registered_model_name}")
                return None
            
            # Filter by stage if specified
            if stage:
                model_versions = [
                    mv for mv in model_versions 
                    if mv.current_stage == stage
                ]
            
            if not model_versions:
                logger.warning(f"No model versions in {stage} stage for {registered_model_name}")
                return None
            
            # Find best model based on metric
            best_version = None
            best_metric_value = float('inf')
            
            for version in model_versions:
                run_id = version.run_id
                run = self.client.get_run(run_id)
                
                if metric in run.data.metrics:
                    metric_value = run.data.metrics[metric]
                    if metric_value < best_metric_value:
                        best_metric_value = metric_value
                        best_version = version
            
            if best_version:
                # Load model
                model_uri = f"models:/{registered_model_name}/{best_version.version}"
                
                if 'xgboost' in model_name.lower():
                    model = mlflow.xgboost.load_model(model_uri)
                else:
                    model = mlflow.sklearn.load_model(model_uri)
                
                logger.info(
                    f"Loaded best {model_name} model for {target} "
                    f"(version {best_version.version}, {metric}: {best_metric_value:.4f})"
                )
                
                return model
            
            logger.warning(f"No model found with metric {metric} for {registered_model_name}")
            return None
            
        except Exception as e:
            logger.error(f"Error loading best model: {e}")
            return None
    
    def promote_model(
        self,
        model_name: str,
        target: str,
        version: str,
        stage: str = "Production"
    ) -> bool:
        """Promote a model version to a specific stage."""
        
        try:
            registered_model_name = f"{model_name}_{target}"
            
            self.client.transition_model_version_stage(
                name=registered_model_name,
                version=version,
                stage=stage
            )
            
            logger.info(
                f"Promoted {registered_model_name} version {version} to {stage}"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error promoting model: {e}")
            return False
    
    def compare_models(
        self,
        model_names: List[str],
        target: str,
        metrics: List[str] = None
    ) -> pd.DataFrame:
        """Compare different models for a target variable."""
        
        if metrics is None:
            metrics = ['val_rmse', 'val_mae', 'val_r2']
        
        comparison_data = []
        
        for model_name in model_names:
            registered_model_name = f"{model_name}_{target}"
            
            try:
                # Get latest version
                model_versions = self.client.search_model_versions(
                    f"name='{registered_model_name}'"
                )
                
                if model_versions:
                    latest_version = max(model_versions, key=lambda x: int(x.version))
                    run_id = latest_version.run_id
                    run = self.client.get_run(run_id)
                    
                    model_data = {
                        'model_name': model_name,
                        'version': latest_version.version,
                        'run_id': run_id,
                        'stage': latest_version.current_stage
                    }
                    
                    # Add metrics
                    for metric in metrics:
                        model_data[metric] = run.data.metrics.get(metric, np.nan)
                    
                    comparison_data.append(model_data)
                    
            except Exception as e:
                logger.error(f"Error getting data for {model_name}: {e}")
                continue
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            return df.sort_values(metrics[0] if metrics else 'val_rmse')
        else:
            return pd.DataFrame()
    
    def cleanup_old_runs(
        self,
        max_runs: int = 100,
        older_than_days: int = 30
    ):
        """Clean up old MLflow runs."""
        
        try:
            # Get all runs for the experiment
            runs = self.client.search_runs(
                experiment_ids=[self.experiment_id],
                max_results=1000
            )
            
            # Filter runs to delete
            cutoff_date = datetime.now() - timedelta(days=older_than_days)
            runs_to_delete = []
            
            for run in runs:
                run_date = datetime.fromtimestamp(run.info.start_time / 1000)
                if run_date < cutoff_date:
                    runs_to_delete.append(run.info.run_id)
            
            # Keep only the most recent runs
            if len(runs) > max_runs:
                sorted_runs = sorted(runs, key=lambda x: x.info.start_time, reverse=True)
                runs_to_delete.extend([
                    run.info.run_id for run in sorted_runs[max_runs:]
                ])
            
            # Delete runs
            for run_id in runs_to_delete:
                self.client.delete_run(run_id)
            
            logger.info(f"Cleaned up {len(runs_to_delete)} old runs")
            
        except Exception as e:
            logger.error(f"Error cleaning up runs: {e}")
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get summary of the current experiment."""
        
        try:
            # Get experiment info
            experiment = self.client.get_experiment(self.experiment_id)
            
            # Get all runs
            runs = self.client.search_runs(
                experiment_ids=[self.experiment_id],
                max_results=1000
            )
            
            # Calculate summary statistics
            summary = {
                'experiment_name': experiment.name,
                'experiment_id': self.experiment_id,
                'total_runs': len(runs),
                'active_runs': len([r for r in runs if r.info.status == 'RUNNING']),
                'completed_runs': len([r for r in runs if r.info.status == 'FINISHED']),
                'failed_runs': len([r for r in runs if r.info.status == 'FAILED']),
                'tracking_uri': self.tracking_uri
            }
            
            # Get model registry info
            try:
                registered_models = self.client.search_registered_models()
                summary['registered_models'] = len(registered_models)
                
                model_stages = {}
                for model in registered_models:
                    for version in model.latest_versions:
                        stage = version.current_stage
                        model_stages[stage] = model_stages.get(stage, 0) + 1
                
                summary['models_by_stage'] = model_stages
                
            except Exception:
                summary['registered_models'] = 0
                summary['models_by_stage'] = {}
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting experiment summary: {e}")
            return {}

# Global MLflow manager instance
_mlflow_manager = None

def get_mlflow_manager() -> MLflowManager:
    """Get or create global MLflow manager instance."""
    global _mlflow_manager
    
    if _mlflow_manager is None:
        _mlflow_manager = MLflowManager()
    
    return _mlflow_manager

def initialize_mlflow_manager(
    tracking_uri: str = None,
    experiment_name: str = "football-fouls-analytics"
) -> MLflowManager:
    """Initialize MLflow manager with custom configuration."""
    global _mlflow_manager
    
    _mlflow_manager = MLflowManager(tracking_uri, experiment_name)
    return _mlflow_manager
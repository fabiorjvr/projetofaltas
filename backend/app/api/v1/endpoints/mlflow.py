from fastapi import APIRouter, HTTPException, Depends, Query, Path
from typing import Dict, List, Optional, Any
import pandas as pd
from datetime import datetime, timedelta

from ....core.auth import get_current_user
from ....ml.mlflow_config import get_mlflow_manager
from ....schemas.user import User
from ....schemas.response import (
    StandardResponse,
    PaginatedResponse,
    create_response,
    create_paginated_response
)

router = APIRouter()

@router.get("/experiments/summary", response_model=StandardResponse)
async def get_experiment_summary(
    current_user: User = Depends(get_current_user)
) -> StandardResponse:
    """
    Get MLflow experiment summary.
    """
    try:
        mlflow_manager = get_mlflow_manager()
        summary = mlflow_manager.get_experiment_summary()
        
        return create_response(
            data=summary,
            message="Experiment summary retrieved successfully"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get experiment summary: {str(e)}"
        )

@router.get("/models/compare", response_model=StandardResponse)
async def compare_models(
    target: str = Query(..., description="Target variable to compare models for"),
    models: List[str] = Query(..., description="List of model names to compare"),
    metrics: Optional[List[str]] = Query(
        default=["val_rmse", "val_mae", "val_r2"],
        description="Metrics to compare"
    ),
    current_user: User = Depends(get_current_user)
) -> StandardResponse:
    """
    Compare different models for a target variable.
    """
    try:
        mlflow_manager = get_mlflow_manager()
        comparison_df = mlflow_manager.compare_models(
            model_names=models,
            target=target,
            metrics=metrics
        )
        
        if comparison_df.empty:
            return create_response(
                data=[],
                message="No models found for comparison"
            )
        
        # Convert DataFrame to list of dictionaries
        comparison_data = comparison_df.to_dict('records')
        
        return create_response(
            data=comparison_data,
            message=f"Model comparison for {target} retrieved successfully"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to compare models: {str(e)}"
        )

@router.get("/models/best", response_model=StandardResponse)
async def get_best_model_info(
    model_name: str = Query(..., description="Model name"),
    target: str = Query(..., description="Target variable"),
    metric: str = Query(default="val_rmse", description="Metric to optimize for"),
    stage: str = Query(default="Production", description="Model stage"),
    current_user: User = Depends(get_current_user)
) -> StandardResponse:
    """
    Get information about the best model for a target variable.
    """
    try:
        mlflow_manager = get_mlflow_manager()
        
        # Get model versions
        registered_model_name = f"{model_name}_{target}"
        model_versions = mlflow_manager.client.search_model_versions(
            f"name='{registered_model_name}'"
        )
        
        if not model_versions:
            return create_response(
                data=None,
                message=f"No model versions found for {registered_model_name}"
            )
        
        # Filter by stage if specified
        if stage:
            model_versions = [
                mv for mv in model_versions 
                if mv.current_stage == stage
            ]
        
        if not model_versions:
            return create_response(
                data=None,
                message=f"No model versions in {stage} stage for {registered_model_name}"
            )
        
        # Find best model based on metric
        best_version = None
        best_metric_value = float('inf')
        best_run_info = None
        
        for version in model_versions:
            run_id = version.run_id
            run = mlflow_manager.client.get_run(run_id)
            
            if metric in run.data.metrics:
                metric_value = run.data.metrics[metric]
                if metric_value < best_metric_value:
                    best_metric_value = metric_value
                    best_version = version
                    best_run_info = run
        
        if best_version and best_run_info:
            model_info = {
                'model_name': model_name,
                'target': target,
                'version': best_version.version,
                'stage': best_version.current_stage,
                'run_id': best_version.run_id,
                'best_metric': {
                    'name': metric,
                    'value': best_metric_value
                },
                'all_metrics': best_run_info.data.metrics,
                'parameters': best_run_info.data.params,
                'tags': best_run_info.data.tags,
                'creation_timestamp': best_version.creation_timestamp,
                'last_updated_timestamp': best_version.last_updated_timestamp
            }
            
            return create_response(
                data=model_info,
                message=f"Best {model_name} model for {target} retrieved successfully"
            )
        else:
            return create_response(
                data=None,
                message=f"No model found with metric {metric} for {registered_model_name}"
            )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get best model info: {str(e)}"
        )

@router.post("/models/promote", response_model=StandardResponse)
async def promote_model(
    model_name: str = Query(..., description="Model name"),
    target: str = Query(..., description="Target variable"),
    version: str = Query(..., description="Model version to promote"),
    stage: str = Query(default="Production", description="Stage to promote to"),
    current_user: User = Depends(get_current_user)
) -> StandardResponse:
    """
    Promote a model version to a specific stage.
    """
    try:
        mlflow_manager = get_mlflow_manager()
        
        success = mlflow_manager.promote_model(
            model_name=model_name,
            target=target,
            version=version,
            stage=stage
        )
        
        if success:
            return create_response(
                data={
                    'model_name': model_name,
                    'target': target,
                    'version': version,
                    'stage': stage,
                    'promoted_at': datetime.now().isoformat()
                },
                message=f"Model {model_name}_{target} version {version} promoted to {stage} successfully"
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="Failed to promote model"
            )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to promote model: {str(e)}"
        )

@router.get("/runs/recent", response_model=StandardResponse)
async def get_recent_runs(
    limit: int = Query(default=10, ge=1, le=100, description="Number of recent runs to retrieve"),
    experiment_name: Optional[str] = Query(default=None, description="Filter by experiment name"),
    current_user: User = Depends(get_current_user)
) -> StandardResponse:
    """
    Get recent MLflow runs.
    """
    try:
        mlflow_manager = get_mlflow_manager()
        
        # Get experiment ID
        experiment_id = mlflow_manager.experiment_id
        if experiment_name:
            experiment = mlflow_manager.client.get_experiment_by_name(experiment_name)
            if experiment:
                experiment_id = experiment.experiment_id
        
        # Get recent runs
        runs = mlflow_manager.client.search_runs(
            experiment_ids=[experiment_id],
            max_results=limit,
            order_by=["start_time DESC"]
        )
        
        runs_data = []
        for run in runs:
            run_data = {
                'run_id': run.info.run_id,
                'experiment_id': run.info.experiment_id,
                'status': run.info.status,
                'start_time': datetime.fromtimestamp(run.info.start_time / 1000).isoformat(),
                'end_time': datetime.fromtimestamp(run.info.end_time / 1000).isoformat() if run.info.end_time else None,
                'metrics': run.data.metrics,
                'parameters': run.data.params,
                'tags': run.data.tags
            }
            runs_data.append(run_data)
        
        return create_response(
            data=runs_data,
            message=f"Retrieved {len(runs_data)} recent runs successfully"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get recent runs: {str(e)}"
        )

@router.get("/runs/{run_id}", response_model=StandardResponse)
async def get_run_details(
    run_id: str = Path(..., description="MLflow run ID"),
    current_user: User = Depends(get_current_user)
) -> StandardResponse:
    """
    Get detailed information about a specific MLflow run.
    """
    try:
        mlflow_manager = get_mlflow_manager()
        
        # Get run details
        run = mlflow_manager.client.get_run(run_id)
        
        run_details = {
            'run_id': run.info.run_id,
            'experiment_id': run.info.experiment_id,
            'status': run.info.status,
            'start_time': datetime.fromtimestamp(run.info.start_time / 1000).isoformat(),
            'end_time': datetime.fromtimestamp(run.info.end_time / 1000).isoformat() if run.info.end_time else None,
            'duration_seconds': (run.info.end_time - run.info.start_time) / 1000 if run.info.end_time else None,
            'metrics': run.data.metrics,
            'parameters': run.data.params,
            'tags': run.data.tags,
            'artifact_uri': run.info.artifact_uri
        }
        
        # Get artifacts list
        try:
            artifacts = mlflow_manager.client.list_artifacts(run_id)
            run_details['artifacts'] = [
                {
                    'path': artifact.path,
                    'is_dir': artifact.is_dir,
                    'file_size': artifact.file_size
                }
                for artifact in artifacts
            ]
        except Exception:
            run_details['artifacts'] = []
        
        return create_response(
            data=run_details,
            message=f"Run {run_id} details retrieved successfully"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get run details: {str(e)}"
        )

@router.delete("/runs/cleanup", response_model=StandardResponse)
async def cleanup_old_runs(
    max_runs: int = Query(default=100, ge=10, le=1000, description="Maximum number of runs to keep"),
    older_than_days: int = Query(default=30, ge=1, le=365, description="Delete runs older than this many days"),
    current_user: User = Depends(get_current_user)
) -> StandardResponse:
    """
    Clean up old MLflow runs.
    """
    try:
        mlflow_manager = get_mlflow_manager()
        
        # Get runs count before cleanup
        runs_before = mlflow_manager.client.search_runs(
            experiment_ids=[mlflow_manager.experiment_id],
            max_results=1000
        )
        
        # Perform cleanup
        mlflow_manager.cleanup_old_runs(
            max_runs=max_runs,
            older_than_days=older_than_days
        )
        
        # Get runs count after cleanup
        runs_after = mlflow_manager.client.search_runs(
            experiment_ids=[mlflow_manager.experiment_id],
            max_results=1000
        )
        
        cleanup_info = {
            'runs_before': len(runs_before),
            'runs_after': len(runs_after),
            'runs_deleted': len(runs_before) - len(runs_after),
            'max_runs_limit': max_runs,
            'older_than_days': older_than_days,
            'cleanup_timestamp': datetime.now().isoformat()
        }
        
        return create_response(
            data=cleanup_info,
            message=f"Cleanup completed. Deleted {cleanup_info['runs_deleted']} runs"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cleanup runs: {str(e)}"
        )

@router.get("/models/registry", response_model=StandardResponse)
async def get_model_registry(
    current_user: User = Depends(get_current_user)
) -> StandardResponse:
    """
    Get all registered models in MLflow model registry.
    """
    try:
        mlflow_manager = get_mlflow_manager()
        
        # Get all registered models
        registered_models = mlflow_manager.client.search_registered_models()
        
        models_data = []
        for model in registered_models:
            model_data = {
                'name': model.name,
                'creation_timestamp': model.creation_timestamp,
                'last_updated_timestamp': model.last_updated_timestamp,
                'description': model.description,
                'tags': model.tags,
                'latest_versions': []
            }
            
            # Get latest versions for each stage
            for version in model.latest_versions:
                version_data = {
                    'version': version.version,
                    'stage': version.current_stage,
                    'creation_timestamp': version.creation_timestamp,
                    'last_updated_timestamp': version.last_updated_timestamp,
                    'run_id': version.run_id,
                    'status': version.status,
                    'status_message': version.status_message
                }
                model_data['latest_versions'].append(version_data)
            
            models_data.append(model_data)
        
        return create_response(
            data=models_data,
            message=f"Retrieved {len(models_data)} registered models successfully"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model registry: {str(e)}"
        )

@router.get("/health", response_model=StandardResponse)
async def check_mlflow_health(
    current_user: User = Depends(get_current_user)
) -> StandardResponse:
    """
    Check MLflow server health and connectivity.
    """
    try:
        mlflow_manager = get_mlflow_manager()
        
        # Test basic connectivity
        experiment_summary = mlflow_manager.get_experiment_summary()
        
        health_info = {
            'status': 'healthy',
            'tracking_uri': mlflow_manager.tracking_uri,
            'experiment_name': experiment_summary.get('experiment_name'),
            'experiment_id': experiment_summary.get('experiment_id'),
            'total_runs': experiment_summary.get('total_runs', 0),
            'registered_models': experiment_summary.get('registered_models', 0),
            'check_timestamp': datetime.now().isoformat()
        }
        
        return create_response(
            data=health_info,
            message="MLflow server is healthy and accessible"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"MLflow server health check failed: {str(e)}"
        )
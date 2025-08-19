#!/usr/bin/env python3
"""
MLflow Setup and Management Script

This script helps initialize and manage MLflow for the football fouls analytics project.
"""

import os
import sys
import time
import subprocess
import requests
from pathlib import Path
import argparse
import logging
from typing import Dict, Any

# Add parent directory to path to import our modules
sys.path.append(str(Path(__file__).parent.parent))

from backend.app.ml.mlflow_config import get_mlflow_manager, initialize_mlflow_manager
from backend.app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_mlflow_server(url: str = "http://localhost:5000", timeout: int = 60) -> bool:
    """Check if MLflow server is running and accessible."""
    
    logger.info(f"Checking MLflow server at {url}...")
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                logger.info("MLflow server is running and accessible")
                return True
        except requests.exceptions.RequestException:
            pass
        
        time.sleep(2)
    
    logger.error(f"MLflow server not accessible after {timeout} seconds")
    return False

def start_mlflow_services():
    """Start MLflow services using Docker Compose."""
    
    logger.info("Starting MLflow services...")
    
    try:
        # Start services
        result = subprocess.run([
            "docker-compose", 
            "-f", "docker-compose.mlflow.yml", 
            "up", "-d"
        ], check=True, capture_output=True, text=True)
        
        logger.info("MLflow services started successfully")
        logger.info(result.stdout)
        
        # Wait for services to be ready
        if check_mlflow_server():
            logger.info("MLflow setup completed successfully!")
            logger.info("MLflow UI available at: http://localhost:5000")
            logger.info("MinIO Console available at: http://localhost:9001")
            return True
        else:
            logger.error("MLflow server failed to start properly")
            return False
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start MLflow services: {e}")
        logger.error(e.stderr)
        return False
    except FileNotFoundError:
        logger.error("Docker Compose not found. Please install Docker and Docker Compose.")
        return False

def stop_mlflow_services():
    """Stop MLflow services."""
    
    logger.info("Stopping MLflow services...")
    
    try:
        result = subprocess.run([
            "docker-compose", 
            "-f", "docker-compose.mlflow.yml", 
            "down"
        ], check=True, capture_output=True, text=True)
        
        logger.info("MLflow services stopped successfully")
        logger.info(result.stdout)
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to stop MLflow services: {e}")
        logger.error(e.stderr)
        return False

def restart_mlflow_services():
    """Restart MLflow services."""
    
    logger.info("Restarting MLflow services...")
    
    if stop_mlflow_services():
        time.sleep(5)  # Wait a bit before restarting
        return start_mlflow_services()
    
    return False

def check_mlflow_status():
    """Check status of MLflow services."""
    
    logger.info("Checking MLflow services status...")
    
    try:
        result = subprocess.run([
            "docker-compose", 
            "-f", "docker-compose.mlflow.yml", 
            "ps"
        ], check=True, capture_output=True, text=True)
        
        print("\nMLflow Services Status:")
        print("=" * 50)
        print(result.stdout)
        
        # Check if MLflow server is accessible
        if check_mlflow_server(timeout=10):
            print("\n✅ MLflow server is accessible at http://localhost:5000")
        else:
            print("\n❌ MLflow server is not accessible")
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to check MLflow services status: {e}")
        return False

def initialize_mlflow_experiment():
    """Initialize MLflow experiment and test connection."""
    
    logger.info("Initializing MLflow experiment...")
    
    try:
        # Initialize MLflow manager
        mlflow_manager = initialize_mlflow_manager(
            tracking_uri="http://localhost:5000",
            experiment_name="football-fouls-analytics"
        )
        
        # Get experiment summary
        summary = mlflow_manager.get_experiment_summary()
        
        logger.info("MLflow experiment initialized successfully")
        logger.info(f"Experiment: {summary.get('experiment_name')}")
        logger.info(f"Experiment ID: {summary.get('experiment_id')}")
        logger.info(f"Total runs: {summary.get('total_runs', 0)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize MLflow experiment: {e}")
        return False

def view_mlflow_logs():
    """View MLflow services logs."""
    
    logger.info("Viewing MLflow services logs...")
    
    try:
        subprocess.run([
            "docker-compose", 
            "-f", "docker-compose.mlflow.yml", 
            "logs", "-f"
        ], check=True)
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to view logs: {e}")
    except KeyboardInterrupt:
        logger.info("\nStopped viewing logs")

def cleanup_mlflow_data():
    """Clean up MLflow data (use with caution)."""
    
    logger.warning("This will remove all MLflow data including experiments, runs, and models!")
    confirm = input("Are you sure you want to continue? (yes/no): ")
    
    if confirm.lower() != 'yes':
        logger.info("Cleanup cancelled")
        return False
    
    logger.info("Cleaning up MLflow data...")
    
    try:
        # Stop services first
        stop_mlflow_services()
        
        # Remove volumes
        subprocess.run([
            "docker-compose", 
            "-f", "docker-compose.mlflow.yml", 
            "down", "-v"
        ], check=True, capture_output=True, text=True)
        
        logger.info("MLflow data cleaned up successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to cleanup MLflow data: {e}")
        return False

def setup_environment():
    """Setup environment variables for MLflow."""
    
    logger.info("Setting up environment variables...")
    
    env_vars = {
        'MLFLOW_TRACKING_URI': 'http://localhost:5000',
        'MLFLOW_S3_ENDPOINT_URL': 'http://localhost:9000',
        'AWS_ACCESS_KEY_ID': 'minioadmin',
        'AWS_SECRET_ACCESS_KEY': 'minioadmin123'
    }
    
    # Create .env file for MLflow
    env_file = Path('.env.mlflow')
    with open(env_file, 'w') as f:
        for key, value in env_vars.items():
            f.write(f"{key}={value}\n")
    
    logger.info(f"Environment variables saved to {env_file}")
    logger.info("To use these variables, run: source .env.mlflow")
    
    return True

def main():
    """Main function to handle command line arguments."""
    
    parser = argparse.ArgumentParser(
        description="MLflow Setup and Management Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/mlflow_setup.py start          # Start MLflow services
  python scripts/mlflow_setup.py stop           # Stop MLflow services
  python scripts/mlflow_setup.py status         # Check services status
  python scripts/mlflow_setup.py init           # Initialize experiment
  python scripts/mlflow_setup.py logs           # View logs
  python scripts/mlflow_setup.py cleanup        # Clean up all data
        """
    )
    
    parser.add_argument(
        'command',
        choices=['start', 'stop', 'restart', 'status', 'init', 'logs', 'cleanup', 'env'],
        help='Command to execute'
    )
    
    args = parser.parse_args()
    
    # Change to project root directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    success = False
    
    if args.command == 'start':
        success = start_mlflow_services()
        if success:
            setup_environment()
            initialize_mlflow_experiment()
    
    elif args.command == 'stop':
        success = stop_mlflow_services()
    
    elif args.command == 'restart':
        success = restart_mlflow_services()
        if success:
            initialize_mlflow_experiment()
    
    elif args.command == 'status':
        success = check_mlflow_status()
    
    elif args.command == 'init':
        success = initialize_mlflow_experiment()
    
    elif args.command == 'logs':
        view_mlflow_logs()
        success = True
    
    elif args.command == 'cleanup':
        success = cleanup_mlflow_data()
    
    elif args.command == 'env':
        success = setup_environment()
    
    if success:
        logger.info(f"Command '{args.command}' completed successfully")
        sys.exit(0)
    else:
        logger.error(f"Command '{args.command}' failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
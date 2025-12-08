"""Experiment tracking with MLflow integration."""
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("Warning: MLflow not available. Install with: pip install mlflow")

from config.config_loader import get_config


class ExperimentLogger:
    """Log experiments and results."""
    
    def __init__(self, experiment_name: Optional[str] = None):
        """
        Initialize experiment logger.
        
        Args:
            experiment_name: Name of the experiment
        """
        self.config = get_config()
        exp_config = self.config.get_experiment_config()
        self.log_directory = Path(exp_config.get('logging', {}).get('log_directory', 'experiments/logs'))
        self.log_directory.mkdir(parents=True, exist_ok=True)
        self.use_mlflow = exp_config.get('logging', {}).get('mlflow_integration', False) and MLFLOW_AVAILABLE
        
        if self.use_mlflow:
            if experiment_name:
                mlflow.set_experiment(experiment_name)
            self.mlflow_run = None
    
    def start_run(self, run_name: Optional[str] = None):
        """Start a new MLflow run."""
        if self.use_mlflow:
            self.mlflow_run = mlflow.start_run(run_name=run_name)
    
    def end_run(self):
        """End the current MLflow run."""
        if self.use_mlflow and self.mlflow_run:
            mlflow.end_run()
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters."""
        if self.use_mlflow:
            mlflow.log_params(params)
        
        # Also save to local file
        params_file = self.log_directory / f"params_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(params_file, 'w') as f:
            json.dump(params, f, indent=2, default=str)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics."""
        if self.use_mlflow:
            mlflow.log_metrics(metrics, step=step)
        
        # Also save to local file
        metrics_file = self.log_directory / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
    
    def log_artifact(self, file_path: str, artifact_path: Optional[str] = None):
        """Log an artifact."""
        if self.use_mlflow:
            mlflow.log_artifact(file_path, artifact_path)
    
    def log_dataframe(self, df: pd.DataFrame, name: str):
        """Log a DataFrame as CSV."""
        file_path = self.log_directory / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(file_path, index=True)
        
        if self.use_mlflow:
            mlflow.log_artifact(str(file_path), "dataframes")


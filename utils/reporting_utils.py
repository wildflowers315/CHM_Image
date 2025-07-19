import json
from pathlib import Path
import numpy as np
from typing import Dict

def save_metrics_and_importance(metrics: dict, importance_data: dict, output_dir: str) -> None:
    """
    Save training metrics and feature importance to JSON file, ensuring all values are JSON serializable.
    """
    # Convert any non-serializable values to Python native types
    serializable_metrics = {}
    for key, value in metrics.items():
        if hasattr(value, 'item'):  # Handle numpy/torch numbers
            serializable_metrics[key] = value.item()
        else:
            serializable_metrics[key] = float(value)
    
    serializable_importance = {}
    for key, value in importance_data.items():
        if hasattr(value, 'item'):  # Handle numpy/torch numbers
            serializable_importance[key] = value.item()
        else:
            serializable_importance[key] = float(value)
    
    # Combine metrics and importance data
    output_data = {
        "train_metrics": serializable_metrics,
        "feature_importance": serializable_importance
    }
    
    # Create output path
    output_path = Path(output_dir) / "model_evaluation.json"
    
    # Save to JSON
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=4)
    print(f"Saved model evaluation data to: {output_path}")

def save_training_metrics(train_metrics: Dict, importance_data: Dict, metrics_file: str):
    """
    Save training metrics and importance data to JSON file.
    """
    def convert_numpy_types(obj):
        """
        Convert numpy types to Python types for JSON serialization.
        """
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif hasattr(obj, 'item'):  # Handle torch/numpy scalars
            return obj.item()
        else:
            return obj
    
    output_data = {
        "training_metrics": convert_numpy_types(train_metrics),
        "feature_importance": convert_numpy_types(importance_data)
    }
    
    with open(metrics_file, 'w') as f:
        json.dump(output_data, f, indent=4)
    
    print(f"Saved training metrics to: {metrics_file}")

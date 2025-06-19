"""Base training infrastructure for all model types."""

import os
import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm


class BaseTrainer(ABC):
    """Abstract base class for all model trainers."""
    
    def __init__(self, 
                 model_type: str,
                 output_dir: str,
                 device: str = 'cpu',
                 logger: Optional['TrainingLogger'] = None):
        """
        Initialize base trainer.
        
        Args:
            model_type: Type of model (rf, mlp, 2d_unet, 3d_unet)
            output_dir: Directory to save outputs
            device: Device for training (cpu/cuda)
            logger: Training logger instance
        """
        self.model_type = model_type
        self.output_dir = Path(output_dir)
        self.device = device
        self.logger = logger
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.model = None
        self.training_metrics = {}
        self.best_metrics = {}
        
    @abstractmethod
    def load_data(self, data_source: Union[str, List[str]], **kwargs) -> Any:
        """Load training data from source."""
        pass
    
    @abstractmethod
    def prepare_data(self, data: Any, **kwargs) -> Tuple[Any, Any]:
        """Prepare data for training (features, targets)."""
        pass
    
    @abstractmethod
    def create_model(self, **kwargs) -> Any:
        """Create model instance."""
        pass
    
    @abstractmethod
    def train_model(self, features: Any, targets: Any, **kwargs) -> Dict[str, float]:
        """Train the model and return metrics."""
        pass
    
    @abstractmethod
    def predict(self, features: Any, **kwargs) -> np.ndarray:
        """Generate predictions."""
        pass
    
    @abstractmethod
    def save_model(self, filepath: str) -> None:
        """Save trained model."""
        pass
    
    @abstractmethod
    def load_model(self, filepath: str) -> None:
        """Load trained model."""
        pass
    
    def full_training_workflow(self, 
                             data_source: Union[str, List[str]],
                             **kwargs) -> Dict[str, Any]:
        """
        Execute complete training workflow.
        
        Args:
            data_source: Path to data or list of patch files
            **kwargs: Additional training parameters
            
        Returns:
            Training results and metrics
        """
        start_time = time.time()
        
        if self.logger:
            self.logger.log_info(f"Starting training workflow for {self.model_type}")
        
        # 1. Load data
        data = self.load_data(data_source, **kwargs)
        
        # 2. Prepare data
        features, targets = self.prepare_data(data, **kwargs)
        
        # 3. Create model
        self.model = self.create_model(**kwargs)
        
        # 4. Train model
        metrics = self.train_model(features, targets, **kwargs)
        
        # 5. Save results
        training_time = time.time() - start_time
        results = {
            'model_type': self.model_type,
            'training_time': training_time,
            'metrics': metrics,
            'output_dir': str(self.output_dir),
            'parameters': kwargs
        }
        
        # Save training results
        results_path = self.output_dir / 'training_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        if self.logger:
            self.logger.log_info(f"Training completed in {training_time:.2f}s")
            self.logger.save_log(self.output_dir / 'training.log')
        
        return results
    
    def evaluate_model(self, 
                      test_features: Any,
                      test_targets: Any,
                      **kwargs) -> Dict[str, float]:
        """
        Evaluate trained model on test data.
        
        Args:
            test_features: Test features
            test_targets: Test targets
            **kwargs: Additional evaluation parameters
            
        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        predictions = self.predict(test_features, **kwargs)
        
        # Calculate metrics (import from evaluate_predictions)
        from evaluate_predictions import calculate_metrics
        
        # Flatten arrays for metric calculation
        if hasattr(test_targets, 'flatten'):
            test_targets_flat = test_targets.flatten()
            predictions_flat = predictions.flatten()
        else:
            test_targets_flat = test_targets
            predictions_flat = predictions
        
        # Remove invalid values
        valid_mask = np.isfinite(test_targets_flat) & np.isfinite(predictions_flat)
        valid_mask &= (test_targets_flat > 0) & (predictions_flat > 0)
        
        if valid_mask.sum() == 0:
            return {'error': 'No valid predictions'}
        
        metrics = calculate_metrics(
            test_targets_flat[valid_mask],
            predictions_flat[valid_mask]
        )
        
        return metrics
"""Random Forest trainer for sparse GEDI pixel training."""

import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from typing import Any, Dict, List, Tuple, Union

from ..core.base_trainer import BaseTrainer
from ..data.datasets import SparseGEDIDataset


class RandomForestTrainer(BaseTrainer):
    """Random Forest trainer for canopy height prediction."""
    
    def __init__(self, 
                 output_dir: str,
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 random_state: int = 42,
                 n_jobs: int = -1,
                 logger=None):
        """
        Initialize Random Forest trainer.
        
        Args:
            output_dir: Directory to save outputs
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            random_state: Random seed
            n_jobs: Number of parallel jobs
            logger: Training logger
        """
        super().__init__('rf', output_dir, 'cpu', logger)
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.n_jobs = n_jobs
        
    def load_data(self, data_source: Union[str, List[str]], **kwargs) -> SparseGEDIDataset:
        """Load patch files and create sparse GEDI dataset."""
        if isinstance(data_source, str):
            # Single patch file
            patch_files = [data_source]
        else:
            # List of patch files
            patch_files = data_source
        
        min_gedi_pixels = kwargs.get('min_gedi_pixels', 10)
        
        if self.logger:
            self.logger.log_info(f"Loading sparse GEDI data from {len(patch_files)} patches")
        
        dataset = SparseGEDIDataset(patch_files, min_gedi_pixels)
        
        if self.logger:
            self.logger.log_info(f"Found {len(dataset)} valid GEDI samples")
        
        return dataset
    
    def prepare_data(self, data: SparseGEDIDataset, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features and targets from dataset."""
        features, targets = data.get_features_and_targets()
        
        if self.logger:
            self.logger.log_info(f"Features shape: {features.shape}")
            self.logger.log_info(f"Targets shape: {targets.shape}")
            self.logger.log_info(f"Target range: {targets.min():.2f} to {targets.max():.2f}")
        
        return features, targets
    
    def create_model(self, **kwargs) -> RandomForestRegressor:
        """Create Random Forest model."""
        model_params = {
            'n_estimators': kwargs.get('n_estimators', self.n_estimators),
            'max_depth': kwargs.get('max_depth', self.max_depth),
            'random_state': self.random_state,
            'n_jobs': self.n_jobs
        }
        
        if self.logger:
            self.logger.log_info(f"Creating Random Forest with params: {model_params}")
        
        return RandomForestRegressor(**model_params)
    
    def train_model(self, 
                   features: np.ndarray, 
                   targets: np.ndarray, 
                   **kwargs) -> Dict[str, float]:
        """Train Random Forest model."""
        # Split data
        test_size = kwargs.get('test_size', 0.2)
        X_train, X_test, y_train, y_test = train_test_split(
            features, targets, 
            test_size=test_size, 
            random_state=self.random_state
        )
        
        if self.logger:
            self.logger.log_info(f"Training on {len(X_train)} samples, testing on {len(X_test)}")
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        # Get feature importance
        self.feature_importance = self.model.feature_importances_
        
        # Generate predictions for metrics
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        # Calculate detailed metrics
        from evaluate_predictions import calculate_metrics
        
        train_metrics = calculate_metrics(y_train, train_pred)
        test_metrics = calculate_metrics(y_test, test_pred)
        
        metrics = {
            'train_r2': train_score,
            'test_r2': test_score,
            'train_rmse': train_metrics.get('rmse', 0),
            'test_rmse': test_metrics.get('rmse', 0),
            'train_mae': train_metrics.get('mae', 0),
            'test_mae': test_metrics.get('mae', 0),
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test),
            'n_features': features.shape[1]
        }
        
        if self.logger:
            self.logger.log_info(f"Training R²: {train_score:.4f}")
            self.logger.log_info(f"Test R²: {test_score:.4f}")
            self.logger.log_info(f"Test RMSE: {metrics['test_rmse']:.4f}")
        
        return metrics
    
    def predict(self, features: np.ndarray, **kwargs) -> np.ndarray:
        """Generate predictions using trained model."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        return self.model.predict(features)
    
    def save_model(self, filepath: str) -> None:
        """Save trained Random Forest model."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        joblib.dump(self.model, filepath)
        
        if self.logger:
            self.logger.log_info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load trained Random Forest model."""
        self.model = joblib.load(filepath)
        
        if self.logger:
            self.logger.log_info(f"Model loaded from {filepath}")
    
    def get_feature_importance(self, feature_names: List[str] = None) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Args:
            feature_names: Optional list of feature names
            
        Returns:
            Dictionary of feature importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        importance = self.model.feature_importances_
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importance))]
        
        return dict(zip(feature_names, importance))
    
    def predict_patch(self, patch_file: str, **kwargs) -> np.ndarray:
        """
        Generate predictions for a full patch.
        
        Args:
            patch_file: Path to patch file
            **kwargs: Additional parameters
            
        Returns:
            Prediction array with same spatial dimensions as patch
        """
        import rasterio
        
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        with rasterio.open(patch_file) as src:
            data = src.read()
            height, width = src.height, src.width
            
            # Separate features and target
            features = data[:-1]  # All bands except last
            
            # Reshape for prediction
            features_flat = features.reshape(features.shape[0], -1).T
            
            # Generate predictions
            predictions_flat = self.model.predict(features_flat)
            
            # Reshape back to spatial dimensions
            predictions = predictions_flat.reshape(height, width)
        
        return predictions
"""Unified training workflow that handles all model types."""

import os
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..core.utils import validate_training_inputs, create_output_directory, get_model_info
from ..core.callbacks import TrainingLogger


class UnifiedTrainer:
    """
    Unified trainer that automatically selects appropriate trainer based on model type.
    
    This replaces the monolithic train_predict_map.py with a clean, modular interface.
    """
    
    def __init__(self, 
                 model_type: str,
                 output_dir: str = "chm_outputs",
                 device: str = "auto",
                 logger: Optional[TrainingLogger] = None):
        """
        Initialize unified trainer.
        
        Args:
            model_type: Type of model ('rf', 'mlp', '2d_unet', '3d_unet')
            output_dir: Base output directory  
            device: Device for training ('cpu', 'cuda', 'auto')
            logger: Optional training logger
        """
        self.model_type = model_type.lower()
        self.base_output_dir = Path(output_dir)
        self.device = self._determine_device(device)
        self.logger = logger or TrainingLogger()
        
        # Create model-specific output directory
        self.output_dir = create_output_directory(output_dir, self.model_type)
        
        # Initialize model-specific trainer
        self.trainer = self._create_trainer()
        
        self.logger.log_info(f"Initialized {self.model_type} trainer")
        self.logger.log_info(f"Output directory: {self.output_dir}")
        self.logger.log_info(f"Device: {self.device}")
    
    def _determine_device(self, device: str) -> str:
        """Determine the appropriate device for training."""
        if device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return device
    
    def _create_trainer(self):
        """Create appropriate trainer based on model type."""
        trainer_kwargs = {
            'output_dir': str(self.output_dir),
            'logger': self.logger
        }
        
        if self.model_type == 'rf':
            from ..models.rf_trainer import RandomForestTrainer
            return RandomForestTrainer(**trainer_kwargs)
        elif self.model_type == 'mlp':
            from ..models.mlp_trainer import MLPTrainer
            return MLPTrainer(device=self.device, **trainer_kwargs)
        elif self.model_type == '2d_unet':
            from ..models.unet_2d_trainer import UNet2DTrainer
            return UNet2DTrainer(device=self.device, **trainer_kwargs)
        elif self.model_type == '3d_unet':
            from ..models.unet_3d_trainer import UNet3DTrainer
            return UNet3DTrainer(device=self.device, **trainer_kwargs)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def train(self, 
              data_source: str,
              **kwargs) -> Dict[str, Any]:
        """
        Execute complete training workflow.
        
        Args:
            data_source: Path to patch file or directory containing patches
            **kwargs: Training parameters (augment, validation_split, etc.)
            
        Returns:
            Training results dictionary
        """
        # Resolve patch files
        patch_files = self._resolve_patch_files(data_source)
        
        # Validate inputs
        validate_training_inputs(patch_files, self.model_type, **kwargs)
        
        # Log training start
        self.logger.log_info(f"Starting training with {len(patch_files)} patches")
        
        # Execute training workflow
        results = self.trainer.full_training_workflow(patch_files, **kwargs)
        
        return results
    
    def _resolve_patch_files(self, data_source: str) -> List[str]:
        """Resolve patch files from data source."""
        data_path = Path(data_source)
        
        if data_path.is_file():
            # Single patch file
            return [str(data_path)]
        elif data_path.is_dir():
            # Directory containing patches
            patch_files = []
            for pattern in ['*.tif', '*.tiff']:
                patch_files.extend(glob.glob(str(data_path / pattern)))
            
            if not patch_files:
                raise ValueError(f"No patch files found in {data_path}")
            
            return sorted(patch_files)
        else:
            raise ValueError(f"Data source not found: {data_source}")
    
    def predict(self, 
                patch_files: List[str],
                output_dir: Optional[str] = None,
                **kwargs) -> List[str]:
        """
        Generate predictions for patch files.
        
        Args:
            patch_files: List of patch file paths
            output_dir: Output directory for predictions
            **kwargs: Prediction parameters
            
        Returns:
            List of prediction file paths
        """
        if output_dir is None:
            output_dir = self.output_dir / "predictions"
        
        # This would be implemented in each specific trainer
        return self.trainer.predict_patches(patch_files, output_dir, **kwargs)
    
    def evaluate(self, 
                 test_data: str,
                 **kwargs) -> Dict[str, float]:
        """
        Evaluate trained model on test data.
        
        Args:
            test_data: Path to test patch file(s)
            **kwargs: Evaluation parameters
            
        Returns:
            Evaluation metrics
        """
        test_files = self._resolve_patch_files(test_data)
        return self.trainer.evaluate_patches(test_files, **kwargs)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model and training setup."""
        info = get_model_info(self.model_type)
        info.update({
            'output_dir': str(self.output_dir),
            'device': self.device,
            'trainer_class': type(self.trainer).__name__
        })
        return info
import os
import time
import json
import numpy as np  # type: ignore
import torch  # type: ignore
import torch.nn as nn  # type: ignore
from torch.utils.data import DataLoader  # type: ignore
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from tqdm import tqdm

from data.patch_loader import load_patch_data
from data.augmentation import AugmentedPatchDataset
from models.height_2d_unet import Height2DUNet
from training.core.callbacks import EarlyStoppingCallback2, TrainingLogger2

class EnhancedUNetTrainer:
    """
    Enhanced U-Net training with proper multi-patch batch processing.
    
    Features:
    - True multi-patch training (not just first patch)
    - Configurable batch size with gradient accumulation
    - Cross-patch validation strategy
    - Memory-efficient data loading
    - Early stopping and checkpointing
    """
    
    def __init__(self, model_type: str = "2d_unet", device: str = "auto"):
        """Initialize enhanced U-Net trainer."""
        self.model_type = model_type
        
        if device == "auto":
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        print(f"🚀 Enhanced U-Net Trainer initialized for {model_type} on {self.device}")
        
    def create_data_loaders(self, patch_files: List[str], 
                          validation_split: float = 0.2,
                          batch_size: int = 8,
                          augment: bool = True,
                          num_workers: int = 2,
                          supervision_mode: str = "gedi_only",
                          band_selection: str = "all") -> Tuple[DataLoader, DataLoader]:
        """
        Create train/validation data loaders with augmentation.
        
        Features:
        - Cross-patch validation (patches split between train/val)
        - Augmented training data (12x increase)
        - Memory-efficient streaming
        - Balanced GEDI pixel sampling
        
        Args:
            patch_files: List of patch file paths
            validation_split: Fraction of patches for validation
            batch_size: Batch size for training
            augment: Enable data augmentation
            num_workers: Number of data loading workers
            supervision_mode: Supervision mode for patch loading
            band_selection: Band selection for patch loading
        
        Returns:
            Training and validation DataLoaders
        """
        # Split patches spatially for validation
        np.random.shuffle(patch_files)  # Randomize order
        split_idx = int(len(patch_files) * (1 - validation_split))
        
        train_files = patch_files[:split_idx]
        val_files = patch_files[split_idx:]
        
        print(f"📊 Data split: {len(train_files)} train patches, {len(val_files)} validation patches")
        
        # Create datasets
        train_dataset = AugmentedPatchDataset(
            train_files, 
            augment=augment, 
            validation_mode=False,
            supervision_mode=supervision_mode,
            band_selection=band_selection
        )
        
        val_dataset = AugmentedPatchDataset(
            val_files, 
            augment=False,  # No augmentation for validation
            validation_mode=True,
            supervision_mode=supervision_mode,
            band_selection=band_selection
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
            drop_last=True  # Ensure consistent batch sizes
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
            drop_last=False
        )
        
        print(f"📈 Training: {len(train_loader)} batches of {batch_size}")
        print(f"📊 Validation: {len(val_loader)} batches of {batch_size}")
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader, model: nn.Module, 
                   optimizer: torch.optim.Optimizer, criterion: nn.Module,
                   epoch: int, logger: Optional[TrainingLogger2] = None) -> float:
        """Train single epoch with proper batch processing."""
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        epoch_start_time = time.time()
        
        for batch_idx, (features, targets, masks) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            # Move to device
            features = features.to(self.device)
            targets = targets.to(self.device)
            masks = masks.to(self.device)
            
            # Handle different input shapes for 2D
            if self.model_type == "2d_unet":
                # For 2D U-Net: handle both (B, C, H, W) and (B, C, T, H, W)
                if len(features.shape) == 5:  # (B, C, T, H, W) -> (B, C*T, H, W)
                    B, C, T, H, W = features.shape
                    features = features.view(B, C * T, H, W)
                    

            # Forward pass
            optimizer.zero_grad()
            predictions = model(features)
            
            # Ensure predictions match target dimensions
            if len(predictions.shape) > len(targets.shape):
                predictions = predictions.squeeze(1)  # Remove channel dimension if present
                
            # Compute loss only on valid GEDI pixels
            valid_loss = self._compute_masked_loss(predictions, targets, masks, criterion)
            
            # Backward pass
            valid_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Update metrics
            loss_value = valid_loss.item()
            total_loss += loss_value
            num_batches += 1
            
            # Log batch metrics
            if logger:
                logger.log_batch(epoch, batch_idx, loss_value, features.size(0))
                
            # Memory cleanup
            del features, targets, masks, predictions, valid_loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        epoch_time = time.time() - epoch_start_time
        
        return avg_loss
    
    def validate_epoch(self, val_loader: DataLoader, model: nn.Module, 
                      criterion: nn.Module) -> float:
        """Validate model performance on validation set."""
        model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for features, targets, masks in val_loader:
                # Move to device
                features = features.to(self.device)
                targets = targets.to(self.device)
                masks = masks.to(self.device)
                
                # Handle different input shapes
                if self.model_type == "2d_unet":
                    if len(features.shape) == 5:  # (B, C, T, H, W) -> (B, C*T, H, W)
                        B, C, T, H, W = features.shape
                        features = features.view(B, C * T, H, W)
                        
                        
                # Forward pass
                predictions = model(features)
                
                # Ensure predictions match target dimensions
                if len(predictions.shape) > len(targets.shape):
                    predictions = predictions.squeeze(1)
                    
                # Compute loss
                valid_loss = self._compute_masked_loss(predictions, targets, masks, criterion)
                total_loss += valid_loss.item()
                num_batches += 1
                
                # Memory cleanup
                del features, targets, masks, predictions, valid_loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def _compute_masked_loss(self, predictions: torch.Tensor, targets: torch.Tensor, 
                           masks: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
        """Compute loss only on valid GEDI pixels."""
        # Apply mask to get valid pixels
        valid_predictions = predictions[masks]
        valid_targets = targets[masks]
        
        if len(valid_predictions) == 0:
            # No valid pixels, return zero loss
            return torch.tensor(0.0, requires_grad=True, device=self.device)
        
        # Compute loss
        loss = criterion(valid_predictions, valid_targets)
        return loss
    
    def train_multi_patch_unet(self, patch_files: List[str], 
                             output_dir: str, 
                             epochs: int = 100,
                             batch_size: int = 8,
                             learning_rate: float = 1e-3,
                             weight_decay: float = 1e-4,
                             validation_split: float = 0.2,
                             early_stopping_patience: int = 15,
                             augment: bool = True,
                             save_checkpoints: bool = True,
                             checkpoint_freq: int = 10,
                             supervision_mode: str = "gedi_only",
                             band_selection: str = "all") -> Dict:
        """
        Complete multi-patch U-Net training workflow.
        
        Args:
            patch_files: List of patch file paths
            output_dir: Output directory for results
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            validation_split: Fraction for validation
            early_stopping_patience: Epochs to wait before stopping
            augment: Enable data augmentation
            save_checkpoints: Save periodic checkpoints
            checkpoint_freq: Frequency of checkpoint saving
            supervision_mode: Supervision mode for patch loading
            band_selection: Band selection for patch loading
        
        Returns:
            Training results and model artifacts
        """
        # Create output directories
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        models_dir = output_path / 'models'
        checkpoints_dir = output_path / 'checkpoints'
        models_dir.mkdir(exist_ok=True)
        checkpoints_dir.mkdir(exist_ok=True)
        
        # Initialize training components
        logger = TrainingLogger2(output_dir)
        early_stopping = EarlyStoppingCallback2(
            patience=early_stopping_patience,
            checkpoint_dir=str(checkpoints_dir)
        )
        
        # Create data loaders
        train_loader, val_loader = self.create_data_loaders(
            patch_files,
            validation_split=validation_split,
            batch_size=batch_size,
            augment=augment,
            supervision_mode=supervision_mode,
            band_selection=band_selection
        )
        
        # Get sample to determine input dimensions
        sample_features, _, _ = next(iter(train_loader))
        
        if self.model_type == "2d_unet" or self.model_type == "shift_aware_unet":
            # For 2D U-Net and ShiftAwareUNet: handle both (B, C, H, W) and (B, C, T, H, W)
            if len(sample_features.shape) == 5:  # (B, C, T, H, W) -> (B, C*T, H, W)
                in_channels = sample_features.shape[1] * sample_features.shape[2]
            else:  # (B, C, H, W)
                in_channels = sample_features.shape[1]
            
            if self.model_type == "2d_unet":
                model = Height2DUNet(in_channels=in_channels)
            elif self.model_type == "shift_aware_unet":
                from models.trainers.shift_aware_trainer import ShiftAwareUNet
                model = ShiftAwareUNet(in_channels=in_channels) # Assuming ShiftAwareUNet takes in_channels

            
        model = model.to(self.device)
        
        # Initialize optimizer and criterion
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        criterion = nn.MSELoss()
        
        # Training loop
        logger.start_training()
        
        print(f"🚀 Starting enhanced {self.model_type.upper()} training...")
        print(f"📊 Model: {in_channels} input channels")
        print(f"📈 Training samples: {len(train_loader) * batch_size}")
        print(f"🎯 Validation samples: {len(val_loader) * batch_size}")
        
        best_val_loss = float('inf')
        training_results = {
            'epochs_completed': 0,
            'best_val_loss': float('inf'),
            'final_train_loss': 0.0,
            'early_stopped': False,
            'total_training_time': 0.0
        }
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Training phase
            train_loss = self.train_epoch(
                train_loader, model, optimizer, criterion, epoch, logger
            )
            
            # Validation phase
            val_loss = self.validate_epoch(val_loader, model, criterion)
            
            # Update metrics
            epoch_time = time.time() - epoch_start_time
            current_lr = optimizer.param_groups[0]['lr']
            
            # Log epoch metrics
            logger.log_epoch(epoch, train_loss, val_loss, current_lr, epoch_time)
            
            # Check early stopping
            should_stop = early_stopping(epoch, val_loss, model, optimizer)
            
            # Save periodic checkpoints
            if save_checkpoints and (epoch + 1) % checkpoint_freq == 0:
                checkpoint_path = checkpoints_dir / f'epoch_{epoch+1:03d}.pth'
                early_stopping.save_checkpoint(model, optimizer, epoch, str(checkpoint_path))
                
            # Update best validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                
            # Only update total_training_time if logger.start_time is not None
            total_training_time = time.time() - logger.start_time if logger and logger.start_time is not None else 0.0
            training_results.update({
                'epochs_completed': epoch + 1,
                'best_val_loss': best_val_loss,
                'final_train_loss': train_loss,
                'total_training_time': total_training_time
            })
            
            if should_stop:
                training_results['early_stopped'] = True
                break
        
        # Save final model
        final_model_path = models_dir / 'final_model.pth'
        torch.save(model.state_dict(), final_model_path)
        
        # Generate training visualizations and reports
        logger.generate_loss_curves()
        metrics_path = logger.export_metrics()
        
        # Save training configuration
        config = {
            'model_type': self.model_type,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'validation_split': validation_split,
            'early_stopping_patience': early_stopping_patience,
            'augmentation_enabled': augment,
            'input_channels': in_channels,
            'device': str(self.device)
        }
        
        config_path = output_path / 'training_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Enhanced {self.model_type.upper()} training completed!")
        print(f"📈 Best validation loss: {best_val_loss:.6f}")
        print(f"💾 Models saved to: {models_dir}")
        print(f"📊 Logs saved to: {logger.logs_dir}")
        
        return training_results

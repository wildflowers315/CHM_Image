import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
try:
    import torch
    import torch.optim as optim
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. 3D U-Net will not be functional.")

from dl_models import MLPRegressionModel, create_normalized_dataloader
import rasterio
from rasterio.mask import geometry_mask
from shapely.geometry import Point, box
from shapely.ops import transform
import geopandas as gpd
import os
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import warnings
import argparse
from tqdm import tqdm
import glob
import json
import joblib
import time
warnings.filterwarnings('ignore')

# Import multi-patch functionality
from data.multi_patch import (
    PatchInfo, PatchRegistry, PredictionMerger,
    load_multi_patch_gedi_data, generate_multi_patch_summary,
    count_gedi_samples_per_patch
)

# Import enhanced spatial merger
try:
    from utils.spatial_utils import EnhancedSpatialMerger
    USE_ENHANCED_MERGER = True
except ImportError:
    USE_ENHANCED_MERGER = False
    print("Warning: Enhanced spatial merger not available, using default merger")

from evaluate_predictions import calculate_metrics
try:
    # Try importing 3D U-Net directly
    exec(open('models/3d_unet.py').read())
except (ImportError, FileNotFoundError):
    print("Warning: 3D U-Net model not found. Creating placeholder functions.")
    def Height3DUNet(*args, **kwargs):
        raise ImportError("3D U-Net not available")
    def create_3d_unet(*args, **kwargs):
        raise ImportError("3D U-Net not available")

# 2D U-Net Model for non-temporal data
class Height2DUNet(nn.Module):
    """2D U-Net for canopy height prediction from non-temporal patches."""
    
    def __init__(self, in_channels, n_classes=1, base_channels=64):
        super().__init__()
        
        # Encoder
        self.encoder1 = self.conv_block(in_channels, base_channels)
        self.encoder2 = self.conv_block(base_channels, base_channels * 2)
        self.encoder3 = self.conv_block(base_channels * 2, base_channels * 4)
        self.encoder4 = self.conv_block(base_channels * 4, base_channels * 8)
        
        # Bottleneck
        self.bottleneck = self.conv_block(base_channels * 8, base_channels * 16)
        
        # Decoder
        self.decoder4 = self.upconv_block(base_channels * 16, base_channels * 8)
        self.decoder3 = self.upconv_block(base_channels * 16, base_channels * 4)  # 16 = 8 + 8 from skip
        self.decoder2 = self.upconv_block(base_channels * 8, base_channels * 2)   # 8 = 4 + 4 from skip
        self.decoder1 = self.upconv_block(base_channels * 4, base_channels)       # 4 = 2 + 2 from skip
        
        # Final prediction
        self.final_conv = nn.Conv2d(base_channels, n_classes, kernel_size=1)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)  # (B, 64, H, W)
        e2 = self.encoder2(nn.MaxPool2d(2)(e1))  # (B, 128, H/2, W/2)
        e3 = self.encoder3(nn.MaxPool2d(2)(e2))  # (B, 256, H/4, W/4)
        e4 = self.encoder4(nn.MaxPool2d(2)(e3))  # (B, 512, H/8, W/8)
        
        # Bottleneck
        b = self.bottleneck(nn.MaxPool2d(2)(e4))  # (B, 1024, H/16, W/16)
        
        # Decoder with skip connections
        d4 = self.decoder4(b)  # (B, 512, H/8, W/8)
        d4 = torch.cat([d4, e4], dim=1)  # (B, 1024, H/8, W/8)
        
        d3 = self.decoder3(d4)  # (B, 256, H/4, W/4)
        d3 = torch.cat([d3, e3], dim=1)  # (B, 512, H/4, W/4)
        
        d2 = self.decoder2(d3)  # (B, 128, H/2, W/2)
        d2 = torch.cat([d2, e2], dim=1)  # (B, 256, H/2, W/2)
        
        d1 = self.decoder1(d2)  # (B, 64, H, W)
        
        # Final prediction
        out = self.final_conv(d1)  # (B, 1, H, W)
        
        return out.squeeze(1)  # (B, H, W)

def create_2d_unet(in_channels: int, n_classes: int = 1, base_channels: int = 64):
    """Create 2D U-Net model."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for 2D U-Net")
    return Height2DUNet(in_channels, n_classes, base_channels)
from data.normalization import (
    normalize_sentinel1, normalize_sentinel2, normalize_srtm_elevation,
    normalize_srtm_slope, normalize_srtm_aspect, normalize_alos2_dn,
    normalize_canopy_height, normalize_ndvi
)

def modified_huber_loss(pred: torch.Tensor, target: torch.Tensor, 
                       mask: Optional[torch.Tensor] = None, 
                       delta: float = 1.0, shift_radius: int = 1) -> torch.Tensor:
    """
    Modified Huber loss for 3D patches with shift awareness for sparse GEDI data.
    
    Args:
        pred: Predicted values [batch, height, width]
        target: Target values [batch, height, width] (sparse GEDI)
        mask: Valid pixel mask [batch, height, width]
        delta: Huber loss threshold
        shift_radius: Radius for spatial shift compensation
        
    Returns:
        Loss value
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for modified Huber loss")
    
    def huber_loss(x, y, delta=1.0):
        diff = x - y
        abs_diff = diff.abs()
        quadratic = torch.min(abs_diff, torch.tensor(delta, device=x.device))
        linear = abs_diff - quadratic
        return 0.5 * quadratic.pow(2) + delta * linear
    
    def generate_shifts(radius):
        """Generate all possible shifts within given radius"""
        shifts = [(0, 0)]  # Always include no shift
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                # Only include shifts within radius (using Euclidean distance)
                if (dx*dx + dy*dy) <= radius*radius:
                    shifts.append((dx, dy))
        return shifts
    
    # Generate shifts based on radius
    shifts = generate_shifts(shift_radius)
    
    best_loss = float('inf')
    for dx, dy in shifts:
        # Shift target
        shifted_target = torch.roll(target, shifts=(dx, dy), dims=(1, 2))
        
        # Compute loss only on valid GEDI pixels
        if mask is not None:
            shifted_mask = torch.roll(mask, shifts=(dx, dy), dims=(1, 2))
            valid_pixels = shifted_mask & (shifted_target > 0)  # GEDI pixels
        else:
            valid_pixels = shifted_target > 0
        
        if valid_pixels.sum() > 0:
            loss = huber_loss(
                pred[valid_pixels], 
                shifted_target[valid_pixels], 
                delta
            ).mean()
            best_loss = min(best_loss, loss.item())
    
    return torch.tensor(best_loss, requires_grad=True)

# ============================================================================
# Enhanced Training Components: Data Augmentation and Training Infrastructure
# ============================================================================

class AugmentedPatchDataset(Dataset):
    """
    PyTorch dataset with comprehensive spatial augmentation for canopy height modeling.
    
    Features:
    - 12x augmentation: 3 flips × 4 rotations per patch
    - Geospatial consistency: Apply same transforms to features and GEDI targets
    - Memory efficient: On-the-fly augmentation generation
    - Configurable augmentation factor
    """
    
    def __init__(self, patch_files: List[str], augment: bool = True, 
                 augment_factor: int = 12, validation_mode: bool = False):
        """
        Initialize augmented dataset.
        
        Args:
            patch_files: List of patch TIF file paths
            augment: Enable spatial augmentation (default: True)
            augment_factor: Number of augmentations per patch (default: 12)
            validation_mode: If True, only use original patches (no augmentation)
        """
        self.patch_files = patch_files
        self.augment = augment and not validation_mode
        self.augment_factor = augment_factor if self.augment else 1
        self.validation_mode = validation_mode
        
        # Pre-load patch info for memory estimation
        self.patch_info = []
        for patch_file in patch_files:
            if os.path.exists(patch_file):
                self.patch_info.append({
                    'file': patch_file,
                    'name': os.path.basename(patch_file)
                })
        
        print(f"🎯 Dataset initialized: {len(self.patch_info)} patches × {self.augment_factor} augmentations = {len(self)} samples")
        
    def __len__(self):
        return len(self.patch_info) * self.augment_factor
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get augmented patch data.
        
        Returns:
            - features: Augmented feature tensor (C, H, W) or (C, T, H, W)
            - target: Augmented GEDI target tensor (H, W)  
            - mask: Valid pixel mask (H, W)
        """
        # Determine patch and augmentation
        patch_idx = idx // self.augment_factor
        augment_id = idx % self.augment_factor
        
        patch_info = self.patch_info[patch_idx]
        
        # Load patch data
        features, target, _ = load_patch_data(patch_info['file'], normalize_bands=True)
        
        # Apply augmentation if enabled
        if self.augment and augment_id > 0:
            features, target = self.apply_spatial_augmentation(features, target, augment_id)
        
        # Create mask for valid GEDI pixels
        mask = (target > 0) & np.isfinite(target)
        
        # Ensure 256x256 dimensions for U-Net compatibility
        features, target, mask = self._ensure_256x256(features, target, mask)
        
        # Convert to tensors with positive strides (fix for negative stride error)
        features_tensor = torch.FloatTensor(features.copy())
        target_tensor = torch.FloatTensor(target.copy())
        mask_tensor = torch.BoolTensor(mask.copy())
        
        return features_tensor, target_tensor, mask_tensor
    
    def apply_spatial_augmentation(self, features: np.ndarray, target: np.ndarray, 
                                 augment_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply consistent spatial augmentation to features and target.
        
        Augmentation combinations:
        - ID 0: No augmentation (original)
        - ID 1-3: Horizontal, vertical, both flips  
        - ID 4-15: Above + 90°, 180°, 270° rotations
        
        Args:
            features: Feature array (C, H, W) or (C, T, H, W)
            target: Target array (H, W)
            augment_id: Augmentation identifier (0-11)
            
        Returns:
            Augmented features and target
        """
        if augment_id == 0:
            return features, target
        
        # Determine flip and rotation
        flip_id = (augment_id - 1) % 3 + 1  # 1, 2, 3
        rotation_id = (augment_id - 1) // 3  # 0, 1, 2, 3
        
        # Apply flips using np.flip to avoid negative strides
        if flip_id == 1:  # Horizontal flip
            if len(features.shape) == 3:  # (C, H, W)
                features = np.flip(features, axis=2).copy()
            else:  # (C, T, H, W)
                features = np.flip(features, axis=3).copy()
            target = np.flip(target, axis=1).copy()
        elif flip_id == 2:  # Vertical flip
            if len(features.shape) == 3:  # (C, H, W)
                features = np.flip(features, axis=1).copy()
            else:  # (C, T, H, W)
                features = np.flip(features, axis=2).copy()
            target = np.flip(target, axis=0).copy()
        elif flip_id == 3:  # Both flips
            if len(features.shape) == 3:  # (C, H, W)
                features = np.flip(features, axis=(1, 2)).copy()
            else:  # (C, T, H, W)
                features = np.flip(features, axis=(2, 3)).copy()
            target = np.flip(target, axis=(0, 1)).copy()
        
        # Apply rotations (k * 90 degrees)
        if rotation_id > 0:
            for _ in range(rotation_id):
                if len(features.shape) == 3:  # (C, H, W)
                    features = np.rot90(features, axes=(1, 2)).copy()
                else:  # (C, T, H, W)
                    features = np.rot90(features, axes=(2, 3)).copy()
                target = np.rot90(target).copy()
        
        return features, target
    
    def _ensure_256x256(self, features: np.ndarray, target: np.ndarray, 
                       mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Ensure arrays are exactly 256x256 for U-Net compatibility."""
        from scipy.ndimage import zoom
        
        target_size = 256
        
        if len(features.shape) == 3:  # (C, H, W)
            if features.shape[1] != target_size or features.shape[2] != target_size:
                scale_h = target_size / features.shape[1]
                scale_w = target_size / features.shape[2]
                
                # Resize features
                resized_features = np.zeros((features.shape[0], target_size, target_size), dtype=features.dtype)
                for i in range(features.shape[0]):
                    resized_features[i] = zoom(features[i], (scale_h, scale_w), order=1)
                features = resized_features
                
                # Resize target and mask
                target = zoom(target, (scale_h, scale_w), order=0)
                mask = zoom(mask.astype(np.float32), (scale_h, scale_w), order=0) > 0.5
                
        elif len(features.shape) == 4:  # (C, T, H, W)
            if features.shape[2] != target_size or features.shape[3] != target_size:
                scale_h = target_size / features.shape[2]
                scale_w = target_size / features.shape[3]
                
                # Resize features
                resized_features = np.zeros((features.shape[0], features.shape[1], target_size, target_size), dtype=features.dtype)
                for i in range(features.shape[0]):
                    for j in range(features.shape[1]):
                        resized_features[i, j] = zoom(features[i, j], (scale_h, scale_w), order=1)
                features = resized_features
                
                # Resize target and mask
                target = zoom(target, (scale_h, scale_w), order=0)
                mask = zoom(mask.astype(np.float32), (scale_h, scale_w), order=0) > 0.5
        
        return features, target, mask

class EarlyStoppingCallback:
    """
    Patience-based early stopping with best model preservation.
    
    Features:
    - Configurable patience (default: 15 epochs)
    - Best validation loss tracking
    - Automatic model checkpoint saving
    - Learning rate scheduling integration
    """
    
    def __init__(self, patience: int = 15, min_delta: float = 1e-4, 
                 restore_best_weights: bool = True, checkpoint_dir: str = None):
        """
        Initialize early stopping callback.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Restore best model weights on stop
            checkpoint_dir: Directory to save checkpoints
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.checkpoint_dir = checkpoint_dir
        
        self.best_loss = float('inf')
        self.best_weights = None
        self.epochs_without_improvement = 0
        self.best_epoch = 0
        
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            
    def __call__(self, epoch: int, val_loss: float, model: nn.Module, 
                 optimizer: torch.optim.Optimizer = None) -> bool:
        """
        Check early stopping criteria.
        
        Args:
            epoch: Current epoch number
            val_loss: Current validation loss
            model: Model to potentially save
            optimizer: Optimizer state to save
            
        Returns:
            True if training should stop, False otherwise
        """
        improved = val_loss < (self.best_loss - self.min_delta)
        
        if improved:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
            
            # Save best weights
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
            # Save checkpoint
            if self.checkpoint_dir:
                checkpoint_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
                self.save_checkpoint(model, optimizer, epoch, checkpoint_path, is_best=True)
                
            print(f"🎯 New best validation loss: {val_loss:.6f} (epoch {epoch})")
            
        else:
            self.epochs_without_improvement += 1
            
        # Check if we should stop
        should_stop = self.epochs_without_improvement >= self.patience
        
        if should_stop:
            print(f"⏹️  Early stopping triggered after {self.patience} epochs without improvement")
            print(f"📈 Best validation loss: {self.best_loss:.6f} (epoch {self.best_epoch})")
            
            # Restore best weights
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
                print("✅ Restored best model weights")
                
        return should_stop
    
    def save_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                       epoch: int, checkpoint_path: str, is_best: bool = False):
        """Save comprehensive training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'best_loss': self.best_loss,
            'best_epoch': self.best_epoch,
            'epochs_without_improvement': self.epochs_without_improvement
        }
        
        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            # Also save as latest checkpoint
            latest_path = os.path.join(os.path.dirname(checkpoint_path), 'latest.pth')
            torch.save(checkpoint, latest_path)

class TrainingLogger:
    """
    Comprehensive training metrics tracking and visualization.
    
    Features:
    - Loss curve tracking (train/validation)
    - Training time and resource monitoring
    - Automatic visualization generation
    - JSON metrics export
    """
    
    def __init__(self, output_dir: str, log_frequency: int = 10):
        """Initialize training logger."""
        self.output_dir = Path(output_dir)
        self.logs_dir = self.output_dir / 'logs'
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_frequency = log_frequency
        self.epoch_logs = []
        self.batch_logs = []
        self.start_time = None
        
    def start_training(self):
        """Mark the start of training."""
        self.start_time = time.time()
        
    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, 
                  learning_rate: float, epoch_time: float, gpu_memory: float = None):
        """Log epoch-level metrics."""
        log_entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': learning_rate,
            'epoch_time': epoch_time,
            'total_time': time.time() - self.start_time if self.start_time else 0
        }
        
        if gpu_memory is not None:
            log_entry['gpu_memory_gb'] = gpu_memory
            
        self.epoch_logs.append(log_entry)
        
        # Print progress
        if epoch % self.log_frequency == 0:
            print(f"Epoch {epoch:3d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                  f"LR: {learning_rate:.2e} | Time: {epoch_time:.1f}s")
    
    def log_batch(self, epoch: int, batch_idx: int, batch_loss: float, batch_size: int):
        """Log batch-level metrics."""
        log_entry = {
            'epoch': epoch,
            'batch_idx': batch_idx,
            'batch_loss': batch_loss,
            'batch_size': batch_size,
            'timestamp': time.time()
        }
        
        self.batch_logs.append(log_entry)
    
    def generate_loss_curves(self) -> str:
        """Generate and save loss curve visualizations."""
        if not self.epoch_logs:
            return None
            
        try:
            import matplotlib.pyplot as plt
            
            epochs = [log['epoch'] for log in self.epoch_logs]
            train_losses = [log['train_loss'] for log in self.epoch_logs]
            val_losses = [log['val_loss'] for log in self.epoch_logs]
            
            plt.figure(figsize=(12, 8))
            
            # Loss curves
            plt.subplot(2, 2, 1)
            plt.plot(epochs, train_losses, label='Training', alpha=0.8)
            plt.plot(epochs, val_losses, label='Validation', alpha=0.8)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Learning rate
            plt.subplot(2, 2, 2)
            learning_rates = [log['learning_rate'] for log in self.epoch_logs]
            plt.plot(epochs, learning_rates)
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Schedule')
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
            
            # Training time
            plt.subplot(2, 2, 3)
            epoch_times = [log['epoch_time'] for log in self.epoch_logs]
            plt.plot(epochs, epoch_times)
            plt.xlabel('Epoch')
            plt.ylabel('Time (seconds)')
            plt.title('Epoch Training Time')
            plt.grid(True, alpha=0.3)
            
            # GPU memory (if available)
            plt.subplot(2, 2, 4)
            if 'gpu_memory_gb' in self.epoch_logs[0]:
                gpu_memory = [log['gpu_memory_gb'] for log in self.epoch_logs]
                plt.plot(epochs, gpu_memory)
                plt.xlabel('Epoch')
                plt.ylabel('GPU Memory (GB)')
                plt.title('GPU Memory Usage')
            else:
                # Total time curve
                total_times = [log['total_time'] for log in self.epoch_logs]
                plt.plot(epochs, total_times)
                plt.xlabel('Epoch')
                plt.ylabel('Total Time (seconds)')
                plt.title('Cumulative Training Time')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save figure
            curves_path = self.logs_dir / 'loss_curves.png'
            plt.savefig(curves_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return str(curves_path)
            
        except ImportError:
            print("Warning: matplotlib not available for loss curve generation")
            return None
    
    def export_metrics(self) -> str:
        """Export comprehensive training metrics to JSON."""
        metrics = {
            'training_summary': {
                'total_epochs': len(self.epoch_logs),
                'total_time': self.epoch_logs[-1]['total_time'] if self.epoch_logs else 0,
                'final_train_loss': self.epoch_logs[-1]['train_loss'] if self.epoch_logs else None,
                'final_val_loss': self.epoch_logs[-1]['val_loss'] if self.epoch_logs else None,
                'best_val_loss': min(log['val_loss'] for log in self.epoch_logs) if self.epoch_logs else None
            },
            'epoch_logs': self.epoch_logs,
            'batch_logs': self.batch_logs
        }
        
        # Save metrics
        metrics_path = self.logs_dir / 'training_log.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
            
        return str(metrics_path)

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
                          num_workers: int = 2) -> Tuple[DataLoader, DataLoader]:
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
            validation_mode=False
        )
        
        val_dataset = AugmentedPatchDataset(
            val_files, 
            augment=False,  # No augmentation for validation
            validation_mode=True
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
                   epoch: int, logger: TrainingLogger = None) -> float:
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
            
            # Handle different input shapes for 2D vs 3D U-Net
            if self.model_type == "2d_unet":
                # For 2D U-Net: handle both (B, C, H, W) and (B, C, T, H, W)
                if len(features.shape) == 5:  # (B, C, T, H, W) -> (B, C*T, H, W)
                    B, C, T, H, W = features.shape
                    features = features.view(B, C * T, H, W)
                    
            elif self.model_type == "3d_unet":
                # For 3D U-Net: ensure (B, C, T, H, W) format
                if len(features.shape) == 4:  # (B, C, H, W) -> expand temporal
                    features = features.unsqueeze(2)  # Add temporal dimension
                    
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
                        
                elif self.model_type == "3d_unet":
                    if len(features.shape) == 4:  # (B, C, H, W) -> expand temporal
                        features = features.unsqueeze(2)
                        
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
                             checkpoint_freq: int = 10) -> Dict:
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
        logger = TrainingLogger(output_dir)
        early_stopping = EarlyStoppingCallback(
            patience=early_stopping_patience,
            checkpoint_dir=str(checkpoints_dir)
        )
        
        # Create data loaders
        train_loader, val_loader = self.create_data_loaders(
            patch_files,
            validation_split=validation_split,
            batch_size=batch_size,
            augment=augment
        )
        
        # Get sample to determine input dimensions
        sample_features, _, _ = next(iter(train_loader))
        
        if self.model_type == "2d_unet":
            # Handle temporal dimension for 2D U-Net
            if len(sample_features.shape) == 5:  # (B, C, T, H, W)
                in_channels = sample_features.shape[1] * sample_features.shape[2]
            else:  # (B, C, H, W)
                in_channels = sample_features.shape[1]
            model = create_2d_unet(in_channels=in_channels)
        else:  # 3d_unet
            in_channels = sample_features.shape[1] if len(sample_features.shape) == 5 else sample_features.shape[1]
            model = create_3d_unet(in_channels=in_channels)
            
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
                
            training_results.update({
                'epochs_completed': epoch + 1,
                'best_val_loss': best_val_loss,
                'final_train_loss': train_loss,
                'total_training_time': time.time() - logger.start_time
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

def load_patch_data(patch_path: str, normalize_bands: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Load patch data from GeoTIFF file with improved temporal and normalization support.
    
    Args:
        patch_path: Path to patch GeoTIFF file
        normalize_bands: Whether to apply band-specific normalization
        
    Returns:
        features: Feature array [bands, height, width]
        gedi_target: GEDI target array [height, width]
        band_info: Dictionary mapping band names to indices
    """
    with rasterio.open(patch_path) as src:
        data = src.read()  # [bands, height, width]
        band_descriptions = src.descriptions
        
        # Create band mapping
        band_info = {desc: i for i, desc in enumerate(band_descriptions) if desc}
        
        # Find GEDI band
        gedi_idx = None
        for i, desc in enumerate(band_descriptions):
            if desc and 'rh' in desc.lower():
                gedi_idx = i
                break
        
        if gedi_idx is None:
            raise ValueError("No GEDI (rh) band found in patch data")
        
        # Extract GEDI target
        gedi_target = data[gedi_idx].astype(np.float32)
        
        # Extract features (all bands except GEDI and forest mask)
        feature_indices = []
        for i, desc in enumerate(band_descriptions):
            if desc and desc not in ['rh', 'forest_mask']:
                feature_indices.append(i)
        
        features = data[feature_indices].astype(np.float32)
        
        # Crop to 256x256 if needed (handle 257x257 patches)
        if features.shape[1] > 256 or features.shape[2] > 256:
            print(f"Cropping patch from {features.shape[1]}x{features.shape[2]} to 256x256")
            features = features[:, :256, :256]
            gedi_target = gedi_target[:256, :256]
        
        # Apply improved normalization with temporal support
        if normalize_bands:
            features = apply_band_normalization(features, band_descriptions, feature_indices)
    
    return features, gedi_target, band_info

def apply_band_normalization(features: np.ndarray, band_descriptions: list, feature_indices: list) -> np.ndarray:
    """Apply band-specific normalization with temporal support."""
    
    for i, idx in enumerate(feature_indices):
        desc = band_descriptions[idx]
        if not desc:
            continue
            
        # Handle temporal bands (with _M## suffix)
        base_desc = desc.split('_M')[0] if '_M' in desc else desc
        
        # Apply normalization based on base description
        if base_desc.startswith('S1_'):
            # Sentinel-1 normalization: (val + 25) / 25
            features[i] = (features[i] + 25) / 25
        elif base_desc in ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']:
            # Sentinel-2 reflectance: val / 10000, clip to [0,1]
            features[i] = np.clip(features[i] / 10000.0, 0, 1)
        elif base_desc == 'NDVI':
            # NDVI: clip to [-1, 1]
            features[i] = np.clip(features[i], -1, 1)
        elif 'elevation' in desc.lower():
            features[i] = normalize_srtm_elevation(features[i])
        elif 'slope' in desc.lower():
            features[i] = normalize_srtm_slope(features[i])
        elif 'aspect' in desc.lower():
            features[i] = normalize_srtm_aspect(features[i])
        elif base_desc.startswith('ALOS2_'):
            # ALOS2: keep as-is or apply light normalization if needed
            features[i] = features[i]  # No normalization for now
        elif base_desc.startswith('ch_'):
            features[i] = normalize_canopy_height(features[i])
        
        # Replace any remaining NaN/inf values
        features[i] = np.nan_to_num(features[i], nan=0.0, posinf=0.0, neginf=0.0)
    
    return features

def extract_sparse_gedi_pixels(features: np.ndarray, gedi_target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract feature vectors only for pixels with valid GEDI data.
    
    Args:
        features: Feature array [bands, height, width]
        gedi_target: GEDI target array [height, width]
        
    Returns:
        X: Feature matrix [n_valid_pixels, n_bands]
        y: Target vector [n_valid_pixels]
    """
    # Find valid GEDI pixels (not NaN and > 0)
    valid_mask = ~np.isnan(gedi_target) & (gedi_target > 0)
    valid_indices = np.where(valid_mask)
    
    if len(valid_indices[0]) == 0:
        raise ValueError("No valid GEDI pixels found in patch")
    
    # Extract features for valid pixels only
    X = features[:, valid_indices[0], valid_indices[1]].T  # Shape: (n_pixels, n_bands)
    y = gedi_target[valid_indices]  # Shape: (n_pixels,)
    
    print(f"Extracted {len(y)} valid GEDI pixels from {gedi_target.size} total pixels ({len(y)/gedi_target.size*100:.2f}%)")
    
    return X, y

def detect_temporal_mode(band_descriptions: list) -> bool:
    """
    Detect if patch data is temporal based on band naming convention.
    
    Args:
        band_descriptions: List of band descriptions
        
    Returns:
        True if temporal data detected, False otherwise
    """
    temporal_indicators = ['_M01', '_M02', '_M03', '_M04', '_M05', '_M06',
                          '_M07', '_M08', '_M09', '_M10', '_M11', '_M12']
    
    for desc in band_descriptions:
        if desc and any(indicator in desc for indicator in temporal_indicators):
            return True
    
    return False

def load_patches_from_directory(patches_dir: str, pattern: str = "*.tif") -> List[Tuple[np.ndarray, np.ndarray, str]]:
    """
    Load all patch files from directory.
    
    Args:
        patches_dir: Directory containing patch files
        pattern: File pattern to match
        
    Returns:
        List of (features, gedi_target, patch_name) tuples
    """
    patch_files = glob.glob(os.path.join(patches_dir, pattern))
    patch_data = []
    
    for patch_file in tqdm(patch_files, desc="Loading patches"):
        try:
            features, gedi_target, _ = load_patch_data(patch_file)
            patch_name = os.path.basename(patch_file)
            patch_data.append((features, gedi_target, patch_name))
        except Exception as e:
            print(f"Error loading patch {patch_file}: {e}")
            continue
    
    return patch_data

def load_training_data(csv_path: str, mask_path: Optional[str] = None,
                      feature_names: Optional[list] = None, ch_col: str = 'rh') -> Tuple[np.ndarray, np.ndarray]:
    """
    Load training data from CSV file and optionally mask with forest mask.
    
    Args:
        csv_path: Path to training data CSV
        mask_path: Optional path to forest mask TIF
        
    Returns:
        X: Feature matrix
        y: Target variable (rh)
    """
    # Read training data
    df = pd.read_csv(csv_path)
    
    # Create GeoDataFrame from points
    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df['longitude'], df['latitude'])],
        crs="EPSG:4326"
    )
    
    if mask_path:
        with rasterio.open(mask_path) as mask_src:
            # Check CRS
            mask_crs = mask_src.crs
            if mask_crs != gdf.crs:
                gdf = gdf.to_crs(mask_crs)
            
            # Get bounds of mask
            mask_bounds = box(*mask_src.bounds)
            
            # First filter points by mask bounds
            gdf_masked = gdf[gdf.geometry.within(mask_bounds)]
            
            if len(gdf_masked) == 0:
                raise ValueError("No training points fall within the mask bounds")
            else:
                gdf = gdf_masked
            
            # Convert points to pixel coordinates
            pts_pixels = []
            valid_indices = []
            for idx, point in enumerate(gdf.geometry):
                row, col = rasterio.transform.rowcol(mask_src.transform, 
                                                   point.x, 
                                                   point.y)
                if (0 <= row < mask_src.height and 
                    0 <= col < mask_src.width):
                    pts_pixels.append((row, col))
                    valid_indices.append(idx)
            
            if not pts_pixels:
                raise ValueError("No training points could be mapped to valid pixels")
            
            # Read forest mask values at pixel locations
            mask_values = [mask_src.read(1)[r, c] for r, c in pts_pixels]
            
            # Filter points by mask values
            mask_indices = [i for i, v in enumerate(mask_values) if v == 1]
            if not mask_indices:
                raise ValueError("No training points fall within the forest mask")
            
            final_indices = [valid_indices[i] for i in mask_indices]
            gdf = gdf.iloc[final_indices]
    
    # Convert back to original CRS if needed
    if mask_path and mask_crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    
    # Separate features and target
    df = pd.DataFrame(gdf.drop(columns='geometry'))
    y = df[ch_col].values
    
    # Get feature columns in same order as feature_names
    if feature_names is not None:
        missing_features = set(feature_names) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing features in training data: {missing_features}")
        X = df[feature_names].values
    else:
        X = df.drop([ch_col, 'longitude', 'latitude'], axis=1, errors='ignore').values
    
    return X, y

def load_prediction_data(stack_path: str, mask_path: Optional[str] = None, feature_names: Optional[list] = None) -> Tuple[np.ndarray, rasterio.DatasetReader]:
    """
    Load prediction data from stack TIF and optionally apply forest mask.
    
    Args:
        stack_path: Path to stack TIF file
        mask_path: Optional path to forest mask TIF
        feature_names: Optional list of feature names for filtering bands
        
    Returns:
        X: Feature matrix for prediction
        src: Rasterio dataset for writing results
    """
    if feature_names is None:
        raise ValueError("feature_names must be provided to ensure consistent features between training and prediction")
    # Read stack file
    with rasterio.open(stack_path) as src:
        stack = src.read()
        stack_crs = src.crs
        
        # Get band descriptions if available
        band_descriptions = src.descriptions
        
        # Filter bands based on feature names if provided
        # Create a mapping of band descriptions to indices
        band_indices = []
        for i, desc in enumerate(band_descriptions):
            if desc in feature_names:
                band_indices.append(i)
        
        if len(band_indices) != len(feature_names):
            missing_features = set(feature_names) - set(band_descriptions)
            raise ValueError(f"Could not find all feature names in stack bands. Missing features: {missing_features}")
        
        # Select only the bands that match feature names
        stack = stack[band_indices]
        
        # Reshape stack to 2D array (bands x pixels)
        n_bands, height, width = stack.shape
        X = stack.reshape(n_bands, -1).T
        
        # Apply mask if provided
        if mask_path:
            with rasterio.open(mask_path) as mask_src:
                # Check CRS
                if mask_src.crs != stack_crs:
                    raise ValueError(f"CRS mismatch: stack {stack_crs} != mask {mask_src.crs}")
                
                # Check dimensions
                if mask_src.shape != (height, width):
                    raise ValueError(f"Shape mismatch: stack {(height, width)} != mask {mask_src.shape}")
                
                mask = mask_src.read(1)
                mask = mask.reshape(-1)
                X = X[mask == 1]
        
        src_copy = rasterio.open(stack_path)
        return X, src_copy

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
    """
    Save training metrics and feature importance to JSON file.
    
    Args:
        metrics: Dictionary of training metrics
        importance_data: Dictionary of feature importance data
        output_dir: Directory to save JSON file
    """
    import json
    from pathlib import Path
    
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

def train_2d_unet(patch_path: str, model_params: Dict = None, training_params: Dict = None) -> Tuple[object, dict]:
    """
    Train 2D U-Net model on single patch with data augmentation.
    
    Args:
        patch_path: Path to patch TIF file
        model_params: Model hyperparameters
        training_params: Training hyperparameters
        
    Returns:
        Trained model and metrics
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for 2D U-Net training")
    
    # Default parameters
    if model_params is None:
        model_params = {'base_channels': 32}  # Smaller for 2D
    if training_params is None:
        training_params = {
            'epochs': 50,
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'huber_delta': 1.0,
            'shift_radius': 1
        }
    
    print(f"Loading patch data for 2D U-Net training...")
    features, gedi_target, band_info = load_patch_data(patch_path)
    
    # Resize to 256x256 if needed
    if features.shape[1] != 256 or features.shape[2] != 256:
        from scipy.ndimage import zoom
        scale_h = 256 / features.shape[1]
        scale_w = 256 / features.shape[2]
        
        resized_features = np.zeros((features.shape[0], 256, 256), dtype=features.dtype)
        for i in range(features.shape[0]):
            resized_features[i] = zoom(features[i], (scale_h, scale_w), order=1)
        
        resized_gedi = zoom(gedi_target, (scale_h, scale_w), order=0)
        features, gedi_target = resized_features, resized_gedi
    
    n_bands = features.shape[0]
    print(f"Training 2D U-Net on {n_bands} bands, patch size: {features.shape[1]}x{features.shape[2]}")
    
    # Initialize model
    model = create_2d_unet(
        in_channels=n_bands,
        n_classes=1,
        base_channels=model_params['base_channels']
    )
    
    # Set up training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=training_params['learning_rate'],
        weight_decay=training_params['weight_decay']
    )
    
    # Convert to tensors
    features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)  # [1, bands, H, W]
    target_tensor = torch.FloatTensor(gedi_target).unsqueeze(0).to(device)  # [1, H, W]
    
    # Create mask for valid GEDI pixels
    valid_mask = ~torch.isnan(target_tensor) & (target_tensor > 0)
    
    best_loss = float('inf')
    metrics = {}
    
    # Training loop
    for epoch in tqdm(range(training_params['epochs']), desc="Training 2D U-Net"):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        pred = model(features_tensor)  # [1, H, W]
        
        # Compute loss only on valid GEDI pixels
        if valid_mask.sum() > 0:
            valid_pred = pred[valid_mask]
            valid_target = target_tensor[valid_mask]
            loss = nn.MSELoss()(valid_pred, valid_target)
        else:
            loss = torch.tensor(0.0, requires_grad=True, device=device)
        
        # Backward pass
        if loss.item() > 0:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        current_loss = loss.item()
        if current_loss < best_loss:
            best_loss = current_loss
            metrics = {
                'train_loss': current_loss,
                'epoch': epoch,
                'valid_gedi_pixels': valid_mask.sum().item()
            }
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{training_params['epochs']}: Loss = {current_loss:.4f}")
    
    print(f"2D U-Net training complete. Best loss: {best_loss:.4f}")
    return model, metrics

def train_3d_unet(patch_path: str, model_params: Dict = None, training_params: Dict = None) -> Tuple[object, dict]:
    """
    Train 3D U-Net model on temporal patch with data augmentation.
    
    Args:
        patch_path: Path to temporal patch TIF file
        model_params: Model hyperparameters
        training_params: Training hyperparameters
        
    Returns:
        Trained model and metrics
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for 3D U-Net training")
    
    # Default parameters
    if model_params is None:
        model_params = {'base_channels': 32}  # Smaller for memory
    if training_params is None:
        training_params = {
            'epochs': 30,
            'learning_rate': 1e-4,
            'weight_decay': 1e-4,
            'huber_delta': 1.0,
            'shift_radius': 1
        }
    
    print(f"Loading temporal patch data for 3D U-Net training...")
    features, gedi_target, band_info = load_patch_data(patch_path)
    
    # Import the improved temporal dataset from our previous work
    from train_temporal_fixed import ImprovedTemporalDataset, MaskedTemporalUNet
    
    # Use the improved temporal dataset
    dataset = ImprovedTemporalDataset(patch_path, patch_size=256, augment=True)
    
    # Initialize model
    model = MaskedTemporalUNet(in_channels=15, n_classes=1).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=training_params['learning_rate'],
        weight_decay=training_params['weight_decay']
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_loss = float('inf')
    metrics = {}
    
    # Training loop
    for epoch in tqdm(range(training_params['epochs']), desc="Training 3D U-Net"):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        # Train on augmented patches
        for i in range(len(dataset)):
            features, target, gedi_mask, availability = dataset[i]
            
            features = features.unsqueeze(0).to(device)  # Add batch dim
            target = target.unsqueeze(0).to(device)
            gedi_mask = gedi_mask.unsqueeze(0).to(device)
            availability = availability.unsqueeze(0).to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            pred = model(features, availability)
            
            # Compute loss only on valid GEDI pixels
            if gedi_mask.sum() > 0:
                valid_pred = pred[gedi_mask > 0]
                valid_target = target[gedi_mask > 0]
                loss = nn.MSELoss()(valid_pred, valid_target)
            else:
                loss = torch.tensor(0.0, requires_grad=True, device=device)
            
            # Backward pass
            if loss.item() > 0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        if avg_loss < best_loss:
            best_loss = avg_loss
            metrics = {
                'train_loss': avg_loss,
                'epoch': epoch
            }
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{training_params['epochs']}: Loss = {avg_loss:.4f}")
    
    print(f"3D U-Net training complete. Best loss: {best_loss:.4f}")
    return model, metrics

def train_model(X: np.ndarray, y: np.ndarray, model_type: str = 'rf', 
                batch_size: int = 64, test_size: float = 0.2, feature_names: Optional[list] = None,
                n_bands: Optional[int] = None) -> Tuple[object, dict, dict]:
    """
    Training function for traditional models (RF/MLP only).
    U-Net models are handled separately in main().
    
    Args:
        X: Feature matrix (extracted from patch for GEDI pixels)
        y: Target variable (extracted from patch for GEDI pixels)
        model_type: Type of model ('rf' or 'mlp')
        batch_size: Batch size for MLP training
        test_size: Proportion of data to use for validation
        feature_names: Optional list of feature names
        
    Returns:
        Trained model, training metrics, and feature importance/weights
    """
    if model_type not in ['rf', 'mlp']:
        raise ValueError(f"train_model only supports 'rf' and 'mlp', got '{model_type}'")
    
    if X.shape[0] < 10:  # Need minimum samples for train/val split
        raise ValueError(f"Insufficient GEDI pixels ({X.shape[0]}) for {model_type.upper()} training. Need at least 10.")
    
    print(f"Training {model_type.upper()} on {X.shape[0]} GEDI pixels with {X.shape[1]} features")
    
    # Split data for traditional models
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    if model_type == 'rf':
        # Train Random Forest model
        model = RandomForestRegressor(
            n_estimators=500,
            min_samples_leaf=5,
            max_features='sqrt',
            n_jobs=-1,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Get predictions
        y_pred = model.predict(X_val)
        train_metrics = calculate_metrics(y_pred, y_val)
        
        # Get feature importance
        importance = model.feature_importances_
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importance))]
        
        importance_data = {
            name: float(imp) for name, imp in zip(feature_names, importance)
        }
        
    else:  # MLP model
        # Create normalized dataloaders
        train_loader, val_loader, scaler_mean, scaler_std = create_normalized_dataloader(
            X_train, X_val, y_train, y_val, batch_size=batch_size, n_bands=n_bands
        )
        
        # Initialize model
        model = MLPRegressionModel(input_size=X.shape[1])
        if torch.cuda.is_available():
            model = model.cuda()
            
        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters())
        num_epochs = 100
        best_val_loss = float('inf')
        
        # Training loop with tqdm progress bar
        for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
            model.train()
            for batch_X, batch_y in train_loader:
                if torch.cuda.is_available():
                    batch_X, batch_y = batch_X.cuda(), batch_y.cuda()
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            # Validation
            model.eval()
            val_predictions = []
            val_targets = []
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    if torch.cuda.is_available():
                        batch_X, batch_y = batch_X.cuda(), batch_y.cuda()
                    outputs = model(batch_X)
                    val_predictions.extend(outputs.cpu().numpy())
                    val_targets.extend(batch_y.cpu().numpy())
            
            val_predictions = np.array(val_predictions)
            val_targets = np.array(val_targets)
            val_metrics = calculate_metrics(val_predictions, val_targets)
            val_loss = val_metrics['RMSE']
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                train_metrics = val_metrics
        
        # Get feature importance (using weights of first layer as proxy)
        with torch.no_grad():
            weights = model.layers[0].weight.abs().mean(dim=0).cpu().numpy()
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(len(weights))]
            importance_data = {}
            for name, weight in zip(feature_names, weights):
                # Convert numpy.float32 to Python float
                weight_value = weight.item() if hasattr(weight, 'item') else float(weight)
                importance_data[name] = weight_value
        
        # Store normalization parameters with model
        model.scaler_mean = scaler_mean
        model.scaler_std = scaler_std
    
    # Sort importance by value
    importance_data = dict(sorted(importance_data.items(), key=lambda x: x[1], reverse=True))
    
    # Print metrics and top features
    for metric, value in train_metrics.items():
        print(f"{metric}: {value:.3f}")
    
    print("\nTop 5 Important Features:")
    for name, imp in list(importance_data.items())[:5]:
        print(f"{name}: {imp:.3f}")
    
    return model, train_metrics, importance_data

def save_predictions(predictions: np.ndarray, src: rasterio.DatasetReader, output_path: str,
                    mask_path: Optional[str] = None) -> None:
    """
    Save predictions to a GeoTIFF file.
    
    Args:
        predictions: Model predictions
        src: Source rasterio dataset for metadata
        output_path: Path to save predictions
        mask_path: Optional path to forest mask TIF
    """
    # Create output profile
    profile = src.profile.copy()
    profile.update(count=1, dtype='float32')
    
    # Initialize prediction array
    height, width = src.height, src.width
    pred_array = np.zeros((height, width), dtype='float32')
    
    if mask_path:
        # Apply predictions only to masked areas
        with rasterio.open(mask_path) as mask_src:
            # Check CRS
            if mask_src.crs != src.crs:
                raise ValueError(f"CRS mismatch: source {src.crs} != mask {mask_src.crs}")
            
            mask = mask_src.read(1)
            mask_idx = np.where(mask.reshape(-1) == 1)[0]
            pred_array.reshape(-1)[mask_idx] = predictions
    else:
        # Apply predictions to all pixels
        pred_array = predictions.reshape(height, width)
    
    try:
        # Save predictions
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(pred_array, 1)
    finally:
        src.close()

def train_2d_unet(features: np.ndarray, gedi_target: np.ndarray, 
                  epochs: int = 50, learning_rate: float = 1e-3, weight_decay: float = 1e-4,
                  base_channels: int = 32, huber_delta: float = 1.0, shift_radius: int = 1) -> Tuple[nn.Module, Dict]:
    """
    Train 2D U-Net model on patch data.
    
    Args:
        features: Feature array [bands, height, width]
        gedi_target: GEDI target array [height, width]
        epochs: Number of training epochs
        learning_rate: Learning rate
        weight_decay: Weight decay
        base_channels: Base channels for U-Net
        huber_delta: Huber loss delta
        shift_radius: Spatial shift radius for GEDI alignment
        
    Returns:
        model: Trained 2D U-Net model
        metrics: Training metrics dictionary
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for 2D U-Net training")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training 2D U-Net on device: {device}")
    
    # Create model
    model = create_2d_unet(in_channels=features.shape[0], n_classes=1, base_channels=base_channels)
    model = model.to(device)
    
    # Setup optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Convert data to tensors
    features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)  # Add batch dim
    gedi_tensor = torch.FloatTensor(gedi_target).unsqueeze(0).to(device)   # Add batch dim
    
    # Create valid mask for GEDI pixels
    valid_mask = ~torch.isnan(gedi_tensor) & (gedi_tensor > 0)
    
    print(f"Valid GEDI pixels: {valid_mask.sum().item()}/{gedi_tensor.numel()}")
    
    # Training loop
    model.train()
    train_losses = []
    
    for epoch in tqdm(range(epochs), desc="Training 2D U-Net"):
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(features_tensor)  # [1, H, W]
        
        # Calculate modified Huber loss with shift awareness
        loss = modified_huber_loss(predictions, gedi_tensor, valid_mask, huber_delta, shift_radius)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")
    
    # Calculate final metrics on valid GEDI pixels
    model.eval()
    with torch.no_grad():
        final_predictions = model(features_tensor)
        
        # Extract predictions and targets for valid GEDI pixels only
        valid_preds = final_predictions[valid_mask].cpu().numpy()
        valid_targets = gedi_tensor[valid_mask].cpu().numpy()
        
        # Calculate metrics using existing function
        metrics = calculate_metrics(valid_preds, valid_targets)
        metrics['train_loss'] = np.mean(train_losses[-10:])  # Average of last 10 epochs
        metrics['final_loss'] = train_losses[-1]
    
    return model, metrics

def separate_temporal_nontemporal_bands(features: np.ndarray, band_descriptions: list) -> Tuple[np.ndarray, np.ndarray]:
    """
    Separate temporal and non-temporal bands from feature array.
    
    Args:
        features: Feature array [bands, height, width]
        band_descriptions: List of band descriptions
        
    Returns:
        temporal_features: Temporal bands [temporal_bands, height, width]
        nontemporal_features: Non-temporal bands [nontemporal_bands, height, width]
    """
    temporal_indices = []
    nontemporal_indices = []
    
    for i, desc in enumerate(band_descriptions):
        if desc and desc not in ['rh', 'forest_mask']:
            # Check if band has monthly suffix (_M01 to _M12)
            if any(f'_M{m:02d}' in desc for m in range(1, 13)):
                temporal_indices.append(i)
            else:
                nontemporal_indices.append(i)
    
    temporal_features = features[temporal_indices] if temporal_indices else np.empty((0, features.shape[1], features.shape[2]))
    nontemporal_features = features[nontemporal_indices] if nontemporal_indices else np.empty((0, features.shape[1], features.shape[2]))
    
    print(f"Separated bands: {len(temporal_indices)} temporal, {len(nontemporal_indices)} non-temporal")
    
    return temporal_features, nontemporal_features

def train_3d_unet(features: np.ndarray, gedi_target: np.ndarray,
                  epochs: int = 50, learning_rate: float = 1e-3, weight_decay: float = 1e-4,
                  base_channels: int = 32, huber_delta: float = 1.0, shift_radius: int = 1) -> Tuple[nn.Module, Dict]:
    """
    Train 3D U-Net model on temporal patch data.
    
    Args:
        features: Feature array [bands, height, width] (all bands including non-temporal)
        gedi_target: GEDI target array [height, width]
        epochs: Number of training epochs
        learning_rate: Learning rate
        weight_decay: Weight decay
        base_channels: Base channels for U-Net
        huber_delta: Huber loss delta
        shift_radius: Spatial shift radius for GEDI alignment
        
    Returns:
        model: Trained 3D U-Net model
        metrics: Training metrics dictionary
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for 3D U-Net training")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training 3D U-Net on device: {device}")
    
    # Load band descriptions to separate temporal/non-temporal bands
    # For now, assume the temporal structure: 180 temporal bands (15 bands × 12 months)
    n_bands = features.shape[0]
    h, w = features.shape[1], features.shape[2]
    
    # Expected structure: 180 temporal + 14 non-temporal = 194 total feature bands
    n_temporal_expected = 180  # 15 bands × 12 months
    n_nontemporal_expected = n_bands - n_temporal_expected
    
    if n_bands >= n_temporal_expected:
        # Separate temporal and non-temporal bands
        temporal_features = features[:n_temporal_expected]  # First 180 bands
        nontemporal_features = features[n_temporal_expected:] if n_bands > n_temporal_expected else np.empty((0, h, w))
        
        print(f"Using expected temporal structure: {n_temporal_expected} temporal + {len(nontemporal_features)} non-temporal bands")
        
        # Reshape temporal data: (180, h, w) -> (15, 12, h, w)
        n_channels_per_month = 15  # S1(2) + S2(11) + ALOS2(2) = 15
        n_months = 12
        
        temporal_features_reshaped = temporal_features.reshape(n_channels_per_month, n_months, h, w)
        
        # Handle missing values in temporal data
        temporal_features_reshaped = np.nan_to_num(temporal_features_reshaped, nan=0.0, posinf=0.0, neginf=0.0)
        
        print(f"Reshaped temporal data: {n_temporal_expected} bands -> {n_channels_per_month} channels × {n_months} months")
        
        # For 3D U-Net, we'll use temporal features. Non-temporal can be added as extra channels if needed
        if len(nontemporal_features) > 0:
            # Repeat non-temporal features across time dimension
            nontemporal_expanded = np.tile(nontemporal_features[:, np.newaxis, :, :], (1, n_months, 1, 1))
            # Combine temporal and non-temporal
            combined_features = np.concatenate([temporal_features_reshaped, nontemporal_expanded], axis=0)
            n_total_channels = n_channels_per_month + len(nontemporal_features)
        else:
            combined_features = temporal_features_reshaped
            n_total_channels = n_channels_per_month
    else:
        raise ValueError(f"Insufficient bands for temporal processing: got {n_bands}, expected at least {n_temporal_expected}")
    
    print(f"Final 3D input shape: {n_total_channels} channels × {n_months} months × {h}×{w}")
    
    # Create model - use smaller base_channels to avoid memory issues and temporal dimension problems
    # For temporal data with only 12 months, we need to be careful with downsampling
    model_base_channels = min(base_channels, 16)  # Use smaller channels for 3D
    print(f"Creating 3D U-Net with {model_base_channels} base channels (reduced for temporal processing)")
    
    try:
        model = create_3d_unet(in_channels=n_total_channels, n_classes=1, base_channels=model_base_channels)
        model = model.to(device)
        
        # Test the model with a small input to verify it works
        test_input = torch.randn(1, n_total_channels, n_months, 32, 32).to(device)
        with torch.no_grad():
            test_output = model(test_input)
            print(f"Model test successful: {test_input.shape} -> {test_output.shape}")
    except Exception as e:
        print(f"Model creation failed with error: {e}")
        print("Falling back to simplified temporal processing...")
        
        # Fallback: reduce temporal dimension by averaging or use 2D approach
        # Average across temporal dimension to create 2D input
        averaged_features = np.mean(combined_features, axis=1)  # Average across time
        print(f"Fallback: Using temporal average, shape: {averaged_features.shape}")
        
        # Use 2D U-Net instead
        model = create_2d_unet(in_channels=averaged_features.shape[0], n_classes=1, base_channels=base_channels)
        model = model.to(device)
        
        # Update the combined_features for training
        combined_features = averaged_features
        print("Using 2D U-Net with temporally averaged features as fallback")
    
    # Setup optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Convert data to tensors - handle both 3D and 2D cases
    if len(combined_features.shape) == 4:  # 3D case: (channels, time, h, w)
        features_tensor = torch.FloatTensor(combined_features).unsqueeze(0).to(device)  # Add batch dim
    else:  # 2D fallback case: (channels, h, w)
        features_tensor = torch.FloatTensor(combined_features).unsqueeze(0).to(device)  # Add batch dim
    
    gedi_tensor = torch.FloatTensor(gedi_target).unsqueeze(0).to(device)           # Add batch dim
    
    # Create valid mask for GEDI pixels
    valid_mask = ~torch.isnan(gedi_tensor) & (gedi_tensor > 0)
    
    print(f"Valid GEDI pixels: {valid_mask.sum().item()}/{gedi_tensor.numel()}")
    
    # Training loop
    model.train()
    train_losses = []
    
    for epoch in tqdm(range(epochs), desc="Training 3D U-Net"):
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(features_tensor)  # [1, H, W]
        
        # Calculate modified Huber loss with shift awareness
        loss = modified_huber_loss(predictions, gedi_tensor, valid_mask, huber_delta, shift_radius)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")
    
    # Calculate final metrics on valid GEDI pixels
    model.eval()
    with torch.no_grad():
        final_predictions = model(features_tensor)
        
        # Extract predictions and targets for valid GEDI pixels only
        valid_preds = final_predictions[valid_mask].cpu().numpy()
        valid_targets = gedi_tensor[valid_mask].cpu().numpy()
        
        # Calculate metrics using existing function
        metrics = calculate_metrics(valid_preds, valid_targets)
        metrics['train_loss'] = np.mean(train_losses[-10:])  # Average of last 10 epochs
        metrics['final_loss'] = train_losses[-1]
    
    return model, metrics

def parse_args():
    parser = argparse.ArgumentParser(description='Unified patch-based training for all model types')
    
    # Input modes - either single patch or multi-patch directory
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--patch-path', type=str,
                           help='Path to single patch TIF file with GEDI rh band')
    input_group.add_argument('--patch-dir', type=str,
                           help='Directory containing multiple patch TIF files')
    
    # Multi-patch options
    parser.add_argument('--patch-pattern', type=str, default='*.tif',
                       help='File pattern for multi-patch mode (e.g., "*_temporal_*.tif")')
    parser.add_argument('--merge-predictions', action='store_true',
                       help='Merge individual patch predictions into continuous map')
    parser.add_argument('--merge-strategy', type=str, default='first', 
                       choices=['average', 'maximum', 'minimum', 'first', 'last'],
                       help='Strategy for merging overlapping predictions')
    parser.add_argument('--create-spatial-mosaic', action='store_true',
                       help='Create proper spatial mosaic (same as --merge-predictions but clearer name)')
    parser.add_argument('--mosaic-name', type=str, default=None,
                       help='Custom name for spatial mosaic output file')
    
    # Output settings
    parser.add_argument('--output-dir', type=str, default='chm_outputs',
                       help='Output directory for models and predictions')
    
    # Model selection
    parser.add_argument('--model', type=str, default='rf', choices=['rf', 'mlp', '2d_unet', '3d_unet'],
                       help='Model type: random forest (rf), MLP (mlp), 2D U-Net (2d_unet), or 3D U-Net (3d_unet)')
    
    # Traditional model parameters (RF/MLP)
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Proportion of GEDI pixels to use for validation (RF/MLP only)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    
    # Neural network parameters (all U-Nets)
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (U-Net models)')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                       help='Learning rate (U-Net models)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay (U-Net models)')
    parser.add_argument('--base-channels', type=int, default=32,
                       help='Base number of channels in U-Net models')
    
    # Advanced training parameters
    parser.add_argument('--huber-delta', type=float, default=1.0,
                       help='Huber loss delta parameter (U-Net models)')
    parser.add_argument('--shift-radius', type=int, default=1,
                       help='Spatial shift radius for GEDI alignment (U-Net models)')
    
    # Generation and evaluation
    parser.add_argument('--generate-prediction', action='store_true',
                       help='Generate prediction map after training')
    parser.add_argument('--prediction-output', type=str, default=None,
                       help='Output path for prediction TIF (auto-generated if not specified)')
    
    return parser.parse_args()

def main():
    """Unified main function for all model types using patch-based input."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine if we're in single-patch or multi-patch mode
    if args.patch_path:
        # Single patch mode (existing functionality)
        print(f"Training {args.model.upper()} model using single patch: {args.patch_path}")
        train_single_patch(args)
    else:
        # Multi-patch mode (new functionality)
        print(f"Training {args.model.upper()} model using multiple patches from: {args.patch_dir}")
        train_multi_patch(args)


def train_single_patch(args):
    """Train model on a single patch (existing functionality)."""
    # Load patch data
    print("Loading patch data...")
    features, gedi_target, band_info = load_patch_data(args.patch_path, normalize_bands=True)
    
    # Detect temporal mode
    band_descriptions = list(band_info.keys())
    is_temporal = detect_temporal_mode(band_descriptions)
    
    print(f"Patch data: {features.shape[0]} bands, {features.shape[1]}x{features.shape[2]} pixels")
    print(f"Temporal mode detected: {is_temporal}")
    print(f"Valid GEDI pixels: {np.sum(~np.isnan(gedi_target) & (gedi_target > 0))}/{gedi_target.size}")
    
    # Validate model-data compatibility
    if args.model == '2d_unet' and is_temporal:
        raise ValueError("2D U-Net cannot be used with temporal data. Use '3d_unet' or enable non-temporal mode in chm_main.py")
    if args.model == '3d_unet' and not is_temporal:
        raise ValueError("3D U-Net requires temporal data. Use '2d_unet' or enable temporal mode in chm_main.py")
    
    # Train model based on type
    if args.model in ['rf', 'mlp']:
        # Extract sparse GEDI pixels for traditional models
        print("Extracting sparse GEDI pixels for RF/MLP training...")
        X, y = extract_sparse_gedi_pixels(features, gedi_target)
        
        # Create feature names from band descriptions
        feature_names = [desc for desc in band_descriptions if desc and desc not in ['rh', 'forest_mask']]
        
        # Train traditional model
        model, train_metrics, importance_data = train_model(
            X, y,
            model_type=args.model,
            batch_size=args.batch_size,
            test_size=args.test_size,
            feature_names=feature_names
        )
        
        # Save model
        if args.model == 'rf':
            model_path = os.path.join(args.output_dir, 'rf_model.pkl')
            joblib.dump(model, model_path)
        else:  # MLP
            model_path = os.path.join(args.output_dir, 'mlp_model.pth')
            if TORCH_AVAILABLE:
                torch.save(model.state_dict(), model_path)
        
        print(f"Saved {args.model.upper()} model to: {model_path}")
        
    elif args.model == '2d_unet':
        # Train 2D U-Net
        print("Training 2D U-Net...")
        model, train_metrics = train_2d_unet(
            features, gedi_target,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            base_channels=args.base_channels,
            huber_delta=args.huber_delta,
            shift_radius=args.shift_radius
        )
        
        # Save model
        model_path = os.path.join(args.output_dir, '2d_unet_model.pth')
        if TORCH_AVAILABLE:
            torch.save(model.state_dict(), model_path)
        print(f"Saved 2D U-Net model to: {model_path}")
        
        importance_data = {}  # U-Net doesn't have traditional feature importance
        
    elif args.model == '3d_unet':
        # Train 3D U-Net
        print("Training 3D U-Net...")
        model, train_metrics = train_3d_unet(
            features, gedi_target,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            base_channels=args.base_channels,
            huber_delta=args.huber_delta,
            shift_radius=args.shift_radius
        )
        
        # Save model
        model_path = os.path.join(args.output_dir, '3d_unet_model.pth')
        if TORCH_AVAILABLE:
            torch.save(model.state_dict(), model_path)
        print(f"Saved 3D U-Net model to: {model_path}")
        
        importance_data = {}  # U-Net doesn't have traditional feature importance
    
    # Save training metrics
    metrics_file = os.path.join(args.output_dir, 'training_metrics.json')
    save_training_metrics(train_metrics, importance_data, metrics_file)
    
    # Generate prediction if requested
    if args.generate_prediction:
        print("Generating prediction...")
        
        if args.model in ['rf', 'mlp']:
            # Generate full prediction map for traditional models
            print("Generating full prediction map for traditional model...")
            
            # Prepare full feature data
            full_features = features.reshape(features.shape[0], -1).T  # (n_pixels, n_bands)
            
            # Remove any NaN/inf values
            valid_mask = np.all(np.isfinite(full_features), axis=1)
            
            if args.model == 'rf':
                predictions = np.full(valid_mask.shape[0], np.nan)
                predictions[valid_mask] = model.predict(full_features[valid_mask])
            else:  # MLP
                if TORCH_AVAILABLE:
                    model.eval()
                    with torch.no_grad():
                        input_tensor = torch.FloatTensor(full_features[valid_mask])
                        if torch.cuda.is_available():
                            input_tensor = input_tensor.cuda()
                            model = model.cuda()
                        pred_tensor = model(input_tensor)
                        pred_values = pred_tensor.cpu().numpy().flatten()
                        
                    predictions = np.full(valid_mask.shape[0], np.nan)
                    predictions[valid_mask] = pred_values
                else:
                    raise ImportError("PyTorch is required for MLP prediction")
            
            # Reshape to original dimensions
            predictions = predictions.reshape(features.shape[1], features.shape[2])
            
        else:  # U-Net models
            # Store original dimensions for output resizing
            original_height, original_width = features.shape[-2], features.shape[-1]
            
            # Resize features to 256x256 if needed for U-Net models
            if features.shape[-2] != 256 or features.shape[-1] != 256:
                from scipy.ndimage import zoom
                print(f"Resizing patch from {features.shape[-2]}x{features.shape[-1]} to 256x256")
                
                if len(features.shape) == 3:  # (bands, height, width)
                    scale_h = 256 / features.shape[1]
                    scale_w = 256 / features.shape[2]
                    resized_features = np.zeros((features.shape[0], 256, 256), dtype=features.dtype)
                    for i in range(features.shape[0]):
                        resized_features[i] = zoom(features[i], (scale_h, scale_w), order=1)
                    features = resized_features
                elif len(features.shape) == 4:  # (bands, time, height, width)
                    scale_h = 256 / features.shape[2]
                    scale_w = 256 / features.shape[3]
                    resized_features = np.zeros((features.shape[0], features.shape[1], 256, 256), dtype=features.dtype)
                    for i in range(features.shape[0]):
                        for j in range(features.shape[1]):
                            resized_features[i, j] = zoom(features[i, j], (scale_h, scale_w), order=1)
                    features = resized_features
            
            # Prepare input data
            if args.model == '2d_unet':
                # Non-temporal mode (collapse time dimension)
                if len(features.shape) == 4:  # (bands, time, height, width)
                    combined_features = features.reshape(-1, features.shape[2], features.shape[3])
                else:  # (bands, height, width)
                    combined_features = features
                    
                input_tensor = torch.FloatTensor(combined_features).unsqueeze(0)
                
            else:  # 3d_unet
                # Temporal mode - first try to determine if we need fallback
                n_bands = features.shape[0]
                n_temporal_expected = int(12 * (n_bands // 12))  # Expected temporal bands
                
                if n_bands >= n_temporal_expected:
                    # Reshape for 3D processing: (bands, height, width) -> (new_bands, time, height, width)
                    n_features_per_month = n_bands // 12
                    combined_features = features[:n_temporal_expected].reshape(n_features_per_month, 12, features.shape[1], features.shape[2])
                    
                    try:
                        # Test if model accepts 3D input
                        test_input = torch.randn(1, combined_features.shape[0], 12, 32, 32)
                        # Try to load the model and see if it expects 3D or 2D input
                        with torch.no_grad():
                            if hasattr(model, 'encoder1'):  # 2D U-Net (fallback was used)
                                # Use temporal averaging for prediction too
                                averaged_features = np.mean(combined_features, axis=1)  # Average across time
                                input_tensor = torch.FloatTensor(averaged_features).unsqueeze(0)
                                print(f"Using temporal averaging for prediction: {averaged_features.shape}")
                            else:  # 3D U-Net
                                input_tensor = torch.FloatTensor(combined_features).unsqueeze(0)
                    except:
                        # Default to temporal averaging if we can't determine
                        averaged_features = np.mean(combined_features, axis=1)  # Average across time
                        input_tensor = torch.FloatTensor(averaged_features).unsqueeze(0)
                        print(f"Using temporal averaging for prediction (fallback): {averaged_features.shape}")
                else:
                    raise ValueError(f"Insufficient bands for temporal processing: got {n_bands}, expected at least {n_temporal_expected}")
            
            if torch.cuda.is_available():
                input_tensor = input_tensor.cuda()
                model = model.cuda()
            
            predictions = model(input_tensor)
            predictions = predictions.squeeze().cpu().numpy()
            
            # Resize predictions back to original dimensions if needed
            if predictions.shape != (original_height, original_width):
                from scipy.ndimage import zoom
                scale_h = original_height / predictions.shape[0]
                scale_w = original_width / predictions.shape[1]
                predictions = zoom(predictions, (scale_h, scale_w), order=1)
                print(f"Resized prediction back to original dimensions: {predictions.shape}")
    
        # Save prediction
        if args.prediction_output is None:
            patch_name = os.path.splitext(os.path.basename(args.patch_path))[0]
            pred_filename = f"prediction_{args.model}_{patch_name}.tif"
            prediction_path = os.path.join(args.output_dir, pred_filename)
        else:
            prediction_path = args.prediction_output
        
        # Save with same georeference as input patch
        with rasterio.open(args.patch_path) as src:
            profile = src.profile.copy()
            profile.update(count=1, dtype='float32')
            
            with rasterio.open(prediction_path, 'w', **profile) as dst:
                dst.write(predictions.astype('float32'), 1)
        
        print(f"Saved prediction to: {prediction_path}")
    
    print("Single-patch training completed successfully!")


def train_multi_patch_from_files(args):
    """Train model on multiple patches specified as file list with unified training and optional prediction merging."""
    print(f"🚀 Starting multi-patch training with {args.model.upper()} from file list")
    
    # Parse file list from command line argument
    patch_files = [f.strip() for f in args.patch_files.split(',')]
    
    # Validate that all files exist
    valid_files = []
    for file_path in patch_files:
        if os.path.exists(file_path):
            valid_files.append(file_path)
        else:
            print(f"⚠️  Warning: File not found: {file_path}")
    
    if not valid_files:
        raise ValueError("No valid patch files found in the provided list")
    
    print(f"📊 Found {len(valid_files)} valid patch files out of {len(patch_files)} provided")
    
    # Create PatchInfo objects from the file list
    patch_registry = PatchRegistry()
    patches = []
    
    for file_path in tqdm(valid_files, desc="Processing patch metadata"):
        try:
            patch_info = PatchInfo.from_file(file_path)
            patches.append(patch_info)
            patch_registry.add_patch(patch_info)
        except Exception as e:
            print(f"⚠️  Warning: Could not process {file_path}: {e}")
            continue
    
    if not patches:
        raise ValueError("No valid patches could be processed from the file list")
    
    print(f"📊 Successfully processed {len(patches)} patches")
    
    # Continue with the same logic as train_multi_patch
    # Validate patch consistency
    is_consistent = patch_registry.validate_consistency()
    if not is_consistent:
        print("⚠️  Warning: Inconsistencies detected in patches. Proceeding with caution...")
    
    # Generate and save patch summary
    summary_df = generate_multi_patch_summary(patches)
    summary_path = os.path.join(args.output_dir, 'patch_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"💾 Saved patch summary to: {summary_path}")
    
    # Get summary statistics
    summary_stats = patch_registry.get_patch_summary()
    print(f"📈 Dataset summary:")
    print(f"  - Total patches: {summary_stats['total_patches']}")
    print(f"  - Temporal patches: {summary_stats['temporal_patches']}")
    print(f"  - Non-temporal patches: {summary_stats['non_temporal_patches']}")
    print(f"  - Total area: {summary_stats['total_area_km2']:.1f} km²")
    print(f"  - Reference CRS: {summary_stats['reference_crs']}")
    print(f"  - Reference resolution: {summary_stats['reference_resolution']}m")
    print(f"  - Reference bands: {summary_stats['reference_bands']}")
    
    # Detect temporal mode from first patch
    reference_patch = patches[0]
    is_temporal = reference_patch.temporal_mode
    print(f"🕐 Temporal mode detected: {is_temporal}")
    
    # Validate model-data compatibility
    if args.model == '2d_unet' and is_temporal:
        raise ValueError("2D U-Net cannot be used with temporal data. Use '3d_unet' or use non-temporal patches.")
    if args.model == '3d_unet' and not is_temporal:
        raise ValueError("3D U-Net requires temporal data. Use '2d_unet' or use temporal patches.")
    
    # Load multi-patch training data
    print("📖 Loading training data from all patches...")
    # Apply GEDI filtering only in training mode
    min_gedi_samples = args.min_gedi_samples if args.mode == 'train' else 0
    combined_features, combined_targets = load_multi_patch_gedi_data(patches, target_band='rh', min_gedi_samples=min_gedi_samples)
    
    print(f"🎯 Combined training dataset:")
    print(f"  - Features shape: {combined_features.shape}")
    print(f"  - Targets shape: {combined_targets.shape}")
    print(f"  - Target range: {combined_targets.min():.1f} - {combined_targets.max():.1f}m")
    
    # Train unified model (same logic as train_multi_patch)
    print(f"🏋️ Training unified {args.model.upper()} model...")
    
    if args.model in ['rf', 'mlp']:
        # Traditional models can train directly on combined features
        feature_names = [f'band_{i+1}' for i in range(combined_features.shape[1])]
        
        model, train_metrics, importance_data = train_model(
            combined_features, combined_targets,
            model_type=args.model,
            batch_size=args.batch_size,
            test_size=args.test_size,
            feature_names=feature_names
        )
        
        # Save model
        if args.model == 'rf':
            model_path = os.path.join(args.output_dir, 'multi_patch_rf_model.pkl')
            joblib.dump(model, model_path)
        else:  # MLP
            model_path = os.path.join(args.output_dir, 'multi_patch_mlp_model.pth')
            if TORCH_AVAILABLE:
                torch.save(model.state_dict(), model_path)
        
        print(f"💾 Saved {args.model.upper()} model to: {model_path}")
        
    else:
        # U-Net models need special handling for multi-patch training
        print("⚠️  U-Net multi-patch training requires patch-level processing")
        print("    Training on first patch and applying to all patches...")
        
        # Load first patch for U-Net training
        first_patch_features, first_patch_target, band_info = load_patch_data(
            reference_patch.file_path, normalize_bands=True
        )
        
        if args.model == '2d_unet':
            model, train_metrics = train_2d_unet(
                first_patch_features, first_patch_target,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                base_channels=args.base_channels,
                huber_delta=args.huber_delta,
                shift_radius=args.shift_radius
            )
            model_path = os.path.join(args.output_dir, 'multi_patch_2d_unet_model.pth')
        else:  # 3d_unet
            model, train_metrics = train_3d_unet(
                first_patch_features, first_patch_target,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                base_channels=args.base_channels,
                huber_delta=args.huber_delta,
                shift_radius=args.shift_radius
            )
            model_path = os.path.join(args.output_dir, 'multi_patch_3d_unet_model.pth')
        
        if TORCH_AVAILABLE:
            torch.save(model.state_dict(), model_path)
        print(f"💾 Saved {args.model.upper()} model to: {model_path}")
        
        importance_data = {}
    
    # Save training metrics
    metrics_file = os.path.join(args.output_dir, 'multi_patch_training_metrics.json')
    save_training_metrics(train_metrics, importance_data, metrics_file)
    
    # Generate predictions for all patches if requested
    if args.generate_prediction:
        print("🔮 Generating predictions for all patches...")
        
        patch_predictions = {}
        
        for i, patch_info in enumerate(tqdm(patches, desc="Generating predictions")):
            try:
                prediction_array = generate_patch_prediction(
                    model, patch_info, args.model, is_temporal
                )
                
                # Save individual patch prediction
                patch_pred_filename = f"prediction_{args.model}_{patch_info.patch_id}.tif"
                patch_pred_path = os.path.join(args.output_dir, patch_pred_filename)
                
                # Save with same georeference as input patch
                with rasterio.open(patch_info.file_path) as src:
                    profile = src.profile.copy()
                    profile.update(count=1, dtype='float32')
                    
                    with rasterio.open(patch_pred_path, 'w', **profile) as dst:
                        dst.write(prediction_array.astype('float32'), 1)
                
                patch_predictions[patch_info.patch_id] = patch_pred_path
                
            except Exception as e:
                print(f"⚠️  Error generating prediction for {patch_info.patch_id}: {e}")
                continue
        
        print(f"✅ Generated predictions for {len(patch_predictions)}/{len(patches)} patches")
        
        # Create spatial mosaic if requested
        if (args.merge_predictions or args.create_spatial_mosaic) and patch_predictions:
            print(f"🔗 Creating spatial mosaic using '{args.merge_strategy}' strategy...")
            
            if USE_ENHANCED_MERGER:
                # Use enhanced spatial merger with improved NaN handling
                merger = EnhancedSpatialMerger(merge_strategy=args.merge_strategy)
                
                # Determine output filename
                if hasattr(args, 'mosaic_name') and args.mosaic_name:
                    if not args.mosaic_name.endswith('.tif'):
                        mosaic_filename = f"{args.mosaic_name}.tif"
                    else:
                        mosaic_filename = args.mosaic_name
                else:
                    mosaic_filename = f'spatial_mosaic_{args.model}.tif'
                
                merged_output_path = os.path.join(args.output_dir, mosaic_filename)
                
                merged_path = merger.merge_predictions_from_files(
                    patch_predictions, merged_output_path
                )
            else:
                # Fallback to original merger
                merger = PredictionMerger(patches, merge_strategy=args.merge_strategy)
                merged_output_path = os.path.join(args.output_dir, f'merged_prediction_{args.model}.tif')
                
                merged_path = merger.merge_predictions_from_files(
                    patch_predictions, merged_output_path
                )
            
            print(f"🗺️  Spatial mosaic saved to: {merged_path}")
    
    print("🎉 Multi-patch training from file list completed successfully!")


def train_multi_patch(args):
    """Train model on multiple patches with unified training and optional prediction merging."""
    print(f"🚀 Starting multi-patch training with {args.model.upper()}")
    
    # Initialize patch registry and discover patches
    patch_registry = PatchRegistry()
    patches = patch_registry.discover_patches(args.patch_dir, args.patch_pattern)
    
    if not patches:
        raise ValueError(f"No patches found in {args.patch_dir} matching pattern {args.patch_pattern}")
    
    print(f"📊 Discovered {len(patches)} patches")
    
    # Validate patch consistency
    is_consistent = patch_registry.validate_consistency()
    if not is_consistent:
        print("⚠️  Warning: Inconsistencies detected in patches. Proceeding with caution...")
    
    # Generate and save patch summary
    summary_df = generate_multi_patch_summary(patches)
    summary_path = os.path.join(args.output_dir, 'patch_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"💾 Saved patch summary to: {summary_path}")
    
    # Get summary statistics
    summary_stats = patch_registry.get_patch_summary()
    print(f"📈 Dataset summary:")
    print(f"  - Total patches: {summary_stats['total_patches']}")
    print(f"  - Temporal patches: {summary_stats['temporal_patches']}")
    print(f"  - Non-temporal patches: {summary_stats['non_temporal_patches']}")
    print(f"  - Total area: {summary_stats['total_area_km2']:.1f} km²")
    print(f"  - Reference CRS: {summary_stats['reference_crs']}")
    print(f"  - Reference resolution: {summary_stats['reference_resolution']}m")
    print(f"  - Reference bands: {summary_stats['reference_bands']}")
    
    # Detect temporal mode from first patch
    reference_patch = patches[0]
    is_temporal = reference_patch.temporal_mode
    print(f"🕐 Temporal mode detected: {is_temporal}")
    
    # Validate model-data compatibility
    if args.model == '2d_unet' and is_temporal:
        raise ValueError("2D U-Net cannot be used with temporal data. Use '3d_unet' or use non-temporal patches.")
    if args.model == '3d_unet' and not is_temporal:
        raise ValueError("3D U-Net requires temporal data. Use '2d_unet' or use temporal patches.")
    
    # Load multi-patch training data
    print("📖 Loading training data from all patches...")
    # Apply GEDI filtering only in training mode
    min_gedi_samples = args.min_gedi_samples if args.mode == 'train' else 0
    combined_features, combined_targets = load_multi_patch_gedi_data(patches, target_band='rh', min_gedi_samples=min_gedi_samples)
    
    print(f"🎯 Combined training dataset:")
    print(f"  - Features shape: {combined_features.shape}")
    print(f"  - Targets shape: {combined_targets.shape}")
    print(f"  - Target range: {combined_targets.min():.1f} - {combined_targets.max():.1f}m")
    
    # Train unified model
    print(f"🏋️ Training unified {args.model.upper()} model...")
    
    if args.model in ['rf', 'mlp']:
        # Traditional models can train directly on combined features
        feature_names = [f'band_{i+1}' for i in range(combined_features.shape[1])]
        
        model, train_metrics, importance_data = train_model(
            combined_features, combined_targets,
            model_type=args.model,
            batch_size=args.batch_size,
            test_size=args.test_size,
            feature_names=feature_names
        )
        
        # Save model
        if args.model == 'rf':
            model_path = os.path.join(args.output_dir, 'multi_patch_rf_model.pkl')
            joblib.dump(model, model_path)
        else:  # MLP
            model_path = os.path.join(args.output_dir, 'multi_patch_mlp_model.pth')
            if TORCH_AVAILABLE:
                torch.save(model.state_dict(), model_path)
        
        print(f"💾 Saved {args.model.upper()} model to: {model_path}")
        
    else:
        # U-Net models need special handling for multi-patch training
        # For now, we'll train on the combined dataset by creating synthetic patches
        print("⚠️  U-Net multi-patch training requires patch-level processing")
        print("    Training on first patch and applying to all patches...")
        
        # Load first patch for U-Net training
        first_patch_features, first_patch_target, band_info = load_patch_data(
            reference_patch.file_path, normalize_bands=True
        )
        
        if args.model == '2d_unet':
            model, train_metrics = train_2d_unet(
                first_patch_features, first_patch_target,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                base_channels=args.base_channels,
                huber_delta=args.huber_delta,
                shift_radius=args.shift_radius
            )
            model_path = os.path.join(args.output_dir, 'multi_patch_2d_unet_model.pth')
        else:  # 3d_unet
            model, train_metrics = train_3d_unet(
                first_patch_features, first_patch_target,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                base_channels=args.base_channels,
                huber_delta=args.huber_delta,
                shift_radius=args.shift_radius
            )
            model_path = os.path.join(args.output_dir, 'multi_patch_3d_unet_model.pth')
        
        if TORCH_AVAILABLE:
            torch.save(model.state_dict(), model_path)
        print(f"💾 Saved {args.model.upper()} model to: {model_path}")
        
        importance_data = {}
    
    # Save training metrics
    metrics_file = os.path.join(args.output_dir, 'multi_patch_training_metrics.json')
    save_training_metrics(train_metrics, importance_data, metrics_file)
    
    # Generate predictions for all patches if requested
    if args.generate_prediction:
        print("🔮 Generating predictions for all patches...")
        
        patch_predictions = {}
        
        for i, patch_info in enumerate(tqdm(patches, desc="Generating predictions")):
            try:
                prediction_array = generate_patch_prediction(
                    model, patch_info, args.model, is_temporal
                )
                
                # Save individual patch prediction
                patch_pred_filename = f"prediction_{args.model}_{patch_info.patch_id}.tif"
                patch_pred_path = os.path.join(args.output_dir, patch_pred_filename)
                
                # Save with same georeference as input patch
                with rasterio.open(patch_info.file_path) as src:
                    profile = src.profile.copy()
                    profile.update(count=1, dtype='float32')
                    
                    with rasterio.open(patch_pred_path, 'w', **profile) as dst:
                        dst.write(prediction_array.astype('float32'), 1)
                
                patch_predictions[patch_info.patch_id] = patch_pred_path
                
            except Exception as e:
                print(f"⚠️  Error generating prediction for {patch_info.patch_id}: {e}")
                continue
        
        print(f"✅ Generated predictions for {len(patch_predictions)}/{len(patches)} patches")
        
        # Create spatial mosaic if requested
        if (args.merge_predictions or args.create_spatial_mosaic) and patch_predictions:
            print(f"🔗 Creating spatial mosaic using '{args.merge_strategy}' strategy...")
            
            if USE_ENHANCED_MERGER:
                # Use enhanced spatial merger with improved NaN handling
                merger = EnhancedSpatialMerger(merge_strategy=args.merge_strategy)
                
                # Determine output filename
                if hasattr(args, 'mosaic_name') and args.mosaic_name:
                    if not args.mosaic_name.endswith('.tif'):
                        mosaic_filename = f"{args.mosaic_name}.tif"
                    else:
                        mosaic_filename = args.mosaic_name
                else:
                    mosaic_filename = f'spatial_mosaic_{args.model}.tif'
                
                merged_output_path = os.path.join(args.output_dir, mosaic_filename)
                
                merged_path = merger.merge_predictions_from_files(
                    patch_predictions, merged_output_path
                )
            else:
                # Fallback to original merger
                merger = PredictionMerger(patches, merge_strategy=args.merge_strategy)
                merged_output_path = os.path.join(args.output_dir, f'merged_prediction_{args.model}.tif')
                
                merged_path = merger.merge_predictions_from_files(
                    patch_predictions, merged_output_path
                )
            
            print(f"🗺️  Spatial mosaic saved to: {merged_path}")
    
    print("🎉 Multi-patch training completed successfully!")


def generate_patch_prediction(model, patch_info: PatchInfo, model_type: str, is_temporal: bool) -> np.ndarray:
    """Generate prediction for a single patch using trained model."""
    # Load patch data
    features, _, _ = load_patch_data(patch_info.file_path, normalize_bands=True)
    
    # Store original dimensions for reshaping output
    original_height, original_width = features.shape[-2], features.shape[-1]
    
    if model_type in ['rf', 'mlp']:
        # Prepare full feature data for traditional models
        full_features = features.reshape(features.shape[0], -1).T  # (n_pixels, n_bands)
        
        # Remove any NaN/inf values
        valid_mask = np.all(np.isfinite(full_features), axis=1)
        
        if model_type == 'rf':
            predictions = np.full(valid_mask.shape[0], np.nan)
            predictions[valid_mask] = model.predict(full_features[valid_mask])
        else:  # MLP
            if TORCH_AVAILABLE:
                model.eval()
                with torch.no_grad():
                    input_tensor = torch.FloatTensor(full_features[valid_mask])
                    if torch.cuda.is_available():
                        input_tensor = input_tensor.cuda()
                        model = model.cuda()
                    pred_tensor = model(input_tensor)
                    pred_values = pred_tensor.cpu().numpy().flatten()
                    
                predictions = np.full(valid_mask.shape[0], np.nan)
                predictions[valid_mask] = pred_values
            else:
                raise ImportError("PyTorch is required for MLP prediction")
        
        # Reshape to original dimensions
        predictions = predictions.reshape(features.shape[1], features.shape[2])
        
    else:  # U-Net models
        # Resize features to 256x256 if needed for U-Net models
        if features.shape[-2] != 256 or features.shape[-1] != 256:
            from scipy.ndimage import zoom
            print(f"Resizing patch from {features.shape[-2]}x{features.shape[-1]} to 256x256")
            
            if len(features.shape) == 3:  # (bands, height, width)
                scale_h = 256 / features.shape[1]
                scale_w = 256 / features.shape[2]
                resized_features = np.zeros((features.shape[0], 256, 256), dtype=features.dtype)
                for i in range(features.shape[0]):
                    resized_features[i] = zoom(features[i], (scale_h, scale_w), order=1)
                features = resized_features
            elif len(features.shape) == 4:  # (bands, time, height, width)
                scale_h = 256 / features.shape[2]
                scale_w = 256 / features.shape[3]
                resized_features = np.zeros((features.shape[0], features.shape[1], 256, 256), dtype=features.dtype)
                for i in range(features.shape[0]):
                    for j in range(features.shape[1]):
                        resized_features[i, j] = zoom(features[i, j], (scale_h, scale_w), order=1)
                features = resized_features
        
        if model_type == '2d_unet':
            # Non-temporal mode
            if len(features.shape) == 4:  # (bands, time, height, width)
                combined_features = features.reshape(-1, features.shape[2], features.shape[3])
            else:  # (bands, height, width)
                combined_features = features
                
            input_tensor = torch.FloatTensor(combined_features).unsqueeze(0)
            
        else:  # 3d_unet
            # Temporal mode
            n_bands = features.shape[0]
            n_temporal_expected = int(12 * (n_bands // 12))
            
            if n_bands >= n_temporal_expected:
                n_features_per_month = n_bands // 12
                combined_features = features[:n_temporal_expected].reshape(
                    n_features_per_month, 12, features.shape[1], features.shape[2]
                )
                input_tensor = torch.FloatTensor(combined_features).unsqueeze(0)
            else:
                raise ValueError(f"Insufficient bands for temporal processing: got {n_bands}")
        
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
            model = model.cuda()
        
        model.eval()
        with torch.no_grad():
            predictions = model(input_tensor)
            predictions = predictions.squeeze().cpu().numpy()
            
        # Resize predictions back to original dimensions if needed
        if predictions.shape != (original_height, original_width):
            from scipy.ndimage import zoom
            scale_h = original_height / predictions.shape[0]
            scale_w = original_width / predictions.shape[1]
            predictions = zoom(predictions, (scale_h, scale_w), order=1)
    
    return predictions


def save_training_metrics(train_metrics: Dict, importance_data: Dict, metrics_file: str):
    """Save training metrics and importance data to JSON file."""
    def convert_numpy_types(obj):
        """Convert numpy types to Python types for JSON serialization."""
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Unified Patch-Based Canopy Height Model Training and Prediction')
    
    # Input modes - single patch, multi-patch directory, or file list
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--patch-path', type=str,
                           help='Path to single patch TIF file with GEDI rh band')
    input_group.add_argument('--patch-dir', type=str,
                           help='Directory containing multiple patch TIF files')
    input_group.add_argument('--patch-files', type=str,
                           help='Comma-separated list of patch TIF file paths')
    
    # Model selection
    parser.add_argument('--model', type=str, required=True,
                       choices=['rf', 'mlp', '2d_unet', '3d_unet'],
                       help='Model type to train')
    
    # Output configuration
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for models and predictions')
    
    # Multi-patch specific options
    parser.add_argument('--patch-pattern', type=str, default='*.tif',
                       help='File pattern to match patches (e.g., "*_temporal_*.tif")')
    parser.add_argument('--merge-predictions', action='store_true',
                       help='Merge individual patch predictions into single GeoTIFF')
    parser.add_argument('--merge-strategy', type=str, default='average',
                       choices=['average', 'maximum', 'first'],
                       help='Strategy for merging overlapping predictions')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs for neural networks')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                       help='Learning rate for neural networks')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for neural networks')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Proportion of data to use for validation')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay for neural networks')
    
    # Model parameters
    parser.add_argument('--n-estimators', type=int, default=100,
                       help='Number of estimators for Random Forest')
    parser.add_argument('--max-depth', type=int, default=10,
                       help='Maximum depth for Random Forest')
    parser.add_argument('--hidden-layers', type=str, default='128,64',
                       help='Hidden layer sizes for MLP (comma-separated)')
    parser.add_argument('--base-channels', type=int, default=32,
                       help='Base channels for U-Net models')
    parser.add_argument('--huber-delta', type=float, default=1.0,
                       help='Delta parameter for Huber loss')
    parser.add_argument('--shift-radius', type=int, default=1,
                       help='Spatial shift radius for GEDI alignment')
    
    # Enhanced Training Options (New Features)
    parser.add_argument('--augment', action='store_true',
                       help='Enable data augmentation (12x spatial transformations)')
    parser.add_argument('--augment-factor', type=int, default=12,
                       help='Number of augmentations per patch (default: 12)')
    parser.add_argument('--validation-split', type=float, default=0.2,
                       help='Fraction of patches for validation (default: 0.2)')
    parser.add_argument('--early-stopping-patience', type=int, default=15,
                       help='Epochs to wait before early stopping (default: 15)')
    parser.add_argument('--checkpoint-freq', type=int, default=10,
                       help='Save checkpoint every N epochs (default: 10)')
    parser.add_argument('--resume-from', type=str,
                       help='Resume training from checkpoint file')
    parser.add_argument('--use-enhanced-training', action='store_true',
                       help='Use enhanced training pipeline with full multi-patch support')
    
    # Prediction options
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'predict'],
                       help='Operation mode: train models or generate predictions only')
    parser.add_argument('--model-path', type=str,
                       help='Path to pre-trained model for prediction mode')
    parser.add_argument('--generate-prediction', action='store_true',
                       help='Generate prediction maps after training')
    parser.add_argument('--save-model', action='store_true',
                       help='Save trained model to disk')
    
    # GEDI filtering options
    parser.add_argument('--min-gedi-samples', type=int, default=10,
                       help='Minimum number of valid GEDI samples per patch for training (default: 10)')
    
    # Verbose output
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.verbose:
        print("🚀 Starting unified patch-based training system")
        print(f"Model: {args.model}")
        print(f"Output directory: {args.output_dir}")
    
    # Route to appropriate training mode
    if args.patch_path:
        # Single patch mode
        if args.verbose:
            print(f"📄 Single patch mode: {args.patch_path}")
        
        # Check if patch exists
        if not os.path.exists(args.patch_path):
            print(f"❌ Error: Patch file not found: {args.patch_path}")
            exit(1)
        
        # Load patch data
        try:
            features, gedi_target, band_info = load_patch_data(args.patch_path, normalize_bands=True)
            
            # Determine temporal mode automatically
            temporal_mode = detect_temporal_mode(band_info)
            
            if args.verbose:
                print(f"✅ Loaded patch: {features.shape[0]} bands, {features.shape[1]}x{features.shape[2]} pixels")
                print(f"📊 Mode detected: {'Temporal' if temporal_mode else 'Non-temporal'}")
            
        except Exception as e:
            print(f"❌ Error loading patch: {e}")
            exit(1)
        
        # Train model
        try:
            if args.model == 'rf':
                model, metrics, importance = train_random_forest(
                    features, gedi_target,
                    n_estimators=args.n_estimators,
                    max_depth=args.max_depth
                )
            elif args.model == 'mlp':
                hidden_sizes = [int(x.strip()) for x in args.hidden_layers.split(',')]
                model, metrics, importance = train_mlp(
                    features, gedi_target,
                    hidden_layers=hidden_sizes,
                    epochs=args.epochs,
                    learning_rate=args.learning_rate
                )
            elif args.model == '2d_unet':
                model, metrics = train_2d_unet(
                    features, gedi_target,
                    epochs=args.epochs,
                    learning_rate=args.learning_rate
                )
                importance = {}
            elif args.model == '3d_unet':
                if not temporal_mode:
                    print("⚠️  Warning: Using 3D U-Net on non-temporal data may not be optimal")
                model, metrics = train_3d_unet(
                    features, gedi_target,
                    epochs=args.epochs,
                    learning_rate=args.learning_rate
                )
                importance = {}
            
            if args.verbose:
                print(f"✅ Training completed: {args.model}")
                print(f"📈 Final metrics: {metrics}")
            
        except Exception as e:
            print(f"❌ Error during training: {e}")
            exit(1)
        
        # Save model if requested
        if args.save_model:
            model_filename = f"model_{args.model}_{Path(args.patch_path).stem}.pkl"
            model_path = os.path.join(args.output_dir, model_filename)
            
            if args.model in ['rf']:
                joblib.dump(model, model_path)
            elif args.model in ['mlp', '2d_unet', '3d_unet']:
                torch.save(model.state_dict(), model_path)
            
            if args.verbose:
                print(f"💾 Model saved to: {model_path}")
        
        # Save training metrics
        metrics_filename = f"metrics_{args.model}_{Path(args.patch_path).stem}.json"
        metrics_path = os.path.join(args.output_dir, metrics_filename)
        save_training_metrics(metrics, importance, metrics_path)
        
        # Generate prediction if requested
        if args.generate_prediction:
            try:
                if args.model in ['rf', 'mlp']:
                    # Full patch prediction for traditional models
                    full_features = features.reshape(features.shape[0], -1).T
                    valid_mask = np.all(np.isfinite(full_features), axis=1)
                    
                    predictions = np.full(valid_mask.shape[0], np.nan)
                    if args.model == 'rf':
                        predictions[valid_mask] = model.predict(full_features[valid_mask])
                    else:  # MLP
                        model.eval()
                        with torch.no_grad():
                            input_tensor = torch.FloatTensor(full_features[valid_mask])
                            if torch.cuda.is_available():
                                input_tensor = input_tensor.cuda()
                            pred_tensor = model(input_tensor)
                            predictions[valid_mask] = pred_tensor.cpu().numpy().flatten()
                    
                    predictions = predictions.reshape(features.shape[1], features.shape[2])
                
                else:  # U-Net models
                    # Store original dimensions for output resizing
                    original_height, original_width = features.shape[-2], features.shape[-1]
                    
                    # Resize features to 256x256 if needed for U-Net models
                    if features.shape[-2] != 256 or features.shape[-1] != 256:
                        from scipy.ndimage import zoom
                        print(f"Resizing patch from {features.shape[-2]}x{features.shape[-1]} to 256x256")
                        
                        if len(features.shape) == 3:  # (bands, height, width)
                            scale_h = 256 / features.shape[1]
                            scale_w = 256 / features.shape[2]
                            resized_features = np.zeros((features.shape[0], 256, 256), dtype=features.dtype)
                            for i in range(features.shape[0]):
                                resized_features[i] = zoom(features[i], (scale_h, scale_w), order=1)
                            features = resized_features
                        elif len(features.shape) == 4:  # (bands, time, height, width)
                            scale_h = 256 / features.shape[2]
                            scale_w = 256 / features.shape[3]
                            resized_features = np.zeros((features.shape[0], features.shape[1], 256, 256), dtype=features.dtype)
                            for i in range(features.shape[0]):
                                for j in range(features.shape[1]):
                                    resized_features[i, j] = zoom(features[i, j], (scale_h, scale_w), order=1)
                            features = resized_features
                    
                    model.eval()
                    with torch.no_grad():
                        if args.model == '2d_unet':
                            if len(features.shape) == 4:
                                combined_features = features.reshape(-1, features.shape[2], features.shape[3])
                            else:
                                combined_features = features
                            input_tensor = torch.FloatTensor(combined_features).unsqueeze(0)
                        else:  # 3d_unet
                            n_bands = features.shape[0]
                            n_features_per_month = n_bands // 12
                            temporal_features = features[:n_features_per_month*12].reshape(
                                n_features_per_month, 12, features.shape[1], features.shape[2]
                            )
                            input_tensor = torch.FloatTensor(temporal_features).unsqueeze(0)
                        
                        if torch.cuda.is_available():
                            input_tensor = input_tensor.cuda()
                        
                        predictions = model(input_tensor).squeeze().cpu().numpy()
                        
                        # Resize predictions back to original dimensions if needed
                        if predictions.shape != (original_height, original_width):
                            from scipy.ndimage import zoom
                            scale_h = original_height / predictions.shape[0]
                            scale_w = original_width / predictions.shape[1]
                            predictions = zoom(predictions, (scale_h, scale_w), order=1)
                            print(f"Resized prediction back to original dimensions: {predictions.shape}")
                
                # Save prediction
                pred_filename = f"prediction_{args.model}_{Path(args.patch_path).stem}.tif"
                pred_path = os.path.join(args.output_dir, pred_filename)
                
                with rasterio.open(args.patch_path) as src:
                    profile = src.profile.copy()
                    profile.update(count=1, dtype='float32')
                    
                    with rasterio.open(pred_path, 'w', **profile) as dst:
                        dst.write(predictions.astype('float32'), 1)
                
                if args.verbose:
                    print(f"🗺️  Prediction saved to: {pred_path}")
                    print(f"📊 Prediction range: {np.nanmin(predictions):.2f} - {np.nanmax(predictions):.2f}m")
                
            except Exception as e:
                print(f"❌ Error generating prediction: {e}")
    
    elif args.patch_dir:
        # Multi-patch mode
        if args.verbose:
            print(f"📁 Multi-patch mode: {args.patch_dir}")
            print(f"🔍 Pattern: {args.patch_pattern}")
        
        try:
            # Check if enhanced training is requested for U-Net models
            if args.use_enhanced_training and args.model in ['2d_unet', '3d_unet']:
                # Use enhanced U-Net training pipeline
                patch_files = glob.glob(os.path.join(args.patch_dir, args.patch_pattern))
                if not patch_files:
                    print(f"❌ No patches found matching pattern: {args.patch_pattern}")
                    exit(1)
                
                print(f"🚀 Using enhanced training pipeline for {args.model.upper()}")
                trainer = EnhancedUNetTrainer(model_type=args.model)
                
                training_results = trainer.train_multi_patch_unet(
                    patch_files=patch_files,
                    output_dir=args.output_dir,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    validation_split=args.validation_split,
                    early_stopping_patience=args.early_stopping_patience,
                    augment=args.augment,
                    checkpoint_freq=args.checkpoint_freq
                )
                
                print(f"🎉 Enhanced training completed!")
                print(f"📊 Results: {training_results}")
                
            else:
                # Use traditional training pipeline
                train_multi_patch(args)
                
        except Exception as e:
            print(f"❌ Error in multi-patch training: {e}")
            exit(1)
    
    elif args.patch_files:
        # Multi-patch mode with file list
        if args.verbose:
            print(f"📋 Multi-patch mode: File list")
            print(f"🔍 Files: {args.patch_files}")
        
        try:
            # Check if enhanced training is requested for U-Net models
            if args.use_enhanced_training and args.model in ['2d_unet', '3d_unet']:
                # Use enhanced U-Net training pipeline
                patch_files = [f.strip() for f in args.patch_files.split(',')]
                
                # Validate patch files exist
                valid_files = []
                for patch_file in patch_files:
                    if os.path.exists(patch_file):
                        valid_files.append(patch_file)
                    else:
                        print(f"⚠️  Warning: Patch file not found: {patch_file}")
                
                if not valid_files:
                    print(f"❌ No valid patch files found")
                    exit(1)
                
                print(f"🚀 Using enhanced training pipeline for {args.model.upper()}")
                trainer = EnhancedUNetTrainer(model_type=args.model)
                
                training_results = trainer.train_multi_patch_unet(
                    patch_files=valid_files,
                    output_dir=args.output_dir,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    validation_split=args.validation_split,
                    early_stopping_patience=args.early_stopping_patience,
                    augment=args.augment,
                    checkpoint_freq=args.checkpoint_freq
                )
                
                print(f"🎉 Enhanced training completed!")
                print(f"📊 Results: {training_results}")
                
            else:
                # Use traditional training pipeline
                train_multi_patch_from_files(args)
                
        except Exception as e:
            print(f"❌ Error in multi-patch training: {e}")
            exit(1)
    
    print("🎉 Training completed successfully!")

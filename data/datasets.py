"""PyTorch datasets for patch-based and sparse GEDI training."""

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional
from pathlib import Path

from .augmentation import SpatialAugmentation, ensure_256x256


class AugmentedPatchDataset(Dataset):
    """
    PyTorch dataset with comprehensive spatial augmentation.
    
    FIXED: Handles negative stride tensor issues in augmentation.
    """
    
    def __init__(self, 
                 patch_files: List[str],
                 augment: bool = True,
                 augment_factor: int = 12,
                 validation_mode: bool = False):
        """
        Initialize augmented patch dataset.
        
        Args:
            patch_files: List of patch file paths
            augment: Whether to apply augmentation
            augment_factor: Number of augmentation combinations (max 12)
            validation_mode: If True, disable augmentation
        """
        self.patch_files = patch_files
        self.augment = augment and not validation_mode
        self.augment_factor = augment_factor if self.augment else 1
        self.validation_mode = validation_mode
        
        # Initialize augmentation
        self.spatial_aug = SpatialAugmentation(augment_factor)
        
        # Cache patch info for efficiency
        self._patch_info = {}
        self._load_patch_info()
        
    def _load_patch_info(self):
        """Load basic info about each patch for efficiency."""
        for i, patch_file in enumerate(self.patch_files):
            try:
                with rasterio.open(patch_file) as src:
                    self._patch_info[i] = {
                        'shape': (src.count, src.height, src.width),
                        'dtype': src.dtypes[0],
                        'crs': src.crs,
                        'transform': src.transform
                    }
            except Exception as e:
                print(f"Warning: Could not load info for {patch_file}: {e}")
                self._patch_info[i] = None
    
    def __len__(self) -> int:
        """Return total number of samples (patches × augmentations)."""
        return len(self.patch_files) * self.augment_factor
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get augmented patch sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (features, target, mask) tensors
        """
        # Determine patch and augmentation
        patch_idx = idx // self.augment_factor
        augment_id = idx % self.augment_factor if self.augment else 0
        
        patch_file = self.patch_files[patch_idx]
        
        # Load patch data
        with rasterio.open(patch_file) as src:
            # Read all bands
            data = src.read()  # (bands, height, width)
            
            # Separate features and target
            features = data[:-1]  # All bands except last
            target = data[-1]     # Last band (GEDI heights)
        
        # Handle temporal data detection
        if self._is_temporal_patch(patch_file):
            features = self._reshape_temporal_features(features)
        
        # Apply augmentation if enabled
        if self.augment and augment_id > 0:
            features, target = self.spatial_aug.apply_augmentation(
                features, target, augment_id
            )
        
        # Create mask for valid GEDI pixels
        mask = (target > 0) & np.isfinite(target)
        
        # Ensure 256x256 dimensions for U-Net compatibility
        features, target, mask = ensure_256x256(features, target, mask)
        
        # Convert to tensors with positive strides
        features_tensor = torch.FloatTensor(features.copy())
        target_tensor = torch.FloatTensor(target.copy())
        mask_tensor = torch.BoolTensor(mask.copy())
        
        return features_tensor, target_tensor, mask_tensor
    
    def _is_temporal_patch(self, patch_file: str) -> bool:
        """Check if patch contains temporal data based on filename/bands."""
        # Check for temporal indicators in filename
        filename = Path(patch_file).name
        return 'temporal' in filename.lower() or '_M01' in filename
    
    def _reshape_temporal_features(self, features: np.ndarray) -> np.ndarray:
        """
        Reshape features for temporal processing.
        
        Args:
            features: Features array (bands, height, width)
            
        Returns:
            Reshaped features (channels, time, height, width) or (channels, height, width)
        """
        # For temporal data, group bands by month
        # Assuming 12 months with multiple bands per month
        total_bands = features.shape[0]
        
        if total_bands % 12 == 0:
            # Temporal data: reshape to (bands_per_month, 12, height, width)
            bands_per_month = total_bands // 12
            height, width = features.shape[1], features.shape[2]
            
            features_temporal = features.reshape(
                bands_per_month, 12, height, width
            )
            return features_temporal
        else:
            # Non-temporal data: return as is
            return features
    
    def get_patch_info(self, patch_idx: int) -> dict:
        """Get information about a specific patch."""
        return self._patch_info.get(patch_idx, {})
    
    def get_augmentation_info(self, idx: int) -> str:
        """Get human-readable augmentation description for sample."""
        augment_id = idx % self.augment_factor if self.augment else 0
        return self.spatial_aug.get_augmentation_info(augment_id)


class SparseGEDIDataset(Dataset):
    """Dataset for sparse GEDI pixel extraction from patches."""
    
    def __init__(self, 
                 patch_files: List[str],
                 min_gedi_pixels: int = 10):
        """
        Initialize sparse GEDI dataset.
        
        Args:
            patch_files: List of patch file paths
            min_gedi_pixels: Minimum GEDI pixels required per patch
        """
        self.patch_files = patch_files
        self.min_gedi_pixels = min_gedi_pixels
        
        # Extract valid samples
        self.valid_samples = []
        self._extract_valid_samples()
    
    def _extract_valid_samples(self):
        """Extract valid GEDI pixels from all patches."""
        for patch_file in self.patch_files:
            try:
                with rasterio.open(patch_file) as src:
                    data = src.read()
                    
                    features = data[:-1]  # All bands except last
                    target = data[-1]     # Last band (GEDI heights)
                    
                    # Find valid GEDI pixels
                    valid_mask = (target > 0) & np.isfinite(target)
                    valid_indices = np.where(valid_mask)
                    
                    if len(valid_indices[0]) >= self.min_gedi_pixels:
                        # Extract features and targets for valid pixels
                        valid_features = features[:, valid_indices[0], valid_indices[1]].T
                        valid_targets = target[valid_indices[0], valid_indices[1]]
                        
                        # Store samples
                        for i in range(len(valid_targets)):
                            self.valid_samples.append({
                                'features': valid_features[i],
                                'target': valid_targets[i],
                                'patch_file': patch_file,
                                'pixel_location': (valid_indices[0][i], valid_indices[1][i])
                            })
                            
            except Exception as e:
                print(f"Warning: Could not process {patch_file}: {e}")
    
    def __len__(self) -> int:
        """Return number of valid GEDI samples."""
        return len(self.valid_samples)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, float]:
        """
        Get sparse GEDI sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (features, target)
        """
        sample = self.valid_samples[idx]
        return sample['features'], sample['target']
    
    def get_features_and_targets(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get all features and targets as arrays for sklearn."""
        if not self.valid_samples:
            return np.array([]), np.array([])
        
        features = np.array([sample['features'] for sample in self.valid_samples])
        targets = np.array([sample['target'] for sample in self.valid_samples])
        
        return features, targets
    
    def get_patch_summary(self) -> dict:
        """Get summary of samples per patch."""
        summary = {}
        for sample in self.valid_samples:
            patch_file = sample['patch_file']
            if patch_file not in summary:
                summary[patch_file] = 0
            summary[patch_file] += 1
        return summary
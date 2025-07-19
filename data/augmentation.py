"""Spatial augmentation functions with fixed negative stride handling."""

import numpy as np
from typing import Tuple
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple
from .patch_loader import load_patch_data

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
                 augment_factor: int = 12, validation_mode: bool = False,
                 supervision_mode: str = "gedi_only", band_selection: str = "all",
                 features: np.ndarray = None, target: np.ndarray = None):
        """
        Initialize augmented dataset.
        
        Args:
            patch_files: List of patch TIF file paths
            augment: Enable spatial augmentation (default: True)
            augment_factor: Number of augmentations per patch (default: 12)
            validation_mode: If True, only use original patches (no augmentation)
            supervision_mode: "reference" or "gedi_only"
            band_selection: "all", "embedding", "original", "auxiliary"
        """
        self.patch_files = patch_files
        self.augment = augment and not validation_mode
        self.augment_factor = augment_factor if self.augment else 1
        self.validation_mode = validation_mode
        self.supervision_mode = supervision_mode
        self.band_selection = band_selection
        self.features = features
        self.target = target
        
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
        
        # Load patch data with correct supervision_mode and band_selection
        features = self.features
        target = self.target
        
        # features, target, _ = load_patch_data(
        #     patch_info['file'], 
        #     supervision_mode=self.supervision_mode, 
        #     band_selection=self.band_selection,
        #     normalize_bands=True
        # )
        
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

class SpatialAugmentation:
    """Spatial augmentation with 12x transformations (3 flips × 4 rotations)."""
    
    def __init__(self, augment_factor: int = 12):
        """
        Initialize spatial augmentation.
        
        Args:
            augment_factor: Number of augmentation combinations (max 12)
        """
        self.augment_factor = min(augment_factor, 12)
    
    def apply_augmentation(self, 
                          features: np.ndarray, 
                          target: np.ndarray, 
                          augment_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply consistent spatial augmentation to features and target.
        
        FIXED: Creates copies to avoid negative stride tensor issues
        
        Augmentation combinations:
        - ID 0: No augmentation (original)
        - ID 1-3: Horizontal, vertical, both flips  
        - ID 4-11: Above + 90°, 180°, 270° rotations
        
        Args:
            features: Feature array (C, H, W) or (C, T, H, W)
            target: Target array (H, W)
            augment_id: Augmentation identifier (0-11)
            
        Returns:
            Augmented features and target (with positive strides)
        """
        if augment_id == 0:
            return features.copy(), target.copy()
        
        # Determine flip and rotation
        flip_id = (augment_id - 1) % 3 + 1  # 1, 2, 3
        rotation_id = (augment_id - 1) // 3  # 0, 1, 2, 3
        
        # Create working copies to avoid negative strides
        features_aug = features.copy()
        target_aug = target.copy()
        
        # Apply flips
        if flip_id == 1:  # Horizontal flip
            if len(features_aug.shape) == 3:  # (C, H, W)
                features_aug = np.flip(features_aug, axis=2).copy()
            else:  # (C, T, H, W)
                features_aug = np.flip(features_aug, axis=3).copy()
            target_aug = np.flip(target_aug, axis=1).copy()
        elif flip_id == 2:  # Vertical flip
            if len(features_aug.shape) == 3:  # (C, H, W)
                features_aug = np.flip(features_aug, axis=1).copy()
            else:  # (C, T, H, W)
                features_aug = np.flip(features_aug, axis=2).copy()
            target_aug = np.flip(target_aug, axis=0).copy()
        elif flip_id == 3:  # Both flips
            if len(features_aug.shape) == 3:  # (C, H, W)
                features_aug = np.flip(features_aug, axis=(1, 2)).copy()
            else:  # (C, T, H, W)
                features_aug = np.flip(features_aug, axis=(2, 3)).copy()
            target_aug = np.flip(target_aug, axis=(0, 1)).copy()
        
        # Apply rotations (k * 90 degrees)
        if rotation_id > 0:
            for _ in range(rotation_id):
                if len(features_aug.shape) == 3:  # (C, H, W)
                    features_aug = np.rot90(features_aug, axes=(1, 2)).copy()
                else:  # (C, T, H, W)
                    features_aug = np.rot90(features_aug, axes=(2, 3)).copy()
                target_aug = np.rot90(target_aug).copy()
        
        return features_aug, target_aug
    
    def get_augmentation_info(self, augment_id: int) -> str:
        """Get human-readable description of augmentation."""
        if augment_id == 0:
            return "Original (no augmentation)"
        
        flip_id = (augment_id - 1) % 3 + 1
        rotation_id = (augment_id - 1) // 3
        
        flip_names = {1: "Horizontal flip", 2: "Vertical flip", 3: "Both flips"}
        rotation_names = {0: "", 1: " + 90° rotation", 2: " + 180° rotation", 3: " + 270° rotation"}
        
        return flip_names[flip_id] + rotation_names[rotation_id]


def ensure_256x256(features: np.ndarray, 
                   target: np.ndarray, 
                   mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Ensure arrays are exactly 256x256 for U-Net compatibility.
    
    Args:
        features: Feature array
        target: Target array 
        mask: Mask array
        
    Returns:
        Resized arrays with 256x256 spatial dimensions
    """
    from scipy.ndimage import zoom
    
    target_size = 256
    
    if len(features.shape) == 3:  # (C, H, W)
        current_h, current_w = features.shape[1], features.shape[2]
    else:  # (C, T, H, W)
        current_h, current_w = features.shape[2], features.shape[3]
    
    if current_h == target_size and current_w == target_size:
        return features, target, mask
    
    # Calculate zoom factors
    zoom_h = target_size / current_h
    zoom_w = target_size / current_w
    
    # Resize features
    if len(features.shape) == 3:  # (C, H, W)
        zoom_factors = (1.0, zoom_h, zoom_w)
    else:  # (C, T, H, W)
        zoom_factors = (1.0, 1.0, zoom_h, zoom_w)
    
    features_resized = zoom(features, zoom_factors, order=1)
    
    # Resize target and mask
    target_resized = zoom(target, (zoom_h, zoom_w), order=1)
    mask_resized = zoom(mask.astype(float), (zoom_h, zoom_w), order=0) > 0.5
    
    return features_resized, target_resized, mask_resized
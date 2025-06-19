"""Spatial augmentation functions with fixed negative stride handling."""

import numpy as np
from typing import Tuple


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
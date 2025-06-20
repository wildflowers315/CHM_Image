"""Data preprocessing utilities for patch-based training."""

import numpy as np
import rasterio
from typing import Tuple, Optional, List, Dict
from pathlib import Path

from data.normalization import normalize_band_data


class PatchPreprocessor:
    """Preprocessing utilities for patch data."""
    
    def __init__(self, target_size: int = 256):
        """
        Initialize patch preprocessor.
        
        Args:
            target_size: Target spatial dimensions (e.g., 256 for 256x256)
        """
        self.target_size = target_size
        
    def load_and_preprocess_patch(self, 
                                 patch_file: str,
                                 normalize: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and preprocess a single patch file.
        
        Args:
            patch_file: Path to patch file
            normalize: Whether to apply normalization
            
        Returns:
            Tuple of (features, target, mask)
        """
        with rasterio.open(patch_file) as src:
            # Read all bands
            data = src.read()  # (bands, height, width)
            
            # Separate features and target
            features = data[:-1]  # All bands except last
            target = data[-1]     # Last band (GEDI heights)
        
        # Apply normalization if requested
        if normalize:
            features = self._normalize_features(features, patch_file)
        
        # Create mask for valid GEDI pixels
        mask = (target > 0) & np.isfinite(target)
        
        # Ensure consistent dimensions
        features, target, mask = self._ensure_target_size(features, target, mask)
        
        return features, target, mask
    
    def _normalize_features(self, 
                           features: np.ndarray, 
                           patch_file: str) -> np.ndarray:
        """
        Normalize feature bands using appropriate strategies.
        
        Args:
            features: Feature array (bands, height, width)
            patch_file: Patch file path for band identification
            
        Returns:
            Normalized features
        """
        # Detect band types from filename
        filename = Path(patch_file).name
        is_temporal = 'temporal' in filename.lower() or '_M01' in filename
        
        # Get band names (this is a simplified approach)
        band_names = self._get_band_names(features.shape[0], is_temporal)
        
        # Apply normalization band by band
        normalized_features = np.zeros_like(features)
        
        for i, band_name in enumerate(band_names):
            band_data = features[i]
            
            # Use normalization from data.normalization module
            try:
                normalized_band = normalize_band_data(band_data, band_name)
                normalized_features[i] = normalized_band
            except Exception as e:
                print(f"Warning: Could not normalize band {band_name}: {e}")
                # Fallback: simple standardization
                if np.std(band_data) > 0:
                    normalized_features[i] = (band_data - np.mean(band_data)) / np.std(band_data)
                else:
                    normalized_features[i] = band_data
        
        return normalized_features
    
    def _get_band_names(self, num_bands: int, is_temporal: bool) -> List[str]:
        """
        Get band names based on number of bands and temporal mode.
        
        Args:
            num_bands: Number of feature bands
            is_temporal: Whether data is temporal
            
        Returns:
            List of band names
        """
        # This is a simplified mapping - in practice, you'd want more sophisticated detection
        base_bands = []
        
        if is_temporal:
            # Temporal data: bands repeated for each month
            if num_bands == 196:  # Full temporal
                # S1: 24 bands (VV, VH × 12 months)
                base_bands.extend([f'S1_VV_M{m:02d}' for m in range(1, 13)])
                base_bands.extend([f'S1_VH_M{m:02d}' for m in range(1, 13)])
                
                # S2: 132 bands (11 bands × 12 months)
                s2_bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12', 'NDVI']
                for month in range(1, 13):
                    base_bands.extend([f'S2_{band}_M{month:02d}' for band in s2_bands])
                
                # ALOS2: 24 bands (HH, HV × 12 months)
                base_bands.extend([f'ALOS2_HH_M{m:02d}' for m in range(1, 13)])
                base_bands.extend([f'ALOS2_HV_M{m:02d}' for m in range(1, 13)])
                
                # Other bands (DEM, etc.)
                remaining = num_bands - len(base_bands)
                base_bands.extend([f'other_{i}' for i in range(remaining)])
            else:
                # Simplified temporal
                base_bands = [f'band_{i}' for i in range(num_bands)]
        else:
            # Non-temporal data: ~31 bands
            base_bands = [
                'S1_VV', 'S1_VH',  # Sentinel-1
                'S2_B2', 'S2_B3', 'S2_B4', 'S2_B5', 'S2_B6',  # Sentinel-2
                'S2_B7', 'S2_B8', 'S2_B8A', 'S2_B11', 'S2_B12', 'S2_NDVI',
                'ALOS2_HH', 'ALOS2_HV',  # ALOS-2
                'DEM', 'slope', 'aspect', 'canopy_height'  # Topographic + height
            ]
            
            # Pad if needed
            while len(base_bands) < num_bands:
                base_bands.append(f'band_{len(base_bands)}')
        
        return base_bands[:num_bands]
    
    def _ensure_target_size(self, 
                           features: np.ndarray, 
                           target: np.ndarray, 
                           mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Ensure arrays have target spatial dimensions.
        
        Args:
            features: Feature array
            target: Target array
            mask: Mask array
            
        Returns:
            Resized arrays
        """
        from scipy.ndimage import zoom
        
        if len(features.shape) == 3:  # (C, H, W)
            current_h, current_w = features.shape[1], features.shape[2]
        else:  # (C, T, H, W)
            current_h, current_w = features.shape[2], features.shape[3]
        
        if current_h == self.target_size and current_w == self.target_size:
            return features, target, mask
        
        # Calculate zoom factors
        zoom_h = self.target_size / current_h
        zoom_w = self.target_size / current_w
        
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
    
    def validate_patch(self, patch_file: str) -> Dict[str, any]:
        """
        Validate a patch file and return information.
        
        Args:
            patch_file: Path to patch file
            
        Returns:
            Validation results dictionary
        """
        try:
            with rasterio.open(patch_file) as src:
                shape = (src.count, src.height, src.width)
                dtype = src.dtypes[0]
                
                # Read target band (last band)
                target = src.read(src.count)
                
                # Check for valid GEDI pixels
                valid_mask = (target > 0) & np.isfinite(target)
                n_valid_pixels = valid_mask.sum()
                total_pixels = target.size
                
                # Check data ranges
                target_range = (np.nanmin(target[valid_mask]), np.nanmax(target[valid_mask])) if n_valid_pixels > 0 else (0, 0)
                
                return {
                    'valid': True,
                    'shape': shape,
                    'dtype': str(dtype),
                    'n_valid_gedi_pixels': int(n_valid_pixels),
                    'total_pixels': int(total_pixels),
                    'gedi_coverage_percent': float(n_valid_pixels / total_pixels * 100),
                    'target_range': target_range,
                    'has_temporal_data': self._detect_temporal(patch_file)
                }
                
        except Exception as e:
            return {
                'valid': False,
                'error': str(e)
            }
    
    def _detect_temporal(self, patch_file: str) -> bool:
        """Detect if patch contains temporal data."""
        filename = Path(patch_file).name
        return 'temporal' in filename.lower() or '_M01' in filename
    
    def batch_validate_patches(self, patch_files: List[str]) -> Dict[str, any]:
        """
        Validate multiple patch files and return summary.
        
        Args:
            patch_files: List of patch file paths
            
        Returns:
            Batch validation summary
        """
        results = {}
        valid_patches = []
        invalid_patches = []
        total_gedi_pixels = 0
        
        for patch_file in patch_files:
            result = self.validate_patch(patch_file)
            results[patch_file] = result
            
            if result['valid']:
                valid_patches.append(patch_file)
                total_gedi_pixels += result['n_valid_gedi_pixels']
            else:
                invalid_patches.append(patch_file)
        
        summary = {
            'total_patches': len(patch_files),
            'valid_patches': len(valid_patches),
            'invalid_patches': len(invalid_patches),
            'total_gedi_pixels': total_gedi_pixels,
            'average_gedi_per_patch': total_gedi_pixels / len(valid_patches) if valid_patches else 0,
            'validation_results': results
        }
        
        return summary
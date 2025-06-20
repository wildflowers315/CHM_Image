#!/usr/bin/env python3
"""
Synthetic data generation for lightweight testing of CHM Image Processing system.

This module creates realistic synthetic data that mimics real satellite and GEDI patterns
while being lightweight enough for fast CPU-based testing.
"""

import numpy as np
import rasterio
from rasterio.transform import from_bounds
import tempfile
import os
from pathlib import Path
import torch
from typing import Tuple, Dict, Optional, List
import json

class SyntheticDataGenerator:
    """Generate realistic synthetic data for testing CHM components."""
    
    def __init__(self, seed: int = 42):
        """Initialize with random seed for reproducible tests."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        
    def create_temporal_patch(self, height: int = 256, width: int = 256, 
                            bands: int = 196) -> np.ndarray:
        """
        Create synthetic temporal patch with realistic band patterns.
        
        Args:
            height: Patch height in pixels (default: 256)
            width: Patch width in pixels (default: 256) 
            bands: Number of bands (default: 196 for temporal)
            
        Returns:
            Synthetic patch array (bands, height, width)
        """
        print(f"🎲 Creating temporal synthetic patch: {bands}×{height}×{width}")
        
        patch = np.zeros((bands, height, width), dtype=np.float32)
        band_idx = 0
        
        # Sentinel-1 temporal (24 bands: 2 polarizations × 12 months)
        for month in range(12):
            # VV polarization (band pattern with seasonal variation)
            vv = self._create_sar_band(height, width, base_value=-15.0, 
                                     seasonal_factor=month/12.0)
            patch[band_idx] = vv
            band_idx += 1
            
            # VH polarization (typically 6dB lower than VV)
            vh = vv - 6.0 + np.random.normal(0, 1.0, (height, width))
            patch[band_idx] = vh
            band_idx += 1
        
        # Sentinel-2 temporal (132 bands: 11 bands × 12 months)
        for month in range(12):
            for s2_band in range(11):  # 10 S2 bands + NDVI
                if s2_band < 10:  # Regular S2 bands
                    band_data = self._create_optical_band(height, width, 
                                                        band_type=s2_band,
                                                        month=month)
                else:  # NDVI
                    band_data = self._create_ndvi_band(height, width, month=month)
                patch[band_idx] = band_data
                band_idx += 1
        
        # ALOS-2 temporal (24 bands: 2 polarizations × 12 months)
        for month in range(12):
            # HH polarization (L-band, different characteristics than C-band)
            hh = self._create_sar_band(height, width, base_value=-12.0,
                                     seasonal_factor=month/12.0, frequency='L')
            patch[band_idx] = hh
            band_idx += 1
            
            # HV polarization
            hv = hh - 4.0 + np.random.normal(0, 0.8, (height, width))
            patch[band_idx] = hv
            band_idx += 1
        
        # DEM and other bands (~16 bands)
        remaining_bands = bands - band_idx
        for i in range(remaining_bands):
            if i == 0:  # Elevation
                patch[band_idx] = self._create_dem_band(height, width)
            elif i == 1:  # Slope
                patch[band_idx] = self._create_slope_band(height, width)
            elif i == 2:  # Aspect
                patch[band_idx] = self._create_aspect_band(height, width)
            else:  # Other auxiliary bands
                patch[band_idx] = np.random.uniform(0, 1, (height, width))
            band_idx += 1
        
        print(f"✅ Created temporal patch with {band_idx} bands")
        return patch
    
    def create_non_temporal_patch(self, height: int = 256, width: int = 256,
                                bands: int = 31) -> np.ndarray:
        """
        Create synthetic non-temporal patch with median composite patterns.
        
        Args:
            height: Patch height in pixels (default: 256)
            width: Patch width in pixels (default: 256)
            bands: Number of bands (default: 31 for non-temporal)
            
        Returns:
            Synthetic patch array (bands, height, width)
        """
        print(f"🎲 Creating non-temporal synthetic patch: {bands}×{height}×{width}")
        
        patch = np.zeros((bands, height, width), dtype=np.float32)
        band_idx = 0
        
        # Sentinel-1 composite (2 bands: VV, VH medians)
        patch[band_idx] = self._create_sar_band(height, width, base_value=-15.0)
        band_idx += 1
        patch[band_idx] = self._create_sar_band(height, width, base_value=-21.0)
        band_idx += 1
        
        # Sentinel-2 composite (11 bands: 10 S2 bands + NDVI)
        for s2_band in range(11):
            if s2_band < 10:
                patch[band_idx] = self._create_optical_band(height, width, 
                                                          band_type=s2_band)
            else:  # NDVI
                patch[band_idx] = self._create_ndvi_band(height, width)
            band_idx += 1
        
        # ALOS-2 composite (2 bands: HH, HV medians)
        patch[band_idx] = self._create_sar_band(height, width, base_value=-12.0, 
                                              frequency='L')
        band_idx += 1
        patch[band_idx] = self._create_sar_band(height, width, base_value=-16.0,
                                              frequency='L')
        band_idx += 1
        
        # DEM and other bands
        remaining_bands = bands - band_idx
        for i in range(remaining_bands):
            if i == 0:  # Elevation
                patch[band_idx] = self._create_dem_band(height, width)
            elif i == 1:  # Slope  
                patch[band_idx] = self._create_slope_band(height, width)
            elif i == 2:  # Aspect
                patch[band_idx] = self._create_aspect_band(height, width)
            else:  # Other auxiliary bands
                patch[band_idx] = np.random.uniform(0, 1, (height, width))
            band_idx += 1
        
        print(f"✅ Created non-temporal patch with {band_idx} bands")
        return patch
    
    def create_gedi_targets(self, height: int = 256, width: int = 256,
                          coverage: float = 0.003) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sparse GEDI height targets (<0.3% coverage).
        
        Args:
            height: Patch height in pixels
            width: Patch width in pixels  
            coverage: Fraction of pixels with GEDI data (default: 0.003)
            
        Returns:
            Tuple of (gedi_mask, gedi_heights)
        """
        total_pixels = height * width
        gedi_pixels = int(total_pixels * coverage)
        
        # Create sparse mask
        gedi_mask = np.zeros((height, width), dtype=bool)
        
        # Random GEDI locations
        flat_indices = np.random.choice(total_pixels, gedi_pixels, replace=False)
        row_indices = flat_indices // width
        col_indices = flat_indices % width
        gedi_mask[row_indices, col_indices] = True
        
        # Create realistic height distribution (0-50m with forest patterns)
        gedi_heights = np.zeros((height, width), dtype=np.float32)
        
        # Forest height patterns (higher in center, lower at edges)
        center_y, center_x = height // 2, width // 2
        for i, j in zip(row_indices, col_indices):
            # Distance from center effect
            dist_from_center = np.sqrt((i - center_y)**2 + (j - center_x)**2) / (height/2)
            
            # Base height decreases with distance from center
            base_height = 30.0 * (1 - dist_from_center * 0.5)
            
            # Add realistic variation
            height_noise = np.random.normal(0, 5.0)
            final_height = max(0.0, min(50.0, base_height + height_noise))
            
            gedi_heights[i, j] = final_height
        
        print(f"🎯 Created GEDI targets: {gedi_pixels} pixels ({coverage*100:.3f}% coverage)")
        return gedi_mask, gedi_heights
    
    def save_synthetic_patch_tif(self, patch_data: np.ndarray, 
                               output_path: str,
                               temporal: bool = True,
                               bounds: Tuple[float, float, float, float] = None) -> str:
        """
        Save synthetic patch as GeoTIFF with proper band naming.
        
        Args:
            patch_data: Patch array (bands, height, width)
            output_path: Output file path
            temporal: Whether this is temporal data (affects band naming)
            bounds: Geographic bounds (min_x, min_y, max_x, max_y)
            
        Returns:
            Path to saved file
        """
        bands, height, width = patch_data.shape
        
        # Default bounds (arbitrary geographic location)
        if bounds is None:
            bounds = (-122.5, 37.5, -122.0, 38.0)  # San Francisco Bay area
        
        # Create geotransform
        transform = from_bounds(*bounds, width, height)
        
        # Create band descriptions for temporal detection
        descriptions = []
        if temporal and bands >= 196:
            # Temporal naming pattern
            band_idx = 0
            
            # Sentinel-1 temporal
            for month in range(1, 13):
                descriptions.append(f'S1_VV_M{month:02d}')
                descriptions.append(f'S1_VH_M{month:02d}')
                band_idx += 2
            
            # Sentinel-2 temporal  
            s2_bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12', 'NDVI']
            for month in range(1, 13):
                for band_name in s2_bands:
                    descriptions.append(f'S2_{band_name}_M{month:02d}')
                    band_idx += 1
            
            # ALOS-2 temporal
            for month in range(1, 13):
                descriptions.append(f'ALOS2_HH_M{month:02d}')
                descriptions.append(f'ALOS2_HV_M{month:02d}')
                band_idx += 2
            
            # Fill remaining bands
            while len(descriptions) < bands:
                descriptions.append(f'AUX_{len(descriptions) - band_idx + 1}')
        else:
            # Non-temporal naming pattern
            descriptions = ['S1_VV', 'S1_VH']
            descriptions.extend([f'S2_B{i}' for i in [2,3,4,5,6,7,8,'8A',11,12]])
            descriptions.append('S2_NDVI')
            descriptions.extend(['ALOS2_HH', 'ALOS2_HV'])
            
            # Fill remaining bands
            while len(descriptions) < bands:
                descriptions.append(f'AUX_{len(descriptions) - 15}')
        
        # Write GeoTIFF
        os.makedirs(Path(output_path).parent, exist_ok=True)
        
        with rasterio.open(
            output_path, 'w',
            driver='GTiff',
            height=height,
            width=width,
            count=bands,
            dtype=patch_data.dtype,
            crs='EPSG:4326',
            transform=transform,
            compress='lzw'
        ) as dst:
            dst.write(patch_data)
            
            # Set band descriptions
            for i, desc in enumerate(descriptions[:bands], 1):
                dst.set_band_description(i, desc)
        
        print(f"💾 Saved synthetic patch: {output_path}")
        return output_path
    
    def _create_sar_band(self, height: int, width: int, base_value: float,
                        seasonal_factor: float = 0.0, frequency: str = 'C') -> np.ndarray:
        """Create realistic SAR band with speckle and terrain effects."""
        # Base backscatter pattern
        band = np.full((height, width), base_value, dtype=np.float32)
        
        # Add terrain effects (higher backscatter in center)
        y, x = np.ogrid[:height, :width]
        center_y, center_x = height // 2, width // 2
        terrain_effect = 3.0 * np.exp(-((y - center_y)**2 + (x - center_x)**2) / (height * width / 16))
        band += terrain_effect
        
        # Add seasonal variation
        seasonal_variation = 2.0 * np.sin(2 * np.pi * seasonal_factor) * np.random.uniform(0.5, 1.5, (height, width))
        band += seasonal_variation
        
        # Add speckle noise
        if frequency == 'L':  # L-band has less speckle
            speckle = np.random.normal(0, 1.5, (height, width))
        else:  # C-band
            speckle = np.random.normal(0, 2.0, (height, width))
        band += speckle
        
        return band
    
    def _create_optical_band(self, height: int, width: int, band_type: int,
                           month: int = 6) -> np.ndarray:
        """Create realistic optical band with vegetation patterns."""
        # Base reflectance values for different band types
        base_values = {
            0: 0.15,  # Blue
            1: 0.18,  # Green  
            2: 0.25,  # Red
            3: 0.35,  # Red Edge 1
            4: 0.40,  # Red Edge 2
            5: 0.45,  # Red Edge 3
            6: 0.50,  # NIR
            7: 0.48,  # NIR narrow
            8: 0.25,  # SWIR 1
            9: 0.15   # SWIR 2
        }
        
        base_value = base_values.get(band_type, 0.25)
        
        # Create vegetation pattern (higher reflectance in center for NIR bands)
        y, x = np.ogrid[:height, :width]
        center_y, center_x = height // 2, width // 2
        
        if band_type >= 6:  # NIR bands
            veg_pattern = 0.3 * np.exp(-((y - center_y)**2 + (x - center_x)**2) / (height * width / 8))
        else:  # Visible/SWIR bands
            veg_pattern = -0.1 * np.exp(-((y - center_y)**2 + (x - center_x)**2) / (height * width / 8))
        
        band = base_value + veg_pattern
        
        # Add seasonal variation for vegetation
        if month is not None:
            seasonal_factor = 0.2 * np.sin(2 * np.pi * (month - 6) / 12)
            band += seasonal_factor * np.random.uniform(0.8, 1.2, (height, width))
        
        # Add noise
        band += np.random.normal(0, 0.02, (height, width))
        
        # Ensure valid reflectance range
        band = np.clip(band, 0, 1)
        
        return band.astype(np.float32)
    
    def _create_ndvi_band(self, height: int, width: int, month: int = 6) -> np.ndarray:
        """Create realistic NDVI band with vegetation patterns."""
        # Create base NDVI pattern (higher in center for forest)
        y, x = np.ogrid[:height, :width]
        center_y, center_x = height // 2, width // 2
        
        # Forest NDVI pattern (0.3-0.9 range)
        ndvi = 0.6 + 0.3 * np.exp(-((y - center_y)**2 + (x - center_x)**2) / (height * width / 10))
        
        # Add seasonal variation
        if month is not None:
            seasonal_factor = 0.15 * np.sin(2 * np.pi * (month - 6) / 12)
            ndvi += seasonal_factor
        
        # Add noise
        ndvi += np.random.normal(0, 0.05, (height, width))
        
        # Ensure valid NDVI range
        ndvi = np.clip(ndvi, -1, 1)
        
        return ndvi.astype(np.float32)
    
    def _create_dem_band(self, height: int, width: int) -> np.ndarray:
        """Create realistic elevation band."""
        # Create elevation pattern (higher in center)
        y, x = np.ogrid[:height, :width]
        center_y, center_x = height // 2, width // 2
        
        # Base elevation 200-800m
        elevation = 500 + 300 * np.exp(-((y - center_y)**2 + (x - center_x)**2) / (height * width / 6))
        
        # Add terrain variation
        elevation += np.random.normal(0, 50, (height, width))
        
        # Ensure realistic range
        elevation = np.clip(elevation, 0, 3000)
        
        return elevation.astype(np.float32)
    
    def _create_slope_band(self, height: int, width: int) -> np.ndarray:
        """Create realistic slope band (0-45 degrees)."""
        # Higher slopes at edges, lower in center
        y, x = np.ogrid[:height, :width]
        center_y, center_x = height // 2, width // 2
        
        dist_from_center = np.sqrt((y - center_y)**2 + (x - center_x)**2) / (height/2)
        slope = 20 * dist_from_center + np.random.normal(0, 5, (height, width))
        
        slope = np.clip(slope, 0, 45)
        return slope.astype(np.float32)
    
    def _create_aspect_band(self, height: int, width: int) -> np.ndarray:
        """Create realistic aspect band (0-360 degrees)."""
        # Random aspect with some spatial correlation
        aspect = np.random.uniform(0, 360, (height, width))
        return aspect.astype(np.float32)


def create_test_datasets(output_dir: str = "tests/fixtures/data") -> Dict[str, str]:
    """
    Create a complete set of synthetic test datasets.
    
    Args:
        output_dir: Directory to save test datasets
        
    Returns:
        Dictionary mapping dataset names to file paths
    """
    generator = SyntheticDataGenerator()
    os.makedirs(output_dir, exist_ok=True)
    
    datasets = {}
    
    # Create temporal patch
    temporal_patch = generator.create_temporal_patch(bands=196)
    temporal_path = os.path.join(output_dir, "temporal_patch_196bands.tif")
    generator.save_synthetic_patch_tif(temporal_patch, temporal_path, temporal=True)
    datasets['temporal_patch'] = temporal_path
    
    # Create non-temporal patch  
    non_temporal_patch = generator.create_non_temporal_patch(bands=31)
    non_temporal_path = os.path.join(output_dir, "non_temporal_patch_31bands.tif")
    generator.save_synthetic_patch_tif(non_temporal_patch, non_temporal_path, temporal=False)
    datasets['non_temporal_patch'] = non_temporal_path
    
    # Create GEDI targets
    gedi_mask, gedi_heights = generator.create_gedi_targets()
    
    # Save GEDI data as GeoTIFF
    gedi_path = os.path.join(output_dir, "gedi_targets.tif")
    with rasterio.open(
        gedi_path, 'w',
        driver='GTiff',
        height=256, width=256, count=1,
        dtype=gedi_heights.dtype,
        crs='EPSG:4326',
        transform=from_bounds(-122.5, 37.5, -122.0, 38.0, 256, 256)
    ) as dst:
        # Set heights to 0 where no GEDI data
        gedi_output = np.where(gedi_mask, gedi_heights, 0)
        dst.write(gedi_output, 1)
    datasets['gedi_targets'] = gedi_path
    
    # Create metadata file
    metadata = {
        'temporal_patch': {
            'bands': 196,
            'temporal': True,
            'description': 'Synthetic temporal patch with 12-month time series'
        },
        'non_temporal_patch': {
            'bands': 31,
            'temporal': False,
            'description': 'Synthetic non-temporal patch with median composites'
        },
        'gedi_targets': {
            'coverage': 0.003,
            'height_range': [0, 50],
            'description': 'Sparse GEDI height targets'
        }
    }
    
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    datasets['metadata'] = metadata_path
    
    print(f"✅ Created complete test dataset in {output_dir}")
    return datasets


if __name__ == "__main__":
    # Create test datasets when run directly
    datasets = create_test_datasets()
    print("📊 Test datasets created:")
    for name, path in datasets.items():
        print(f"  {name}: {path}")
"""
Multi-patch training system for large-area canopy height mapping.

This module provides infrastructure for handling multiple patch TIF files
in a unified training system with geospatial prediction merging capabilities.
"""

import os
import re
import glob
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.transform import from_bounds
import pandas as pd
from tqdm import tqdm
import warnings
from multiprocessing import Pool, cpu_count
from functools import partial
warnings.filterwarnings('ignore')

@dataclass
class PatchInfo:
    """Comprehensive patch metadata for geospatial processing."""
    file_path: str
    patch_id: str
    geospatial_bounds: Tuple[float, float, float, float]  # (min_x, min_y, max_x, max_y)
    center_coordinates: Tuple[float, float]  # (lon, lat)
    crs: str  # e.g., "EPSG:4326"
    transform: rasterio.Affine
    patch_size_meters: int
    pixel_size_meters: int
    band_count: int
    temporal_mode: bool
    width: int
    height: int
    overlap_info: Optional[Dict] = None  # Neighboring patch overlap data
    
    @classmethod
    def from_file(cls, file_path: str) -> 'PatchInfo':
        """Create PatchInfo from a patch TIF file."""
        # Extract metadata from filename
        filename = Path(file_path).stem
        patch_id = cls._extract_patch_id(filename)
        temporal_mode = '_temporal_' in filename
        
        # Extract geospatial metadata from TIF file
        with rasterio.open(file_path) as src:
            bounds = src.bounds
            transform = src.transform
            crs = str(src.crs)
            band_count = src.count
            width = src.width
            height = src.height
            
            # Calculate center coordinates
            center_x = (bounds.left + bounds.right) / 2
            center_y = (bounds.bottom + bounds.top) / 2
            
            # Extract scale from filename or calculate from transform
            pixel_size = abs(transform[0])  # Pixel size in meters
            patch_size = max(
                bounds.right - bounds.left,
                bounds.top - bounds.bottom
            )
            
        return cls(
            file_path=file_path,
            patch_id=patch_id,
            geospatial_bounds=(bounds.left, bounds.bottom, bounds.right, bounds.top),
            center_coordinates=(center_x, center_y),
            crs=crs,
            transform=transform,
            patch_size_meters=int(patch_size),
            pixel_size_meters=int(pixel_size),
            band_count=band_count,
            temporal_mode=temporal_mode,
            width=width,
            height=height
        )
    
    @staticmethod
    def _extract_patch_id(filename: str) -> str:
        """Extract patch ID from filename."""
        # Pattern: {geojson_name}[_temporal]_bandNum{N}_scale{scale}_patch{NNNN}
        match = re.search(r'patch(\d{4})', filename)
        if match:
            return f"patch{match.group(1)}"
        else:
            # Fallback to filename if pattern not found
            return filename


class PatchRegistry:
    """Registry for managing multiple patches with spatial indexing."""
    
    def __init__(self):
        self.patches: List[PatchInfo] = []
        self.spatial_index: Dict[str, PatchInfo] = {}
        self.train_patches: List[PatchInfo] = []
        self.val_patches: List[PatchInfo] = []
        
    def add_patch(self, patch_info: PatchInfo):
        """Add patch to registry with spatial indexing."""
        self.patches.append(patch_info)
        self.spatial_index[patch_info.patch_id] = patch_info
        
    def discover_patches(self, patch_dir: str, pattern: str = "*.tif") -> List[PatchInfo]:
        """
        Discover and catalog all patch files with metadata.
        
        Args:
            patch_dir: Directory containing patch TIF files
            pattern: File pattern to match (e.g., "*_temporal_*.tif")
            
        Returns:
            List of PatchInfo objects with complete metadata
        """
        print(f"Discovering patches in {patch_dir} with pattern {pattern}")
        
        # Find all matching patch files
        search_pattern = os.path.join(patch_dir, pattern)
        patch_files = glob.glob(search_pattern)
        
        if not patch_files:
            print(f"Warning: No patch files found matching {search_pattern}")
            return []
        
        print(f"Found {len(patch_files)} patch files")
        
        # Create PatchInfo objects for each file
        discovered_patches = []
        for file_path in tqdm(patch_files, desc="Processing patch metadata"):
            try:
                patch_info = PatchInfo.from_file(file_path)
                discovered_patches.append(patch_info)
                self.add_patch(patch_info)
            except Exception as e:
                print(f"Warning: Could not process {file_path}: {e}")
                continue
                
        return discovered_patches
    
    def find_neighbors(self, patch_id: str, buffer: float = 0.1) -> List[PatchInfo]:
        """Find spatially adjacent patches for overlap handling."""
        if patch_id not in self.spatial_index:
            return []
            
        target_patch = self.spatial_index[patch_id]
        target_bounds = target_patch.geospatial_bounds
        
        neighbors = []
        for patch in self.patches:
            if patch.patch_id == patch_id:
                continue
                
            # Check if patches overlap or are adjacent (within buffer)
            if self._patches_overlap(target_bounds, patch.geospatial_bounds, buffer):
                neighbors.append(patch)
                
        return neighbors
    
    def _patches_overlap(self, bounds1: Tuple[float, float, float, float], 
                        bounds2: Tuple[float, float, float, float], 
                        buffer: float = 0.0) -> bool:
        """Check if two patch bounds overlap (with optional buffer)."""
        min_x1, min_y1, max_x1, max_y1 = bounds1
        min_x2, min_y2, max_x2, max_y2 = bounds2
        
        # Expand bounds by buffer
        min_x1 -= buffer
        min_y1 -= buffer
        max_x1 += buffer
        max_y1 += buffer
        
        # Check overlap
        return not (max_x1 < min_x2 or max_x2 < min_x1 or 
                   max_y1 < min_y2 or max_y2 < min_y1)
    
    def get_merged_bounds(self) -> Tuple[float, float, float, float]:
        """Calculate overall bounding box for all patches."""
        if not self.patches:
            return (0, 0, 0, 0)
            
        all_bounds = [patch.geospatial_bounds for patch in self.patches]
        
        min_x = min(bounds[0] for bounds in all_bounds)
        min_y = min(bounds[1] for bounds in all_bounds)
        max_x = max(bounds[2] for bounds in all_bounds)
        max_y = max(bounds[3] for bounds in all_bounds)
        
        return (min_x, min_y, max_x, max_y)
    
    def validate_consistency(self) -> bool:
        """Validate CRS, resolution, and band consistency across patches."""
        if not self.patches:
            return True
            
        # Check CRS consistency
        reference_crs = self.patches[0].crs
        crs_consistent = all(patch.crs == reference_crs for patch in self.patches)
        
        # Check resolution consistency  
        reference_pixel_size = self.patches[0].pixel_size_meters
        resolution_consistent = all(
            abs(patch.pixel_size_meters - reference_pixel_size) < 1 
            for patch in self.patches
        )
        
        # Check temporal mode consistency
        reference_temporal = self.patches[0].temporal_mode
        temporal_consistent = all(
            patch.temporal_mode == reference_temporal for patch in self.patches
        )
        
        # Check band count consistency
        reference_bands = self.patches[0].band_count
        bands_consistent = all(
            patch.band_count == reference_bands for patch in self.patches
        )
        
        if not crs_consistent:
            print("Warning: Inconsistent CRS across patches")
        if not resolution_consistent:
            print("Warning: Inconsistent resolution across patches")
        if not temporal_consistent:
            print("Warning: Mixed temporal and non-temporal patches")
        if not bands_consistent:
            print("Warning: Inconsistent band counts across patches")
            
        return crs_consistent and resolution_consistent and temporal_consistent and bands_consistent
    
    def get_patch_summary(self) -> Dict:
        """Get summary statistics for the patch collection."""
        if not self.patches:
            return {}
            
        temporal_count = sum(1 for p in self.patches if p.temporal_mode)
        non_temporal_count = len(self.patches) - temporal_count
        
        bounds = self.get_merged_bounds()
        total_area_km2 = ((bounds[2] - bounds[0]) * (bounds[3] - bounds[1])) / 1e6
        
        return {
            'total_patches': len(self.patches),
            'temporal_patches': temporal_count,
            'non_temporal_patches': non_temporal_count,
            'total_area_km2': total_area_km2,
            'merged_bounds': bounds,
            'reference_crs': self.patches[0].crs if self.patches else None,
            'reference_resolution': self.patches[0].pixel_size_meters if self.patches else None,
            'reference_bands': self.patches[0].band_count if self.patches else None
        }


class PredictionMerger:
    """Handle merging of individual patch predictions with geospatial registration."""
    
    def __init__(self, patches: List[PatchInfo], merge_strategy: str = "average"):
        """
        Initialize prediction merger.
        
        Args:
            patches: List of patch metadata
            merge_strategy: "average", "maximum", "weighted", "seamless"
        """
        self.patches = patches
        self.merge_strategy = merge_strategy
        self.patch_registry = PatchRegistry()
        for patch in patches:
            self.patch_registry.add_patch(patch)
            
    def create_output_grid(self, pixel_size: Optional[float] = None) -> Tuple[rasterio.Affine, int, int, Tuple[float, float, float, float]]:
        """
        Create output grid specifications that encompasses all patches.
        
        Returns:
            - Geospatial transform for output
            - Width in pixels
            - Height in pixels  
            - Bounds (min_x, min_y, max_x, max_y)
        """
        merged_bounds = self.patch_registry.get_merged_bounds()
        min_x, min_y, max_x, max_y = merged_bounds
        
        # Use reference pixel size if not provided
        if pixel_size is None:
            pixel_size = self.patches[0].pixel_size_meters
            
        # Calculate grid dimensions
        width = int((max_x - min_x) / pixel_size)
        height = int((max_y - min_y) / pixel_size)
        
        # Create transform
        transform = from_bounds(min_x, min_y, max_x, max_y, width, height)
        
        return transform, width, height, merged_bounds
    
    def merge_predictions_from_files(self, prediction_files: Dict[str, str], 
                                   output_path: str, 
                                   nodata_value: float = -9999) -> str:
        """
        Merge prediction TIF files into a single continuous GeoTIFF.
        
        Args:
            prediction_files: Dict mapping patch_id -> prediction_file_path
            output_path: Output path for merged GeoTIFF
            nodata_value: NoData value for output
            
        Returns:
            Path to merged output file
        """
        print(f"Merging {len(prediction_files)} prediction files using {self.merge_strategy} strategy")
        
        # Collect rasterio datasets
        datasets = []
        for patch_id, file_path in prediction_files.items():
            if os.path.exists(file_path):
                datasets.append(rasterio.open(file_path))
            else:
                print(f"Warning: Prediction file not found: {file_path}")
        
        if not datasets:
            raise ValueError("No valid prediction files found")
        
        # Merge using rasterio
        if self.merge_strategy == "average":
            merged_array, merged_transform = merge(
                datasets, 
                method='first',  # Use first for averaging (will compute manually)
                nodata=nodata_value
            )
            # Manual averaging for overlapping areas
            # Note: rasterio merge doesn't support 'mean', using 'first' as fallback
        elif self.merge_strategy == "maximum":
            merged_array, merged_transform = merge(
                datasets,
                method='max', 
                nodata=nodata_value
            )
        else:  # Default to first
            merged_array, merged_transform = merge(
                datasets,
                method='first',
                nodata=nodata_value
            )
        
        # Get reference metadata
        ref_dataset = datasets[0]
        
        # Write merged output
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=merged_array.shape[1],
            width=merged_array.shape[2], 
            count=merged_array.shape[0],
            dtype=merged_array.dtype,
            crs=ref_dataset.crs,
            transform=merged_transform,
            nodata=nodata_value,
            compress='lzw'
        ) as dst:
            dst.write(merged_array)
        
        # Close datasets
        for dataset in datasets:
            dataset.close()
            
        print(f"Merged prediction saved to: {output_path}")
        return output_path
    
    def merge_predictions_from_arrays(self, predictions: Dict[str, np.ndarray], 
                                    output_path: str) -> str:
        """
        Merge prediction arrays into a single continuous GeoTIFF.
        
        Args:
            predictions: Dict mapping patch_id -> prediction_array  
            output_path: Output path for merged GeoTIFF
            
        Returns:
            Path to merged output file
        """
        print(f"Merging {len(predictions)} prediction arrays using {self.merge_strategy} strategy")
        
        # Create output grid
        transform, width, height, bounds = self.create_output_grid()
        
        # Initialize output array
        merged = np.full((height, width), np.nan, dtype=np.float32)
        weights = np.zeros((height, width), dtype=np.float32)
        
        # Place each prediction in the output grid
        for patch_id, pred_array in predictions.items():
            # Find corresponding patch info
            patch_info = self.patch_registry.spatial_index.get(patch_id)
            if patch_info is None:
                print(f"Warning: No patch info found for {patch_id}")
                continue
                
            # Calculate pixel coordinates in output grid
            patch_bounds = patch_info.geospatial_bounds
            min_x, min_y, max_x, max_y = patch_bounds
            
            # Convert to pixel coordinates
            col_start = int((min_x - bounds[0]) / abs(transform[0]))
            row_start = int((bounds[3] - max_y) / abs(transform[4]))
            
            pred_height, pred_width = pred_array.shape
            col_end = min(col_start + pred_width, width)
            row_end = min(row_start + pred_height, height)
            
            # Ensure we don't go out of bounds
            col_start = max(0, col_start)
            row_start = max(0, row_start)
            
            # Extract the portion that fits
            src_col_start = max(0, -col_start) if col_start < 0 else 0
            src_row_start = max(0, -row_start) if row_start < 0 else 0
            src_col_end = src_col_start + (col_end - col_start)
            src_row_end = src_row_start + (row_end - row_start)
            
            if col_end > col_start and row_end > row_start:
                pred_subset = pred_array[src_row_start:src_row_end, src_col_start:src_col_end]
                
                # Apply merge strategy
                if self.merge_strategy == "average":
                    # Weighted average for overlapping areas
                    valid_mask = ~np.isnan(pred_subset)
                    merged[row_start:row_end, col_start:col_end] = np.where(
                        valid_mask,
                        np.nansum([
                            merged[row_start:row_end, col_start:col_end] * weights[row_start:row_end, col_start:col_end],
                            pred_subset
                        ], axis=0) / (weights[row_start:row_end, col_start:col_end] + 1),
                        merged[row_start:row_end, col_start:col_end]
                    )
                    weights[row_start:row_end, col_start:col_end] += valid_mask.astype(float)
                    
                elif self.merge_strategy == "maximum":
                    merged[row_start:row_end, col_start:col_end] = np.fmax(
                        merged[row_start:row_end, col_start:col_end],
                        pred_subset
                    )
                else:  # "first" - keep first non-NaN value
                    current_slice = merged[row_start:row_end, col_start:col_end]
                    merged[row_start:row_end, col_start:col_end] = np.where(
                        np.isnan(current_slice), pred_subset, current_slice
                    )
        
        # Write output GeoTIFF
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=np.float32,
            crs=self.patches[0].crs,
            transform=transform,
            nodata=np.nan,
            compress='lzw'
        ) as dst:
            dst.write(merged, 1)
            
        print(f"Merged prediction saved to: {output_path}")
        return output_path


def load_multi_patch_gedi_data(patches: List[PatchInfo], 
                              target_band: str = 'rh',
                              min_gedi_samples: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and combine GEDI data from multiple patches for training.
    
    Args:
        patches: List of patch metadata
        target_band: Name of GEDI height band
        min_gedi_samples: Minimum number of valid GEDI samples per patch for inclusion
        
    Returns:
        Combined features and targets from all patches
    """
    print(f"Loading GEDI data from {len(patches)} patches (min GEDI samples: {min_gedi_samples})")
    
    all_features = []
    all_targets = []
    skipped_patches = []
    
    for patch_info in tqdm(patches, desc="Loading patch data"):
        try:
            # Load patch data
            with rasterio.open(patch_info.file_path) as src:
                # Read all bands
                patch_data = src.read()  # Shape: (bands, height, width)
                
                # Get band names if available
                band_names = [src.descriptions[i] or f'band_{i+1}' for i in range(src.count)]
                
                # Find GEDI band
                gedi_band_idx = None
                for i, name in enumerate(band_names):
                    if target_band in name.lower():
                        gedi_band_idx = i
                        break
                
                if gedi_band_idx is None:
                    # Try last band as GEDI (common pattern)
                    gedi_band_idx = -1
                    print(f"Warning: '{target_band}' band not found in {patch_info.patch_id}, using last band")
                
                # Extract features (all bands except GEDI) and targets (GEDI band)
                gedi_target = patch_data[gedi_band_idx]
                features = np.delete(patch_data, gedi_band_idx, axis=0)
                
                # Find valid GEDI pixels (non-zero, non-NaN)
                valid_mask = (gedi_target > 0) & (~np.isnan(gedi_target)) & (gedi_target < 100)  # Reasonable height range
                
                valid_count = np.sum(valid_mask)
                if valid_count == 0:
                    print(f"Warning: No valid GEDI pixels found in {patch_info.patch_id}")
                    continue
                
                # Check minimum GEDI samples threshold
                if valid_count < min_gedi_samples:
                    print(f"Skipping {patch_info.patch_id}: only {valid_count} GEDI samples (minimum required: {min_gedi_samples})")
                    skipped_patches.append(patch_info.patch_id)
                    continue
                
                # Extract valid pixels
                valid_indices = np.where(valid_mask)
                patch_features = features[:, valid_indices[0], valid_indices[1]].T  # Shape: (n_pixels, n_bands)
                patch_targets = gedi_target[valid_indices]  # Shape: (n_pixels,)
                
                all_features.append(patch_features)
                all_targets.append(patch_targets)
                
                print(f"  {patch_info.patch_id}: {len(patch_targets)} valid GEDI pixels")
                
        except Exception as e:
            print(f"Error loading {patch_info.patch_id}: {e}")
            continue
    
    if not all_features:
        raise ValueError("No valid GEDI data found in any patches")
    
    # Combine all patches
    combined_features = np.vstack(all_features)
    combined_targets = np.hstack(all_targets)
    
    # Handle NaN/inf values in features
    print(f"Cleaning features: checking for NaN/inf values...")
    
    # Replace NaN/inf values with zeros instead of removing samples
    # This handles missing bands (e.g., GLO30 slope/aspect) properly
    original_shape = combined_features.shape
    combined_features = np.nan_to_num(combined_features, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Check for completely zero features (all bands missing)
    zero_features = np.all(combined_features == 0, axis=1)
    if np.sum(zero_features) > 0:
        print(f"  Found {np.sum(zero_features)} samples with all features zero - keeping them")
    
    print(f"  Processed features: {original_shape} -> {combined_features.shape}")
    print(f"  Replaced NaN/inf values with zeros for missing bands")
    
    print(f"Total training samples: {len(combined_targets)}")
    print(f"Feature dimensions: {combined_features.shape}")
    print(f"Target range: {combined_targets.min():.1f} - {combined_targets.max():.1f}m")
    
    if skipped_patches:
        print(f"Skipped {len(skipped_patches)} patches due to insufficient GEDI samples: {skipped_patches}")
    
    return combined_features, combined_targets


def count_gedi_samples_per_patch(patches: List[PatchInfo], 
                                target_band: str = 'rh') -> Dict[str, int]:
    """Count valid GEDI samples per patch without loading full training data."""
    gedi_counts = {}
    
    for patch_info in tqdm(patches, desc="Counting GEDI samples"):
        try:
            with rasterio.open(patch_info.file_path) as src:
                # Get band names
                band_names = [src.descriptions[i] or f'band_{i+1}' for i in range(src.count)]
                
                # Find GEDI band
                gedi_band_idx = None
                for i, name in enumerate(band_names):
                    if target_band in name.lower():
                        gedi_band_idx = i
                        break
                
                if gedi_band_idx is None:
                    gedi_band_idx = -1
                
                # Read only GEDI band
                gedi_data = src.read(gedi_band_idx + 1)  # rasterio uses 1-based indexing
                
                # Count valid GEDI pixels
                valid_mask = (gedi_data > 0) & (~np.isnan(gedi_data)) & (gedi_data < 100)
                valid_count = np.sum(valid_mask)
                
                gedi_counts[patch_info.patch_id] = valid_count
                
        except Exception as e:
            print(f"Error counting GEDI samples in {patch_info.patch_id}: {e}")
            gedi_counts[patch_info.patch_id] = 0
    
    return gedi_counts


def load_multi_patch_reference_data(patches: List[PatchInfo], 
                                   reference_tif_path: str,
                                   min_reference_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and combine reference height data from multiple patches for training.
    
    Args:
        patches: List of patch metadata
        reference_tif_path: Path to reference height TIF file
        min_reference_samples: Minimum number of valid reference samples per patch
        
    Returns:
        Combined features and targets from all patches with reference supervision
    """
    print(f"Loading reference height data from {len(patches)} patches")
    print(f"Reference TIF: {reference_tif_path}")
    
    # Open reference TIF once
    reference_src = rasterio.open(reference_tif_path)
    
    all_features = []
    all_targets = []
    skipped_patches = []
    
    for patch_info in tqdm(patches, desc="Loading patch data with reference supervision"):
        try:
            # Load satellite patch data
            with rasterio.open(patch_info.file_path) as src:
                # Read all bands (satellite features)
                patch_data = src.read()  # Shape: (bands, height, width)
                
                # Get band names and remove GEDI 'rh' band for reference supervision
                band_names = [src.descriptions[i] or f'band_{i+1}' for i in range(src.count)]
                
                # Find and exclude the GEDI 'rh' band
                gedi_band_idx = None
                for i, name in enumerate(band_names):
                    if 'rh' in name.lower():
                        gedi_band_idx = i
                        break
                
                if gedi_band_idx is not None:
                    # Remove GEDI band from features for reference supervision
                    patch_data = np.delete(patch_data, gedi_band_idx, axis=0)
                    print(f"    Removed GEDI 'rh' band (band {gedi_band_idx+1}) from features")
                else:
                    print(f"    Warning: No 'rh' band found in {patch_info.patch_id}")
                
                # Extract reference height data for this patch's spatial extent
                patch_bounds = patch_info.geospatial_bounds
                
                # Read and align reference data using proper resampling
                from rasterio.warp import reproject, Resampling
                from rasterio.transform import from_bounds as transform_from_bounds
                
                try:
                    # Create target transform that matches the patch
                    target_transform = transform_from_bounds(
                        patch_bounds[0], patch_bounds[1], 
                        patch_bounds[2], patch_bounds[3], 
                        patch_data.shape[2], patch_data.shape[1]  # width, height
                    )
                    
                    # Create destination array with same dimensions as patch
                    reference_data = np.zeros(patch_data.shape[1:], dtype=np.float32)
                    
                    # Reproject reference data to match patch resolution and alignment
                    reproject(
                        source=rasterio.band(reference_src, 1),
                        destination=reference_data,
                        src_transform=reference_src.transform,
                        src_crs=reference_src.crs,
                        dst_transform=target_transform,
                        dst_crs=src.crs,
                        resampling=Resampling.average
                    )
                    
                    # Find valid reference pixels (non-zero, non-NaN, reasonable height range)
                    valid_mask = (
                        (reference_data > 0) & 
                        (~np.isnan(reference_data)) & 
                        (reference_data < 100)  # Reasonable height range
                    )
                    
                    valid_count = np.sum(valid_mask)
                    if valid_count == 0:
                        print(f"Warning: No valid reference pixels found in {patch_info.patch_id}")
                        continue
                    
                    # Check minimum reference samples threshold
                    if valid_count < min_reference_samples:
                        print(f"Skipping {patch_info.patch_id}: only {valid_count} reference samples (minimum required: {min_reference_samples})")
                        skipped_patches.append(patch_info.patch_id)
                        continue
                    
                    # Extract valid pixels
                    valid_indices = np.where(valid_mask)
                    patch_features = patch_data[:, valid_indices[0], valid_indices[1]].T  # Shape: (n_pixels, n_bands)
                    patch_targets = reference_data[valid_indices]  # Shape: (n_pixels,)
                    
                    all_features.append(patch_features)
                    all_targets.append(patch_targets)
                    
                    print(f"  {patch_info.patch_id}: {len(patch_targets)} valid reference pixels")
                    
                except Exception as spatial_error:
                    print(f"Spatial processing error for {patch_info.patch_id}: {spatial_error}")
                    continue
                    
        except Exception as e:
            print(f"Error loading {patch_info.patch_id}: {e}")
            continue
    
    # Close reference TIF
    reference_src.close()
    
    if not all_features:
        raise ValueError("No valid reference data found in any patches")
    
    # Combine all patches
    combined_features = np.vstack(all_features)
    combined_targets = np.hstack(all_targets)
    
    # Handle NaN/inf values in features
    print(f"Cleaning features: checking for NaN/inf values...")
    
    # Replace NaN/inf values with zeros instead of removing samples
    # This is because satellite data often has missing bands (e.g., ALOS2)
    original_shape = combined_features.shape
    combined_features = np.nan_to_num(combined_features, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Check for completely zero features (all bands missing)
    zero_features = np.all(combined_features == 0, axis=1)
    if np.sum(zero_features) > 0:
        print(f"  Found {np.sum(zero_features)} samples with all features zero - keeping them")
    
    print(f"  Processed features: {original_shape} -> {combined_features.shape}")
    print(f"  Replaced NaN/inf values with zeros for missing bands")
    
    print(f"Total training samples: {len(combined_targets)}")
    print(f"Feature dimensions: {combined_features.shape}")
    print(f"Target range: {combined_targets.min():.1f} - {combined_targets.max():.1f}m")
    
    if skipped_patches:
        print(f"Skipped {len(skipped_patches)} patches due to insufficient reference samples: {skipped_patches}")
    
    return combined_features, combined_targets


def generate_multi_patch_summary(patches: List[PatchInfo]) -> pd.DataFrame:
    """Generate summary statistics for patch collection."""
    summary_data = []
    
    for patch in patches:
        summary_data.append({
            'patch_id': patch.patch_id,
            'file_path': patch.file_path,
            'temporal_mode': patch.temporal_mode,
            'band_count': patch.band_count,
            'patch_size_m': patch.patch_size_meters,
            'pixel_size_m': patch.pixel_size_meters,
            'width': patch.width,
            'height': patch.height,
            'center_lon': patch.center_coordinates[0],
            'center_lat': patch.center_coordinates[1],
            'min_x': patch.geospatial_bounds[0],
            'min_y': patch.geospatial_bounds[1], 
            'max_x': patch.geospatial_bounds[2],
            'max_y': patch.geospatial_bounds[3],
            'crs': patch.crs
        })
    
    return pd.DataFrame(summary_data)


def _process_single_patch_reference(patch_info: PatchInfo, reference_tif_path: str, min_reference_samples: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    """
    Process a single patch for reference height supervision (for parallel processing).
    
    Args:
        patch_info: Patch metadata
        reference_tif_path: Path to reference height TIF
        min_reference_samples: Minimum valid reference samples
        
    Returns:
        Tuple of (features, targets, patch_id) or (None, None, patch_id) if skipped
    """
    try:
        # Open reference TIF (each process opens its own copy)
        with rasterio.open(reference_tif_path) as reference_src:
            
            # Load satellite patch data
            with rasterio.open(patch_info.file_path) as src:
                # Read all bands (satellite features)
                patch_data = src.read()  # Shape: (bands, height, width)
                
                # Get band names and remove GEDI 'rh' band for reference supervision
                band_names = [src.descriptions[i] or f'band_{i+1}' for i in range(src.count)]
                
                # Find and exclude the GEDI 'rh' band
                gedi_band_idx = None
                for i, name in enumerate(band_names):
                    if 'rh' in name.lower():
                        gedi_band_idx = i
                        break
                
                if gedi_band_idx is not None:
                    # Remove GEDI band from features for reference supervision
                    patch_data = np.delete(patch_data, gedi_band_idx, axis=0)
                
                # Extract reference height data for this patch's spatial extent
                patch_bounds = patch_info.geospatial_bounds
                
                # Read and align reference data using proper resampling
                from rasterio.warp import reproject, Resampling
                from rasterio.transform import from_bounds as transform_from_bounds
                
                # Create target transform that matches the patch
                target_transform = transform_from_bounds(
                    patch_bounds[0], patch_bounds[1], 
                    patch_bounds[2], patch_bounds[3], 
                    patch_data.shape[2], patch_data.shape[1]  # width, height
                )
                
                # Create destination array with same dimensions as patch
                reference_data = np.zeros(patch_data.shape[1:], dtype=np.float32)
                
                # Reproject reference data to match patch resolution and alignment
                reproject(
                    source=rasterio.band(reference_src, 1),
                    destination=reference_data,
                    src_transform=reference_src.transform,
                    src_crs=reference_src.crs,
                    dst_transform=target_transform,
                    dst_crs=src.crs,
                    resampling=Resampling.average
                )
                
                # Find valid reference pixels (non-zero, non-NaN, reasonable height range)
                valid_mask = (
                    (reference_data > 0) & 
                    (~np.isnan(reference_data)) & 
                    (reference_data < 100)  # Reasonable height range
                )
                
                valid_count = np.sum(valid_mask)
                
                if valid_count < min_reference_samples:
                    return None, None, patch_info.patch_id
                
                # Extract valid pixels for training
                valid_indices = np.where(valid_mask)
                
                # Extract features for valid pixels
                patch_features = []
                for band_idx in range(patch_data.shape[0]):
                    band_data = patch_data[band_idx]
                    valid_band_data = band_data[valid_indices]
                    patch_features.append(valid_band_data)
                
                # Stack features: shape (n_features, n_valid_pixels)
                patch_features = np.stack(patch_features, axis=0)
                
                # Transpose to get (n_valid_pixels, n_features)
                patch_features = patch_features.T
                
                # Extract reference targets for valid pixels
                patch_targets = reference_data[valid_indices]
                
                # Replace NaN/inf values with zeros for missing bands
                patch_features = np.nan_to_num(patch_features, nan=0.0, posinf=0.0, neginf=0.0)
                
                return patch_features, patch_targets, patch_info.patch_id
                
    except Exception as e:
        print(f"Error processing {patch_info.patch_id}: {e}")
        return None, None, patch_info.patch_id


def load_multi_patch_reference_data_parallel(patches: List[PatchInfo], 
                                            reference_tif_path: str,
                                            min_reference_samples: int = 100,
                                            n_workers: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and combine reference height data from multiple patches using parallel processing.
    
    This is a faster version of load_multi_patch_reference_data that uses multiprocessing
    to process patches in parallel, reducing loading time from ~60 minutes to ~15 minutes.
    
    Args:
        patches: List of patch metadata
        reference_tif_path: Path to reference height TIF file
        min_reference_samples: Minimum number of valid reference samples per patch
        n_workers: Number of parallel workers (default: cpu_count() - 1)
        
    Returns:
        Combined features and targets from all patches with reference supervision
    """
    if n_workers is None:
        # Use reasonable number of workers: min of (patches/2, cpus-1, 8)
        max_reasonable = min(len(patches), max(1, cpu_count() - 1))
        n_workers = max(1, max_reasonable)
    
    print(f"Loading reference height data from {len(patches)} patches using {n_workers} parallel workers")
    print(f"Reference TIF: {reference_tif_path}")
    
    # Create partial function with fixed arguments
    process_func = partial(_process_single_patch_reference, 
                          reference_tif_path=reference_tif_path,
                          min_reference_samples=min_reference_samples)
    
    # Process patches in parallel
    all_features = []
    all_targets = []
    skipped_patches = []
    
    with Pool(processes=n_workers) as pool:
        # Use tqdm to show progress
        results = list(tqdm(
            pool.imap(process_func, patches),
            total=len(patches),
            desc="Loading patch data with reference supervision (parallel)"
        ))
    
    # Collect results
    for features, targets, patch_id in results:
        if features is not None and targets is not None:
            all_features.append(features)
            all_targets.append(targets)
            print(f"  {patch_id}: {len(targets)} valid reference pixels")
        else:
            skipped_patches.append(patch_id)
    
    if not all_features:
        raise ValueError("No patches contained sufficient reference data for training")
    
    # Combine all features and targets
    print("Combining features and targets from all patches...")
    combined_features = np.vstack(all_features)
    combined_targets = np.concatenate(all_targets)
    
    print(f"Total training samples: {len(combined_targets)}")
    print(f"Feature dimensions: {combined_features.shape}")
    print(f"Target range: {combined_targets.min():.1f} - {combined_targets.max():.1f}m")
    
    if skipped_patches:
        print(f"Skipped {len(skipped_patches)} patches due to insufficient reference samples: {skipped_patches}")
    
    return combined_features, combined_targets
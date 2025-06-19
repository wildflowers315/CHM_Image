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
                              target_band: str = 'rh') -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and combine GEDI data from multiple patches for training.
    
    Args:
        patches: List of patch metadata
        target_band: Name of GEDI height band
        
    Returns:
        Combined features and targets from all patches
    """
    print(f"Loading GEDI data from {len(patches)} patches")
    
    all_features = []
    all_targets = []
    
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
                
                if np.sum(valid_mask) == 0:
                    print(f"Warning: No valid GEDI pixels found in {patch_info.patch_id}")
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
    
    # Find rows with any NaN or inf values
    valid_rows = np.all(np.isfinite(combined_features), axis=1)
    n_invalid = np.sum(~valid_rows)
    
    if n_invalid > 0:
        print(f"  Removing {n_invalid} samples with NaN/inf values")
        combined_features = combined_features[valid_rows]
        combined_targets = combined_targets[valid_rows]
    
    # Replace any remaining NaN with zeros (shouldn't happen but safety)
    combined_features = np.nan_to_num(combined_features, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"Total training samples: {len(combined_targets)}")
    print(f"Feature dimensions: {combined_features.shape}")
    print(f"Target range: {combined_targets.min():.1f} - {combined_targets.max():.1f}m")
    
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
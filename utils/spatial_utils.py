#!/usr/bin/env python3
"""
Enhanced spatial merger that integrates with the existing pipeline
"""

import os
import numpy as np
import rasterio
from rasterio.merge import merge
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class EnhancedSpatialMerger:
    """Enhanced spatial merger with improved NaN handling and mosaic creation."""
    
    def __init__(self, merge_strategy: str = "first"):
        """
        Initialize enhanced spatial merger.
        
        Args:
            merge_strategy: "first", "last", "min", "max", "sum", "count"
        """
        self.merge_strategy = merge_strategy
        
        # Map common strategy names to rasterio merge methods
        self.strategy_mapping = {
            'average': 'first',  # Will handle averaging manually
            'maximum': 'max',
            'minimum': 'min', 
            'first': 'first',
            'last': 'last'
        }
        
    def clean_prediction_data(self, data: np.ndarray) -> np.ndarray:
        """Clean prediction data by handling NaN and infinite values."""
        # Replace NaN and inf with 0 (no data)
        cleaned = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        return cleaned
        
    def merge_predictions_from_files(self, prediction_files: Dict[str, str], 
                                   output_path: str,
                                   nodata_value: float = 0.0) -> str:
        """
        Create spatial mosaic from prediction files with enhanced error handling.
        
        Args:
            prediction_files: Dict mapping patch_id -> prediction_file_path
            output_path: Output path for merged GeoTIFF
            nodata_value: NoData value for output
            
        Returns:
            Path to merged output file
        """
        print(f"🗺️  Creating spatial mosaic from {len(prediction_files)} predictions...")
        print(f"📊 Strategy: {self.merge_strategy}")
        
        # Collect and validate prediction files
        valid_files = []
        datasets = []
        
        for patch_id, file_path in prediction_files.items():
            if os.path.exists(file_path):
                try:
                    src = rasterio.open(file_path)
                    
                    # Check for valid data
                    data = src.read(1)
                    valid_pixels = np.sum(np.isfinite(data) & (data != 0))
                    
                    if valid_pixels > 0:
                        valid_files.append(file_path)
                        datasets.append(src)
                        print(f"   ✅ {Path(file_path).name}: {valid_pixels} valid pixels")
                    else:
                        print(f"   ⚠️  {Path(file_path).name}: No valid data")
                        src.close()
                        
                except Exception as e:
                    print(f"   ❌ {Path(file_path).name}: Error reading file - {e}")
            else:
                print(f"   ❌ File not found: {file_path}")
        
        if not datasets:
            raise ValueError("No valid prediction files found for merging")
        
        print(f"📁 Processing {len(datasets)} valid prediction files...")
        
        try:
            # Get merge method
            rasterio_method = self.strategy_mapping.get(self.merge_strategy, 'first')
            
            # Create spatial mosaic
            if self.merge_strategy == 'average' and len(datasets) > 1:
                # Special handling for averaging
                merged_array, merged_transform = self._create_averaged_mosaic(
                    datasets, nodata_value
                )
            else:
                # Use rasterio merge directly
                merged_array, merged_transform = merge(
                    datasets,
                    method=rasterio_method,
                    nodata=nodata_value
                )
            
            # Clean the merged data
            if merged_array.ndim == 3:
                for i in range(merged_array.shape[0]):
                    merged_array[i] = self.clean_prediction_data(merged_array[i])
            else:
                merged_array = self.clean_prediction_data(merged_array)
            
            # Get metadata from first dataset
            output_meta = datasets[0].meta.copy()
            output_meta.update({
                'driver': 'GTiff',
                'height': merged_array.shape[-2],
                'width': merged_array.shape[-1],
                'transform': merged_transform,
                'nodata': nodata_value,
                'compress': 'lzw'
            })
            
            # Write output
            os.makedirs(Path(output_path).parent, exist_ok=True)
            
            with rasterio.open(output_path, 'w', **output_meta) as dst:
                if merged_array.ndim == 3:
                    dst.write(merged_array)
                else:
                    dst.write(merged_array, 1)
            
            # Close datasets
            for dataset in datasets:
                dataset.close()
            
            # Report results
            with rasterio.open(output_path) as result:
                result_data = result.read(1)
                valid_data = result_data[result_data != nodata_value]
                
                print(f"✅ Spatial mosaic created successfully!")
                print(f"📊 Output shape: {result.shape} pixels")
                print(f"📊 Bounds: {result.bounds}")
                print(f"📊 Coverage: {result.bounds.right - result.bounds.left:.6f} × {result.bounds.top - result.bounds.bottom:.6f} degrees")
                
                if len(valid_data) > 0:
                    print(f"📊 Height range: {valid_data.min():.2f} to {valid_data.max():.2f} meters")
                    print(f"📊 Valid pixels: {len(valid_data)} / {result_data.size}")
                else:
                    print("⚠️  No valid height data in mosaic")
            
            return output_path
            
        except Exception as e:
            # Close datasets on error
            for dataset in datasets:
                dataset.close()
            raise Exception(f"Error creating spatial mosaic: {e}")
    
    def _create_averaged_mosaic(self, datasets: List, nodata_value: float) -> Tuple[np.ndarray, rasterio.Affine]:
        """Create averaged mosaic with proper handling of overlapping areas."""
        
        # First, create a mosaic to get the output shape and transform
        mosaic_first, out_transform = merge(datasets, method='first', nodata=nodata_value)
        
        if len(datasets) == 1:
            return mosaic_first, out_transform
        
        # Initialize arrays for averaging
        output_shape = mosaic_first.shape
        sum_array = np.zeros(output_shape, dtype=np.float64)
        count_array = np.zeros(output_shape, dtype=np.int32)
        
        print(f"📊 Computing average across {len(datasets)} overlapping predictions...")
        
        # Process each dataset
        for i, dataset in enumerate(datasets):
            try:
                # Read the data
                data = dataset.read(1)
                
                # Clean the data
                data = self.clean_prediction_data(data)
                
                # Get the window in the output mosaic for this dataset
                from rasterio.windows import from_bounds
                window = from_bounds(
                    dataset.bounds.left, dataset.bounds.bottom,
                    dataset.bounds.right, dataset.bounds.top,
                    out_transform
                )
                
                # Convert window to integer bounds
                row_start = max(0, int(window.row_off))
                row_end = min(output_shape[-2], int(window.row_off + window.height))
                col_start = max(0, int(window.col_off))
                col_end = min(output_shape[-1], int(window.col_off + window.width))
                
                # Resize data to fit the window if needed
                target_height = row_end - row_start
                target_width = col_end - col_start
                
                if data.shape != (target_height, target_width):
                    from scipy.ndimage import zoom
                    scale_h = target_height / data.shape[0]
                    scale_w = target_width / data.shape[1]
                    data = zoom(data, (scale_h, scale_w), order=1)
                
                # Add to sum where data is valid (not zero/nodata)
                valid_mask = (data != nodata_value) & (data != 0) & np.isfinite(data)
                
                if output_shape[0] == 1:  # Single band
                    sum_array[0, row_start:row_end, col_start:col_end][valid_mask] += data[valid_mask]
                    count_array[0, row_start:row_end, col_start:col_end][valid_mask] += 1
                else:
                    sum_array[row_start:row_end, col_start:col_end][valid_mask] += data[valid_mask]
                    count_array[row_start:row_end, col_start:col_end][valid_mask] += 1
                    
            except Exception as e:
                print(f"⚠️  Warning: Error processing dataset {i}: {e}")
                continue
        
        # Compute average where we have data
        average_array = np.full_like(sum_array, nodata_value, dtype=np.float32)
        
        if output_shape[0] == 1:  # Single band
            valid_pixels = count_array[0] > 0
            average_array[0][valid_pixels] = (sum_array[0][valid_pixels] / count_array[0][valid_pixels]).astype(np.float32)
        else:
            valid_pixels = count_array > 0
            average_array[valid_pixels] = (sum_array[valid_pixels] / count_array[valid_pixels]).astype(np.float32)
        
        total_valid = np.sum(count_array > 0)
        print(f"📊 Averaged {total_valid} pixels across overlapping areas")
        
        return average_array, out_transform

def integrate_enhanced_merger():
    """Integration function to replace the existing PredictionMerger with enhanced version."""
    return EnhancedSpatialMerger

# Export for use in train_predict_map.py
__all__ = ['EnhancedSpatialMerger', 'integrate_enhanced_merger']
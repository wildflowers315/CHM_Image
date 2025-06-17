"""
Module for handling large area processing with patches.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_origin
from config.resolution_config import get_patch_size, get_pixel_count, validate_scale

def collect_area_patches(
    patches: List[Dict],
    output_dir: str,
    scale: int = 10
) -> Dict[str, np.ndarray]:
    """
    Collect and merge predictions from multiple patches into a single array.
    
    Args:
        patches: List of patch dictionaries with predictions
        output_dir: Directory containing patch files
        scale: Resolution in meters (default: 10)
        
    Returns:
        Dictionary mapping patch IDs to merged predictions
    """
    validate_scale(scale)
    pixel_size = get_pixel_count(scale)
    
    # Initialize dictionary to store merged predictions
    merged_data = {}
    
    for i, patch in enumerate(patches):
        patch_id = f"patch_{i:04d}"
        
        # Skip extruded patches
        if patch['is_extruded']:
            continue
        
        # Initialize array for this patch
        patch_data = np.zeros((pixel_size, pixel_size), dtype=np.float32)
        
        # Load patch data
        patch_file = f"{output_dir}/{patch_id}.tif"
        with rasterio.open(patch_file) as src:
            patch_data = src.read(1)  # Read first band
        
        merged_data[patch_id] = patch_data
    
    return merged_data

def merge_patch_predictions(
    patches: List[Dict],
    predictions: Dict[str, np.ndarray],
    scale: int = 10
) -> np.ndarray:
    """
    Merge patch predictions into a single array.
    
    Args:
        patches: List of patch dictionaries
        predictions: Dictionary mapping patch IDs to predictions
        scale: Resolution in meters (default: 10)
        
    Returns:
        Merged predictions array
    """
    validate_scale(scale)
    
    # Find bounds of entire area
    min_x = min(patch['x'] for patch in patches)
    min_y = min(patch['y'] for patch in patches)
    max_x = max(patch['x'] + patch['width'] for patch in patches)
    max_y = max(patch['y'] + patch['height'] for patch in patches)
    
    # Calculate dimensions in pixels
    width_px = int((max_x - min_x) / scale)
    height_px = int((max_y - min_y) / scale)
    
    # Initialize merged array
    merged = np.zeros((height_px, width_px), dtype=np.float32)
    
    # Merge each patch
    for i, patch in enumerate(patches):
        patch_id = f"patch_{i:04d}"
        if patch_id not in predictions:
            continue
        
        # Calculate pixel coordinates
        x_px = int((patch['x'] - min_x) / scale)
        y_px = int((patch['y'] - min_y) / scale)
        width_px = int(patch['width'] / scale)
        height_px = int(patch['height'] / scale)
        
        # Insert patch data
        merged[y_px:y_px + height_px, x_px:x_px + width_px] = predictions[patch_id]
    
    return merged 
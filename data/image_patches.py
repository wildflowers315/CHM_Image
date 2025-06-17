"""
Module for creating and managing 3D image patches from satellite data.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import rasterio
from rasterio.windows import Window
from config.resolution_config import get_patch_size, get_pixel_count, validate_scale
from data.normalization import (
    normalize_sentinel1, normalize_sentinel2, normalize_srtm_elevation,
    normalize_srtm_slope, normalize_srtm_aspect, normalize_alos2_dn,
    normalize_canopy_height, normalize_ndvi
)

class Patch:
    """Class representing a single image patch with its metadata."""
    def __init__(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        scale: int = 10,
        is_extruded: bool = False
    ):
        """
        Initialize a patch.
        
        Args:
            x: X coordinate of top-left corner in meters
            y: Y coordinate of top-left corner in meters
            width: Width of patch in meters
            height: Height of patch in meters
            scale: Resolution in meters (default: 10)
            is_extruded: Whether this patch extends beyond original bounds
        """
        validate_scale(scale)
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.scale = scale
        self.is_extruded = is_extruded
        self.data: Optional[np.ndarray] = None
        self.forest_mask: Optional[np.ndarray] = None
        self.pixel_count = get_pixel_count(scale)
    
    def get_window(self) -> Window:
        """Get rasterio window for this patch."""
        col_off = int(self.x / self.scale)
        row_off = int(self.y / self.scale)
        width = int(self.width / self.scale)
        height = int(self.height / self.scale)
        return Window(col_off=col_off, row_off=row_off, width=width, height=height)
    
    def load_data(self, src: rasterio.DatasetReader, forest_mask_src: Optional[rasterio.DatasetReader] = None) -> np.ndarray:
        """
        Load patch data from a raster source.
        
        Args:
            src: Open rasterio dataset
            forest_mask_src: Optional forest mask dataset
            
        Returns:
            Numpy array of patch data
        """
        window = self.get_window()
        self.data = src.read(window=window)
        
        # Load forest mask if provided
        if forest_mask_src is not None:
            self.forest_mask = forest_mask_src.read(1, window=window) > 0
            # Apply forest mask to data
            if self.data is not None:
                for band in range(self.data.shape[0]):
                    self.data[band][~self.forest_mask] = np.nan
        
        return self.data

def create_patch_grid(
    bounds: Tuple[float, float, float, float],
    patch_size: Optional[int] = None,
    scale: int = 10,
    overlap: float = 0.1
) -> List[Patch]:
    """
    Create a grid of patches covering the given bounds.
    
    Args:
        bounds: (minx, miny, maxx, maxy) in meters
        patch_size: Size of each patch in meters (default: based on scale)
        scale: Resolution in meters (default: 10)
        overlap: Overlap between patches as a fraction (0.0 to 1.0)
        
    Returns:
        List of Patch objects
    """
    validate_scale(scale)
    if patch_size is None:
        patch_size = get_patch_size(scale)
    
    minx, miny, maxx, maxy = bounds
    width = maxx - minx
    height = maxy - miny
    
    # Calculate stride (distance between patch starts)
    stride = int(patch_size * (1 - overlap))
    
    # Calculate number of patches needed
    n_patches_x = max(1, int(np.ceil(width / stride)))
    n_patches_y = max(1, int(np.ceil(height / stride)))
    
    print(f"Creating {n_patches_x * n_patches_y} patches with size {patch_size}m and {overlap*100}% overlap")
    
    patches = []
    for i in range(n_patches_x):
        for j in range(n_patches_y):
            # Calculate patch position with stride
            x = minx + i * stride
            y = miny + j * stride
            
            # Check if this patch extends beyond bounds
            is_extruded = (
                x + patch_size > maxx or
                y + patch_size > maxy
            )
            
            # For extruded patches, adjust size to fit bounds
            actual_width = min(patch_size, maxx - x)
            actual_height = min(patch_size, maxy - y)
            
            patches.append(Patch(
                x=x,
                y=y,
                width=patch_size if not is_extruded else actual_width,
                height=patch_size if not is_extruded else actual_height,
                scale=scale,
                is_extruded=is_extruded
            ))
    
    return patches

def normalize_patch_data(
    patch: Patch,
    data_type: str,
    band_indices: Optional[List[int]] = None
) -> np.ndarray:
    """
    Normalize patch data based on data type.
    
    Args:
        patch: Patch object with loaded data
        data_type: Type of data ('sentinel1', 'sentinel2', etc.)
        band_indices: Optional list of band indices to normalize
        
    Returns:
        Normalized data array
    """
    if patch.data is None:
        raise ValueError("Patch data must be loaded before normalization")
    
    if band_indices is None:
        band_indices = list(range(patch.data.shape[0]))
    
    normalized = patch.data.astype(np.float32).copy()
    for idx in band_indices:
        if data_type == 'sentinel1':
            normalized[idx] = (patch.data[idx].astype(np.float32) + 25) / 25
        elif data_type == 'sentinel2':
            normalized[idx] = patch.data[idx].astype(np.float32) / 10000
        elif data_type == 'srtm_elevation':
            normalized[idx] = patch.data[idx].astype(np.float32) / 2000
        elif data_type == 'srtm_slope':
            normalized[idx] = patch.data[idx].astype(np.float32) / 50
        elif data_type == 'srtm_aspect':
            normalized[idx] = (patch.data[idx].astype(np.float32) - 180) / 180
        elif data_type == 'alos2':
            dn = patch.data[idx].astype(np.float32)
            normalized[idx] = 10 * np.log10(dn ** 2) - 83.0
        elif data_type == 'canopy_height':
            normalized[idx] = patch.data[idx].astype(np.float32) / 50
        elif data_type == 'ndvi':
            normalized[idx] = patch.data[idx].astype(np.float32)
        else:
            raise ValueError(f"Unknown data type: {data_type}")
    
    return normalized

def create_3d_patches(
    patches: List[Patch],
    time_steps: int = 12,
    patch_size: Optional[int] = None,
    scale: int = 10
) -> List[Dict[str, np.ndarray]]:
    """
    Create 3D patches from a list of 2D patches.
    
    Args:
        patches: List of Patch objects
        time_steps: Number of time steps for temporal dimension
        patch_size: Size of patches in meters (default: based on scale)
        scale: Resolution in meters (default: 10)
        
    Returns:
        List of dictionaries containing 3D patch data
    """
    validate_scale(scale)
    if patch_size is None:
        patch_size = get_patch_size(scale)
    
    pixel_size = get_pixel_count(scale)
    
    # Initialize list to store 3D patches
    patches_3d = []
    
    for patch in patches:
        if patch.data is None:
            continue
        
        # Get number of bands
        n_bands = patch.data.shape[0]
        
        # Reshape data to [bands, time_steps, height, width]
        data_3d = np.zeros((n_bands, time_steps, pixel_size, pixel_size), dtype=np.float32)
        
        # Fill temporal dimension with available data
        for t in range(time_steps):
            data_3d[:, t] = patch.data
        
        patches_3d.append({
            'data': data_3d,
            'forest_mask': patch.forest_mask,
            'x': patch.x,
            'y': patch.y,
            'width': patch.width,
            'height': patch.height,
            'is_extruded': patch.is_extruded
        })
    
    return patches_3d

def prepare_training_patches(
    patches: List[Dict[str, np.ndarray]],
    gedi_data: Dict[str, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare training data from 3D patches and GEDI height data.
    
    Args:
        patches: List of 3D patch dictionaries
        gedi_data: Dictionary mapping patch IDs to GEDI height data
        
    Returns:
        Tuple of (input_data, masks, height_data)
    """
    # Initialize lists to store data
    input_data = []
    masks = []
    height_data = []
    
    for i, patch in enumerate(patches):
        patch_id = f"patch_{i:04d}"
        if patch_id not in gedi_data:
            continue
            
        # Only include patches with forest data
        if patch['forest_mask'] is not None and np.any(patch['forest_mask']):
            input_data.append(patch['data'])
            masks.append(patch['forest_mask'])
            height_data.append(gedi_data[patch_id])
    
    return (
        np.stack(input_data) if input_data else np.array([]),
        np.stack(masks) if masks else np.array([]),
        np.stack(height_data) if height_data else np.array([])
    ) 
#!/usr/bin/env python3
"""
Band identification utilities for CHM patches
Provides functions to identify bands by name/description instead of hardcoded indexes
"""

import rasterio
import numpy as np
from typing import Dict, List, Optional, Tuple

def get_band_info(tif_path: str) -> Dict[str, int]:
    """
    Get band information from a TIF file.
    
    Args:
        tif_path: Path to TIF file
        
    Returns:
        Dictionary mapping band names to band indices (0-based)
    """
    with rasterio.open(tif_path) as src:
        descriptions = src.descriptions
        band_info = {}
        
        for i, desc in enumerate(descriptions):
            if desc is not None:
                band_info[desc] = i
                
        return band_info

def find_band_by_name(tif_path: str, band_name: str) -> Optional[int]:
    """
    Find band index by name/description.
    
    Args:
        tif_path: Path to TIF file
        band_name: Name of the band to find
        
    Returns:
        Band index (0-based) or None if not found
    """
    band_info = get_band_info(tif_path)
    return band_info.get(band_name)

def find_satellite_bands(tif_path: str) -> List[int]:
    """
    Find all satellite feature bands (excludes rh and reference_height).
    
    Args:
        tif_path: Path to TIF file
        
    Returns:
        List of band indices (0-based) for satellite features
    """
    band_info = get_band_info(tif_path)
    
    # Exclude special bands
    exclude_bands = {'rh', 'reference_height'}
    
    satellite_bands = []
    for band_name, band_idx in band_info.items():
        if band_name not in exclude_bands:
            satellite_bands.append(band_idx)
    
    return sorted(satellite_bands)

def find_supervision_band(tif_path: str, supervision_mode: str) -> Optional[int]:
    """
    Find supervision band based on mode.
    
    Args:
        tif_path: Path to TIF file
        supervision_mode: "reference" or "gedi_only"
        
    Returns:
        Band index (0-based) or None if not found
    """
    if supervision_mode == "reference":
        return find_band_by_name(tif_path, "reference_height")
    elif supervision_mode == "gedi_only":
        return find_band_by_name(tif_path, "rh")
    else:
        raise ValueError(f"Unknown supervision mode: {supervision_mode}")

def extract_bands_by_name(tif_path: str, supervision_mode: str = "reference") -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract satellite features and supervision target by band names.
    
    Args:
        tif_path: Path to TIF file
        supervision_mode: "reference" or "gedi_only"
        
    Returns:
        Tuple of (satellite_features, supervision_target)
        - satellite_features: Shape (n_bands, height, width)
        - supervision_target: Shape (height, width)
    """
    with rasterio.open(tif_path) as src:
        patch_data = src.read()
    
    # Find satellite bands
    satellite_band_indices = find_satellite_bands(tif_path)
    
    # Find supervision band
    supervision_band_idx = find_supervision_band(tif_path, supervision_mode)
    
    if supervision_band_idx is None:
        raise ValueError(f"Could not find supervision band for mode '{supervision_mode}' in {tif_path}")
    
    # Extract data
    satellite_features = patch_data[satellite_band_indices]
    supervision_target = patch_data[supervision_band_idx]
    
    return satellite_features, supervision_target

def check_patch_compatibility(tif_path: str, supervision_mode: str) -> bool:
    """
    Check if patch is compatible with requested supervision mode.
    
    Args:
        tif_path: Path to TIF file
        supervision_mode: "reference" or "gedi_only"
        
    Returns:
        True if patch has required bands
    """
    try:
        supervision_band_idx = find_supervision_band(tif_path, supervision_mode)
        return supervision_band_idx is not None
    except Exception:
        return False

def print_band_info(tif_path: str):
    """
    Print detailed band information for debugging.
    
    Args:
        tif_path: Path to TIF file
    """
    print(f"=== Band Information for {tif_path} ===")
    
    with rasterio.open(tif_path) as src:
        print(f"Total bands: {src.count}")
        print(f"Shape: {src.shape}")
        print(f"CRS: {src.crs}")
        print()
        
        print("Band Index | Description")
        print("-" * 30)
        for i, desc in enumerate(src.descriptions):
            print(f"{i:>10} | {desc or 'None'}")
    
    print()
    
    # Test supervision modes
    for mode in ["reference", "gedi_only"]:
        compatible = check_patch_compatibility(tif_path, mode)
        supervision_idx = find_supervision_band(tif_path, mode)
        print(f"Supervision mode '{mode}': {'✅' if compatible else '❌'} (band {supervision_idx})")
    
    # Show satellite bands
    satellite_bands = find_satellite_bands(tif_path)
    print(f"Satellite bands: {len(satellite_bands)} bands (indices: {satellite_bands})")
    print()

if __name__ == "__main__":
    # Test with example files
    import glob
    
    # Test with enhanced patches
    enhanced_patches = glob.glob('chm_outputs/enhanced_patches/*05LE4*.tif')
    if enhanced_patches:
        print_band_info(enhanced_patches[0])
    
    # Test with original patches
    original_patches = glob.glob('chm_outputs/*05LE4*.tif')
    if original_patches:
        print_band_info(original_patches[0])
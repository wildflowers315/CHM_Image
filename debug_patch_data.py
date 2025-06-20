#!/usr/bin/env python3
"""
Debug patch data to understand NaN issues
"""

import rasterio
import numpy as np
import sys

def debug_patch(patch_path):
    """Debug a patch file."""
    print(f"🔍 Debugging: {patch_path}")
    
    try:
        with rasterio.open(patch_path) as src:
            data = src.read()
            
        print(f"📊 Shape: {data.shape}")
        print(f"📊 Data type: {data.dtype}")
        
        # Check for NaN/inf values
        nan_count = np.sum(np.isnan(data))
        inf_count = np.sum(np.isinf(data))
        
        print(f"📊 NaN values: {nan_count}")
        print(f"📊 Inf values: {inf_count}")
        
        # Check data range per band
        for i in range(min(5, data.shape[0])):  # First 5 bands
            band_data = data[i]
            valid_mask = ~(np.isnan(band_data) | np.isinf(band_data))
            valid_data = band_data[valid_mask]
            
            if len(valid_data) > 0:
                print(f"📊 Band {i}: {valid_data.min():.3f} to {valid_data.max():.3f} (valid: {len(valid_data)})")
            else:
                print(f"📊 Band {i}: No valid data")
        
        if data.shape[0] > 5:
            print(f"📊 ... and {data.shape[0] - 5} more bands")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    patches = [
        "chm_outputs/dchm_09gd4_bandNum31_scale10_patch0000.tif",
        "chm_outputs/dchm_09gd4_bandNum31_scale10_patch0001.tif", 
        "chm_outputs/dchm_09gd4_bandNum31_scale10_patch0002.tif",
        "chm_outputs/dchm_09gd4_bandNum31_scale10_patch0003.tif"
    ]
    
    for patch in patches:
        debug_patch(patch)
        print()
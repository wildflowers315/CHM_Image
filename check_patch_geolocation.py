#!/usr/bin/env python3
"""
Check geospatial information of patches to understand their spatial arrangement
"""

import rasterio
import glob
from pathlib import Path

def check_patch_geolocation(patch_path):
    """Check the geospatial properties of a patch."""
    print(f"🔍 {Path(patch_path).name}")
    
    with rasterio.open(patch_path) as src:
        print(f"   📊 Shape: {src.shape} (W×H)")
        print(f"   📊 CRS: {src.crs}")
        print(f"   📊 Transform: {src.transform}")
        print(f"   📊 Bounds: {src.bounds}")
        print(f"   📊 Resolution: {src.res}")
        
        # Calculate corner coordinates
        bounds = src.bounds
        print(f"   📍 Top-left: ({bounds.left:.6f}, {bounds.top:.6f})")
        print(f"   📍 Bottom-right: ({bounds.right:.6f}, {bounds.bottom:.6f})")
        print(f"   📏 Width: {bounds.right - bounds.left:.6f}")
        print(f"   📏 Height: {bounds.top - bounds.bottom:.6f}")
        print()

def main():
    print("🗺️  Checking patch geolocations...")
    
    # Find patch files
    patch_files = glob.glob("chm_outputs/dchm_09gd4_*.tif")
    patch_files.sort()
    
    for patch_file in patch_files:
        check_patch_geolocation(patch_file)
    
    print("📐 Spatial Analysis:")
    if len(patch_files) >= 2:
        # Compare first two patches to understand spatial relationship
        with rasterio.open(patch_files[0]) as src1, rasterio.open(patch_files[1]) as src2:
            bounds1 = src1.bounds
            bounds2 = src2.bounds
            
            # Check if patches are adjacent or overlapping
            print(f"   Patch 0 vs Patch 1:")
            print(f"   📏 X gap: {bounds2.left - bounds1.right:.6f}")
            print(f"   📏 Y gap: {bounds1.bottom - bounds2.top:.6f}")
            
            if abs(bounds2.left - bounds1.right) < 0.001:
                print("   🔗 Patches are horizontally adjacent")
            elif abs(bounds1.bottom - bounds2.top) < 0.001:
                print("   🔗 Patches are vertically adjacent")
            else:
                print("   🔍 Patches have spatial gap or overlap")

if __name__ == "__main__":
    main()
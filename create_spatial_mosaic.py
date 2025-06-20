#!/usr/bin/env python3
"""
Create spatial mosaic from predictions based on their geographic locations
"""

import rasterio
from rasterio.merge import merge
from rasterio.plot import show
import glob
from pathlib import Path
import argparse
import numpy as np

def create_spatial_mosaic(prediction_dir, output_path, patch_pattern="prediction_dchm_09gd4_*.tif"):
    """Create a spatial mosaic from prediction patches."""
    
    print("🗺️  Creating spatial mosaic from predictions...")
    
    # Find prediction files
    prediction_files = glob.glob(str(Path(prediction_dir) / patch_pattern))
    prediction_files.sort()
    
    print(f"📁 Found {len(prediction_files)} prediction files:")
    for f in prediction_files:
        print(f"   - {Path(f).name}")
    
    if len(prediction_files) < 2:
        print("❌ Need at least 2 prediction files to create mosaic")
        return False
    
    # Open all prediction files
    src_files_to_mosaic = []
    for fp in prediction_files:
        src = rasterio.open(fp)
        src_files_to_mosaic.append(src)
        print(f"📊 {Path(fp).name}: {src.shape} pixels, bounds: {src.bounds}")
    
    # Create mosaic
    print("🔧 Merging predictions...")
    mosaic, out_trans = merge(src_files_to_mosaic)
    
    # Get metadata from first file
    out_meta = src_files_to_mosaic[0].meta.copy()
    
    # Update metadata for mosaic
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2], 
        "transform": out_trans,
        "compress": "lzw"
    })
    
    # Write mosaic
    print(f"💾 Saving mosaic to: {output_path}")
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(mosaic)
    
    # Close source files
    for src in src_files_to_mosaic:
        src.close()
    
    # Get mosaic info
    with rasterio.open(output_path) as mosaic_src:
        mosaic_data = mosaic_src.read(1)
        valid_data = mosaic_data[mosaic_data != 0]  # Exclude no-data
        
        print(f"✅ Mosaic created successfully!")
        print(f"📊 Mosaic shape: {mosaic_src.shape} pixels")
        print(f"📊 Mosaic bounds: {mosaic_src.bounds}")
        print(f"📊 Mosaic CRS: {mosaic_src.crs}")
        print(f"📊 Pixel resolution: {mosaic_src.res}")
        print(f"📊 Coverage: {mosaic_src.bounds.right - mosaic_src.bounds.left:.6f} × {mosaic_src.bounds.top - mosaic_src.bounds.bottom:.6f} degrees")
        
        if len(valid_data) > 0:
            print(f"📊 Height range: {valid_data.min():.2f} to {valid_data.max():.2f} meters")
            print(f"📊 Valid pixels: {len(valid_data)} / {mosaic_data.size}")
        else:
            print("⚠️  No valid height data in mosaic")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Create spatial mosaic from prediction patches")
    parser.add_argument('--prediction-dir', required=True, help='Directory containing prediction files')
    parser.add_argument('--output-path', required=True, help='Output mosaic file path')
    parser.add_argument('--patch-pattern', default='prediction_dchm_09gd4_*.tif', help='Pattern to match prediction files')
    
    args = parser.parse_args()
    
    print(f"🚀 Spatial Mosaic Creator")
    print(f"📁 Prediction directory: {args.prediction_dir}")
    print(f"📄 Output path: {args.output_path}")
    print(f"🔍 Pattern: {args.patch_pattern}")
    print()
    
    success = create_spatial_mosaic(args.prediction_dir, args.output_path, args.patch_pattern)
    
    if success:
        print(f"\n🎉 Success! Spatial mosaic saved to: {args.output_path}")
    else:
        print(f"\n❌ Failed to create spatial mosaic")

if __name__ == "__main__":
    main()
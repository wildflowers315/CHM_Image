#!/usr/bin/env python3
"""
Compare the averaged prediction vs spatial mosaic
"""

import rasterio
import numpy as np

def compare_results():
    """Compare the two different aggregation approaches."""
    
    print("🔍 Comparing Results:")
    print("=" * 50)
    
    # Read averaged prediction (my incorrect approach)
    with rasterio.open("safe_predictions/prediction_average.tif") as src:
        avg_data = src.read(1)
        avg_shape = src.shape
        avg_bounds = src.bounds
        
    print(f"📊 Averaged Prediction (incorrect):")
    print(f"   Shape: {avg_shape} pixels")
    print(f"   Bounds: {avg_bounds}")
    print(f"   Coverage: {avg_bounds.right - avg_bounds.left:.6f} × {avg_bounds.top - avg_bounds.bottom:.6f} degrees")
    print(f"   Height range: {avg_data.min():.2f} to {avg_data.max():.2f} meters")
    print()
    
    # Read spatial mosaic (correct approach)
    with rasterio.open("spatial_mosaic.tif") as src:
        mosaic_data = src.read(1)
        mosaic_shape = src.shape  
        mosaic_bounds = src.bounds
        
    print(f"📊 Spatial Mosaic (correct):")
    print(f"   Shape: {mosaic_shape} pixels")
    print(f"   Bounds: {mosaic_bounds}")
    print(f"   Coverage: {mosaic_bounds.right - mosaic_bounds.left:.6f} × {mosaic_bounds.top - mosaic_bounds.bottom:.6f} degrees")
    print(f"   Height range: {mosaic_data.min():.2f} to {mosaic_data.max():.2f} meters")
    print()
    
    print("🎯 Key Differences:")
    print(f"   ✅ Spatial mosaic is {mosaic_shape[0] // avg_shape[0]}x taller ({mosaic_shape[0]} vs {avg_shape[0]} pixels)")
    print(f"   ✅ Spatial mosaic covers {(mosaic_bounds.top - mosaic_bounds.bottom) / (avg_bounds.top - avg_bounds.bottom):.1f}x more area")
    print(f"   ✅ Spatial mosaic preserves spatial relationships between patches")
    print(f"   ✅ Spatial mosaic shows individual patch contributions")
    print()
    
    print("📍 What the averaged version did wrong:")
    print("   ❌ Averaged all 4 patches pixel-by-pixel into single 257×257 image")
    print("   ❌ Lost spatial arrangement and geographic relationships")
    print("   ❌ Created artificial blending between non-adjacent areas")
    print()
    
    print("📍 What the spatial mosaic does correctly:")
    print("   ✅ Arranges patches according to their actual geographic positions")
    print("   ✅ Creates proper 1025×257 pixel map covering full area")
    print("   ✅ Preserves spatial continuity and relationships")
    print("   ✅ Handles overlapping areas appropriately")

if __name__ == "__main__":
    compare_results()
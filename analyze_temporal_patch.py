#!/usr/bin/env python3
"""
Analyze temporal patch structure for Paul's 2025 methodology
"""

import rasterio
import numpy as np
import pandas as pd
import os
from pathlib import Path

def analyze_temporal_patch(patch_path):
    """Analyze the temporal patch structure and band organization."""
    
    print(f"🔍 Analyzing temporal patch: {patch_path}")
    
    if not os.path.exists(patch_path):
        print(f"❌ File not found: {patch_path}")
        return
    
    with rasterio.open(patch_path) as src:
        print(f"\n📊 Basic Information:")
        print(f"   Shape: {src.height} x {src.width} pixels")
        print(f"   Bands: {src.count}")
        print(f"   CRS: {src.crs}")
        print(f"   Resolution: {src.res}")
        print(f"   Bounds: {src.bounds}")
        
        # Get band descriptions
        descriptions = src.descriptions
        print(f"\n🏷️  Band Descriptions:")
        
        # Categorize bands
        s1_bands = []
        s2_bands = []
        alos2_bands = []
        other_bands = []
        
        for i, desc in enumerate(descriptions):
            band_num = i + 1
            if desc:
                if desc.startswith('S1_'):
                    s1_bands.append((band_num, desc))
                elif any(desc.startswith(prefix) for prefix in ['B2_', 'B3_', 'B4_', 'B5_', 'B6_', 'B7_', 'B8_', 'B8A_', 'B11_', 'B12_', 'NDVI_']):
                    s2_bands.append((band_num, desc))
                elif desc.startswith('ALOS2_'):
                    alos2_bands.append((band_num, desc))
                else:
                    other_bands.append((band_num, desc))
            else:
                other_bands.append((band_num, f"Band_{band_num}"))
        
        print(f"\n📈 Band Categories:")
        print(f"   Sentinel-1: {len(s1_bands)} bands")
        print(f"   Sentinel-2: {len(s2_bands)} bands") 
        print(f"   ALOS-2: {len(alos2_bands)} bands")
        print(f"   Other: {len(other_bands)} bands")
        print(f"   Total: {len(s1_bands) + len(s2_bands) + len(alos2_bands) + len(other_bands)} bands")
        
        # Show sample bands from each category
        if s1_bands:
            print(f"\n🛰️  Sentinel-1 Bands (showing first 5):")
            for band_num, desc in s1_bands[:5]:
                print(f"   Band {band_num:3d}: {desc}")
            if len(s1_bands) > 5:
                print(f"   ... and {len(s1_bands)-5} more")
        
        if s2_bands:
            print(f"\n🌍 Sentinel-2 Bands (showing first 10):")
            for band_num, desc in s2_bands[:10]:
                print(f"   Band {band_num:3d}: {desc}")
            if len(s2_bands) > 10:
                print(f"   ... and {len(s2_bands)-10} more")
        
        if alos2_bands:
            print(f"\n📡 ALOS-2 Bands (showing first 5):")
            for band_num, desc in alos2_bands[:5]:
                print(f"   Band {band_num:3d}: {desc}")
            if len(alos2_bands) > 5:
                print(f"   ... and {len(alos2_bands)-5} more")
        
        if other_bands:
            print(f"\n🔧 Other Bands:")
            for band_num, desc in other_bands:
                print(f"   Band {band_num:3d}: {desc}")
        
        # Analyze temporal structure for each sensor
        print(f"\n📅 Temporal Structure Analysis:")
        
        # Check Sentinel-1 months
        s1_months = set()
        for _, desc in s1_bands:
            if '_M' in desc:
                month = desc.split('_M')[1][:2]
                s1_months.add(month)
        print(f"   S1 months: {sorted(s1_months)} ({len(s1_months)} months)")
        
        # Check Sentinel-2 months
        s2_months = set()
        for _, desc in s2_bands:
            if '_M' in desc:
                month = desc.split('_M')[1][:2]
                s2_months.add(month)
        print(f"   S2 months: {sorted(s2_months)} ({len(s2_months)} months)")
        
        # Check ALOS-2 months
        alos2_months = set()
        for _, desc in alos2_bands:
            if '_M' in desc:
                month = desc.split('_M')[1][:2]
                alos2_months.add(month)
        print(f"   ALOS2 months: {sorted(alos2_months)} ({len(alos2_months)} months)")
        
        # Data quality check
        print(f"\n🔍 Data Quality Check:")
        
        # Sample a small window for statistics
        window = rasterio.windows.Window(0, 0, min(100, src.width), min(100, src.height))
        
        for category, bands in [("S1", s1_bands[:3]), ("S2", s2_bands[:3]), ("ALOS2", alos2_bands[:3]), ("Other", other_bands[:3])]:
            if bands:
                print(f"\n   {category} Sample Statistics:")
                for band_num, desc in bands:
                    data = src.read(band_num, window=window)
                    valid_data = data[~np.isnan(data)]
                    if len(valid_data) > 0:
                        print(f"     {desc}: min={valid_data.min():.2f}, max={valid_data.max():.2f}, mean={valid_data.mean():.2f}")
                    else:
                        print(f"     {desc}: All NaN values")
        
        # Check for GEDI/reference data
        print(f"\n🎯 Reference Data Check:")
        gedi_bands = [desc for _, desc in other_bands if 'rh' in desc.lower() or 'gedi' in desc.lower()]
        if gedi_bands:
            print(f"   Found GEDI bands: {gedi_bands}")
            # Find the band number for GEDI data
            for band_num, desc in other_bands:
                if 'rh' in desc.lower():
                    gedi_data = src.read(band_num, window=window)
                    gedi_valid = gedi_data[~np.isnan(gedi_data) & (gedi_data > 0)]
                    if len(gedi_valid) > 0:
                        print(f"     GEDI pixels: {len(gedi_valid)}/{gedi_data.size} ({len(gedi_valid)/gedi_data.size*100:.2f}%)")
                        print(f"     GEDI range: {gedi_valid.min():.2f}m to {gedi_valid.max():.2f}m")
                        print(f"     GEDI mean: {gedi_valid.mean():.2f}m")
                    else:
                        print(f"     No valid GEDI data in sample window")
                    break
        else:
            print(f"   No GEDI/reference bands found")
        
        return {
            'total_bands': src.count,
            's1_bands': len(s1_bands),
            's2_bands': len(s2_bands), 
            'alos2_bands': len(alos2_bands),
            'other_bands': len(other_bands),
            'shape': (src.height, src.width),
            'crs': src.crs,
            'band_descriptions': descriptions
        }

if __name__ == "__main__":
    # Analyze the temporal patch
    patch_path = "chm_outputs/dchm_09gd4_temporal_bandNum196_scale10_patch0000.tif"
    result = analyze_temporal_patch(patch_path)
    
    if result:
        print(f"\n✅ Analysis complete!")
        print(f"   Total bands: {result['total_bands']}")
        print(f"   Temporal bands: {result['s1_bands'] + result['s2_bands'] + result['alos2_bands']}")
        print(f"   Other bands: {result['other_bands']}")
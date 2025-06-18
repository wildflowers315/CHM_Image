#!/usr/bin/env python3
import rasterio
import numpy as np

patch_path = 'chm_outputs/dchm_09gd4_bandNum31_scale10_patch0000.tif'
with rasterio.open(patch_path) as src:
    print('All band descriptions:')
    for i, desc in enumerate(src.descriptions, 1):
        print(f'  Band {i}: {desc}')
    
    # Look for GEDI data
    gedi_band_idx = None
    for i, desc in enumerate(src.descriptions):
        if desc and 'rh' in desc.lower():
            gedi_band_idx = i
            break
    
    if gedi_band_idx is not None:
        print(f'\nGEDI data found in band {gedi_band_idx + 1}: {src.descriptions[gedi_band_idx]}')
        gedi_data = src.read(gedi_band_idx + 1)
        valid_mask = (~np.isnan(gedi_data)) & (gedi_data != 0)
        non_zero_count = np.count_nonzero(valid_mask)
        total_pixels = gedi_data.size
        print(f'Non-zero GEDI pixels: {non_zero_count}/{total_pixels} ({non_zero_count/total_pixels*100:.2f}%)')
        if non_zero_count > 0:
            valid_gedi = gedi_data[valid_mask]
            print(f'GEDI value range: {np.min(valid_gedi):.2f} to {np.max(valid_gedi):.2f}')
        else:
            print('No valid GEDI data found')
    else:
        print('\nNo GEDI rh band found')
    
    # Check if temporal data exists (multiple S2 bands for same spectral band)
    print(f'\nTemporal analysis:')
    s2_bands = [desc for desc in src.descriptions if desc and desc.startswith('B')]
    print(f'Number of S2-like bands: {len(s2_bands)}')
    print(f'S2 bands: {s2_bands}')
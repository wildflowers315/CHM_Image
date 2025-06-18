#!/usr/bin/env python3
import rasterio
import numpy as np

print('=== TRAINING PATCH ANALYSIS ===')
patch_path = 'chm_outputs/dchm_09gd4_bandNum31_scale10_patch0000.tif'
with rasterio.open(patch_path) as src:
    print(f'Training patch: {patch_path}')
    print(f'Dimensions: {src.width} x {src.height}')
    print(f'Bands: {src.count}')
    print(f'CRS: {src.crs}')
    print(f'Bounds: {src.bounds}')
    print(f'Resolution: {src.res}')
    
    # Check GEDI data distribution
    gedi_band = src.read(31)  # rh band
    valid_gedi = gedi_band[gedi_band > 0]
    print(f'GEDI pixels: {len(valid_gedi)}/{gedi_band.size} ({len(valid_gedi)/gedi_band.size*100:.2f}%)')
    if len(valid_gedi) > 0:
        print(f'GEDI height range: {valid_gedi.min():.2f} - {valid_gedi.max():.2f} m')
        print(f'GEDI height mean: {valid_gedi.mean():.2f} m')

print('\n=== EVALUATION REFERENCE ANALYSIS ===')
ref_path = 'downloads/dchm_09gd4.tif'
try:
    with rasterio.open(ref_path) as src:
        print(f'Reference CHM: {ref_path}')
        print(f'Dimensions: {src.width} x {src.height}')
        print(f'Bands: {src.count}')
        print(f'CRS: {src.crs}')
        print(f'Bounds: {src.bounds}')
        print(f'Resolution: {src.res}')
        
        # Sample reference data
        ref_data = src.read(1)
        valid_ref = ref_data[~np.isnan(ref_data) & (ref_data > 0)]
        print(f'Valid pixels: {len(valid_ref)}/{ref_data.size} ({len(valid_ref)/ref_data.size*100:.2f}%)')
        if len(valid_ref) > 0:
            print(f'Height range: {valid_ref.min():.2f} - {valid_ref.max():.2f} m')
            print(f'Height mean: {valid_ref.mean():.2f} m')
except Exception as e:
    print(f'Error reading reference CHM: {e}')

print('\n=== SPATIAL OVERLAP ANALYSIS ===')
# Check if patch bounds overlap with reference bounds
try:
    with rasterio.open(patch_path) as patch_src, rasterio.open(ref_path) as ref_src:
        patch_bounds = patch_src.bounds
        ref_bounds = ref_src.bounds
        
        overlap_left = max(patch_bounds.left, ref_bounds.left)
        overlap_right = min(patch_bounds.right, ref_bounds.right)
        overlap_bottom = max(patch_bounds.bottom, ref_bounds.bottom)
        overlap_top = min(patch_bounds.top, ref_bounds.top)
        
        if overlap_left < overlap_right and overlap_bottom < overlap_top:
            overlap_area = (overlap_right - overlap_left) * (overlap_top - overlap_bottom)
            patch_area = (patch_bounds.right - patch_bounds.left) * (patch_bounds.top - patch_bounds.bottom)
            print(f'Spatial overlap exists!')
            print(f'Overlap bounds: ({overlap_left:.2f}, {overlap_bottom:.2f}, {overlap_right:.2f}, {overlap_top:.2f})')
            print(f'Overlap area: {overlap_area:.2f} sq units')
            print(f'Patch coverage: {overlap_area/patch_area*100:.2f}%')
        else:
            print('No spatial overlap found between patch and reference CHM')
except Exception as e:
    print(f'Error checking spatial overlap: {e}')
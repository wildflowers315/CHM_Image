#!/usr/bin/env python3
"""
Investigate RGB normalization issues and create properly normalized RGB.
"""

import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt

def analyze_original_rgb_values():
    """Analyze the original RGB values from the patch."""
    
    print("=== ANALYZING ORIGINAL RGB VALUES ===")
    
    patch_path = 'chm_outputs/dchm_09gd4_bandNum31_scale10_patch0000.tif'
    
    with rasterio.open(patch_path) as src:
        # Get RGB bands (B2=Blue, B3=Green, B4=Red)
        b2_data = src.read(3).astype(np.float32)  # Blue
        b3_data = src.read(4).astype(np.float32)  # Green  
        b4_data = src.read(5).astype(np.float32)  # Red
        
        print("Original Sentinel-2 L2A surface reflectance values:")
        print(f"B2 (Blue):  min={b2_data.min():.1f}, max={b2_data.max():.1f}, mean={b2_data.mean():.1f}")
        print(f"B3 (Green): min={b3_data.min():.1f}, max={b3_data.max():.1f}, mean={b3_data.mean():.1f}")
        print(f"B4 (Red):   min={b4_data.min():.1f}, max={b4_data.max():.1f}, mean={b4_data.mean():.1f}")
        
        # These should be in range 0-10000 for Sentinel-2 L2A surface reflectance
        # But it looks like they might be scaled differently
        
        return b4_data, b3_data, b2_data  # Return in RGB order

def scale_adjust_band_demo(band_data, min_val, max_val, contrast=1.0, gamma=1.0):
    """Demo of the scale_adjust_band function from save_evaluation_pdf.py"""
    
    # Handle NaN values
    nan_mask = np.isnan(band_data)
    temp_nodata = -9999
    work_data = band_data.copy()
    
    if np.any(work_data[~nan_mask] == temp_nodata):
        valid_min = np.min(work_data[~nan_mask]) if not nan_mask.all() else -1
        temp_nodata = valid_min - 1

    work_data[nan_mask] = temp_nodata
    work_data = work_data.astype(np.float32)

    # Min/Max scaling
    mask_valid = (work_data != temp_nodata)
    scaled_data = np.zeros_like(work_data, dtype=np.float32)
    if max_val - min_val != 0:
        scaled_data[mask_valid] = (work_data[mask_valid] - min_val) / (max_val - min_val)
    scaled_data[mask_valid] = np.clip(scaled_data[mask_valid], 0, 1)

    # Contrast adjustment
    if contrast != 1.0:
        scaled_data[mask_valid] = 0.5 + contrast * (scaled_data[mask_valid] - 0.5)
        scaled_data[mask_valid] = np.clip(scaled_data[mask_valid], 0, 1)

    # Gamma correction
    if gamma != 1.0 and gamma > 0:
        gamma_mask = mask_valid & (scaled_data > 0)
        with np.errstate(invalid='ignore'):
            scaled_data[gamma_mask] = scaled_data[gamma_mask]**(1.0 / gamma)
        scaled_data[gamma_mask] = np.clip(scaled_data[gamma_mask], 0, 1)

    # Convert to uint8
    scaled_data[~mask_valid] = 0
    scaled_uint8 = (scaled_data * 255).astype(np.uint8)
    return scaled_uint8

def compare_normalization_methods(red_data, green_data, blue_data, output_dir):
    """Compare different normalization approaches."""
    
    print("\n=== COMPARING NORMALIZATION METHODS ===")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Method 1: Our current approach (percentile stretch)
    rgb_percentile = np.zeros((red_data.shape[0], red_data.shape[1], 3), dtype=np.uint8)
    
    for i, (band_data, band_name) in enumerate([(red_data, 'Red'), (green_data, 'Green'), (blue_data, 'Blue')]):
        valid_mask = ~np.isnan(band_data) & (band_data > 0)
        if valid_mask.sum() > 0:
            p2, p98 = np.percentile(band_data[valid_mask], [2, 98])
            stretched = np.clip((band_data - p2) / (p98 - p2) * 255, 0, 255)
            rgb_percentile[:, :, i] = stretched.astype(np.uint8)
            print(f"{band_name} percentile stretch: {p2:.1f} - {p98:.1f} -> 0-255")
    
    # Method 2: save_evaluation_pdf.py approach (0-3000 range with contrast/gamma)
    rgb_pdf_method = np.zeros((red_data.shape[0], red_data.shape[1], 3), dtype=np.uint8)
    
    scale_params = [
        {'min': 0, 'max': 3000, 'contrast': 1.2, 'gamma': 0.8},  # Red (B4)
        {'min': 0, 'max': 3000, 'contrast': 1.2, 'gamma': 0.8},  # Green (B3)
        {'min': 0, 'max': 3000, 'contrast': 1.2, 'gamma': 0.8}   # Blue (B2)
    ]
    
    for i, (band_data, params, band_name) in enumerate([(red_data, scale_params[0], 'Red'), 
                                                         (green_data, scale_params[1], 'Green'), 
                                                         (blue_data, scale_params[2], 'Blue')]):
        scaled = scale_adjust_band_demo(
            band_data, 
            params['min'], 
            params['max'], 
            contrast=params['contrast'], 
            gamma=params['gamma']
        )
        rgb_pdf_method[:, :, i] = scaled
        print(f"{band_name} PDF method: {params['min']}-{params['max']} with contrast={params['contrast']}, gamma={params['gamma']}")
    
    # Method 3: Adaptive range based on actual data
    rgb_adaptive = np.zeros((red_data.shape[0], red_data.shape[1], 3), dtype=np.uint8)
    
    for i, (band_data, band_name) in enumerate([(red_data, 'Red'), (green_data, 'Green'), (blue_data, 'Blue')]):
        valid_mask = ~np.isnan(band_data) & (band_data > 0)
        if valid_mask.sum() > 0:
            data_min = band_data[valid_mask].min()
            data_max = band_data[valid_mask].max()
            
            # Use the PDF method but with adaptive range
            scaled = scale_adjust_band_demo(
                band_data, 
                data_min, 
                data_max, 
                contrast=1.2, 
                gamma=0.8
            )
            rgb_adaptive[:, :, i] = scaled
            print(f"{band_name} adaptive method: {data_min:.1f}-{data_max:.1f} with contrast=1.2, gamma=0.8")
    
    # Method 4: Standard Sentinel-2 scaling (assuming 0-10000 range)
    rgb_standard = np.zeros((red_data.shape[0], red_data.shape[1], 3), dtype=np.uint8)
    
    for i, (band_data, band_name) in enumerate([(red_data, 'Red'), (green_data, 'Green'), (blue_data, 'Blue')]):
        # Standard Sentinel-2 L2A is 0-10000, but we'll use 0-4000 for better visualization
        scaled = scale_adjust_band_demo(
            band_data, 
            0, 
            4000, 
            contrast=1.1, 
            gamma=0.9
        )
        rgb_standard[:, :, i] = scaled
        print(f"{band_name} standard S2: 0-4000 with contrast=1.1, gamma=0.9")
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    axes[0,0].imshow(rgb_percentile)
    axes[0,0].set_title('Method 1: Percentile Stretch\n(Our current approach)')
    axes[0,0].axis('off')
    
    axes[0,1].imshow(rgb_pdf_method)
    axes[0,1].set_title('Method 2: PDF Script Method\n(0-3000, contrast=1.2, gamma=0.8)')
    axes[0,1].axis('off')
    
    axes[1,0].imshow(rgb_adaptive)
    axes[1,0].set_title('Method 3: Adaptive Range\n(Data min-max, contrast=1.2, gamma=0.8)')
    axes[1,0].axis('off')
    
    axes[1,1].imshow(rgb_standard)
    axes[1,1].set_title('Method 4: Standard S2\n(0-4000, contrast=1.1, gamma=0.9)')
    axes[1,1].axis('off')
    
    plt.tight_layout()
    comparison_path = os.path.join(output_dir, 'rgb_normalization_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Comparison saved: {comparison_path}")
    
    # Return the best looking one for further use
    return rgb_adaptive, comparison_path

def create_corrected_rgb_composite(output_dir):
    """Create a properly normalized RGB composite using the best method."""
    
    print("\n=== CREATING CORRECTED RGB COMPOSITE ===")
    
    # Load original data
    red_data, green_data, blue_data = analyze_original_rgb_values()
    
    # Use adaptive method (seemed to work best in comparison)
    rgb_corrected = np.zeros((red_data.shape[0], red_data.shape[1], 3), dtype=np.uint8)
    
    for i, (band_data, band_name) in enumerate([(red_data, 'Red'), (green_data, 'Green'), (blue_data, 'Blue')]):
        valid_mask = ~np.isnan(band_data) & (band_data > 0)
        if valid_mask.sum() > 0:
            # Use data-driven range with moderate enhancement
            data_min = band_data[valid_mask].min()
            data_max = band_data[valid_mask].max()
            
            # Apply the same scaling as save_evaluation_pdf.py but with adaptive range
            scaled = scale_adjust_band_demo(
                band_data, 
                data_min, 
                data_max * 0.8,  # Use 80% of max for better contrast
                contrast=1.15,   # Moderate contrast enhancement
                gamma=0.85       # Slight gamma correction
            )
            rgb_corrected[:, :, i] = scaled
            print(f"{band_name}: {data_min:.1f} - {data_max*0.8:.1f} -> 0-255")
    
    # Save corrected RGB composite
    patch_path = 'chm_outputs/dchm_09gd4_bandNum31_scale10_patch0000.tif'
    corrected_path = os.path.join(output_dir, 'rgb_composite_corrected.tif')
    
    with rasterio.open(patch_path) as src:
        profile = src.profile.copy()
        profile.update({
            'count': 3,
            'dtype': 'uint8',
            'compress': 'lzw'
        })
        
        # Stack in CHW format for rasterio
        rgb_chw = np.transpose(rgb_corrected, (2, 0, 1))
        
        with rasterio.open(corrected_path, 'w', **profile) as dst:
            dst.write(rgb_chw)
            dst.descriptions = ['Red (B4)', 'Green (B3)', 'Blue (B2)']
    
    print(f"✓ Corrected RGB composite saved: {corrected_path}")
    
    # Create preview
    preview_path = os.path.join(output_dir, 'rgb_corrected_preview.png')
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb_corrected)
    plt.title('Corrected RGB Composite\n(Adaptive range with moderate enhancement)')
    plt.axis('off')
    plt.savefig(preview_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Preview saved: {preview_path}")
    
    return corrected_path

def main():
    output_dir = 'chm_outputs/rgb_normalization_investigation'
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== RGB NORMALIZATION INVESTIGATION ===")
    
    # Analyze original values
    red_data, green_data, blue_data = analyze_original_rgb_values()
    
    # Compare normalization methods
    best_rgb, comparison_path = compare_normalization_methods(red_data, green_data, blue_data, output_dir)
    
    # Create corrected composite
    corrected_path = create_corrected_rgb_composite(output_dir)
    
    print(f"\n=== SUMMARY ===")
    print(f"Investigation results saved in: {output_dir}")
    print(f"- Comparison plot: {comparison_path}")
    print(f"- Corrected RGB: {corrected_path}")
    print("\nThe issue was that existing RGB composites bypass the scale_adjust_band normalization.")
    print("The corrected version uses adaptive range with the same contrast/gamma enhancements.")

if __name__ == "__main__":
    main()
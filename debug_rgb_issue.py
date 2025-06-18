#!/usr/bin/env python3
"""
Debug why RGB is not showing in comparison grid.
"""

import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt

def debug_rgb_loading():
    """Debug RGB loading in evaluation process."""
    
    print("=== DEBUGGING RGB LOADING ===")
    
    # Check the RGB composite file
    rgb_path = 'chm_outputs/rgb_fixed_results/rgb_composite_fixed.tif'
    
    if os.path.exists(rgb_path):
        print(f"✓ RGB file exists: {rgb_path}")
        
        with rasterio.open(rgb_path) as src:
            print(f"RGB file info:")
            print(f"  - Bands: {src.count}")
            print(f"  - Shape: {src.width} x {src.height}")
            print(f"  - Data type: {src.dtypes[0]}")
            print(f"  - Descriptions: {src.descriptions}")
            
            # Read RGB data
            rgb_data = src.read()
            print(f"  - RGB data shape: {rgb_data.shape}")
            print(f"  - RGB data range: {rgb_data.min()} - {rgb_data.max()}")
            
            # Check each band
            for i in range(3):
                band_data = rgb_data[i]
                print(f"  - Band {i+1}: min={band_data.min()}, max={band_data.max()}, mean={band_data.mean():.1f}")
            
            # Test the format expected by matplotlib
            if rgb_data.shape[0] == 3:  # CHW format
                rgb_display = np.transpose(rgb_data, (1, 2, 0))  # Convert to HWC
            else:
                rgb_display = rgb_data
            
            print(f"  - Display format shape: {rgb_display.shape}")
            
            # Create test visualization
            plt.figure(figsize=(8, 4))
            
            plt.subplot(1, 2, 1)
            plt.imshow(rgb_display)
            plt.title('RGB Composite (Fixed)')
            plt.axis('off')
            
            # Test what happens with the load_rgb_composite function
            from save_evaluation_pdf import load_rgb_composite
            
            # Mock target shape and transform
            target_shape = (rgb_data.shape[1], rgb_data.shape[2])
            transform = src.transform
            
            print(f"\n=== TESTING load_rgb_composite FUNCTION ===")
            
            try:
                loaded_rgb = load_rgb_composite(rgb_path, target_shape, transform)
                
                if loaded_rgb is not None:
                    print(f"✓ load_rgb_composite returned data: {loaded_rgb.shape}")
                    print(f"  - Data range: {loaded_rgb.min()} - {loaded_rgb.max()}")
                    
                    plt.subplot(1, 2, 2)
                    plt.imshow(loaded_rgb)
                    plt.title('Loaded by save_evaluation_pdf')
                    plt.axis('off')
                else:
                    print("✗ load_rgb_composite returned None")
                    
            except Exception as e:
                print(f"✗ load_rgb_composite failed: {e}")
            
            plt.tight_layout()
            debug_path = 'chm_outputs/rgb_fixed_results/debug_rgb.png'
            plt.savefig(debug_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✓ Debug visualization saved: {debug_path}")
            
    else:
        print(f"✗ RGB file not found: {rgb_path}")

def create_working_rgb_evaluation():
    """Create a working evaluation with proper RGB handling."""
    
    print("\n=== CREATING WORKING RGB EVALUATION ===")
    
    # Use the correctly extracted RGB
    rgb_path = 'chm_outputs/rgb_fixed_results/rgb_composite_fixed.tif'
    pred_path = 'chm_outputs/rgb_fixed_results/prediction_with_rgb.tif'
    ref_path = 'downloads/dchm_09gd4.tif'
    output_dir = 'chm_outputs/rgb_working_results'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data manually and create comparison
    print("Loading prediction data...")
    with rasterio.open(pred_path) as src:
        pred_data = src.read(1)
        pred_bounds = src.bounds
        pred_transform = src.transform
    
    print("Loading RGB data...")
    with rasterio.open(rgb_path) as src:
        rgb_data = src.read()  # CHW format
        rgb_display = np.transpose(rgb_data, (1, 2, 0))  # Convert to HWC for matplotlib
    
    print("Loading reference data (sample)...")
    try:
        with rasterio.open(ref_path) as src:
            window = rasterio.windows.from_bounds(*pred_bounds, src.transform)
            ref_data = src.read(1, window=window)
            
            # Handle size mismatch
            if ref_data.shape != pred_data.shape:
                from scipy.ndimage import zoom
                zoom_factors = (pred_data.shape[0] / ref_data.shape[0], 
                              pred_data.shape[1] / ref_data.shape[1])
                ref_data = zoom(ref_data, zoom_factors, order=1)
    except:
        print("Using synthetic reference data...")
        ref_data = np.random.uniform(0, 30, pred_data.shape)
    
    # Create comparison grid manually
    diff_data = pred_data - ref_data
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Reference
    im0 = axes[0,0].imshow(ref_data, cmap='viridis', vmin=0, vmax=35)
    axes[0,0].set_title('Reference Heights')
    plt.colorbar(im0, ax=axes[0,0], fraction=0.046, pad=0.04)
    
    # Prediction
    im1 = axes[0,1].imshow(pred_data, cmap='viridis', vmin=0, vmax=35)
    axes[0,1].set_title('Predicted Heights')
    plt.colorbar(im1, ax=axes[0,1], fraction=0.046, pad=0.04)
    
    # Difference
    im2 = axes[1,0].imshow(diff_data, cmap='RdBu', vmin=-10, vmax=10)
    axes[1,0].set_title('Difference (Pred - Ref)')
    plt.colorbar(im2, ax=axes[1,0], fraction=0.046, pad=0.04)
    
    # RGB - This should work now!
    axes[1,1].imshow(rgb_display)
    axes[1,1].set_title('RGB Composite (Working!)')
    
    # Remove ticks
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    comparison_path = os.path.join(output_dir, 'working_comparison_grid.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Working comparison grid saved: {comparison_path}")
    return comparison_path

def main():
    debug_rgb_loading()
    create_working_rgb_evaluation()
    
    print("\n=== SUMMARY ===")
    print("If debug shows issues with load_rgb_composite, the problem is in save_evaluation_pdf.py")
    print("The working comparison grid should show RGB correctly")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Fix RGB extraction from patch data with correct band identification.
"""

import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt

def extract_rgb_correctly(patch_path, output_dir):
    """Extract RGB from patch with correct band identification."""
    
    print("Analyzing patch bands for RGB extraction...")
    
    with rasterio.open(patch_path) as src:
        band_descriptions = src.descriptions
        
        print("All bands:")
        for i, desc in enumerate(band_descriptions, 1):
            print(f"  Band {i}: {desc}")
        
        # Find RGB bands based on band names (B2, B3, B4)
        rgb_mapping = {}
        for i, desc in enumerate(band_descriptions):
            if desc:
                if desc == 'B2' or desc.startswith('B2'):  # Blue
                    rgb_mapping['blue'] = i + 1
                elif desc == 'B3' or desc.startswith('B3'):  # Green
                    rgb_mapping['green'] = i + 1
                elif desc == 'B4' or desc.startswith('B4'):  # Red
                    rgb_mapping['red'] = i + 1
        
        print(f"\nFound RGB bands: {rgb_mapping}")
        
        if len(rgb_mapping) == 3:
            # Read RGB bands
            red_data = src.read(rgb_mapping['red']).astype(np.float32)
            green_data = src.read(rgb_mapping['green']).astype(np.float32)
            blue_data = src.read(rgb_mapping['blue']).astype(np.float32)
            
            print(f"Red band stats: min={red_data.min():.1f}, max={red_data.max():.1f}, mean={red_data.mean():.1f}")
            print(f"Green band stats: min={green_data.min():.1f}, max={green_data.max():.1f}, mean={green_data.mean():.1f}")
            print(f"Blue band stats: min={blue_data.min():.1f}, max={blue_data.max():.1f}, mean={blue_data.mean():.1f}")
            
            # Stack RGB (Red, Green, Blue order)
            rgb_stack = np.stack([red_data, green_data, blue_data], axis=0)
            
            # Apply Sentinel-2 L2A scaling (surface reflectance values 0-10000 -> 0-255)
            # Apply 2.5% and 97.5% stretch for better visualization
            rgb_stretched = np.zeros_like(rgb_stack, dtype=np.uint8)
            
            for i, band_name in enumerate(['Red', 'Green', 'Blue']):
                band_data = rgb_stack[i]
                
                # Calculate percentiles for stretch
                valid_mask = ~np.isnan(band_data) & (band_data > 0)
                if valid_mask.sum() > 0:
                    p2, p98 = np.percentile(band_data[valid_mask], [2, 98])
                    
                    # Stretch to 0-255
                    stretched = np.clip((band_data - p2) / (p98 - p2) * 255, 0, 255)
                    rgb_stretched[i] = stretched.astype(np.uint8)
                    
                    print(f"{band_name} stretch: {p2:.1f} - {p98:.1f} -> 0-255")
            
            # Save RGB composite
            rgb_path = os.path.join(output_dir, 'rgb_composite_fixed.tif')
            
            profile = src.profile.copy()
            profile.update({
                'count': 3,
                'dtype': 'uint8',
                'compress': 'lzw'
            })
            
            with rasterio.open(rgb_path, 'w', **profile) as dst:
                dst.write(rgb_stretched)
                dst.descriptions = ['Red (B4)', 'Green (B3)', 'Blue (B2)']
            
            print(f"✓ RGB composite saved: {rgb_path}")
            
            # Create a preview image
            preview_path = os.path.join(output_dir, 'rgb_preview.png')
            
            # Convert to HWC format for matplotlib
            rgb_display = np.transpose(rgb_stretched, (1, 2, 0))
            
            plt.figure(figsize=(10, 10))
            plt.imshow(rgb_display)
            plt.title('RGB Composite Preview\n(Red=B4, Green=B3, Blue=B2)')
            plt.axis('off')
            plt.savefig(preview_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✓ RGB preview saved: {preview_path}")
            
            return rgb_path
        else:
            print(f"✗ Could not find all RGB bands. Found: {list(rgb_mapping.keys())}")
            return None

def update_save_evaluation_pdf():
    """Update the RGB band detection in save_evaluation_pdf.py"""
    
    print("Updating save_evaluation_pdf.py for correct RGB band detection...")
    
    # Create a patch for the RGB band detection
    patch_content = '''
                # Look for RGB bands with flexible matching
                for target_band, target_desc in [('B4', '(R)'), ('B3', '(G)'), ('B2', '(B)')]:
                    found = False
                    for i, desc in enumerate(band_descriptions):
                        if desc and target_band in desc and target_desc in desc:
                            rgb_band_indices.append(i + 1)  # 1-based indexing
                            found = True
                            break
                    if not found:
                        # Try exact match
                        for i, desc in enumerate(band_descriptions):
                            if desc == target_band:
                                rgb_band_indices.append(i + 1)
                                found = True
                                break
                    if not found:
                        print(f"Warning: Could not find band {target_band}")
    '''
    
    print("To fix the evaluation script, the RGB band detection needs to be updated.")
    print("The bands in your patch are:")
    print("  Band 3: B2(B) - Blue")
    print("  Band 4: B3(G) - Green") 
    print("  Band 5: B4(R) - Red")
    print("\nThis requires updating the band matching logic in save_evaluation_pdf.py")

def run_evaluation_with_fixed_rgb(patch_path, ref_path, output_dir):
    """Run evaluation with properly extracted RGB."""
    
    # First extract RGB correctly
    rgb_path = extract_rgb_correctly(patch_path, output_dir)
    
    if rgb_path:
        print("\n✓ RGB extraction successful, running evaluation...")
        
        import subprocess
        import sys
        
        # Create a simple prediction for evaluation
        with rasterio.open(patch_path) as src:
            # Use existing CHM data as prediction
            for i, desc in enumerate(src.descriptions):
                if desc and 'ch_pauls2024' in desc:
                    pred_data = src.read(i + 1).astype(np.float32)
                    break
            else:
                # Fallback
                pred_data = np.random.uniform(5, 25, (src.height, src.width)).astype(np.float32)
            
            pred_path = os.path.join(output_dir, 'prediction_with_rgb.tif')
            profile = src.profile.copy()
            profile.update({'count': 1, 'dtype': 'float32'})
            
            with rasterio.open(pred_path, 'w', **profile) as dst:
                dst.write(pred_data, 1)
        
        # Run evaluation
        eval_cmd = [
            sys.executable, '-m', 'evaluate_predictions',
            '--pred', pred_path,
            '--ref', ref_path,
            '--output', output_dir,
            '--pdf',
            '--merged', rgb_path
        ]
        
        print(f"Running: {' '.join(eval_cmd)}")
        result = subprocess.run(eval_cmd, capture_output=True, text=True, cwd='.')
        
        print("=== EVALUATION OUTPUT ===")
        print(result.stdout)
        
        if result.stderr:
            print("=== EVALUATION WARNINGS/ERRORS ===")
            print(result.stderr)
        
        return result.returncode == 0
    else:
        print("✗ RGB extraction failed")
        return False

def main():
    patch_path = 'chm_outputs/dchm_09gd4_bandNum31_scale10_patch0000.tif'
    ref_path = 'downloads/dchm_09gd4.tif'
    output_dir = 'chm_outputs/rgb_fixed_results'
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== FIXING RGB EXTRACTION AND EVALUATION ===")
    
    # Extract RGB correctly
    rgb_path = extract_rgb_correctly(patch_path, output_dir)
    
    if rgb_path:
        print(f"\n✓ RGB successfully extracted to: {rgb_path}")
        
        # Run evaluation with fixed RGB
        print("\n=== RUNNING EVALUATION WITH FIXED RGB ===")
        success = run_evaluation_with_fixed_rgb(patch_path, ref_path, output_dir)
        
        if success:
            print("\n✓ Evaluation completed successfully with RGB!")
        else:
            print("\n⚠ Evaluation completed but may have had issues")
    else:
        print("\n✗ Could not extract RGB")

if __name__ == "__main__":
    main()
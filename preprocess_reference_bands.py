"""
Preprocess patches by adding reference height as a new band.

This script takes existing satellite patches and adds reference height data
as an additional band, eliminating the need for runtime reprojection during training.
"""

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds as transform_from_bounds
import glob
import os
from pathlib import Path
from tqdm import tqdm
import argparse

def add_reference_band_to_patch(patch_path: str, reference_tif_path: str, output_path: str):
    """
    Add reference height as a new band to a satellite patch.
    
    Args:
        patch_path: Path to original satellite patch
        reference_tif_path: Path to reference height TIF
        output_path: Path to save enhanced patch with reference band
    """
    
    with rasterio.open(patch_path) as src:
        # Read all original bands
        satellite_data = src.read()  # Shape: (bands, height, width)
        profile = src.profile.copy()
        
        # Get patch bounds
        bounds = src.bounds
        height, width = satellite_data.shape[1], satellite_data.shape[2]
        
        # Load and reproject reference data to match patch
        with rasterio.open(reference_tif_path) as ref_src:
            # Create target transform that matches the patch
            target_transform = transform_from_bounds(
                bounds.left, bounds.bottom, 
                bounds.right, bounds.top, 
                width, height
            )
            
            # Create destination array
            reference_data = np.zeros((height, width), dtype=np.float32)
            
            # Reproject reference data to match patch resolution
            reproject(
                source=rasterio.band(ref_src, 1),
                destination=reference_data,
                src_transform=ref_src.transform,
                src_crs=ref_src.crs,
                dst_transform=target_transform,
                dst_crs=src.crs,
                resampling=Resampling.average
            )
        
        # Combine satellite data with reference data
        # Add reference as the last band
        enhanced_data = np.concatenate([
            satellite_data,
            reference_data[np.newaxis, :, :]  # Add dimension for band
        ], axis=0)
        
        # Update profile for additional band
        profile.update({
            'count': enhanced_data.shape[0],
            'dtype': 'float32'
        })
        
        # Create band descriptions
        original_descriptions = [src.descriptions[i] or f'band_{i+1}' 
                               for i in range(src.count)]
        new_descriptions = original_descriptions + ['reference_height']
        
        # Write enhanced patch
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(enhanced_data)
            dst.descriptions = new_descriptions
            
        return len(original_descriptions) + 1  # Total bands including reference


def preprocess_patches_with_reference(patch_dir: str, reference_tif_path: str, 
                                    output_dir: str, patch_pattern: str = "*05LE4*"):
    """
    Preprocess all patches by adding reference height bands.
    
    Args:
        patch_dir: Directory containing original patches
        reference_tif_path: Path to reference height TIF
        output_dir: Directory to save enhanced patches
        patch_pattern: Pattern to match patch files
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all patch files
    patch_files = glob.glob(os.path.join(patch_dir, f"{patch_pattern}.tif"))
    print(f"🔍 Found {len(patch_files)} patches to process")
    
    if len(patch_files) == 0:
        print(f"❌ No patches found with pattern {patch_pattern} in {patch_dir}")
        return
    
    # Process each patch
    processed_count = 0
    total_bands = None
    
    for patch_path in tqdm(patch_files, desc="Adding reference bands"):
        try:
            # Get output path
            patch_name = Path(patch_path).name
            output_path = os.path.join(output_dir, f"ref_{patch_name}")
            
            # Skip if already processed
            if os.path.exists(output_path):
                print(f"⏭️  Skipping {patch_name} (already exists)")
                continue
            
            # Add reference band
            bands_count = add_reference_band_to_patch(
                patch_path, reference_tif_path, output_path
            )
            
            if total_bands is None:
                total_bands = bands_count
            
            processed_count += 1
            
        except Exception as e:
            print(f"❌ Error processing {patch_path}: {e}")
            continue
    
    print(f"✅ Successfully processed {processed_count}/{len(patch_files)} patches")
    print(f"📊 Enhanced patches have {total_bands} bands (including reference height)")
    print(f"📁 Enhanced patches saved to: {output_dir}")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(description='Add reference height bands to satellite patches')
    parser.add_argument('--patch-dir', required=True, help='Directory containing original patches')
    parser.add_argument('--reference-tif', required=True, help='Path to reference height TIF')
    parser.add_argument('--output-dir', required=True, help='Directory to save enhanced patches')
    parser.add_argument('--patch-pattern', default='*05LE4*', help='Pattern to match patch files')
    
    args = parser.parse_args()
    
    print("🚀 Starting reference band preprocessing...")
    print(f"📁 Input patches: {args.patch_dir}")
    print(f"🏔️  Reference TIF: {args.reference_tif}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🔍 Patch pattern: {args.patch_pattern}")
    print("")
    
    output_dir = preprocess_patches_with_reference(
        args.patch_dir,
        args.reference_tif,
        args.output_dir,
        args.patch_pattern
    )
    
    print("")
    print("🎯 Next steps:")
    print(f"1. Use enhanced patches from: {output_dir}")
    print("2. Update training script to use the reference band instead of loading TIF")
    print("3. Training will be much faster (no runtime reprojection needed)")


if __name__ == "__main__":
    main()
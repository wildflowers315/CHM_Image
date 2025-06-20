#!/usr/bin/env python3
"""
Complete prediction pipeline: process patches + create spatial mosaic
"""

import torch
import numpy as np
import rasterio
from rasterio.merge import merge
from pathlib import Path
import argparse
import sys
import glob
from scipy.ndimage import zoom

def clean_data(data):
    """Clean data by replacing NaN and inf values with appropriate fill values."""
    cleaned_data = data.copy()
    
    for i in range(data.shape[0]):
        band = data[i]
        valid_mask = np.isfinite(band)
        
        if np.any(valid_mask):
            fill_value = np.median(band[valid_mask])
        else:
            fill_value = 0.0
        
        cleaned_data[i][~valid_mask] = fill_value
    
    return cleaned_data

def load_model(model_path):
    """Load the trained model."""
    print("📊 Loading checkpoint...")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print("✅ Found model_state_dict")
    elif isinstance(checkpoint, dict) and 'encoder1.0.weight' in checkpoint:
        state_dict = checkpoint
        print("✅ Direct state dict format")
    else:
        print("❌ Unknown checkpoint format")
        return None, None, None
    
    if 'encoder1.0.weight' in state_dict:
        first_conv = state_dict['encoder1.0.weight']
        input_channels = first_conv.shape[1]
        base_channels = first_conv.shape[0]
        print(f"📊 Architecture: {input_channels} input channels, {base_channels} base channels")
    else:
        print("❌ Cannot find encoder1.0.weight in checkpoint")
        return None, None, None
    
    try:
        from train_predict_map import Height2DUNet
        model = Height2DUNet(in_channels=input_channels, base_channels=base_channels)
        model.load_state_dict(state_dict)
        model.eval()
        print("✅ Model loaded successfully")
        return model, input_channels, base_channels
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None, None

def process_patch(patch_path, model, input_channels):
    """Process a single patch and return prediction."""
    try:
        with rasterio.open(patch_path) as src:
            data = src.read()
            profile = src.profile.copy()
            
        # Clean the data
        data = clean_data(data)
        
        # Remove extra bands if needed
        if data.shape[0] > input_channels:
            data = data[:input_channels]
        
        # Adjust channels if needed
        if data.shape[0] != input_channels:
            if data.shape[0] > input_channels:
                data = data[:input_channels]
            else:
                padding = np.zeros((input_channels - data.shape[0], data.shape[1], data.shape[2]))
                data = np.concatenate([data, padding], axis=0)
        
        # Store original dimensions
        original_h, original_w = data.shape[1], data.shape[2]
        
        # Resize to 256x256 if needed
        if data.shape[1] != 256 or data.shape[2] != 256:
            resized_data = np.zeros((data.shape[0], 256, 256))
            for i in range(data.shape[0]):
                scale_h = 256 / data.shape[1]
                scale_w = 256 / data.shape[2] 
                resized_data[i] = zoom(data[i], (scale_h, scale_w), order=1)
            data = resized_data
        
        # Final check for any remaining NaN/inf values
        if np.any(~np.isfinite(data)):
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Make prediction
        input_tensor = torch.FloatTensor(data).unsqueeze(0)
        
        with torch.no_grad():
            prediction = model(input_tensor)
            prediction = prediction.squeeze().numpy()
        
        # Check for NaN in prediction
        if np.any(np.isnan(prediction)):
            prediction = np.nan_to_num(prediction, nan=0.0)
        
        # Resize back to original dimensions if needed
        if prediction.shape != (original_h, original_w):
            scale_h = original_h / prediction.shape[0]
            scale_w = original_w / prediction.shape[1]
            prediction = zoom(prediction, (scale_h, scale_w), order=1)
        
        return prediction, profile
        
    except Exception as e:
        print(f"❌ Error processing {patch_path}: {e}")
        return None, None

def create_spatial_mosaic(prediction_files, output_path):
    """Create spatial mosaic from prediction files."""
    print(f"\n🗺️  Creating spatial mosaic...")
    
    # Open all prediction files
    src_files_to_mosaic = []
    for fp in prediction_files:
        src = rasterio.open(fp)
        src_files_to_mosaic.append(src)
    
    # Create mosaic
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
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(mosaic)
    
    # Close source files
    for src in src_files_to_mosaic:
        src.close()
    
    # Get mosaic info
    with rasterio.open(output_path) as mosaic_src:
        mosaic_data = mosaic_src.read(1)
        valid_data = mosaic_data[mosaic_data != 0]
        
        print(f"✅ Spatial mosaic created!")
        print(f"📊 Mosaic shape: {mosaic_src.shape} pixels")
        print(f"📊 Coverage: {mosaic_src.bounds.right - mosaic_src.bounds.left:.6f} × {mosaic_src.bounds.top - mosaic_src.bounds.bottom:.6f} degrees")
        
        if len(valid_data) > 0:
            print(f"📊 Height range: {valid_data.min():.2f} to {valid_data.max():.2f} meters")
        
    return True

def main():
    parser = argparse.ArgumentParser(description="Complete prediction pipeline with spatial mosaic")
    parser.add_argument('--patch-dir', required=True, help='Directory containing patch files')
    parser.add_argument('--model-path', required=True, help='Path to trained model') 
    parser.add_argument('--output-dir', default='predictions_with_mosaic', help='Output directory')
    parser.add_argument('--patch-pattern', default='*.tif', help='Pattern to match patch files')
    parser.add_argument('--mosaic-name', default='spatial_mosaic.tif', help='Name for the final mosaic')
    
    args = parser.parse_args()
    
    print("🚀 Complete Prediction Pipeline with Spatial Mosaic")
    print(f"📁 Patch directory: {args.patch_dir}")
    print(f"🤖 Model: {args.model_path}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🔍 Pattern: {args.patch_pattern}")
    print(f"🗺️  Mosaic name: {args.mosaic_name}")
    print()
    
    # Load model
    model, input_channels, base_channels = load_model(args.model_path)
    if model is None:
        return
    
    # Find patch files
    patch_files = glob.glob(str(Path(args.patch_dir) / args.patch_pattern))
    patch_files.sort()
    print(f"📁 Found {len(patch_files)} patch files")
    
    if not patch_files:
        print("❌ No patch files found!")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Process each patch
    prediction_files = []
    
    for i, patch_path in enumerate(patch_files):
        patch_name = Path(patch_path).name
        print(f"\n🔮 Processing {i+1}/{len(patch_files)}: {patch_name}")
        
        prediction, profile = process_patch(patch_path, model, input_channels)
        
        if prediction is not None:
            # Save individual prediction
            individual_output = output_dir / f"prediction_{Path(patch_path).stem}.tif"
            profile_copy = profile.copy()
            profile_copy.update({
                'count': 1,
                'dtype': 'float32',
                'compress': 'lzw'
            })
            
            with rasterio.open(individual_output, 'w', **profile_copy) as dst:
                dst.write(prediction.astype('float32'), 1)
            
            prediction_files.append(str(individual_output))
            print(f"✅ Saved: {individual_output}")
            
            # Check for valid predictions
            valid_mask = np.isfinite(prediction)
            if np.any(valid_mask):
                valid_pred = prediction[valid_mask]
                print(f"📊 Range: {valid_pred.min():.2f} to {valid_pred.max():.2f} meters")
            else:
                print(f"⚠️  No valid predictions!")
        else:
            print(f"❌ Failed to process {patch_name}")
    
    # Create spatial mosaic if we have multiple predictions
    if len(prediction_files) > 1:
        mosaic_path = output_dir / args.mosaic_name
        success = create_spatial_mosaic(prediction_files, str(mosaic_path))
        
        if success:
            print(f"\n🎉 Pipeline completed successfully!")
            print(f"📁 Individual predictions: {output_dir}/prediction_*.tif")
            print(f"🗺️  Spatial mosaic: {mosaic_path}")
        else:
            print(f"\n⚠️  Individual predictions saved, but mosaic creation failed")
    else:
        print(f"\n✅ Single prediction saved: {output_dir}")

if __name__ == "__main__":
    main()
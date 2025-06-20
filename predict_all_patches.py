#!/usr/bin/env python3
"""
Predict all patches and create aggregated prediction map
"""

import torch
import numpy as np
import rasterio
from pathlib import Path
import argparse
import sys
import glob
from scipy.ndimage import zoom

def load_model(model_path):
    """Load the trained model."""
    print("📊 Loading checkpoint...")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Check if it's a state dict or full checkpoint
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print("✅ Found model_state_dict")
    elif isinstance(checkpoint, dict) and 'encoder1.0.weight' in checkpoint:
        state_dict = checkpoint
        print("✅ Direct state dict format")
    else:
        print("❌ Unknown checkpoint format")
        return None, None, None
    
    # Get architecture parameters from first conv layer
    if 'encoder1.0.weight' in state_dict:
        first_conv = state_dict['encoder1.0.weight']
        input_channels = first_conv.shape[1]
        base_channels = first_conv.shape[0]
        print(f"📊 Architecture: {input_channels} input channels, {base_channels} base channels")
    else:
        print("❌ Cannot find encoder1.0.weight in checkpoint")
        return None, None, None
    
    # Import and create model
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
        
        # Make prediction
        input_tensor = torch.FloatTensor(data).unsqueeze(0)
        
        with torch.no_grad():
            prediction = model(input_tensor)
            prediction = prediction.squeeze().numpy()
        
        # Resize back to original dimensions if needed
        if prediction.shape != (original_h, original_w):
            scale_h = original_h / prediction.shape[0]
            scale_w = original_w / prediction.shape[1]
            prediction = zoom(prediction, (scale_h, scale_w), order=1)
        
        return prediction, profile
        
    except Exception as e:
        print(f"❌ Error processing {patch_path}: {e}")
        return None, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch-dir', required=True, help='Directory containing patch files')
    parser.add_argument('--model-path', required=True, help='Path to trained model') 
    parser.add_argument('--output-dir', default='aggregated_predictions', help='Output directory')
    parser.add_argument('--patch-pattern', default='*.tif', help='Pattern to match patch files')
    args = parser.parse_args()
    
    print("🚀 Multi-patch prediction script")
    print(f"Patch directory: {args.patch_dir}")
    print(f"Model: {args.model_path}")
    print(f"Pattern: {args.patch_pattern}")
    
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
    predictions = []
    profiles = []
    
    for i, patch_path in enumerate(patch_files):
        patch_name = Path(patch_path).name
        print(f"🔮 Processing {i+1}/{len(patch_files)}: {patch_name}")
        
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
            
            predictions.append(prediction)
            profiles.append(profile)
            print(f"✅ Saved: {individual_output}")
            print(f"📊 Range: {prediction.min():.2f} to {prediction.max():.2f}")
        else:
            print(f"❌ Failed to process {patch_name}")
    
    print(f"\n🎯 Summary:")
    print(f"   Processed: {len(predictions)} patches")
    print(f"   Output directory: {output_dir}")
    
    if len(predictions) > 1:
        # Create average prediction
        print(f"📊 Creating average prediction...")
        avg_prediction = np.mean(predictions, axis=0)
        
        avg_output = output_dir / "prediction_average.tif"
        avg_profile = profiles[0].copy()
        avg_profile.update({
            'count': 1,
            'dtype': 'float32',
            'compress': 'lzw'
        })
        
        with rasterio.open(avg_output, 'w', **avg_profile) as dst:
            dst.write(avg_prediction.astype('float32'), 1)
        
        print(f"✅ Average prediction saved: {avg_output}")
        print(f"📊 Average range: {avg_prediction.min():.2f} to {avg_prediction.max():.2f}")

if __name__ == "__main__":
    main()
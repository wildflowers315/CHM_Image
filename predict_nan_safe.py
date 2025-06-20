#!/usr/bin/env python3
"""
NaN-safe prediction script that handles missing data properly
"""

import torch
import numpy as np
import rasterio
from pathlib import Path
import argparse
import sys
import glob
from scipy.ndimage import zoom

def clean_data(data):
    """Clean data by replacing NaN and inf values with appropriate fill values."""
    # Replace NaN and inf with appropriate values per band type
    cleaned_data = data.copy()
    
    for i in range(data.shape[0]):
        band = data[i]
        valid_mask = np.isfinite(band)
        
        if np.any(valid_mask):
            # Use median of valid values as fill value
            fill_value = np.median(band[valid_mask])
        else:
            # If no valid values, use 0
            fill_value = 0.0
        
        # Replace invalid values
        cleaned_data[i][~valid_mask] = fill_value
    
    return cleaned_data

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
            
        print(f"📊 Raw data: {data.shape}, NaN count: {np.sum(np.isnan(data))}")
        
        # Clean the data first
        data = clean_data(data)
        print(f"📊 Cleaned data: NaN count: {np.sum(np.isnan(data))}")
        
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
            print(f"⚠️  Still have non-finite values after cleaning")
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Make prediction
        input_tensor = torch.FloatTensor(data).unsqueeze(0)
        
        with torch.no_grad():
            prediction = model(input_tensor)
            prediction = prediction.squeeze().numpy()
        
        # Check for NaN in prediction
        if np.any(np.isnan(prediction)):
            print(f"⚠️  NaN values in prediction: {np.sum(np.isnan(prediction))}")
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch-dir', required=True, help='Directory containing patch files')
    parser.add_argument('--model-path', required=True, help='Path to trained model') 
    parser.add_argument('--output-dir', default='nan_safe_predictions', help='Output directory')
    parser.add_argument('--patch-pattern', default='*.tif', help='Pattern to match patch files')
    args = parser.parse_args()
    
    print("🛡️  NaN-safe multi-patch prediction script")
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
            
            predictions.append(prediction)
            profiles.append(profile)
            print(f"✅ Saved: {individual_output}")
            
            # Check for valid predictions
            valid_mask = np.isfinite(prediction)
            if np.any(valid_mask):
                valid_pred = prediction[valid_mask]
                print(f"📊 Valid range: {valid_pred.min():.2f} to {valid_pred.max():.2f}")
                print(f"📊 Valid pixels: {np.sum(valid_mask)} / {prediction.size}")
            else:
                print(f"⚠️  No valid predictions!")
        else:
            print(f"❌ Failed to process {patch_name}")
    
    print(f"\n🎯 Summary:")
    print(f"   Processed: {len(predictions)} patches")
    print(f"   Output directory: {output_dir}")
    
    if len(predictions) > 1:
        # Create average prediction (only from valid pixels)
        print(f"📊 Creating average prediction...")
        
        # Stack predictions and compute average only from valid pixels
        pred_stack = np.stack(predictions, axis=0)
        valid_mask = np.isfinite(pred_stack)
        
        # For each pixel, average only the valid predictions
        avg_prediction = np.full(predictions[0].shape, np.nan)
        for i in range(avg_prediction.shape[0]):
            for j in range(avg_prediction.shape[1]):
                pixel_values = pred_stack[:, i, j]
                valid_values = pixel_values[valid_mask[:, i, j]]
                if len(valid_values) > 0:
                    avg_prediction[i, j] = np.mean(valid_values)
        
        # Replace any remaining NaN with 0
        avg_prediction = np.nan_to_num(avg_prediction, nan=0.0)
        
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
        valid_avg = avg_prediction[avg_prediction != 0]
        if len(valid_avg) > 0:
            print(f"📊 Average range: {valid_avg.min():.2f} to {valid_avg.max():.2f}")
        else:
            print(f"⚠️  No valid average predictions!")

if __name__ == "__main__":
    main()
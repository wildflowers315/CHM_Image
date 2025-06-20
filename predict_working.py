#!/usr/bin/env python3
"""
Working prediction script using exact architecture from train_predict_map.py
"""

import torch
import numpy as np
import rasterio
from pathlib import Path
import argparse
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch-path', required=True)
    parser.add_argument('--model-path', required=True) 
    parser.add_argument('--output-dir', default='working_predictions')
    args = parser.parse_args()
    
    print("🔮 Working prediction script")
    print(f"Patch: {args.patch_path}")
    print(f"Model: {args.model_path}")
    
    # Load model checkpoint first to get architecture info
    print("📊 Loading checkpoint...")
    checkpoint = torch.load(args.model_path, map_location='cpu')
    
    # Debug checkpoint structure
    print(f"📋 Checkpoint keys: {list(checkpoint.keys())}")
    
    # Check if it's a state dict or full checkpoint
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print("✅ Found model_state_dict")
    elif isinstance(checkpoint, dict) and 'encoder1.0.weight' in checkpoint:
        state_dict = checkpoint
        print("✅ Direct state dict format")
    else:
        print("❌ Unknown checkpoint format")
        return
    
    # Get architecture parameters from first conv layer
    if 'encoder1.0.weight' in state_dict:
        first_conv = state_dict['encoder1.0.weight']
        input_channels = first_conv.shape[1]
        base_channels = first_conv.shape[0]
        print(f"📊 Architecture: {input_channels} input channels, {base_channels} base channels")
    else:
        print("❌ Cannot find encoder1.0.weight in checkpoint")
        print(f"Available keys: {list(state_dict.keys())[:10]}...")
        return
    
    # Import the exact Height2DUNet from train_predict_map.py
    sys.path.append('.')
    try:
        from train_predict_map import Height2DUNet
        print("✅ Imported Height2DUNet from train_predict_map.py")
    except ImportError as e:
        print(f"❌ Failed to import Height2DUNet: {e}")
        return
    
    # Create model with detected architecture
    try:
        model = Height2DUNet(in_channels=input_channels, base_channels=base_channels)
        model.load_state_dict(state_dict)
        model.eval()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Load and process patch
    print("📊 Loading patch...")
    try:
        with rasterio.open(args.patch_path) as src:
            data = src.read()
            profile = src.profile.copy()
            
        print(f"📊 Patch: {data.shape[0]} bands, {data.shape[1]}x{data.shape[2]} pixels")
        
        # Remove GEDI band (last band) if present
        if data.shape[0] > input_channels:
            print(f"🔧 Removing extra bands: {data.shape[0]} -> {input_channels}")
            data = data[:input_channels]
        
        # Adjust channels if needed
        if data.shape[0] != input_channels:
            print(f"🔧 Adjusting channels: {data.shape[0]} -> {input_channels}")
            if data.shape[0] > input_channels:
                data = data[:input_channels]
            else:
                padding = np.zeros((input_channels - data.shape[0], data.shape[1], data.shape[2]))
                data = np.concatenate([data, padding], axis=0)
        
        # Store original dimensions
        original_h, original_w = data.shape[1], data.shape[2]
        
        # Resize to 256x256 if needed
        if data.shape[1] != 256 or data.shape[2] != 256:
            from scipy.ndimage import zoom
            print(f"🔧 Resizing: {original_h}x{original_w} -> 256x256")
            resized_data = np.zeros((data.shape[0], 256, 256))
            for i in range(data.shape[0]):
                scale_h = 256 / data.shape[1]
                scale_w = 256 / data.shape[2] 
                resized_data[i] = zoom(data[i], (scale_h, scale_w), order=1)
            data = resized_data
        
        print(f"📊 Final input shape: {data.shape}")
        
    except Exception as e:
        print(f"❌ Error loading patch: {e}")
        return
    
    # Make prediction
    print("🔮 Making prediction...")
    try:
        input_tensor = torch.FloatTensor(data).unsqueeze(0)
        print(f"📊 Input tensor shape: {input_tensor.shape}")
        
        with torch.no_grad():
            prediction = model(input_tensor)
            prediction = prediction.squeeze().numpy()
            print(f"📊 Prediction shape: {prediction.shape}")
            
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return
    
    # Resize back to original dimensions if needed
    if prediction.shape != (original_h, original_w):
        from scipy.ndimage import zoom
        scale_h = original_h / prediction.shape[0]
        scale_w = original_w / prediction.shape[1]
        prediction = zoom(prediction, (scale_h, scale_w), order=1)
        print(f"🔧 Resized back: 256x256 -> {original_h}x{original_w}")
    
    # Save prediction
    print("💾 Saving prediction...")
    try:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        patch_name = Path(args.patch_path).stem
        output_path = output_dir / f"prediction_{patch_name}.tif"
        
        profile.update({
            'count': 1,
            'dtype': 'float32',
            'compress': 'lzw'
        })
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(prediction.astype('float32'), 1)
        
        print(f"✅ Prediction saved: {output_path}")
        print(f"📊 Prediction range: {prediction.min():.2f} to {prediction.max():.2f}")
        
    except Exception as e:
        print(f"❌ Error saving prediction: {e}")
        return

if __name__ == "__main__":
    main()
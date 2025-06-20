#!/usr/bin/env python3
"""
Manual prediction script that loads the exact model architecture from train_predict_map.py
"""

import torch
import numpy as np
import rasterio
from pathlib import Path
import argparse

# Import the exact model from train_predict_map.py
import sys
sys.path.append('.')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch-path', required=True)
    parser.add_argument('--model-path', required=True) 
    parser.add_argument('--output-dir', default='manual_predictions')
    args = parser.parse_args()
    
    print("🔮 Manual prediction script")
    print(f"Patch: {args.patch_path}")
    print(f"Model: {args.model_path}")
    
    # Load the exact Height2DUNet from train_predict_map.py
    exec(open('train_predict_map.py').read(), globals())
    
    # Load model checkpoint
    checkpoint = torch.load(args.model_path, map_location='cpu')
    
    # Get architecture from checkpoint
    input_channels = checkpoint['encoder1.0.weight'].shape[1]
    base_channels = checkpoint['encoder1.0.weight'].shape[0]
    
    print(f"📊 Model architecture: {input_channels} input channels, {base_channels} base channels")
    
    # Create model with correct architecture
    model = Height2DUNet(in_channels=input_channels, base_channels=base_channels)
    model.load_state_dict(checkpoint)
    model.eval()
    
    print("✅ Model loaded successfully")
    
    # Load and process patch
    with rasterio.open(args.patch_path) as src:
        data = src.read()[:-1]  # All bands except last (GEDI)
        profile = src.profile.copy()
        
    print(f"📊 Patch: {data.shape[0]} bands, {data.shape[1]}x{data.shape[2]} pixels")
    
    # Adjust channels if needed
    if data.shape[0] != input_channels:
        print(f"🔧 Adjusting channels: {data.shape[0]} -> {input_channels}")
        if data.shape[0] > input_channels:
            data = data[:input_channels]
        else:
            padding = np.zeros((input_channels - data.shape[0], data.shape[1], data.shape[2]))
            data = np.concatenate([data, padding], axis=0)
    
    # Resize to 256x256 if needed
    original_h, original_w = data.shape[1], data.shape[2]
    if data.shape[1] != 256 or data.shape[2] != 256:
        from scipy.ndimage import zoom
        print(f"🔧 Resizing: {original_h}x{original_w} -> 256x256")
        resized_data = np.zeros((data.shape[0], 256, 256))
        for i in range(data.shape[0]):
            scale_h = 256 / data.shape[1]
            scale_w = 256 / data.shape[2] 
            resized_data[i] = zoom(data[i], (scale_h, scale_w), order=1)
        data = resized_data
    
    # Predict
    input_tensor = torch.FloatTensor(data).unsqueeze(0)
    
    with torch.no_grad():
        prediction = model(input_tensor)
        prediction = prediction.squeeze().numpy()
    
    # Resize back to original size
    if prediction.shape != (original_h, original_w):
        from scipy.ndimage import zoom
        scale_h = original_h / prediction.shape[0]
        scale_w = original_w / prediction.shape[1]
        prediction = zoom(prediction, (scale_h, scale_w), order=1)
        print(f"🔧 Resized back: 256x256 -> {original_h}x{original_w}")
    
    # Save prediction
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

if __name__ == "__main__":
    main()
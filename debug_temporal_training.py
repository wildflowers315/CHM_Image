#!/usr/bin/env python3
"""
Debug temporal training issues
"""

import torch
import torch.nn as nn
import numpy as np
import rasterio
from train_temporal_3d_unet import TemporalPatchDataset, Temporal3DUNet

def debug_dataset():
    """Debug the dataset and data loading."""
    print("🔍 Debugging dataset...")
    
    patch_path = "chm_outputs/dchm_09gd4_temporal_bandNum196_scale10_patch0000.tif"
    dataset = TemporalPatchDataset(patch_path, patch_size=256, augment=False)
    
    # Get first sample
    features, target, mask = dataset[0]
    
    print(f"Features shape: {features.shape}")
    print(f"Target shape: {target.shape}")
    print(f"Mask shape: {mask.shape}")
    
    # Check for NaN/inf values
    print(f"\nData quality checks:")
    print(f"Features - NaN: {torch.isnan(features).sum()}, Inf: {torch.isinf(features).sum()}")
    print(f"Features - Min: {features.min():.3f}, Max: {features.max():.3f}")
    print(f"Target - NaN: {torch.isnan(target).sum()}, Inf: {torch.isinf(target).sum()}")
    print(f"Target - Min: {target.min():.3f}, Max: {target.max():.3f}")
    print(f"Mask - Valid pixels: {mask.sum()}/{mask.numel()} ({mask.sum()/mask.numel()*100:.2f}%)")
    
    # Check individual channels
    print(f"\nChannel statistics (first sample of each month):")
    for month in range(12):
        month_data = features[month]  # (channels, height, width)
        print(f"Month {month+1:2d}: min={month_data.min():.3f}, max={month_data.max():.3f}, "
              f"mean={month_data.mean():.3f}, std={month_data.std():.3f}")
        
        # Check for problematic values
        if torch.isnan(month_data).any():
            print(f"  WARNING: Month {month+1} has NaN values!")
        if torch.isinf(month_data).any():
            print(f"  WARNING: Month {month+1} has Inf values!")
    
    return features, target, mask

def debug_model():
    """Debug the model forward pass."""
    print(f"\n🧠 Debugging model...")
    
    # Create dummy input
    batch_size = 1
    time_steps = 12
    channels = 15
    height = 256
    width = 256
    
    dummy_input = torch.randn(batch_size, time_steps, channels, height, width)
    print(f"Input shape: {dummy_input.shape}")
    
    # Create model
    model = Temporal3DUNet(in_channels=channels, n_classes=1)
    
    # Test forward pass
    try:
        with torch.no_grad():
            output = model(dummy_input)
        print(f"Output shape: {output.shape}")
        print(f"Output - Min: {output.min():.3f}, Max: {output.max():.3f}")
        return True
    except Exception as e:
        print(f"Model forward pass failed: {e}")
        return False

def debug_loss_function():
    """Debug the loss function."""
    print(f"\n📊 Debugging loss function...")
    
    # Create dummy data
    batch_size = 1
    height = 256
    width = 256
    
    # Create realistic predictions and targets
    pred = torch.randn(batch_size, height, width) * 10 + 20  # Heights around 20m
    target = torch.randn(batch_size, height, width) * 5 + 25  # Heights around 25m
    
    # Create sparse mask (like GEDI)
    mask = torch.zeros(batch_size, height, width)
    n_valid = 100  # Only 100 valid pixels
    for b in range(batch_size):
        indices = torch.randperm(height * width)[:n_valid]
        flat_mask = mask[b].view(-1)
        flat_mask[indices] = 1
    
    print(f"Pred shape: {pred.shape}, range: {pred.min():.3f} to {pred.max():.3f}")
    print(f"Target shape: {target.shape}, range: {target.min():.3f} to {target.max():.3f}")
    print(f"Mask: {mask.sum()} valid pixels")
    
    # Test different loss functions
    from train_temporal_3d_unet import modified_huber_loss
    
    try:
        # Simple MSE first
        mse_loss = nn.MSELoss()
        if mask.sum() > 0:
            valid_pred = pred[mask > 0]
            valid_target = target[mask > 0]
            simple_loss = mse_loss(valid_pred, valid_target)
            print(f"Simple MSE loss: {simple_loss.item():.6f}")
        
        # Modified Huber loss
        huber_loss = modified_huber_loss(pred, target, mask, delta=1.0, shift_radius=1)
        print(f"Modified Huber loss: {huber_loss.item():.6f}")
        
        return True
    except Exception as e:
        print(f"Loss function failed: {e}")
        return False

def debug_normalization():
    """Debug data normalization."""
    print(f"\n🔧 Debugging data normalization...")
    
    patch_path = "chm_outputs/dchm_09gd4_temporal_bandNum196_scale10_patch0000.tif"
    
    with rasterio.open(patch_path) as src:
        # Sample some bands
        s1_data = src.read(1)  # S1_VV_M01
        s2_data = src.read(25)  # B2_M01
        alos2_data = src.read(157)  # ALOS2_HH_M01
        gedi_data = src.read(196)  # rh
        
    print(f"Raw data ranges:")
    print(f"S1 (VV): {s1_data.min():.3f} to {s1_data.max():.3f}")
    print(f"S2 (B2): {s2_data.min():.3f} to {s2_data.max():.3f}")
    print(f"ALOS2 (HH): {alos2_data.min():.3f} to {alos2_data.max():.3f}")
    print(f"GEDI: {np.nanmin(gedi_data):.3f} to {np.nanmax(gedi_data):.3f}")
    
    # Check for extreme values
    if np.any(np.abs(s1_data) > 100):
        print(f"WARNING: S1 has extreme values!")
    if np.any(s2_data > 50000):
        print(f"WARNING: S2 has extreme values!")
    if np.any(np.abs(alos2_data) > 100):
        print(f"WARNING: ALOS2 has extreme values!")

if __name__ == "__main__":
    print("🐛 Debugging temporal training issues...\n")
    
    # Debug dataset
    features, target, mask = debug_dataset()
    
    # Debug model
    model_ok = debug_model()
    
    # Debug loss function
    loss_ok = debug_loss_function()
    
    # Debug normalization
    debug_normalization()
    
    print(f"\n📋 Debug Summary:")
    print(f"   Model forward pass: {'✅' if model_ok else '❌'}")
    print(f"   Loss function: {'✅' if loss_ok else '❌'}")
    
    if not model_ok or not loss_ok:
        print(f"\n🔧 Recommendations:")
        if not model_ok:
            print(f"   - Fix model architecture issues")
        if not loss_ok:
            print(f"   - Fix loss function implementation")
        print(f"   - Add gradient clipping")
        print(f"   - Use smaller learning rate")
        print(f"   - Add data normalization")
#!/usr/bin/env python3
"""
Evaluate temporal prediction results
"""

import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def evaluate_temporal_prediction():
    """Evaluate the temporal prediction results."""
    
    print("🔍 Evaluating temporal prediction results...")
    
    # Check if prediction exists
    pred_path = "chm_outputs/improved_temporal_prediction.tif"
    if not os.path.exists(pred_path):
        print(f"❌ Prediction file not found: {pred_path}")
        
        # Check for intermediate checkpoints
        checkpoints = list(Path("chm_outputs").glob("improved_temporal_epoch_*.pth"))
        if checkpoints:
            print(f"✅ Found {len(checkpoints)} training checkpoints")
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.stem.split('_')[-1]))
            print(f"   Latest: {latest_checkpoint}")
            return
        else:
            print("❌ No training checkpoints found")
            return
    
    # Load prediction and reference
    with rasterio.open(pred_path) as src:
        prediction = src.read(1)
        pred_transform = src.transform
        pred_crs = src.crs
    
    reference_path = "downloads/dchm_09gd4.tif"
    if os.path.exists(reference_path):
        with rasterio.open(reference_path) as src:
            # Read a portion that matches the prediction
            reference = src.read(1)
        
        print(f"Prediction shape: {prediction.shape}")
        print(f"Reference shape: {reference.shape}")
        
        # Calculate basic statistics
        pred_valid = prediction[~np.isnan(prediction) & (prediction > 0)]
        ref_valid = reference[~np.isnan(reference) & (reference > 0)]
        
        print(f"\nPrediction statistics:")
        print(f"  Range: {pred_valid.min():.2f}m to {pred_valid.max():.2f}m")
        print(f"  Mean: {pred_valid.mean():.2f}m")
        print(f"  Valid pixels: {len(pred_valid)}/{prediction.size}")
        
        print(f"\nReference statistics:")
        print(f"  Range: {ref_valid.min():.2f}m to {ref_valid.max():.2f}m")
        print(f"  Mean: {ref_valid.mean():.2f}m")
        print(f"  Valid pixels: {len(ref_valid)}/{reference.size}")
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Prediction
        im1 = axes[0].imshow(prediction, cmap='viridis', vmin=0, vmax=50)
        axes[0].set_title(f'Temporal 3D U-Net Prediction\nMean: {pred_valid.mean():.1f}m')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], label='Height (m)')
        
        # Reference (sample portion)
        ref_sample = reference[0:256, 0:256]  # Sample same size as prediction
        im2 = axes[1].imshow(ref_sample, cmap='viridis', vmin=0, vmax=50)
        axes[1].set_title(f'Reference Data (Sample)\nMean: {ref_sample[ref_sample>0].mean():.1f}m')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], label='Height (m)')
        
        plt.tight_layout()
        plt.savefig('chm_outputs/temporal_evaluation.png', dpi=300, bbox_inches='tight')
        print(f"\n✅ Evaluation plot saved: chm_outputs/temporal_evaluation.png")
        
    else:
        print(f"❌ Reference file not found: {reference_path}")

def check_training_progress():
    """Check training progress from logs."""
    
    print("\n📊 Checking training progress...")
    
    # Check for loss plot
    loss_plot = "chm_outputs/improved_temporal_loss.png"
    if os.path.exists(loss_plot):
        print(f"✅ Loss plot available: {loss_plot}")
    
    # Check for checkpoints
    checkpoints = list(Path("chm_outputs").glob("improved_temporal_epoch_*.pth"))
    if checkpoints:
        print(f"✅ Training checkpoints: {len(checkpoints)}")
        for cp in sorted(checkpoints):
            epoch = cp.stem.split('_')[-1]
            print(f"   Epoch {epoch}: {cp}")
    
    # Check for final model
    final_model = "chm_outputs/improved_temporal_final.pth"
    if os.path.exists(final_model):
        print(f"✅ Final model: {final_model}")
    else:
        print(f"⏳ Final model not yet available")

if __name__ == "__main__":
    check_training_progress()
    evaluate_temporal_prediction()
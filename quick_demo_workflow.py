#!/usr/bin/env python3
"""
Quick demonstration of 3D U-Net workflow with minimal training.
"""

import os
import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path

# Import our modules
from train_predict_map import load_patch_data

class QuickPatchDataset(Dataset):
    """Minimal dataset for quick demonstration."""
    
    def __init__(self, features, gedi_target):
        # Resize to 64x64 for speed
        self.features = self._resize(features, 64)
        self.gedi_target = self._resize(gedi_target[None], 64)[0]
        self.n_samples = 8  # Small number of augmented samples
    
    def _resize(self, data, target_size):
        """Simple center crop to target size."""
        if len(data.shape) == 2:
            data = data[None]
        
        _, h, w = data.shape
        start_h = (h - target_size) // 2
        start_w = (w - target_size) // 2
        return data[:, start_h:start_h+target_size, start_w:start_w+target_size]
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        features = torch.FloatTensor(self.features)
        gedi_target = torch.FloatTensor(self.gedi_target)
        
        # Simple augmentation
        if idx > 0 and np.random.random() > 0.5:
            features = torch.flip(features, [2])
            gedi_target = torch.flip(gedi_target, [1])
        
        return features, gedi_target

class SimpleUNet(nn.Module):
    """Minimal U-Net for demonstration."""
    
    def __init__(self, in_channels, base_channels=16):
        super().__init__()
        
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels*2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(base_channels*2, base_channels, 2, stride=2),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 1, 1)
        )
    
    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)
        return x.squeeze(1)

def simple_gedi_loss(pred, target):
    """Simple MSE loss on GEDI pixels."""
    valid_mask = target > 0
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True)
    
    return nn.functional.mse_loss(pred[valid_mask], target[valid_mask])

def quick_train(patch_path, output_dir, epochs=5):
    """Quick training demonstration."""
    
    print(f"Loading patch: {patch_path}")
    features, gedi_target, band_info = load_patch_data(patch_path)
    
    print(f"Patch shape: {features.shape}, GEDI shape: {gedi_target.shape}")
    print(f"GEDI pixels: {(gedi_target > 0).sum()}/{gedi_target.size}")
    
    # Create minimal dataset
    dataset = QuickPatchDataset(features, gedi_target)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Simple model
    model = SimpleUNet(features.shape[0], base_channels=8)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Quick training
    print(f"Training for {epochs} epochs...")
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_features, batch_gedi in dataloader:
            optimizer.zero_grad()
            pred = model(batch_features)
            loss = simple_gedi_loss(pred, batch_gedi)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / max(num_batches, 1)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # Save model
    model_path = os.path.join(output_dir, 'quick_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'losses': losses,
        'model_config': {'in_channels': features.shape[0], 'base_channels': 8}
    }, model_path)
    
    return model, model_path

def quick_predict(model_path, patch_path, output_dir):
    """Quick prediction."""
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    model = SimpleUNet(checkpoint['model_config']['in_channels'], 
                      checkpoint['model_config']['base_channels'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load and process patch
    features, gedi_target, _ = load_patch_data(patch_path)
    dataset = QuickPatchDataset(features, gedi_target)
    
    # Predict
    with torch.no_grad():
        features_tensor = torch.FloatTensor(dataset.features).unsqueeze(0)
        pred = model(features_tensor).squeeze(0).numpy()
    
    # Save prediction
    pred_path = os.path.join(output_dir, 'quick_predictions.tif')
    
    # Create simple georeference (using patch bounds but for 64x64)
    with rasterio.open(patch_path) as src:
        bounds = src.bounds
        transform = rasterio.transform.from_bounds(
            bounds.left, bounds.bottom, bounds.right, bounds.top, 64, 64
        )
        
        with rasterio.open(
            pred_path, 'w',
            driver='GTiff',
            height=64, width=64,
            count=1, dtype=pred.dtype,
            crs=src.crs,
            transform=transform
        ) as dst:
            dst.write(pred, 1)
    
    print(f"Predictions saved: {pred_path}")
    return pred_path

def quick_evaluation(pred_path, ref_path, output_dir):
    """Quick evaluation with basic metrics."""
    
    print("Running quick evaluation...")
    
    try:
        # Load prediction
        with rasterio.open(pred_path) as src:
            pred_data = src.read(1)
            pred_bounds = src.bounds
            pred_crs = src.crs
        
        # Load reference and crop to prediction area
        with rasterio.open(ref_path) as src:
            # Read a small area from reference for comparison
            ref_data = src.read(1, window=rasterio.windows.from_bounds(
                *pred_bounds, src.transform
            ))
        
        # Resize reference to match prediction
        if ref_data.shape != pred_data.shape:
            from scipy.ndimage import zoom
            zoom_factors = (pred_data.shape[0] / ref_data.shape[0], 
                           pred_data.shape[1] / ref_data.shape[1])
            ref_data = zoom(ref_data, zoom_factors, order=1)
        
        # Simple metrics on valid pixels
        valid_mask = (pred_data > 0) & (ref_data > 0) & ~np.isnan(pred_data) & ~np.isnan(ref_data)
        
        if valid_mask.sum() > 0:
            pred_valid = pred_data[valid_mask]
            ref_valid = ref_data[valid_mask]
            
            mae = np.mean(np.abs(pred_valid - ref_valid))
            rmse = np.sqrt(np.mean((pred_valid - ref_valid)**2))
            corr = np.corrcoef(pred_valid, ref_valid)[0, 1] if len(ref_valid) > 1 else 0
            
            print(f"Quick Evaluation Results:")
            print(f"Valid pixels: {valid_mask.sum()}")
            print(f"MAE: {mae:.3f}")
            print(f"RMSE: {rmse:.3f}")
            print(f"Correlation: {corr:.3f}")
            
            # Save comparison plot
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 3, 1)
            plt.imshow(pred_data, cmap='viridis', vmin=0, vmax=30)
            plt.title('Prediction')
            plt.colorbar()
            
            plt.subplot(1, 3, 2)
            plt.imshow(ref_data, cmap='viridis', vmin=0, vmax=30)
            plt.title('Reference')
            plt.colorbar()
            
            plt.subplot(1, 3, 3)
            plt.scatter(ref_valid, pred_valid, alpha=0.6, s=1)
            plt.plot([0, 30], [0, 30], 'r--', alpha=0.8)
            plt.xlabel('Reference')
            plt.ylabel('Prediction')
            plt.title(f'Scatter (r={corr:.3f})')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'quick_evaluation.png'), dpi=150, bbox_inches='tight')
            plt.close()
            
            return True
        else:
            print("No valid pixels for comparison")
            return False
            
    except Exception as e:
        print(f"Evaluation error: {e}")
        return False

def main():
    patch_path = 'chm_outputs/dchm_09gd4_bandNum31_scale10_patch0000.tif'
    ref_path = 'downloads/dchm_09gd4.tif'
    output_dir = 'chm_outputs/quick_demo_results'
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== QUICK 3D U-NET DEMONSTRATION ===")
    
    # Train
    print("\n1. Training...")
    model, model_path = quick_train(patch_path, output_dir, epochs=5)
    
    # Predict
    print("\n2. Predicting...")
    pred_path = quick_predict(model_path, patch_path, output_dir)
    
    # Evaluate
    print("\n3. Evaluating...")
    success = quick_evaluation(pred_path, ref_path, output_dir)
    
    print(f"\nDemo completed! Results in: {output_dir}")
    
    if success:
        print("✓ Training, prediction, and evaluation successful!")
    else:
        print("✗ Some issues occurred, but basic workflow demonstrated")

if __name__ == "__main__":
    main()
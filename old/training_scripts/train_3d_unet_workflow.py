#!/usr/bin/env python3
"""
Complete 3D U-Net training, prediction, and evaluation workflow.
Handles single patch training with data augmentation and proper evaluation.
"""

import os
import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from pathlib import Path
import argparse
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import our modules
from train_predict_map import (
    load_patch_data, modified_huber_loss, load_patches_from_directory
)
from data.normalization import *
from raster_utils import load_and_align_rasters

class PatchDataset(Dataset):
    """Dataset for 3D patch training with augmentation."""
    
    def __init__(self, features, gedi_target, patch_size=256, augment=True, n_augmentations=8):
        """
        Args:
            features: [C, H, W] feature array
            gedi_target: [H, W] GEDI height array
            patch_size: Target patch size (256)
            augment: Whether to apply data augmentation
            n_augmentations: Number of augmented versions per patch
        """
        self.features = features
        self.gedi_target = gedi_target
        self.patch_size = patch_size
        self.augment = augment
        self.n_augmentations = n_augmentations if augment else 1
        
        # Resize to 256x256 if needed
        if features.shape[1] != patch_size or features.shape[2] != patch_size:
            self.features = self._resize_patch(features, patch_size)
            self.gedi_target = self._resize_patch(gedi_target[None], patch_size)[0]
    
    def _resize_patch(self, data, target_size):
        """Resize patch to target size by center cropping or padding."""
        if len(data.shape) == 2:
            data = data[None]  # Add channel dimension
        
        _, h, w = data.shape
        
        if h > target_size or w > target_size:
            # Center crop
            start_h = (h - target_size) // 2
            start_w = (w - target_size) // 2
            data = data[:, start_h:start_h+target_size, start_w:start_w+target_size]
        elif h < target_size or w < target_size:
            # Pad to target size
            pad_h = max(0, target_size - h)
            pad_w = max(0, target_size - w)
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            
            data = np.pad(data, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), 
                         mode='reflect')
        
        return data
    
    def __len__(self):
        return self.n_augmentations
    
    def __getitem__(self, idx):
        features = torch.FloatTensor(self.features)
        gedi_target = torch.FloatTensor(self.gedi_target)
        
        if self.augment and idx > 0:
            # Apply random augmentations
            if np.random.random() > 0.5:
                # Horizontal flip
                features = torch.flip(features, [2])
                gedi_target = torch.flip(gedi_target, [1])
            
            if np.random.random() > 0.5:
                # Vertical flip
                features = torch.flip(features, [1])
                gedi_target = torch.flip(gedi_target, [0])
            
            # Random rotation (90, 180, 270 degrees)
            k = np.random.randint(0, 4)
            if k > 0:
                features = torch.rot90(features, k, [1, 2])
                gedi_target = torch.rot90(gedi_target, k, [0, 1])
            
            # Small random crops and resize back
            if np.random.random() > 0.5:
                crop_size = int(self.patch_size * (0.85 + 0.15 * np.random.random()))
                if crop_size < self.patch_size:
                    # Random crop
                    max_offset = self.patch_size - crop_size
                    h_offset = np.random.randint(0, max_offset + 1)
                    w_offset = np.random.randint(0, max_offset + 1)
                    
                    features_crop = features[:, h_offset:h_offset+crop_size, w_offset:w_offset+crop_size]
                    gedi_crop = gedi_target[h_offset:h_offset+crop_size, w_offset:w_offset+crop_size]
                    
                    # Resize back to original size
                    features = TF.resize(features_crop.unsqueeze(0), (self.patch_size, self.patch_size)).squeeze(0)
                    gedi_target = TF.resize(gedi_crop.unsqueeze(0).unsqueeze(0), (self.patch_size, self.patch_size)).squeeze(0).squeeze(0)
        
        return features, gedi_target

def create_3d_unet_simple(in_channels, base_channels=32):
    """Create a simplified 3D U-Net for single patch training."""
    
    class Simple3DUNet(nn.Module):
        def __init__(self, in_channels, base_channels):
            super().__init__()
            
            # Encoder
            self.enc1 = nn.Sequential(
                nn.Conv2d(in_channels, base_channels, 3, padding=1),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels, base_channels, 3, padding=1),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(inplace=True)
            )
            
            self.enc2 = nn.Sequential(
                nn.MaxPool2d(2),
                nn.Conv2d(base_channels, base_channels*2, 3, padding=1),
                nn.BatchNorm2d(base_channels*2),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels*2, base_channels*2, 3, padding=1),
                nn.BatchNorm2d(base_channels*2),
                nn.ReLU(inplace=True)
            )
            
            self.enc3 = nn.Sequential(
                nn.MaxPool2d(2),
                nn.Conv2d(base_channels*2, base_channels*4, 3, padding=1),
                nn.BatchNorm2d(base_channels*4),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels*4, base_channels*4, 3, padding=1),
                nn.BatchNorm2d(base_channels*4),
                nn.ReLU(inplace=True)
            )
            
            # Decoder
            self.up1 = nn.ConvTranspose2d(base_channels*4, base_channels*2, 2, stride=2)
            self.dec1 = nn.Sequential(
                nn.Conv2d(base_channels*4, base_channels*2, 3, padding=1),
                nn.BatchNorm2d(base_channels*2),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels*2, base_channels*2, 3, padding=1),
                nn.BatchNorm2d(base_channels*2),
                nn.ReLU(inplace=True)
            )
            
            self.up2 = nn.ConvTranspose2d(base_channels*2, base_channels, 2, stride=2)
            self.dec2 = nn.Sequential(
                nn.Conv2d(base_channels*2, base_channels, 3, padding=1),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels, base_channels, 3, padding=1),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(inplace=True)
            )
            
            self.final = nn.Conv2d(base_channels, 1, 1)
            
        def forward(self, x):
            # Encoder
            e1 = self.enc1(x)
            e2 = self.enc2(e1)
            e3 = self.enc3(e2)
            
            # Decoder
            d1 = self.up1(e3)
            d1 = torch.cat([d1, e2], dim=1)
            d1 = self.dec1(d1)
            
            d2 = self.up2(d1)
            d2 = torch.cat([d2, e1], dim=1)
            d2 = self.dec2(d2)
            
            out = self.final(d2)
            return out.squeeze(1)  # Remove channel dimension
    
    return Simple3DUNet(in_channels, base_channels)

def sparse_gedi_loss(pred, target, delta=1.0, shift_radius=1):
    """Modified Huber loss for sparse GEDI data with shift awareness."""
    
    def huber_loss(x, y, delta=1.0):
        diff = x - y
        abs_diff = diff.abs()
        quadratic = torch.min(abs_diff, torch.tensor(delta, device=x.device))
        linear = abs_diff - quadratic
        return 0.5 * quadratic.pow(2) + delta * linear
    
    # Find valid GEDI pixels
    valid_mask = target > 0
    
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True, device=pred.device)
    
    best_loss = float('inf')
    best_tensor = None
    
    # Try different spatial shifts
    for dx in range(-shift_radius, shift_radius + 1):
        for dy in range(-shift_radius, shift_radius + 1):
            # Shift predictions
            if dx == 0 and dy == 0:
                shifted_pred = pred
            else:
                shifted_pred = torch.roll(pred, shifts=(dx, dy), dims=(1, 2))
            
            # Compute loss on valid pixels
            if len(shifted_pred.shape) == 3:  # [B, H, W]
                loss_values = []
                for b in range(shifted_pred.shape[0]):
                    batch_valid = valid_mask[b] if len(valid_mask.shape) == 3 else valid_mask
                    if batch_valid.sum() > 0:
                        loss = huber_loss(
                            shifted_pred[b][batch_valid],
                            target[b][batch_valid] if len(target.shape) == 3 else target[batch_valid],
                            delta
                        ).mean()
                        loss_values.append(loss)
                
                if loss_values:
                    total_loss = torch.stack(loss_values).mean()
                    if total_loss.item() < best_loss:
                        best_loss = total_loss.item()
                        best_tensor = total_loss
            else:  # [H, W]
                if valid_mask.sum() > 0:
                    loss = huber_loss(shifted_pred[valid_mask], target[valid_mask], delta).mean()
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        best_tensor = loss
    
    return best_tensor if best_tensor is not None else torch.tensor(0.0, requires_grad=True, device=pred.device)

def train_3d_unet(patch_path, output_dir, epochs=50, batch_size=4, learning_rate=1e-3):
    """Train 3D U-Net on single patch with augmentation."""
    
    print(f"Loading patch data from: {patch_path}")
    features, gedi_target, band_info = load_patch_data(patch_path)
    
    print(f"Original patch shape: {features.shape}, GEDI shape: {gedi_target.shape}")
    print(f"GEDI pixels with data: {(gedi_target > 0).sum()}/{gedi_target.size}")
    
    # Create dataset with augmentation
    dataset = PatchDataset(features, gedi_target, patch_size=256, augment=True, n_augmentations=32)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = create_3d_unet_simple(features.shape[0], base_channels=32)
    model = model.to(device)
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # Training loop
    model.train()
    train_losses = []
    
    print(f"Starting training for {epochs} epochs...")
    for epoch in tqdm(range(epochs), desc="Training"):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_features, batch_gedi in dataloader:
            batch_features = batch_features.to(device)
            batch_gedi = batch_gedi.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            pred = model(batch_features)
            
            # Compute loss
            loss = sparse_gedi_loss(pred, batch_gedi, delta=1.0, shift_radius=1)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / max(num_batches, 1)
        train_losses.append(avg_loss)
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
    
    # Save model
    model_path = os.path.join(output_dir, '3d_unet_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'band_info': band_info,
        'model_config': {
            'in_channels': features.shape[0],
            'base_channels': 32,
            'patch_size': 256
        }
    }, model_path)
    
    print(f"Model saved to: {model_path}")
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'training_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return model, model_path

def predict_patch(model_path, patch_path, output_dir):
    """Generate predictions for a patch."""
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    model_config = checkpoint['model_config']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_3d_unet_simple(model_config['in_channels'], model_config['base_channels'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load patch data
    features, gedi_target, band_info = load_patch_data(patch_path)
    
    # Resize to 256x256 if needed
    if features.shape[1] != 256 or features.shape[2] != 256:
        dataset = PatchDataset(features, gedi_target, patch_size=256, augment=False)
        features = dataset.features
        gedi_target = dataset.gedi_target
    
    # Make prediction
    with torch.no_grad():
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
        pred = model(features_tensor)
        pred = pred.squeeze(0).cpu().numpy()
    
    # Save prediction as GeoTIFF
    pred_path = os.path.join(output_dir, 'predictions.tif')
    
    # Use original patch georeference
    with rasterio.open(patch_path) as src:
        # Get original transform and adjust for size change
        original_transform = src.transform
        original_bounds = src.bounds
        
        # Calculate new transform for 256x256
        width_scale = src.width / 256
        height_scale = src.height / 256
        new_pixel_width = original_transform[0] * width_scale
        new_pixel_height = original_transform[4] * height_scale
        
        new_transform = rasterio.Affine(
            new_pixel_width, 0, original_bounds.left,
            0, new_pixel_height, original_bounds.top
        )
        
        # Save prediction
        with rasterio.open(
            pred_path, 'w',
            driver='GTiff',
            height=256, width=256,
            count=1, dtype=pred.dtype,
            crs=src.crs,
            transform=new_transform,
            compress='lzw'
        ) as dst:
            dst.write(pred, 1)
    
    print(f"Predictions saved to: {pred_path}")
    return pred_path

def run_evaluation(pred_path, ref_path, output_dir):
    """Run evaluation using the existing evaluation pipeline."""
    
    print(f"Running evaluation...")
    print(f"Prediction: {pred_path}")
    print(f"Reference: {ref_path}")
    
    # Use the existing evaluation pipeline
    import subprocess
    import sys
    
    eval_cmd = [
        sys.executable, '-m', 'evaluate_predictions',
        '--pred', pred_path,
        '--ref', ref_path,
        '--output', output_dir,
        '--pdf'
    ]
    
    try:
        result = subprocess.run(eval_cmd, capture_output=True, text=True, cwd='.')
        print("Evaluation output:")
        print(result.stdout)
        if result.stderr:
            print("Evaluation errors:")
            print(result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running evaluation: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='3D U-Net Training and Evaluation Workflow')
    parser.add_argument('--patch', type=str, required=True,
                       help='Path to training patch TIF file')
    parser.add_argument('--reference', type=str, required=True,
                       help='Path to reference CHM for evaluation')
    parser.add_argument('--output-dir', type=str, default='chm_outputs/3d_unet_results',
                       help='Output directory')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training and use existing model')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train model
    if not args.skip_training:
        print("=== TRAINING PHASE ===")
        model, model_path = train_3d_unet(
            args.patch, 
            args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
    else:
        model_path = os.path.join(args.output_dir, '3d_unet_model.pth')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        print(f"Using existing model: {model_path}")
    
    # Generate predictions
    print("\n=== PREDICTION PHASE ===")
    pred_path = predict_patch(model_path, args.patch, args.output_dir)
    
    # Run evaluation
    print("\n=== EVALUATION PHASE ===")
    success = run_evaluation(pred_path, args.reference, args.output_dir)
    
    if success:
        print(f"\nWorkflow completed successfully!")
        print(f"Results saved in: {args.output_dir}")
    else:
        print(f"\nEvaluation failed. Check outputs in: {args.output_dir}")

if __name__ == "__main__":
    main()
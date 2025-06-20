#!/usr/bin/env python3
"""
Train 3D U-Net for Paul's 2025 temporal canopy height modeling
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import rasterio
import numpy as np
import pandas as pd
import os
import argparse
from pathlib import Path
import time
from datetime import datetime
import json
import matplotlib.pyplot as plt

class TemporalPatchDataset(Dataset):
    """Dataset for temporal patches with 3D structure."""
    
    def __init__(self, patch_path, reference_path=None, patch_size=256, augment=True):
        self.patch_path = patch_path
        self.reference_path = reference_path
        self.patch_size = patch_size
        self.augment = augment
        
        print(f"Loading temporal patch: {patch_path}")
        
        # Load patch data
        with rasterio.open(patch_path) as src:
            self.patch_data = src.read()  # Shape: (bands, height, width)
            self.patch_crs = src.crs
            self.patch_transform = src.transform
            self.band_descriptions = src.descriptions
            
        print(f"Patch shape: {self.patch_data.shape}")
        
        # Organize bands by sensor and time
        self.organize_temporal_bands()
        
        # Load reference data if provided
        if reference_path:
            self.load_reference_data()
        
        # Resize to target patch size if needed
        if self.patch_data.shape[1] != patch_size or self.patch_data.shape[2] != patch_size:
            self.resize_patch()
        
        # Create augmented versions if enabled
        if augment:
            self.create_augmentations()
        else:
            self.augmented_patches = [self.patch_data]
            self.augmented_gedi = [self.gedi_data] if hasattr(self, 'gedi_data') else [None]
    
    def organize_temporal_bands(self):
        """Organize bands into temporal structure for 3D U-Net."""
        
        # Find band indices for each sensor/month
        s1_bands = {}  # {month: [VV_idx, VH_idx]}
        s2_bands = {}  # {month: [B2_idx, B3_idx, ...]}
        alos2_bands = {}  # {month: [HH_idx, HV_idx]}
        other_bands = []
        
        for i, desc in enumerate(self.band_descriptions):
            if desc.startswith('S1_'):
                if '_M' in desc:
                    parts = desc.split('_')
                    month = parts[2][1:]  # Remove 'M' prefix
                    pol = parts[1]  # VV or VH
                    if month not in s1_bands:
                        s1_bands[month] = {}
                    s1_bands[month][pol] = i
            elif any(desc.startswith(prefix) for prefix in ['B2_', 'B3_', 'B4_', 'B5_', 'B6_', 'B7_', 'B8_', 'B8A_', 'B11_', 'B12_', 'NDVI_']):
                if '_M' in desc:
                    parts = desc.split('_')
                    month = parts[1][1:]  # Remove 'M' prefix
                    band = parts[0]
                    if month not in s2_bands:
                        s2_bands[month] = {}
                    s2_bands[month][band] = i
            elif desc.startswith('ALOS2_'):
                if '_M' in desc:
                    parts = desc.split('_')
                    month = parts[2][1:]  # Remove 'M' prefix
                    pol = parts[1]  # HH or HV
                    if month not in alos2_bands:
                        alos2_bands[month] = {}
                    alos2_bands[month][pol] = i
            else:
                other_bands.append(i)
        
        # Store organization info
        self.s1_bands = s1_bands
        self.s2_bands = s2_bands
        self.alos2_bands = alos2_bands
        self.other_bands = other_bands
        
        print(f"Temporal organization:")
        print(f"  S1: {len(s1_bands)} months")
        print(f"  S2: {len(s2_bands)} months")
        print(f"  ALOS2: {len(alos2_bands)} months")
        print(f"  Other: {len(other_bands)} bands")
        
        # Find GEDI band
        self.gedi_band_idx = None
        for i, desc in enumerate(self.band_descriptions):
            if desc == 'rh':
                self.gedi_band_idx = i
                break
        
        if self.gedi_band_idx is not None:
            self.gedi_data = self.patch_data[self.gedi_band_idx]
            print(f"Found GEDI data at band {self.gedi_band_idx + 1}")
            valid_gedi = self.gedi_data[~np.isnan(self.gedi_data) & (self.gedi_data > 0)]
            print(f"  Valid GEDI pixels: {len(valid_gedi)}/{self.gedi_data.size} ({len(valid_gedi)/self.gedi_data.size*100:.2f}%)")
            if len(valid_gedi) > 0:
                print(f"  GEDI range: {valid_gedi.min():.2f}m to {valid_gedi.max():.2f}m")
        else:
            print("Warning: No GEDI reference data found")
    
    def load_reference_data(self):
        """Load reference data for evaluation."""
        if os.path.exists(self.reference_path):
            with rasterio.open(self.reference_path) as src:
                self.reference_data = src.read(1)
            print(f"Loaded reference data: {self.reference_data.shape}")
        else:
            print(f"Warning: Reference file not found: {self.reference_path}")
            self.reference_data = None
    
    def resize_patch(self):
        """Resize patch to target size."""
        from scipy.ndimage import zoom
        
        current_h, current_w = self.patch_data.shape[1], self.patch_data.shape[2]
        scale_h = self.patch_size / current_h
        scale_w = self.patch_size / current_w
        
        print(f"Resizing from {current_h}x{current_w} to {self.patch_size}x{self.patch_size}")
        
        resized_data = np.zeros((self.patch_data.shape[0], self.patch_size, self.patch_size), dtype=self.patch_data.dtype)
        
        for i in range(self.patch_data.shape[0]):
            resized_data[i] = zoom(self.patch_data[i], (scale_h, scale_w), order=1)
        
        self.patch_data = resized_data
        
        # Resize GEDI data if it exists
        if hasattr(self, 'gedi_data'):
            self.gedi_data = zoom(self.gedi_data, (scale_h, scale_w), order=0)  # Nearest neighbor for labels
    
    def create_augmentations(self):
        """Create augmented versions of the patch."""
        augmentations = []
        gedi_augmentations = []
        
        # Original
        augmentations.append(self.patch_data.copy())
        if hasattr(self, 'gedi_data'):
            gedi_augmentations.append(self.gedi_data.copy())
        
        # Rotations
        for angle in [90, 180, 270]:
            k = angle // 90
            rotated = np.rot90(self.patch_data, k=k, axes=(1, 2)).copy()
            augmentations.append(rotated)
            
            if hasattr(self, 'gedi_data'):
                rotated_gedi = np.rot90(self.gedi_data, k=k).copy()
                gedi_augmentations.append(rotated_gedi)
        
        # Flips
        # Horizontal flip
        flipped_h = np.flip(self.patch_data, axis=2).copy()
        augmentations.append(flipped_h)
        if hasattr(self, 'gedi_data'):
            gedi_augmentations.append(np.flip(self.gedi_data, axis=1).copy())
        
        # Vertical flip
        flipped_v = np.flip(self.patch_data, axis=1).copy()
        augmentations.append(flipped_v)
        if hasattr(self, 'gedi_data'):
            gedi_augmentations.append(np.flip(self.gedi_data, axis=0).copy())
        
        # Random crops (create 4 additional crops)
        for _ in range(4):
            if self.patch_size >= 224:  # Only if patch is large enough
                crop_size = int(self.patch_size * 0.9)  # 90% of original
                start_h = np.random.randint(0, self.patch_size - crop_size)
                start_w = np.random.randint(0, self.patch_size - crop_size)
                
                cropped = self.patch_data[:, start_h:start_h+crop_size, start_w:start_w+crop_size]
                # Resize back to original size
                from scipy.ndimage import zoom
                scale = self.patch_size / crop_size
                resized_crop = np.zeros_like(self.patch_data)
                for i in range(cropped.shape[0]):
                    resized_crop[i] = zoom(cropped[i], (scale, scale), order=1)
                
                augmentations.append(resized_crop)
                
                if hasattr(self, 'gedi_data'):
                    cropped_gedi = self.gedi_data[start_h:start_h+crop_size, start_w:start_w+crop_size]
                    resized_gedi = zoom(cropped_gedi, (scale, scale), order=0)
                    gedi_augmentations.append(resized_gedi)
        
        self.augmented_patches = augmentations
        self.augmented_gedi = gedi_augmentations if hasattr(self, 'gedi_data') else [None] * len(augmentations)
        
        print(f"Created {len(augmentations)} augmented patches")
    
    def __len__(self):
        return len(self.augmented_patches)
    
    def __getitem__(self, idx):
        patch = self.augmented_patches[idx].copy()  # Make a copy to avoid negative stride issues
        gedi = self.augmented_gedi[idx].copy() if self.augmented_gedi[idx] is not None else np.zeros((self.patch_size, self.patch_size))
        
        # Organize into temporal structure for 3D U-Net
        # Shape: (time, channels, height, width)
        temporal_data = self.organize_for_3d_unet(patch)
        
        # Clean GEDI data
        gedi = np.nan_to_num(gedi, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Convert to tensors (make copies to ensure positive strides)
        features = torch.FloatTensor(temporal_data.copy())
        target = torch.FloatTensor(gedi.copy())
        
        # Create mask for valid GEDI pixels (original valid pixels before NaN replacement)
        original_gedi = self.augmented_gedi[idx] if self.augmented_gedi[idx] is not None else np.zeros((self.patch_size, self.patch_size))
        mask = torch.FloatTensor(((~np.isnan(original_gedi)) & (original_gedi > 0)).copy())
        
        return features, target, mask
    
    def organize_for_3d_unet(self, patch_data):
        """Organize patch data for 3D U-Net: (time=12, channels, height, width)."""
        
        # Initialize temporal array
        n_months = 12
        # Calculate channels per month: S1(2) + S2(11) + ALOS2(2) = 15 channels per month
        channels_per_month = 15
        temporal_array = np.zeros((n_months, channels_per_month, self.patch_size, self.patch_size), dtype=np.float32)
        
        # Fill temporal data month by month
        for month_idx in range(1, 13):
            month_str = f"{month_idx:02d}"
            channel_idx = 0
            
            # S1 data (2 channels: VV, VH)
            if month_str in self.s1_bands:
                for pol in ['VV', 'VH']:
                    if pol in self.s1_bands[month_str]:
                        band_idx = self.s1_bands[month_str][pol]
                        band_data = patch_data[band_idx]
                        # Replace NaN with zeros and normalize S1 dB values
                        band_data = np.nan_to_num(band_data, nan=0.0, posinf=0.0, neginf=0.0)
                        # S1 is in dB, normalize using (val + 25) / 25 as per normalization.py
                        band_data = (band_data + 25) / 25
                        temporal_array[month_idx-1, channel_idx] = band_data
                    channel_idx += 1
            else:
                channel_idx += 2  # Skip S1 channels if no data
            
            # S2 data (11 channels: B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, NDVI)
            if month_str in self.s2_bands:
                for band in ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12', 'NDVI']:
                    if band in self.s2_bands[month_str]:
                        band_idx = self.s2_bands[month_str][band]
                        band_data = patch_data[band_idx]
                        # Replace NaN with zeros and normalize S2 reflectance values
                        band_data = np.nan_to_num(band_data, nan=0.0, posinf=0.0, neginf=0.0)
                        # Normalize S2 reflectance values
                        if band != 'NDVI':
                            # S2 L2A reflectance should be in range 0-10000, normalize to 0-1
                            band_data = np.clip(band_data / 10000.0, 0, 1)
                        else:
                            # NDVI should be in range -1 to 1, keep as is but clip
                            band_data = np.clip(band_data, -1, 1)
                        temporal_array[month_idx-1, channel_idx] = band_data
                    channel_idx += 1
            else:
                channel_idx += 11  # Skip S2 channels if no data
            
            # ALOS2 data (2 channels: HH, HV)
            if month_str in self.alos2_bands:
                for pol in ['HH', 'HV']:
                    if pol in self.alos2_bands[month_str]:
                        band_idx = self.alos2_bands[month_str][pol]
                        band_data = patch_data[band_idx]
                        # Replace NaN with zeros
                        band_data = np.nan_to_num(band_data, nan=0.0, posinf=0.0, neginf=0.0)
                        temporal_array[month_idx-1, channel_idx] = band_data
                    channel_idx += 1
            else:
                channel_idx += 2  # Skip ALOS2 channels if no data
        
        # Final check: replace any remaining NaN/inf values
        temporal_array = np.nan_to_num(temporal_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        return temporal_array

class Temporal3DUNet(nn.Module):
    """3D U-Net for temporal canopy height modeling."""
    
    def __init__(self, in_channels=15, n_classes=1):
        super().__init__()
        
        # Encoder (no temporal pooling to keep 12-month structure)
        self.encoder1 = self.conv_block_3d(in_channels, 32)
        self.encoder2 = self.conv_block_3d(32, 64)
        self.encoder3 = self.conv_block_3d(64, 128)
        self.encoder4 = self.conv_block_3d(128, 256)
        
        # Bottleneck
        self.bottleneck = self.conv_block_3d(256, 512)
        
        # Decoder (only spatial upsampling)
        self.decoder4 = self.upconv_block_3d(512, 256)
        self.decoder3 = self.upconv_block_3d(512, 128)  # 512 = 256 + 256 from skip
        self.decoder2 = self.upconv_block_3d(256, 64)   # 256 = 128 + 128 from skip
        self.decoder1 = self.upconv_block_3d(128, 32)   # 128 = 64 + 64 from skip
        
        # Temporal aggregation and final prediction
        self.temporal_conv = nn.Sequential(
            nn.Conv3d(32, 16, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 8, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.ReLU(inplace=True)
        )
        
        # Global temporal pooling and final 2D conv
        self.final_conv = nn.Conv2d(8, n_classes, kernel_size=1)
        
    def conv_block_3d(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def upconv_block_3d(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(3,3,3), padding=(1,1,1)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)
        )
    
    def forward(self, x):
        # x shape: (batch, time, channels, height, width)
        # Rearrange to (batch, channels, time, height, width) for 3D conv
        x = x.permute(0, 2, 1, 3, 4)
        
        # Encoder (only spatial pooling, preserve temporal dimension)
        e1 = self.encoder1(x)  # (B, 32, 12, 256, 256)
        e2 = self.encoder2(nn.MaxPool3d((1,2,2))(e1))  # (B, 64, 12, 128, 128)
        e3 = self.encoder3(nn.MaxPool3d((1,2,2))(e2))  # (B, 128, 12, 64, 64)
        e4 = self.encoder4(nn.MaxPool3d((1,2,2))(e3))  # (B, 256, 12, 32, 32)
        
        # Bottleneck
        b = self.bottleneck(nn.MaxPool3d((1,2,2))(e4))  # (B, 512, 12, 16, 16)
        
        # Decoder with skip connections
        d4 = self.decoder4(b)  # (B, 256, 12, 32, 32)
        d4 = torch.cat([d4, e4], dim=1)  # (B, 512, 12, 32, 32)
        
        d3 = self.decoder3(d4)  # (B, 128, 12, 64, 64)
        d3 = torch.cat([d3, e3], dim=1)  # (B, 256, 12, 64, 64)
        
        d2 = self.decoder2(d3)  # (B, 64, 12, 128, 128)
        d2 = torch.cat([d2, e2], dim=1)  # (B, 128, 12, 128, 128)
        
        d1 = self.decoder1(d2)  # (B, 32, 12, 256, 256)
        
        # Temporal processing
        temporal_features = self.temporal_conv(d1)  # (B, 8, 12, 256, 256)
        
        # Global temporal pooling (average across time)
        pooled = torch.mean(temporal_features, dim=2)  # (B, 8, 256, 256)
        
        # Final prediction
        out = self.final_conv(pooled)  # (B, 1, 256, 256)
        
        return out.squeeze(1)  # (B, 256, 256)

def modified_huber_loss(pred, target, mask=None, delta=1.0, shift_radius=1):
    """Modified Huber loss with spatial shift awareness for GEDI alignment."""
    
    def huber_loss(x, y, delta=1.0):
        diff = x - y
        abs_diff = diff.abs()
        quadratic = torch.min(abs_diff, torch.tensor(delta, device=x.device))
        linear = abs_diff - quadratic
        return 0.5 * quadratic.pow(2) + delta * linear
    
    # Generate spatial shifts
    shifts = [(0, 0)]  # No shift
    for dx in range(-shift_radius, shift_radius + 1):
        for dy in range(-shift_radius, shift_radius + 1):
            if dx == 0 and dy == 0:
                continue
            if dx*dx + dy*dy <= shift_radius*shift_radius:
                shifts.append((dx, dy))
    
    best_loss = float('inf')
    
    for dx, dy in shifts:
        # Apply spatial shift to target
        if dx != 0 or dy != 0:
            shifted_target = torch.roll(target, shifts=(dx, dy), dims=(-2, -1))
            if mask is not None:
                shifted_mask = torch.roll(mask, shifts=(dx, dy), dims=(-2, -1))
            else:
                shifted_mask = mask
        else:
            shifted_target = target
            shifted_mask = mask
        
        # Calculate loss
        loss_map = huber_loss(pred, shifted_target, delta)
        
        if shifted_mask is not None:
            # Only compute loss on valid pixels
            valid_loss = loss_map * shifted_mask
            if shifted_mask.sum() > 0:
                loss = valid_loss.sum() / shifted_mask.sum()
            else:
                loss = torch.tensor(0.0, device=pred.device)
        else:
            loss = loss_map.mean()
        
        best_loss = min(best_loss, loss.item())
    
    return torch.tensor(best_loss, device=pred.device, requires_grad=True)

def train_temporal_model(patch_path, reference_path=None, epochs=50, batch_size=4, learning_rate=1e-3):
    """Train temporal 3D U-Net model."""
    
    print(f"🚀 Starting temporal 3D U-Net training")
    print(f"   Patch: {patch_path}")
    print(f"   Reference: {reference_path}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")
    
    # Create dataset
    dataset = TemporalPatchDataset(patch_path, reference_path, patch_size=256, augment=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create model
    model = Temporal3DUNet(in_channels=15, n_classes=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    # Training loop
    training_losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (features, target, mask) in enumerate(dataloader):
            features = features.to(device)
            target = target.to(device)
            mask = mask.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            pred = model(features)
            
            # Compute loss
            loss = modified_huber_loss(pred, target, mask, delta=1.0, shift_radius=1)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 5 == 0:
                print(f"   Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        training_losses.append(avg_loss)
        scheduler.step(avg_loss)
        
        print(f"Epoch {epoch+1}/{epochs}: Average Loss = {avg_loss:.4f}")
        
        # Save model checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"chm_outputs/temporal_3d_unet_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"   Saved checkpoint: {checkpoint_path}")
    
    # Save final model
    final_model_path = "chm_outputs/temporal_3d_unet_final.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'training_losses': training_losses,
        'model_config': {
            'in_channels': 15,
            'n_classes': 1,
            'patch_size': 256
        }
    }, final_model_path)
    
    print(f"✅ Training complete! Model saved: {final_model_path}")
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(training_losses)
    plt.title('Temporal 3D U-Net Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('chm_outputs/temporal_training_loss.png')
    print(f"Training loss plot saved: chm_outputs/temporal_training_loss.png")
    
    return model, training_losses

def predict_with_temporal_model(model_path, patch_path, output_path):
    """Generate predictions using trained temporal model."""
    
    print(f"🔮 Generating temporal predictions")
    print(f"   Model: {model_path}")
    print(f"   Patch: {patch_path}")
    print(f"   Output: {output_path}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = Temporal3DUNet(in_channels=15, n_classes=1).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load data
    dataset = TemporalPatchDataset(patch_path, patch_size=256, augment=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Generate prediction
    with torch.no_grad():
        for features, _, _ in dataloader:
            features = features.to(device)
            pred = model(features)
            prediction = pred.cpu().numpy()[0, 0]  # Get first batch, first channel
            break
    
    # Save prediction as GeoTIFF
    with rasterio.open(patch_path) as src:
        profile = src.profile.copy()
        profile.update({
            'count': 1,
            'dtype': 'float32'
        })
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(prediction, 1)
    
    print(f"✅ Prediction saved: {output_path}")
    return prediction

if __name__ == "__main__":
    # Configuration
    patch_path = "chm_outputs/dchm_09gd4_temporal_bandNum196_scale10_patch0000.tif"
    reference_path = "downloads/dchm_09gd4.tif"
    
    # Train model
    model, losses = train_temporal_model(
        patch_path=patch_path,
        reference_path=reference_path,
        epochs=10,  # Reduced for testing
        batch_size=1,  # Smaller batch size for temporal data
        learning_rate=1e-3
    )
    
    # Generate prediction
    predict_with_temporal_model(
        model_path="chm_outputs/temporal_3d_unet_final.pth",
        patch_path=patch_path,
        output_path="chm_outputs/temporal_prediction.tif"
    )
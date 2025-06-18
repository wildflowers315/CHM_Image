#!/usr/bin/env python3
"""
Improved temporal training with proper NaN masking
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import rasterio
import numpy as np
import os
import matplotlib.pyplot as plt
from train_temporal_3d_unet import TemporalPatchDataset, Temporal3DUNet

class ImprovedTemporalDataset(TemporalPatchDataset):
    """Improved dataset with proper NaN masking."""
    
    def __getitem__(self, idx):
        patch = self.augmented_patches[idx].copy()
        gedi = self.augmented_gedi[idx].copy() if self.augmented_gedi[idx] is not None else np.zeros((self.patch_size, self.patch_size))
        
        # Organize temporal data AND create availability mask
        temporal_data, availability_mask = self.organize_for_3d_unet_with_mask(patch)
        
        # Clean GEDI data
        original_gedi = gedi.copy()
        gedi = np.nan_to_num(gedi, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Convert to tensors
        features = torch.FloatTensor(temporal_data.copy())
        target = torch.FloatTensor(gedi.copy())
        availability = torch.FloatTensor(availability_mask.copy())
        
        # Create mask for valid GEDI pixels
        gedi_mask = torch.FloatTensor(((~np.isnan(original_gedi)) & (original_gedi > 0)).copy())
        
        return features, target, gedi_mask, availability
    
    def organize_for_3d_unet_with_mask(self, patch_data):
        """Organize temporal data and create availability mask."""
        
        n_months = 12
        channels_per_month = 15
        temporal_array = np.zeros((n_months, channels_per_month, self.patch_size, self.patch_size), dtype=np.float32)
        availability_mask = np.zeros((n_months, channels_per_month, self.patch_size, self.patch_size), dtype=np.float32)
        
        for month_idx in range(1, 13):
            month_str = f"{month_idx:02d}"
            channel_idx = 0
            
            # S1 data
            if month_str in self.s1_bands:
                for pol in ['VV', 'VH']:
                    if pol in self.s1_bands[month_str]:
                        band_idx = self.s1_bands[month_str][pol]
                        band_data = patch_data[band_idx]
                        
                        # Create availability mask (1 where data exists, 0 where NaN)
                        valid_mask = ~np.isnan(band_data)
                        availability_mask[month_idx-1, channel_idx] = valid_mask.astype(np.float32)
                        
                        # Normalize and fill NaN with reasonable defaults
                        band_data = np.where(valid_mask, (band_data + 25) / 25, 0.0)  # S1 normalization
                        temporal_array[month_idx-1, channel_idx] = band_data
                    else:
                        # No data available for this polarization
                        availability_mask[month_idx-1, channel_idx] = 0.0
                    channel_idx += 1
            else:
                channel_idx += 2
            
            # S2 data
            if month_str in self.s2_bands:
                for band in ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12', 'NDVI']:
                    if band in self.s2_bands[month_str]:
                        band_idx = self.s2_bands[month_str][band]
                        band_data = patch_data[band_idx]
                        
                        # Create availability mask
                        valid_mask = ~np.isnan(band_data)
                        availability_mask[month_idx-1, channel_idx] = valid_mask.astype(np.float32)
                        
                        # Normalize based on band type
                        if band != 'NDVI':
                            # S2 reflectance normalization
                            band_data = np.where(valid_mask, np.clip(band_data / 10000.0, 0, 1), 0.0)
                        else:
                            # NDVI normalization
                            band_data = np.where(valid_mask, np.clip(band_data, -1, 1), 0.0)
                        
                        temporal_array[month_idx-1, channel_idx] = band_data
                    else:
                        availability_mask[month_idx-1, channel_idx] = 0.0
                    channel_idx += 1
            else:
                channel_idx += 11
            
            # ALOS2 data
            if month_str in self.alos2_bands:
                for pol in ['HH', 'HV']:
                    if pol in self.alos2_bands[month_str]:
                        band_idx = self.alos2_bands[month_str][pol]
                        band_data = patch_data[band_idx]
                        
                        # Create availability mask
                        valid_mask = ~np.isnan(band_data) & (band_data != 0)  # ALOS2 uses 0 for no data
                        availability_mask[month_idx-1, channel_idx] = valid_mask.astype(np.float32)
                        
                        # Keep as is (already in linear scale)
                        band_data = np.where(valid_mask, band_data, 0.0)
                        temporal_array[month_idx-1, channel_idx] = band_data
                    else:
                        availability_mask[month_idx-1, channel_idx] = 0.0
                    channel_idx += 1
            else:
                channel_idx += 2
        
        return temporal_array, availability_mask

class MaskedTemporalUNet(Temporal3DUNet):
    """3D U-Net that uses availability mask."""
    
    def forward(self, x, availability_mask=None):
        # x shape: (batch, time, channels, height, width)
        # availability_mask: (batch, time, channels, height, width)
        
        # Rearrange to (batch, channels, time, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        if availability_mask is not None:
            availability_mask = availability_mask.permute(0, 2, 1, 3, 4)
            # Apply mask to features
            x = x * availability_mask
        
        # Rest is same as parent forward method
        e1 = self.encoder1(x)
        e2 = self.encoder2(nn.MaxPool3d((1,2,2))(e1))
        e3 = self.encoder3(nn.MaxPool3d((1,2,2))(e2))
        e4 = self.encoder4(nn.MaxPool3d((1,2,2))(e3))
        
        b = self.bottleneck(nn.MaxPool3d((1,2,2))(e4))
        
        d4 = self.decoder4(b)
        d4 = torch.cat([d4, e4], dim=1)
        
        d3 = self.decoder3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        
        d2 = self.decoder2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        
        d1 = self.decoder1(d2)
        
        temporal_features = self.temporal_conv(d1)
        pooled = torch.mean(temporal_features, dim=2)
        out = self.final_conv(pooled)
        
        return out.squeeze(1)

def improved_training():
    """Run improved training with proper masking."""
    
    print("🚀 Starting improved temporal training with NaN masking")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create improved dataset
    patch_path = "chm_outputs/dchm_09gd4_temporal_bandNum196_scale10_patch0000.tif"
    dataset = ImprovedTemporalDataset(patch_path, patch_size=256, augment=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Create model
    model = MaskedTemporalUNet(in_channels=15, n_classes=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    # Training loop
    epochs = 3
    losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (features, target, gedi_mask, availability) in enumerate(dataloader):
            features = features.to(device)
            target = target.to(device)
            gedi_mask = gedi_mask.to(device)
            availability = availability.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with availability mask
            pred = model(features, availability)
            
            # Compute loss only on valid GEDI pixels
            if gedi_mask.sum() > 0:
                valid_pred = pred[gedi_mask > 0]
                valid_target = target[gedi_mask > 0]
                loss = nn.MSELoss()(valid_pred, valid_target)
            else:
                loss = torch.tensor(0.0, requires_grad=True, device=device)
            
            # Backward pass
            if loss.item() > 0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 5 == 0:
                print(f"   Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}: Average Loss = {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'loss': avg_loss
            }, f"chm_outputs/improved_temporal_epoch_{epoch+1}.pth")
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'losses': losses,
        'model_config': {'in_channels': 15, 'n_classes': 1}
    }, "chm_outputs/improved_temporal_final.pth")
    
    # Plot losses
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Improved Temporal Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('chm_outputs/improved_temporal_loss.png')
    print("Training complete!")
    
    return model

def generate_improved_prediction():
    """Generate prediction with improved model."""
    
    print("🔮 Generating improved temporal prediction")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    checkpoint = torch.load("chm_outputs/improved_temporal_final.pth", map_location=device)
    model = MaskedTemporalUNet(in_channels=15, n_classes=1).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load data
    patch_path = "chm_outputs/dchm_09gd4_temporal_bandNum196_scale10_patch0000.tif"
    dataset = ImprovedTemporalDataset(patch_path, patch_size=256, augment=False)
    
    with torch.no_grad():
        features, _, _, availability = dataset[0]
        features = features.unsqueeze(0).to(device)
        availability = availability.unsqueeze(0).to(device)
        
        pred = model(features, availability)
        prediction = pred.cpu().numpy()[0]
    
    # Save prediction
    output_path = "chm_outputs/improved_temporal_prediction.tif"
    with rasterio.open(patch_path) as src:
        profile = src.profile.copy()
        profile.update({'count': 1, 'dtype': 'float32'})
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(prediction, 1)
    
    print(f"✅ Improved prediction saved: {output_path}")
    return prediction

if __name__ == "__main__":
    # Run improved training
    model = improved_training()
    
    # Generate prediction
    prediction = generate_improved_prediction()
    
    print("🎉 Improved temporal training and prediction complete!")
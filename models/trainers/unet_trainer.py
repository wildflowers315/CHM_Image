import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds as transform_from_bounds
from typing import Tuple, Dict, List

from models.height_2d_unet import Height2DUNet
from evaluate_predictions import calculate_metrics
from data.multi_patch import PatchInfo

# Assuming this is defined elsewhere or needs to be moved here
def create_2d_unet(in_channels: int, n_classes: int = 1, base_channels: int = 64):
    """Create 2D U-Net model."""
    return Height2DUNet(in_channels, n_classes, base_channels)

# Assuming this is defined elsewhere or needs to be moved here
def modified_huber_loss(predictions, gedi_tensor, valid_mask, huber_delta, shift_radius):
    # Placeholder for modified_huber_loss, needs to be implemented or imported
    # This was present in the original train_2d_unet function, but not defined.
    # For now, I'll use a simple MSE loss on valid pixels.
    if valid_mask.sum() > 0:
        return nn.MSELoss()(predictions[valid_mask], gedi_tensor[valid_mask])
    else:
        return torch.tensor(0.0, requires_grad=True, device=predictions.device)

def train_2d_unet(features: np.ndarray, gedi_target: np.ndarray, 
                  epochs: int = 50, learning_rate: float = 1e-3, weight_decay: float = 1e-4,
                  base_channels: int = 32, huber_delta: float = 1.0, shift_radius: int = 1) -> Tuple[nn.Module, Dict]:
    """
    Train 2D U-Net model on patch data.
    
    Args:
        features: Feature array [bands, height, width]
        gedi_target: GEDI target array [height, width]
        epochs: Number of training epochs
        learning_rate: Learning rate
        weight_decay: Weight decay
        base_channels: Base channels for U-Net
        huber_delta: Huber loss delta
        shift_radius: Spatial shift radius for GEDI alignment
        
    Returns:
        model: Trained 2D U-Net model
        metrics: Training metrics dictionary
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training 2D U-Net on device: {device}")
    
    # Create model
    model = create_2d_unet(in_channels=features.shape[0], n_classes=1, base_channels=base_channels)
    model = model.to(device)
    
    # Setup optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Convert data to tensors
    features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)  # Add batch dim
    gedi_tensor = torch.FloatTensor(gedi_target).unsqueeze(0).to(device)   # Add batch dim
    
    # Create valid mask for GEDI pixels
    valid_mask = ~torch.isnan(gedi_tensor) & (gedi_tensor > 0)
    
    print(f"Valid GEDI pixels: {valid_mask.sum().item()}/{gedi_tensor.numel()}")
    
    # Training loop
    model.train()
    train_losses = []
    
    for epoch in tqdm(range(epochs), desc="Training 2D U-Net"):
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(features_tensor)  # [1, H, W]
        
        # Calculate modified Huber loss with shift awareness
        loss = modified_huber_loss(predictions, gedi_tensor, valid_mask, huber_delta, shift_radius)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")
    
    # Calculate final metrics on valid GEDI pixels
    model.eval()
    with torch.no_grad():
        final_predictions = model(features_tensor)
        
        # Extract predictions and targets for valid GEDI pixels only
        valid_preds = final_predictions[valid_mask].cpu().numpy()
        valid_targets = gedi_tensor[valid_mask].cpu().numpy()
        
        # Calculate metrics using existing function
        metrics = calculate_metrics(valid_preds, valid_targets)
        metrics['train_loss'] = np.mean(train_losses[-10:])  # Average of last 10 epochs
        metrics['final_loss'] = train_losses[-1]
    
    return model, metrics

def train_multi_patch_unet_reference(patches: List[PatchInfo], args, epochs: int = 100, 
                                   learning_rate: float = 1e-3, weight_decay: float = 1e-4,
                                   base_channels: int = 32, validation_split: float = 0.2) -> Tuple[object, dict]:
    """
    Train 2D U-Net with dense reference height supervision using image-based approach.
    
    This function properly trains U-Net on full image patches (256x256) with dense reference masks,
    instead of the incorrect pixel-based approach.
    
    Args:
        patches: List of patch metadata  
        args: Training arguments containing reference_height_path
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for regularization
        base_channels: Base number of channels in U-Net
        validation_split: Fraction of patches for validation
        
    Returns:
        Tuple of (trained_model, training_metrics)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training 2D U-Net on device: {device}")
    
    # Split patches into train/validation
    n_patches = len(patches)
    n_val_patches = int(n_patches * validation_split)
    n_train_patches = n_patches - n_val_patches
    
    # Random split of patches
    patch_indices = np.random.permutation(n_patches)
    train_patch_indices = patch_indices[:n_train_patches]
    val_patch_indices = patch_indices[n_train_patches:]
    
    train_patches = [patches[i] for i in train_patch_indices]
    val_patches = [patches[i] for i in val_patch_indices]
    
    print(f"Training patches: {len(train_patches)}, Validation patches: {len(val_patches)}")
    
    # Load a sample patch to determine input dimensions
    sample_patch = train_patches[0]
    with rasterio.open(sample_patch.file_path) as src:
        sample_data = src.read()
        
        # Remove GEDI band for reference supervision
        band_names = [src.descriptions[i] or f'band_{i+1}' for i in range(src.count)]
        gedi_band_idx = None
        for i, name in enumerate(band_names):
            if 'rh' in name.lower():
                gedi_band_idx = i
                break
        
        if gedi_band_idx is not None:
            sample_data = np.delete(sample_data, gedi_band_idx, axis=0)
        
        n_input_channels = sample_data.shape[0]
        height, width = sample_data.shape[1], sample_data.shape[2]
        
        # Ensure 256x256 dimensions
        if height > 256 or width > 256:
            height, width = 256, 256
    
    print(f"Input dimensions: {n_input_channels} channels, {height}x{width} spatial")
    
    # Initialize 2D U-Net model  
    model = Height2DUNet(in_channels=n_input_channels, n_classes=1, base_channels=base_channels)
    model = model.to(device)
    
    # Setup optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    
    # Training loop with image-based processing
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in tqdm(range(epochs), desc="Training 2D U-Net on image patches"):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        n_train_batches = 0
        
        for patch_info in train_patches:
            try:
                # Load satellite image patch
                with rasterio.open(patch_info.file_path) as src:
                    patch_data = src.read()  # Shape: (bands, height, width)
                    
                    # Remove GEDI band
                    band_names = [src.descriptions[i] or f'band_{i+1}' for i in range(src.count)]
                    gedi_band_idx = None
                    for i, name in enumerate(band_names):
                        if 'rh' in name.lower():
                            gedi_band_idx = i
                            break
                    
                    if gedi_band_idx is not None:
                        patch_data = np.delete(patch_data, gedi_band_idx, axis=0)
                    
                    # Crop to 256x256 if needed
                    if patch_data.shape[1] > 256 or patch_data.shape[2] > 256:
                        patch_data = patch_data[:, :256, :256]
                    
                    # Load corresponding reference height data
                    
                    with rasterio.open(args.reference_height_path) as ref_src:
                        patch_bounds = patch_info.geospatial_bounds
                        
                        # Create target transform that matches the patch
                        target_transform = transform_from_bounds(
                            patch_bounds[0], patch_bounds[1], 
                            patch_bounds[2], patch_bounds[3], 
                            patch_data.shape[2], patch_data.shape[1]  # width, height
                        )
                        
                        # Create destination array with same dimensions as patch
                        reference_data = np.zeros(patch_data.shape[1:], dtype=np.float32)
                        
                        # Reproject reference data to match patch resolution
                        reproject(
                            source=rasterio.band(ref_src, 1),
                            destination=reference_data,
                            src_transform=ref_src.transform,
                            src_crs=ref_src.crs,
                            dst_transform=target_transform,
                            dst_crs=src.crs,
                            resampling=Resampling.average
                        )
                    
                    # Create valid mask (where we have reference data)
                    valid_mask = (reference_data > 0) & (~np.isnan(reference_data)) & (reference_data < 100)
                    
                    if np.sum(valid_mask) < 100:  # Skip patches with too few reference pixels
                        continue
                    
                    # Convert to tensors
                    patch_features = torch.FloatTensor(patch_data).unsqueeze(0).to(device)  # [1, C, H, W]
                    patch_targets = torch.FloatTensor(reference_data).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, H, W]
                    valid_mask_tensor = torch.FloatTensor(valid_mask).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, H, W]
                    
                    # Replace NaN in features with zeros
                    patch_features = torch.nan_to_num(patch_features, nan=0.0)
                    
                    optimizer.zero_grad()
                    
                    # Forward pass
                    predictions = model(patch_features)  # [1, 1, H, W]
                    
                    # Calculate loss only on valid reference pixels
                    if torch.sum(valid_mask_tensor) > 0:
                        masked_predictions = predictions * valid_mask_tensor
                        masked_targets = patch_targets * valid_mask_tensor
                        
                        # Only compute loss where we have valid reference data
                        loss = criterion(masked_predictions[valid_mask_tensor > 0], 
                                       masked_targets[valid_mask_tensor > 0])
                    else:
                        continue  # Skip if no valid pixels
                    
                    # Backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    epoch_train_loss += loss.item()
                    n_train_batches += 1
                    
            except Exception as e:
                print(f"Error processing training patch {patch_info.patch_id}: {e}")
                continue
        
        avg_train_loss = epoch_train_loss / max(1, n_train_batches)
        train_losses.append(avg_train_loss)
        
        # Validation phase  
        model.eval()
        epoch_val_loss = 0.0
        n_val_batches = 0
        
        with torch.no_grad():
            for patch_info in val_patches:
                try:
                    # Same processing as training but without gradients
                    with rasterio.open(patch_info.file_path) as src:
                        patch_data = src.read()
                        
                        # Remove GEDI band
                        band_names = [src.descriptions[i] or f'band_{i+1}' for i in range(src.count)]
                        gedi_band_idx = None
                        for i, name in enumerate(band_names):
                            if 'rh' in name.lower():
                                gedi_band_idx = i
                                break
                        
                        if gedi_band_idx is not None:
                            patch_data = np.delete(patch_data, gedi_band_idx, axis=0)
                        
                        if patch_data.shape[1] > 256 or patch_data.shape[2] > 256:
                            patch_data = patch_data[:, :256, :256]
                    
                        with rasterio.open(args.reference_height_path) as ref_src:
                            patch_bounds = patch_info.geospatial_bounds
                            target_transform = transform_from_bounds(
                                patch_bounds[0], patch_bounds[1], 
                                patch_bounds[2], patch_bounds[3], 
                                patch_data.shape[2], patch_data.shape[1]
                            )
                            
                            reference_data = np.zeros(patch_data.shape[1:], dtype=np.float32)
                            reproject(
                                source=rasterio.band(ref_src, 1),
                                destination=reference_data,
                                src_transform=ref_src.transform,
                                src_crs=ref_src.crs,
                                dst_transform=target_transform,
                                dst_crs=src.crs,
                                resampling=Resampling.average
                            )
                        
                        valid_mask = (reference_data > 0) & (~np.isnan(reference_data)) & (reference_data < 100)
                        
                        if np.sum(valid_mask) < 100:
                            continue
                        
                        patch_features = torch.FloatTensor(patch_data).unsqueeze(0).to(device)
                        patch_targets = torch.FloatTensor(reference_data).unsqueeze(0).unsqueeze(0).to(device)
                        valid_mask_tensor = torch.FloatTensor(valid_mask).unsqueeze(0).unsqueeze(0).to(device)
                        
                        patch_features = torch.nan_to_num(patch_features, nan=0.0)
                        
                        predictions = model(patch_features)
                        
                        if torch.sum(valid_mask_tensor) > 0:
                            masked_predictions = predictions * valid_mask_tensor
                            masked_targets = patch_targets * valid_mask_tensor
                            loss = criterion(masked_predictions[valid_mask_tensor > 0], 
                                           masked_targets[valid_mask_tensor > 0])
                            epoch_val_loss += loss.item()
                            n_val_batches += 1
                            
                except Exception as e:
                    print(f"Error processing validation patch {patch_info.patch_id}: {e}")
                    continue
        
        avg_val_loss = epoch_val_loss / max(1, n_val_batches)
        val_losses.append(avg_val_loss)
        
        # Track best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
        
        # Print progress
        if epoch % 5 == 0:
            print(f"Epoch {epoch}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    # Calculate final metrics
    metrics = {
        'train_loss': train_losses[-1] if train_losses else 0.0,
        'val_loss': val_losses[-1] if val_losses else 0.0,
        'best_val_loss': best_val_loss,
        'n_train_patches': len(train_patches),
        'n_val_patches': len(val_patches),
        'input_channels': n_input_channels
    }
    
    print(f"✅ Image-based reference height training completed!")
    print(f"📈 Best validation loss: {best_val_loss:.4f}")
    print(f"📊 Training patches: {len(train_patches)}, Validation patches: {len(val_patches)}")
    
    return model, metrics

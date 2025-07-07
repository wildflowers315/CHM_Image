"""
Production training script with data augmentation for Scenario 1 reference-only training.

Uses enhanced patches with pre-processed reference bands + spatial augmentation
for 12x training data increase (flips + rotations).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import rasterio
from typing import List, Tuple
import json
import os
import glob
import argparse
from data.augmentation import SpatialAugmentation, ensure_256x256

class AugmentedEnhancedPatchDataset(Dataset):
    """Enhanced patch dataset with spatial augmentation (12x data increase)."""
    
    def __init__(self, patch_dir: str, patch_pattern: str = "ref_*05LE4*", 
                 patch_size: int = 256, use_augmentation: bool = True):
        self.patch_dir = patch_dir
        self.patch_size = patch_size
        self.use_augmentation = use_augmentation
        
        # Initialize augmentation
        self.augmentor = SpatialAugmentation(augment_factor=12)
        
        # Find all enhanced patches
        self.patch_files = glob.glob(os.path.join(patch_dir, f"{patch_pattern}.tif"))
        print(f"🔍 Found {len(self.patch_files)} enhanced patches")
        
        if len(self.patch_files) == 0:
            raise ValueError(f"No enhanced patches found with pattern {patch_pattern} in {patch_dir}")
        
        # Get reference band index from first patch
        with rasterio.open(self.patch_files[0]) as src:
            descriptions = src.descriptions
            self.reference_band_idx = None
            for i, desc in enumerate(descriptions):
                if desc and 'reference' in desc.lower():
                    self.reference_band_idx = i
                    break
            
            if self.reference_band_idx is None:
                raise ValueError("No reference band found. Use enhanced patches with reference heights.")
            
            self.total_bands = src.count
            self.satellite_bands = self.total_bands - 1
            
        # Calculate total samples with augmentation
        base_samples = len(self.patch_files)
        augment_factor = 12 if self.use_augmentation else 1
        self.total_samples = base_samples * augment_factor
        
        print(f"✅ Enhanced patches ready: {self.total_bands} bands total")
        print(f"📊 Base patches: {base_samples}")
        print(f"🔄 Augmentation factor: {augment_factor}x")
        print(f"📈 Total training samples: {self.total_samples}")
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        # Calculate which base patch and which augmentation
        if self.use_augmentation:
            base_patch_idx = idx // 12
            augment_id = idx % 12
        else:
            base_patch_idx = idx
            augment_id = 0
        
        patch_file = self.patch_files[base_patch_idx]
        
        with rasterio.open(patch_file) as src:
            # Read all bands
            all_data = src.read()  # Shape: (bands, height, width)
            
            # Split into satellite features and reference target
            satellite_data = np.delete(all_data, self.reference_band_idx, axis=0)
            reference_data = all_data[self.reference_band_idx]
            
            # Create valid mask (where we have reference data)
            valid_mask = (reference_data > 0) & (~np.isnan(reference_data)) & (reference_data < 100)
            
            # Ensure 256x256 dimensions
            satellite_data, reference_data, valid_mask = ensure_256x256(
                satellite_data, reference_data, valid_mask
            )
            
            # Apply spatial augmentation
            if self.use_augmentation and augment_id > 0:
                satellite_data, reference_data = self.augmentor.apply_augmentation(
                    satellite_data, reference_data, augment_id
                )
                # Need to recompute mask after augmentation
                valid_mask = (reference_data > 0) & (~np.isnan(reference_data)) & (reference_data < 100)
            
            # Convert to tensors
            patch_features = torch.FloatTensor(satellite_data)  # [C, H, W]
            patch_targets = torch.FloatTensor(reference_data).unsqueeze(0)  # [1, H, W]
            valid_mask_tensor = torch.FloatTensor(valid_mask.astype(np.float32)).unsqueeze(0)  # [1, H, W]
            
            # Replace NaN with zeros
            patch_features = torch.nan_to_num(patch_features, nan=0.0)
            patch_targets = torch.nan_to_num(patch_targets, nan=0.0)
            
            return patch_features, patch_targets, valid_mask_tensor


def train_production_model(patch_dir: str, output_dir: str, epochs: int = 50, 
                          learning_rate: float = 1e-3, batch_size: int = 8,
                          base_channels: int = 64, validation_split: float = 0.2,
                          use_augmentation: bool = True) -> Tuple[object, dict]:
    """
    Train production-quality 2D U-Net with data augmentation.
    """
    from train_predict_map import Height2DUNet, TORCH_AVAILABLE
    
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for U-Net training")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Training production model on device: {device}")
    
    # Create augmented dataset
    full_dataset = AugmentedEnhancedPatchDataset(patch_dir, use_augmentation=use_augmentation)
    
    # Split dataset
    n_total = len(full_dataset)
    n_val = int(n_total * validation_split)
    n_train = n_total - n_val
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val]
    )
    
    # Create data loaders with more workers for augmented data
    num_workers = 6 if use_augmentation else 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=True)
    
    print(f"📊 Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
    print(f"📊 Batch size: {batch_size}, Total samples: {n_total}")
    
    # Get sample to determine dimensions
    sample_features, _, _ = full_dataset[0]
    n_input_channels = sample_features.shape[0]
    
    print(f"📐 Input channels: {n_input_channels}, Spatial: 256x256")
    
    # Initialize model with larger capacity for production
    model = Height2DUNet(in_channels=n_input_channels, n_classes=1, base_channels=base_channels)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🧠 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Setup training with advanced optimizers
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    criterion = nn.MSELoss()
    
    # Learning rate scheduling
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=learning_rate/100
    )
    
    # Training metrics tracking
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10
    
    print(f"🏁 Starting production training for {epochs} epochs...")
    print(f"⚙️  Learning rate: {learning_rate}, Weight decay: 1e-2")
    print(f"📈 Scheduler: CosineAnnealingWarmRestarts, Early stopping patience: {patience}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        n_train_batches = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch_features, batch_targets, batch_masks in train_pbar:
            batch_features = batch_features.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)
            batch_masks = batch_masks.to(device, non_blocking=True)
            
            # Skip batches with no valid pixels
            total_valid_pixels = torch.sum(batch_masks)
            if total_valid_pixels == 0:
                continue
            
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(batch_features)
            
            # Apply mask and compute loss
            masked_predictions = predictions * batch_masks
            masked_targets = batch_targets * batch_masks
            loss = criterion(masked_predictions, masked_targets) * (batch_masks.numel() / total_valid_pixels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_train_loss += loss.item()
            n_train_batches += 1
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        avg_train_loss = epoch_train_loss / max(1, n_train_batches)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        n_val_batches = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for batch_features, batch_targets, batch_masks in val_pbar:
                batch_features = batch_features.to(device, non_blocking=True)
                batch_targets = batch_targets.to(device, non_blocking=True)
                batch_masks = batch_masks.to(device, non_blocking=True)
                
                total_valid_pixels = torch.sum(batch_masks)
                if total_valid_pixels == 0:
                    continue
                
                predictions = model(batch_features)
                
                masked_predictions = predictions * batch_masks
                masked_targets = batch_targets * batch_masks
                loss = criterion(masked_predictions, masked_targets) * (batch_masks.numel() / total_valid_pixels)
                
                epoch_val_loss += loss.item()
                n_val_batches += 1
                
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_val_loss = epoch_val_loss / max(1, n_val_batches)
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step()
        
        # Early stopping and best model saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save best model
            best_model_path = os.path.join(output_dir, 'best_production_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"💾 New best model saved! Val loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
        
        print(f"Epoch {epoch+1}/{epochs} - Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}, "
              f"Best: {best_val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"🛑 Early stopping triggered after {patience} epochs without improvement")
            break
    
    # Save final model
    os.makedirs(output_dir, exist_ok=True)
    final_model_path = os.path.join(output_dir, 'final_production_model.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"💾 Final model saved to: {final_model_path}")
    
    # Training metrics
    train_metrics = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'final_train_loss': train_losses[-1] if train_losses else 0,
        'final_val_loss': val_losses[-1] if val_losses else 0,
        'total_epochs': len(train_losses),
        'early_stopped': patience_counter >= patience,
        'input_channels': n_input_channels,
        'model_parameters': total_params,
        'use_augmentation': use_augmentation,
        'augmentation_factor': 12 if use_augmentation else 1
    }
    
    # Save metrics
    metrics_path = os.path.join(output_dir, 'production_training_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(train_metrics, f, indent=2)
    print(f"📊 Training metrics saved to: {metrics_path}")
    
    return model, train_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Production training with data augmentation')
    parser.add_argument('--patch-dir', default='chm_outputs/enhanced_patches/', 
                       help='Directory containing enhanced patches')
    parser.add_argument('--output-dir', default='chm_outputs/production_results/', 
                       help='Output directory')
    parser.add_argument('--epochs', type=int, default=50, help='Maximum number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--base-channels', type=int, default=64, help='Base channels for U-Net')
    parser.add_argument('--no-augmentation', action='store_true', 
                       help='Disable data augmentation')
    
    args = parser.parse_args()
    
    print("🏭 Production Training with Data Augmentation")
    print(f"📁 Enhanced patches: {args.patch_dir}")
    print(f"📁 Output: {args.output_dir}")
    print(f"🔢 Max epochs: {args.epochs}")
    print(f"📊 Batch size: {args.batch_size}")
    print(f"📈 Learning rate: {args.learning_rate}")
    print(f"🧠 Base channels: {args.base_channels}")
    print(f"🔄 Augmentation: {'Disabled' if args.no_augmentation else 'Enabled (12x)'}")
    print("")
    
    model, metrics = train_production_model(
        patch_dir=args.patch_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        base_channels=args.base_channels,
        use_augmentation=not args.no_augmentation
    )
    
    print("")
    print("🎉 Production training completed!")
    print(f"🏆 Best validation loss: {metrics['best_val_loss']:.4f}")
    print(f"📈 Final training loss: {metrics['final_train_loss']:.4f}")
    print(f"📉 Final validation loss: {metrics['final_val_loss']:.4f}")
    print(f"🔢 Total epochs trained: {metrics['total_epochs']}")
    print(f"🧠 Model parameters: {metrics['model_parameters']:,}")
    print(f"⏹️  Early stopped: {metrics['early_stopped']}")
    
    if metrics['use_augmentation']:
        print(f"🔄 Data augmentation: {metrics['augmentation_factor']}x increase")
        print("   Includes: horizontal/vertical flips + 90°/180°/270° rotations")
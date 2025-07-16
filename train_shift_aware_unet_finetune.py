#!/usr/bin/env python3
"""
Fine-tuning script for shift-aware U-Net models
Supports loading pretrained models and adapting to new regions
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import rasterio
from pathlib import Path
import glob
import json
from tqdm import tqdm
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import existing models and training utilities
try:
    from models.trainers.shift_aware_trainer import ShiftAwareTrainer, auto_find_patches, ShiftAwareUNet
    SHIFT_AWARE_AVAILABLE = True
except ImportError:
    SHIFT_AWARE_AVAILABLE = False
    print("Warning: Shift-aware training not available")

# Import multi-patch functionality
from data.multi_patch import load_multi_patch_gedi_data

class TochigiFinetuneDataset(Dataset):
    """Dataset for fine-tuning on Tochigi region GEDI data - using full patches"""
    
    def __init__(self, patch_dir, shift_radius=2):
        self.patch_dir = Path(patch_dir)
        self.shift_radius = shift_radius
        
        # Find Tochigi patches
        self.patch_files = self._find_tochigi_patches()
        print(f"Found {len(self.patch_files)} Tochigi patches for fine-tuning")
        
        # Load patch data
        self.patch_data = self._load_patch_data()
        print(f"Loaded {len(self.patch_data)} patches for fine-tuning")
    
    def _find_tochigi_patches(self):
        """Find Tochigi patches with GEDI data"""
        pattern = str(self.patch_dir / "*09gd4*bandNum31*.tif")
        patches = glob.glob(pattern)
        return sorted(patches)
    
    def _load_patch_data(self):
        """Load full patch data"""
        all_data = []
        
        for patch_file in tqdm(self.patch_files, desc="Loading patches"):
            try:
                # Load patch
                with rasterio.open(patch_file) as src:
                    patch_data = src.read().astype(np.float32)
                    
                # Extract satellite features (first 30 bands) and GEDI targets (band 31)
                if patch_data.shape[0] >= 31:
                    satellite_features = patch_data[:30]  # (30, H, W)
                    gedi_targets = patch_data[30]  # (H, W)
                    
                    # Check if patch has valid GEDI data
                    valid_mask = (gedi_targets > 0) & (gedi_targets <= 100)
                    if np.sum(valid_mask) > 10:  # At least 10 valid pixels
                        all_data.append({
                            'satellite_features': satellite_features,
                            'gedi_targets': gedi_targets,
                            'patch_file': patch_file
                        })
                        
            except Exception as e:
                print(f"Error loading {patch_file}: {e}")
                continue
        
        return all_data
    
    def __len__(self):
        return len(self.patch_data)
    
    def __getitem__(self, idx):
        sample = self.patch_data[idx]
        
        # Convert to tensors
        satellite_features = torch.FloatTensor(sample['satellite_features'])
        gedi_targets = torch.FloatTensor(sample['gedi_targets'])
        
        return satellite_features, gedi_targets

def load_pretrained_shift_aware_unet(model_path, device):
    """Load pretrained shift-aware U-Net model"""
    print(f"Loading pretrained shift-aware U-Net from: {model_path}")
    
    try:
        # Load checkpoint with weights_only=False 
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Create model
        model = ShiftAwareUNet(
            in_channels=30,
            out_channels=1
        )
        
        # Load state dict
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        
        print("✅ Successfully loaded pretrained shift-aware U-Net")
        return model
        
    except Exception as e:
        print(f"❌ Error loading pretrained model: {e}")
        return None

def fine_tune_unet(model, train_loader, val_loader, device, epochs=30, learning_rate=0.00005):
    """Fine-tune pretrained U-Net with conservative settings using full patches"""
    
    # Conservative optimizer for fine-tuning
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Loss function - MSE for patch-level training
    criterion = nn.MSELoss()
    
    # Training metrics
    train_losses = []
    val_losses = []
    val_r2_scores = []
    best_val_r2 = -float('inf')
    patience_counter = 0
    patience = 10  # Early stopping patience
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_features)
            
            # Handle dimension mismatch - predictions are (batch, 1, H, W), targets are (batch, H, W)
            predictions = predictions.squeeze(1)  # Remove channel dimension: (batch, H, W)
            
            # Calculate loss only on valid GEDI pixels
            valid_mask = (batch_targets > 0) & (batch_targets <= 100)
            if torch.sum(valid_mask) > 0:
                loss = criterion(predictions[valid_mask], batch_targets[valid_mask])
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_val_preds = []
        all_val_targets = []
        
        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)
                
                predictions = model(batch_features)
                
                # Handle dimension mismatch - predictions are (batch, 1, H, W), targets are (batch, H, W)
                predictions = predictions.squeeze(1)  # Remove channel dimension: (batch, H, W)
                
                # Calculate loss only on valid GEDI pixels
                valid_mask = (batch_targets > 0) & (batch_targets <= 100)
                if torch.sum(valid_mask) > 0:
                    loss = criterion(predictions[valid_mask], batch_targets[valid_mask])
                    val_loss += loss.item()
                    
                    # Collect predictions and targets for R² calculation
                    all_val_preds.extend(predictions[valid_mask].cpu().numpy().flatten())
                    all_val_targets.extend(batch_targets[valid_mask].cpu().numpy().flatten())
        
        val_loss /= len(val_loader)
        
        # Calculate R²
        if len(all_val_targets) > 0:
            from sklearn.metrics import r2_score
            val_r2 = r2_score(all_val_targets, all_val_preds)
        else:
            val_r2 = -float('inf')
        
        # Record metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_r2_scores.append(val_r2)
        
        # Update learning rate
        scheduler.step()
        
        # Print progress
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val R² = {val_r2:.4f}, LR = {current_lr:.6f}")
        
        # Early stopping and best model saving
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            patience_counter = 0
            
            # Save best fine-tuned model
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_r2': val_r2,
                'val_loss': val_loss,
                'fine_tuned': True,
                'target_region': 'tochigi'
            }, 'chm_outputs/scenario3_tochigi_unet_adaptation/fine_tuned_unet_tochigi_best.pth')
            
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    return train_losses, val_losses, val_r2_scores, best_val_r2

def main():
    parser = argparse.ArgumentParser(description='Fine-tune shift-aware U-Net for target region')
    parser.add_argument('--patch-dir', default='chm_outputs/enhanced_patches/', help='Directory containing patches')
    parser.add_argument('--output-dir', default='chm_outputs/scenario3_tochigi_unet_adaptation/', help='Output directory')
    parser.add_argument('--pretrained-model-path', required=True, help='Path to pretrained U-Net model')
    parser.add_argument('--epochs', type=int, default=30, help='Number of fine-tuning epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.00005, help='Learning rate for fine-tuning')
    parser.add_argument('--shift-radius', type=int, default=2, help='Shift radius for patch extraction')
    parser.add_argument('--max-samples', type=int, default=5000, help='Max samples per patch')
    
    args = parser.parse_args()
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("🔧 Scenario 3: U-Net Fine-tuning for Target Region Adaptation")
    print(f"📅 Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🖥️  Device: {device}")
    print(f"📁 Patch directory: {args.patch_dir}")
    print(f"📊 Output directory: {args.output_dir}")
    print(f"🔄 Pretrained model: {args.pretrained_model_path}")
    print(f"🎯 Shift radius: {args.shift_radius}")
    
    if not SHIFT_AWARE_AVAILABLE:
        print("❌ Shift-aware training not available. Please check imports.")
        return
    
    try:
        # Load pretrained model
        model = load_pretrained_shift_aware_unet(args.pretrained_model_path, device)
        
        if model is None:
            print("❌ Failed to load pretrained model")
            return
        
        # Load target region dataset
        print(f"📂 Loading Tochigi region dataset for fine-tuning...")
        dataset = TochigiFinetuneDataset(
            patch_dir=args.patch_dir,
            shift_radius=args.shift_radius
        )
        
        if len(dataset) == 0:
            print("❌ No valid data found for fine-tuning")
            return
        
        # Create data loaders
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        
        print(f"🔄 Training samples: {len(train_dataset)}")
        print(f"🔄 Validation samples: {len(val_dataset)}")
        
        # Fine-tune model
        print("🚀 Starting fine-tuning...")
        train_losses, val_losses, val_r2_scores, best_val_r2 = fine_tune_unet(
            model, train_loader, val_loader, device, 
            epochs=args.epochs, learning_rate=args.learning_rate
        )
        
        # Save results
        results = {
            'best_val_r2': float(best_val_r2),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_r2_scores': val_r2_scores,
            'target_region': 'tochigi',
            'shift_radius': args.shift_radius,
            'fine_tuning_completed': True
        }
        
        results_path = os.path.join(args.output_dir, 'unet_fine_tuning_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Fine-tuning completed!")
        print(f"📊 Best validation R²: {best_val_r2:.4f}")
        print(f"💾 Results saved to: {results_path}")
        print(f"🎯 Best model saved to: {args.output_dir}/fine_tuned_unet_tochigi_best.pth")
        
    except Exception as e:
        print(f"❌ Error during fine-tuning: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
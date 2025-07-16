#!/usr/bin/env python3
"""
Fine-tuning script for production MLP models
Supports loading pretrained models and adapting to new regions
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import rasterio
from pathlib import Path
import glob
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, QuantileTransformer
import json
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import logging
from collections import Counter

# Import band utilities
from utils.band_utils import extract_bands_by_name, check_patch_compatibility

class AdvancedReferenceHeightMLP(nn.Module):
    """Advanced MLP with residual connections and attention"""
    
    def __init__(self, input_dim=30, hidden_dims=[1024, 512, 256, 128, 64], 
                 dropout_rate=0.4, use_residuals=True):
        super().__init__()
        
        self.use_residuals = use_residuals
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization
            self.norms.append(nn.BatchNorm1d(hidden_dim))
            
            # Dropout
            dropout_p = dropout_rate * (1 - i / len(hidden_dims))  # Decreasing dropout
            self.dropouts.append(nn.Dropout(dropout_p))
            
            prev_dim = hidden_dim
        
        # Output layers
        self.output_norm = nn.BatchNorm1d(prev_dim)
        self.output = nn.Linear(prev_dim, 1)
        
        # Feature attention mechanism
        self.feature_attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim),
            nn.Sigmoid()
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Advanced weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Feature attention
        attention_weights = self.feature_attention(x)
        x = x * attention_weights
        
        # Forward through layers with residuals
        for i, (layer, norm, dropout) in enumerate(zip(self.layers, self.norms, self.dropouts)):
            residual = x if self.use_residuals and layer.in_features == layer.out_features else None
            
            x = layer(x)
            x = norm(x)
            x = torch.relu(x)
            x = dropout(x)
            
            # Add residual connection if applicable
            if residual is not None:
                x = x + residual
        
        # Output
        x = self.output_norm(x)
        x = self.output(x)
        
        return x.squeeze(-1)

class ProductionReferenceDataset(Dataset):
    """Production dataset for reference height training with fine-tuning support"""
    
    def __init__(self, patch_dir, reference_tif_path, max_samples_per_patch=10000, 
                 augment_factor=3, supervision_mode='reference', pretrained_scaler=None):
        self.patch_dir = Path(patch_dir)
        self.reference_tif_path = reference_tif_path
        self.max_samples_per_patch = max_samples_per_patch
        self.augment_factor = augment_factor
        self.supervision_mode = supervision_mode
        self.pretrained_scaler = pretrained_scaler
        
        # Load patches
        self.patch_files = self._find_patches()
        print(f"Found {len(self.patch_files)} patches")
        
        # Load and process data
        self.features, self.targets = self._load_all_data()
        
        # Create scaler (or use pretrained one)
        if pretrained_scaler is not None:
            print("Using pretrained scaler for consistency")
            self.scaler = pretrained_scaler
        else:
            print("Creating new scaler")
            self.scaler = QuantileTransformer(n_quantiles=1000, random_state=42)
            self.scaler.fit(self.features)
        
        # Transform features and handle any remaining NaN values
        self.features = self.scaler.transform(self.features)
        
        # Final NaN check after transformation
        nan_mask = np.isnan(self.features).any(axis=1) | np.isnan(self.targets)
        if np.any(nan_mask):
            print(f"Removing {np.sum(nan_mask)} samples with NaN values after scaling")
            self.features = self.features[~nan_mask]
            self.targets = self.targets[~nan_mask]
        
        # Set feature selection (30 satellite bands)
        self.selected_features = np.ones(30, dtype=bool)
        
        print(f"Dataset loaded: {len(self.features)} samples, {self.supervision_mode} supervision")
    
    def _find_patches(self):
        """Find enhanced patches in directory - only those with GEDI data"""
        if self.supervision_mode == 'gedi_only':
            # Only look for patches with GEDI data (bandNum31)
            pattern = str(self.patch_dir / "ref_*09gd4*bandNum31*.tif")  # Tochigi patches with GEDI
        else:
            # All Tochigi patches
            pattern = str(self.patch_dir / "ref_*09gd4*.tif")  # Tochigi patches
        patches = glob.glob(pattern)
        return sorted(patches)
    
    def _load_all_data(self):
        """Load data from all patches"""
        all_features = []
        all_targets = []
        
        for patch_file in tqdm(self.patch_files, desc="Loading patches"):
            features, targets = self._load_patch_data(patch_file)
            if features is not None and len(features) > 0:
                all_features.append(features)
                all_targets.append(targets)
        
        if not all_features:
            raise ValueError("No valid data found in patches")
        
        features = np.vstack(all_features)
        targets = np.concatenate(all_targets)
        
        return features, targets
    
    def _load_patch_data(self, patch_file):
        """Load data from a single patch"""
        try:
            # Extract bands using band utilities
            satellite_features, supervision_target = extract_bands_by_name(
                patch_file, self.supervision_mode
            )
            
            # Reshape to (pixels, bands)
            H, W = satellite_features.shape[1], satellite_features.shape[2]
            satellite_features = satellite_features.reshape(30, -1).T  # (pixels, 30)
            supervision_target = supervision_target.flatten()
            
            # Filter valid pixels (remove NaN and invalid values)
            if self.supervision_mode == 'reference':
                valid_mask = (supervision_target > 0) & (supervision_target <= 100) & (~np.isnan(supervision_target))
            else:  # gedi_only
                valid_mask = (supervision_target > 0) & (supervision_target <= 100) & (~np.isnan(supervision_target))
            
            # Also remove pixels with NaN in satellite features
            satellite_nan_mask = ~np.isnan(satellite_features).any(axis=1)
            valid_mask = valid_mask & satellite_nan_mask
            
            if not np.any(valid_mask):
                return None, None
            
            valid_features = satellite_features[valid_mask]
            valid_targets = supervision_target[valid_mask]
            
            # Double-check for NaN values
            if np.isnan(valid_features).any() or np.isnan(valid_targets).any():
                print(f"Warning: NaN values found in {patch_file}, skipping")
                return None, None
            
            # Sample if too many pixels
            if len(valid_features) > self.max_samples_per_patch:
                indices = np.random.choice(len(valid_features), self.max_samples_per_patch, replace=False)
                valid_features = valid_features[indices]
                valid_targets = valid_targets[indices]
            
            # Data augmentation for minority classes
            if self.supervision_mode == 'reference':
                valid_features, valid_targets = self._augment_data(valid_features, valid_targets)
            
            return valid_features, valid_targets
            
        except Exception as e:
            print(f"Error loading {patch_file}: {e}")
            return None, None
    
    def _augment_data(self, features, targets):
        """Augment data for minority height classes"""
        # Height stratification
        height_bins = np.array([0, 5, 10, 15, 20, 25, 100])
        bin_indices = np.digitize(targets, height_bins)
        
        augmented_features = [features]
        augmented_targets = [targets]
        
        # Count samples per bin
        bin_counts = Counter(bin_indices)
        max_count = max(bin_counts.values())
        
        # Augment minority classes
        for bin_idx, count in bin_counts.items():
            if count < max_count / 2:  # Minority class
                mask = bin_indices == bin_idx
                minority_features = features[mask]
                minority_targets = targets[mask]
                
                # Add noise for augmentation
                for _ in range(self.augment_factor):
                    noise = np.random.normal(0, 0.05, minority_features.shape)
                    augmented_features.append(minority_features + noise)
                    augmented_targets.append(minority_targets)
        
        return np.vstack(augmented_features), np.concatenate(augmented_targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), torch.FloatTensor([self.targets[idx]])

def load_pretrained_model(model_path, device):
    """Load pretrained model with proper handling"""
    print(f"Loading pretrained model from: {model_path}")
    
    try:
        # Load with weights_only=False to handle sklearn objects
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model_state_dict = checkpoint['model_state_dict']
                scaler = checkpoint.get('scaler', None)
                supervision_mode = checkpoint.get('supervision_mode', 'reference')
            else:
                model_state_dict = checkpoint
                scaler = None
                supervision_mode = 'reference'
        else:
            model_state_dict = checkpoint
            scaler = None
            supervision_mode = 'reference'
        
        # Create model with same architecture
        model = AdvancedReferenceHeightMLP(
            input_dim=30,
            hidden_dims=[1024, 512, 256, 128, 64],
            dropout_rate=0.4
        )
        
        # Load state dict
        model.load_state_dict(model_state_dict)
        model.to(device)
        
        print(f"✅ Successfully loaded pretrained model (supervision: {supervision_mode})")
        return model, scaler, supervision_mode
        
    except Exception as e:
        print(f"❌ Error loading pretrained model: {e}")
        return None, None, None

def fine_tune_model(model, train_loader, val_loader, device, epochs=50, learning_rate=0.0001):
    """Fine-tune pretrained model with conservative settings"""
    
    # Conservative optimizer settings for fine-tuning
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Weighted Huber loss
    def weighted_huber_loss(predictions, targets, delta=1.0):
        residual = torch.abs(predictions - targets)
        weights = 1.0 / (1.0 + targets)  # Higher weights for lower values
        
        mask = residual <= delta
        loss = torch.where(
            mask,
            0.5 * weights * residual ** 2,
            weights * (delta * residual - 0.5 * delta ** 2)
        )
        return loss.mean()
    
    # Training loop with metrics tracking
    train_losses = []
    val_losses = []
    val_r2_scores = []
    best_val_r2 = -float('inf')
    patience_counter = 0
    patience = 15  # Early stopping patience
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device).squeeze()
            
            optimizer.zero_grad()
            predictions = model(batch_features)
            loss = weighted_huber_loss(predictions, batch_targets)
            loss.backward()
            
            # Gradient clipping for stability
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
                batch_targets = batch_targets.to(device).squeeze()
                
                predictions = model(batch_features)
                loss = weighted_huber_loss(predictions, batch_targets)
                val_loss += loss.item()
                
                all_val_preds.extend(predictions.cpu().numpy())
                all_val_targets.extend(batch_targets.cpu().numpy())
        
        val_loss /= len(val_loader)
        
        # Calculate R²
        from sklearn.metrics import r2_score
        val_r2 = r2_score(all_val_targets, all_val_preds)
        
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
                'original_supervision_mode': 'gedi_only'
            }, 'chm_outputs/scenario3_tochigi_mlp_adaptation/fine_tuned_gedi_mlp_best.pth')
            
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    return train_losses, val_losses, val_r2_scores, best_val_r2

def main():
    parser = argparse.ArgumentParser(description='Fine-tune production MLP for target region')
    parser.add_argument('--patch-dir', default='chm_outputs/enhanced_patches/', help='Directory containing patches')
    parser.add_argument('--reference-tif', default='downloads/dchm_09gd4.tif', help='Reference height TIF')
    parser.add_argument('--output-dir', default='chm_outputs/scenario3_tochigi_mlp_adaptation/', help='Output directory')
    parser.add_argument('--pretrained-model-path', required=True, help='Path to pretrained model')
    parser.add_argument('--supervision-mode', choices=['reference', 'gedi_only'], default='gedi_only', 
                       help='Supervision mode for fine-tuning')
    parser.add_argument('--epochs', type=int, default=50, help='Number of fine-tuning epochs')
    parser.add_argument('--batch-size', type=int, default=1024, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.0001, help='Learning rate for fine-tuning')
    parser.add_argument('--max-samples', type=int, default=50000, help='Max samples per patch')
    
    args = parser.parse_args()
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("🔧 Scenario 3: MLP Fine-tuning for Target Region Adaptation")
    print(f"📅 Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🖥️  Device: {device}")
    print(f"📁 Patch directory: {args.patch_dir}")
    print(f"📊 Output directory: {args.output_dir}")
    print(f"🎯 Supervision mode: {args.supervision_mode}")
    print(f"🔄 Pretrained model: {args.pretrained_model_path}")
    
    try:
        # Load pretrained model
        model, pretrained_scaler, original_supervision_mode = load_pretrained_model(
            args.pretrained_model_path, device
        )
        
        if model is None:
            print("❌ Failed to load pretrained model")
            return
        
        # Load target region dataset
        print(f"📂 Loading target region dataset with {args.supervision_mode} supervision...")
        dataset = ProductionReferenceDataset(
            patch_dir=args.patch_dir,
            reference_tif_path=args.reference_tif,
            max_samples_per_patch=args.max_samples,
            augment_factor=1,  # Minimal augmentation for fine-tuning
            supervision_mode=args.supervision_mode,
            pretrained_scaler=pretrained_scaler  # Use pretrained scaler
        )
        
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
        train_losses, val_losses, val_r2_scores, best_val_r2 = fine_tune_model(
            model, train_loader, val_loader, device, 
            epochs=args.epochs, learning_rate=args.learning_rate
        )
        
        # Save results
        results = {
            'best_val_r2': float(best_val_r2),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_r2_scores': val_r2_scores,
            'original_supervision_mode': original_supervision_mode,
            'fine_tune_supervision_mode': args.supervision_mode,
            'target_region': 'tochigi',
            'fine_tuning_completed': True
        }
        
        results_path = os.path.join(args.output_dir, 'fine_tuning_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Fine-tuning completed!")
        print(f"📊 Best validation R²: {best_val_r2:.4f}")
        print(f"💾 Results saved to: {results_path}")
        print(f"🎯 Best model saved to: {args.output_dir}/fine_tuned_gedi_mlp_best.pth")
        
    except Exception as e:
        print(f"❌ Error during fine-tuning: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Production-scale MLP training for reference height prediction
Optimized for sparse supervision with advanced techniques
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
    """Production dataset with advanced preprocessing and augmentation"""
    
    def __init__(self, patch_dir: str, reference_tif_path: str, patch_pattern: str = "*05LE4*",
                 max_samples_per_patch: int = 100000, min_height: float = 0.0, max_height: float = 100.0,
                 use_height_stratification: bool = True, augment_factor: int = 3, supervision_mode: str = "reference"):
        
        self.patch_dir = Path(patch_dir)
        self.reference_tif_path = reference_tif_path
        self.patch_pattern = patch_pattern
        self.max_samples_per_patch = max_samples_per_patch
        self.min_height = min_height
        self.max_height = max_height
        self.use_height_stratification = use_height_stratification
        self.augment_factor = augment_factor
        self.supervision_mode = supervision_mode  # "reference" or "gedi_only"
        
        # Find all patches
        self.patch_files = self._find_patches()
        print(f"Found {len(self.patch_files)} patches for training")
        
        # Load all data
        self.features, self.targets, self.height_bins = self._load_all_data()
        print(f"Loaded {len(self.features)} training samples with {supervision_mode} supervision")
        
        # Advanced preprocessing
        self._preprocess_features()
        
    def _find_patches(self):
        """Find all available patches compatible with supervision mode"""
        pattern = str(self.patch_dir / self.patch_pattern)
        all_files = glob.glob(pattern)
        
        # Filter patches based on compatibility with supervision mode
        compatible_files = []
        skipped_count = 0
        
        for file_path in all_files:
            if check_patch_compatibility(file_path, self.supervision_mode):
                compatible_files.append(file_path)
            else:
                skipped_count += 1
        
        if skipped_count > 0:
            print(f"⚠️  Skipped {skipped_count} patches incompatible with {self.supervision_mode} supervision")
        
        return sorted(compatible_files)
    
    def _load_all_data(self):
        """Load and preprocess all training data"""
        all_features = []
        all_targets = []
        height_bins = []
        
        for patch_file in tqdm(self.patch_files, desc="Loading patches"):
            try:
                features, targets = self._extract_patch_data(patch_file)
                if len(features) > 0:
                    all_features.append(features)
                    all_targets.append(targets)
                    
                    # Create height bins for stratification
                    bins = np.digitize(targets, bins=[0, 5, 10, 15, 20, 30, 50, 100])
                    height_bins.extend(bins.tolist())
                    
            except Exception as e:
                print(f"Error loading {patch_file}: {e}")
                continue
        
        if not all_features:
            raise ValueError("No valid training data found")
        
        # Combine data
        features = np.vstack(all_features)
        targets = np.concatenate(all_targets)
        
        # Data augmentation for minority classes
        if self.augment_factor > 1:
            features, targets, height_bins = self._augment_data(features, targets, height_bins)
        
        # Ensure height_bins is a proper numpy array
        height_bins = np.array(height_bins, dtype=int)
        
        return features, targets, height_bins
    
    def _extract_patch_data(self, patch_file):
        """Extract training data from a single patch using band utilities"""
        try:
            # Extract bands by name using utility function
            satellite_bands, target_band = extract_bands_by_name(patch_file, self.supervision_mode)
        except Exception as e:
            print(f"⚠️  Error extracting bands from {patch_file}: {e}")
            return np.array([]), np.array([])
        
        # Advanced NaN handling for satellite bands
        satellite_bands = np.nan_to_num(satellite_bands, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Get valid target pixels
        valid_mask = (~np.isnan(target_band)) & \
                    (target_band >= self.min_height) & \
                    (target_band <= self.max_height)
        
        if not np.any(valid_mask):
            return np.array([]), np.array([])
        
        # Extract pixels
        satellite_pixels = satellite_bands.reshape(satellite_bands.shape[0], -1).T
        target_pixels = target_band.flatten()
        
        # Apply mask
        valid_pixels = valid_mask.flatten()
        satellite_pixels = satellite_pixels[valid_pixels]
        target_pixels = target_pixels[valid_pixels]
        
        # Height-stratified sampling
        if self.use_height_stratification and len(satellite_pixels) > self.max_samples_per_patch:
            indices = self._stratified_sample(target_pixels, self.max_samples_per_patch)
            satellite_pixels = satellite_pixels[indices]
            target_pixels = target_pixels[indices]
        elif len(satellite_pixels) > self.max_samples_per_patch:
            indices = np.random.choice(len(satellite_pixels), self.max_samples_per_patch, replace=False)
            satellite_pixels = satellite_pixels[indices]
            target_pixels = target_pixels[indices]
        
        return satellite_pixels.astype(np.float32), target_pixels.astype(np.float32)
    
    def _stratified_sample(self, heights, n_samples):
        """Stratified sampling to ensure representation across height ranges"""
        height_bins = np.digitize(heights, bins=[0, 5, 10, 15, 20, 30, 50, 100])
        
        # Calculate samples per bin
        unique_bins, bin_counts = np.unique(height_bins, return_counts=True)
        
        indices = []
        samples_per_bin = max(1, n_samples // len(unique_bins))
        
        for bin_val in unique_bins:
            bin_indices = np.where(height_bins == bin_val)[0]
            n_from_bin = min(samples_per_bin, len(bin_indices))
            selected = np.random.choice(bin_indices, n_from_bin, replace=False)
            indices.extend(selected)
        
        # Fill remaining samples randomly
        remaining = n_samples - len(indices)
        if remaining > 0:
            all_indices = set(range(len(heights)))
            available = list(all_indices - set(indices))
            if len(available) >= remaining:
                additional = np.random.choice(available, remaining, replace=False)
                indices.extend(additional)
        
        return np.array(indices[:n_samples])
    
    def _augment_data(self, features, targets, height_bins):
        """Data augmentation focusing on minority height classes"""
        # Identify minority classes (tall trees)
        bin_counts = Counter(height_bins)
        total_samples = len(targets)
        
        augmented_features = [features]
        augmented_targets = [targets]
        augmented_bins = list(height_bins)  # Convert to list for extension
        
        for bin_val, count in bin_counts.items():
            if bin_val >= 5 and count < total_samples * 0.1:  # Minority class
                bin_mask = np.array(height_bins) == bin_val
                bin_features = features[bin_mask]
                bin_targets = targets[bin_mask]
                
                # Add noise augmentation for minority samples
                for _ in range(self.augment_factor):
                    noise_scale = 0.05  # 5% noise
                    augmented_feat = bin_features + np.random.normal(0, noise_scale, bin_features.shape)
                    augmented_features.append(augmented_feat)
                    augmented_targets.append(bin_targets)
                    augmented_bins.extend([bin_val] * len(bin_targets))
        
        return (np.vstack(augmented_features), 
                np.concatenate(augmented_targets),
                augmented_bins)
    
    def _preprocess_features(self):
        """Advanced feature preprocessing"""
        # Robust scaling
        self.scaler = QuantileTransformer(output_distribution='normal', random_state=42)
        self.features = self.scaler.fit_transform(self.features).astype(np.float32)
        
        # Feature selection based on variance
        feature_vars = np.var(self.features, axis=0)
        self.selected_features = feature_vars > 0.01  # Remove low-variance features
        self.features = self.features[:, self.selected_features]
        
        # Ensure targets are float32
        self.targets = self.targets.astype(np.float32)
        
        print(f"Selected {np.sum(self.selected_features)}/30 features after variance filtering")
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.features[idx]).float(), torch.from_numpy(np.array([self.targets[idx]])).float()


def create_weighted_sampler(targets, height_bins):
    """Create weighted sampler to balance height classes"""
    bin_counts = Counter(height_bins)
    total_samples = len(targets)
    
    weights = []
    for bin_val in height_bins:
        # Inverse frequency weighting
        weight = total_samples / (len(bin_counts) * bin_counts[bin_val])
        weights.append(weight)
    
    return WeightedRandomSampler(weights, len(weights), replacement=True)


def train_production_mlp(dataset, model, device, epochs=200, batch_size=2048, 
                        learning_rate=0.001, use_weighted_sampling=True):
    """Advanced training with multiple techniques"""
    
    # Simple train-test split (avoid stratified due to data structure complexity)
    from sklearn.model_selection import train_test_split
    train_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
    
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Create samplers (disable for now due to complexity)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Advanced optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01, 
                           betas=(0.9, 0.999), eps=1e-8)
    
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate*5, 
                                             steps_per_epoch=len(train_loader), 
                                             epochs=epochs)
    
    # Advanced loss function (Huber with height-dependent weighting)
    def weighted_huber_loss(pred, target):
        # Higher weight for tall trees (important minority class)
        weights = 1.0 + 0.1 * (target / 30.0)  # Increasing weight with height
        huber = nn.SmoothL1Loss(reduction='none')(pred, target)
        return torch.mean(weights * huber)
    
    # Training metrics
    train_losses = []
    val_losses = []
    val_r2_scores = []
    best_val_r2 = -float('inf')
    patience = 25
    patience_counter = 0
    
    print(f"Starting advanced training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_features, batch_targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device).squeeze()
            
            optimizer.zero_grad()
            
            predictions = model(batch_features)
            loss = weighted_huber_loss(predictions, batch_targets)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
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
        
        # Print progress
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val R² = {val_r2:.4f}, LR = {current_lr:.6f}")
        
        # Early stopping and best model saving
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            patience_counter = 0
            
            # Save best model with supervision mode in filename
            model_filename = f'chm_outputs/production_mlp_{dataset.supervision_mode}_best.pth'
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_r2': val_r2,
                'val_loss': val_loss,
                'scaler': dataset.scaler,
                'selected_features': dataset.selected_features,
                'supervision_mode': dataset.supervision_mode
            }, model_filename)
            
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    return train_losses, val_losses, val_r2_scores, best_val_r2


def main():
    parser = argparse.ArgumentParser(description='Production MLP training for reference heights')
    parser.add_argument('--patch-dir', default='chm_outputs/enhanced_patches/', help='Directory containing patches')
    parser.add_argument('--reference-tif', default='downloads/dchm_05LE4.tif', help='Reference height TIF')
    parser.add_argument('--output-dir', default='chm_outputs/production_mlp_results/', help='Output directory')
    parser.add_argument('--supervision-mode', choices=['reference', 'gedi_only'], default='reference', 
                       help='Supervision mode: reference (dense) or gedi_only (sparse)')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=2048, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--max-samples', type=int, default=100000, help='Max samples per patch')
    parser.add_argument('--augment-factor', type=int, default=3, help='Data augmentation factor')
    
    args = parser.parse_args()
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("🚀 Production MLP Training for Reference Heights")
    print(f"📅 Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🖥️  Device: {device}")
    print(f"📁 Patch directory: {args.patch_dir}")
    print(f"📊 Output directory: {args.output_dir}")
    print(f"🎯 Supervision mode: {args.supervision_mode}")
    
    try:
        # Load dataset
        print(f"📂 Loading production dataset with {args.supervision_mode} supervision...")
        dataset = ProductionReferenceDataset(
            patch_dir=args.patch_dir,
            reference_tif_path=args.reference_tif,
            max_samples_per_patch=args.max_samples,
            augment_factor=args.augment_factor,
            supervision_mode=args.supervision_mode
        )
        
        # Create model
        print("🧠 Creating advanced MLP model...")
        input_dim = np.sum(dataset.selected_features)
        model = AdvancedReferenceHeightMLP(
            input_dim=input_dim,
            hidden_dims=[1024, 512, 256, 128, 64],
            dropout_rate=0.4
        ).to(device)
        
        print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Train model
        print("🚀 Starting production training...")
        train_losses, val_losses, val_r2_scores, best_val_r2 = train_production_mlp(
            dataset=dataset,
            model=model,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
        
        print(f"\n🎉 Training completed!")
        print(f"🎯 Supervision mode: {args.supervision_mode}")
        print(f"🏆 Best validation R²: {best_val_r2:.4f}")
        print(f"📈 Improvement over U-Net: {best_val_r2 - 0.074:+.4f}")
        
        if best_val_r2 > 0.5:
            print("🏆 EXCELLENT performance achieved!")
        elif best_val_r2 > 0.3:
            print("👍 GOOD performance achieved!")
        else:
            print("📈 Moderate improvement - consider further optimization")
        
        # Save model path info
        model_path = f'chm_outputs/production_mlp_{args.supervision_mode}_best.pth'
        print(f"💾 Best model saved to: {model_path}")
        
        # Save training results
        results = {
            'supervision_mode': args.supervision_mode,
            'best_val_r2': float(best_val_r2),
            'improvement_over_unet': float(best_val_r2 - 0.074),
            'train_losses': [float(x) for x in train_losses],
            'val_losses': [float(x) for x in val_losses],
            'val_r2_scores': [float(x) for x in val_r2_scores],
            'total_samples': int(len(dataset)),
            'input_features': int(input_dim),
            'model_path': model_path
        }
        
        results_filename = f'production_mlp_{args.supervision_mode}_results.json'
        with open(os.path.join(args.output_dir, results_filename), 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"📁 Results saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
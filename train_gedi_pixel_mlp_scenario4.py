#!/usr/bin/env python3
"""
GEDI Pixel-level MLP training for Scenario 4 (no filter)
Based on train_production_mlp.py but adapted for GEDI CSV pixel data
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import json
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import logging
from collections import Counter
import gc

class AdvancedGEDIMLP(nn.Module):
    """Advanced MLP with residual connections and attention for GEDI pixel data"""
    
    def __init__(self, input_dim=86, hidden_dims=[1024, 512, 256, 128, 64], 
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


class GEDIPixelDataset(Dataset):
    """GEDI pixel dataset from CSV files with reference heights"""
    
    def __init__(self, csv_dir: str, scenario: str = "scenario4", 
                 min_height: float = 0.1, max_height: float = 100.0,
                 band_selection: str = "embedding", max_samples: int = None):
        
        self.csv_dir = Path(csv_dir)
        self.scenario = scenario
        self.min_height = min_height
        self.max_height = max_height
        self.band_selection = band_selection
        self.max_samples = max_samples
        
        print(f"🔬 Loading GEDI pixel dataset for {scenario}")
        print(f"📂 CSV directory: {csv_dir}")
        print(f"🎵 Band selection: {band_selection}")
        
        # Load all CSV data
        self.features, self.targets, self.metadata = self._load_all_csv_data()
        print(f"📊 Loaded {len(self.features)} GEDI pixels")
        
        # Preprocess features
        self._preprocess_features()
        
    def _load_all_csv_data(self):
        """Load and combine all GEDI CSV files with reference heights"""
        csv_pattern = str(self.csv_dir / "*_with_reference.csv")
        csv_files = glob.glob(csv_pattern)
        
        if not csv_files:
            raise ValueError(f"No CSV files found matching pattern: {csv_pattern}")
        
        print(f"Found {len(csv_files)} CSV files:")
        for f in csv_files:
            print(f"  - {Path(f).name}")
        
        all_features = []
        all_targets = []
        all_metadata = []
        
        for csv_file in tqdm(csv_files, desc="Loading CSV files"):
            df = pd.read_csv(csv_file)
            
            # Filter valid data (focus on GEDI rh as target)
            valid_mask = (
                ~df['rh'].isna() &
                (df['rh'] > self.min_height) &
                (df['rh'] <= self.max_height) &
                ~df['reference_height'].isna()  # Keep reference for validation
            )
            
            df_valid = df[valid_mask].copy()
            
            if len(df_valid) == 0:
                print(f"⚠️  No valid data in {Path(csv_file).name}")
                continue
            
            # Extract features based on band selection
            features = self._extract_features(df_valid)
            targets = df_valid['rh'].values  # GEDI height quantile as target
            
            # Store metadata
            metadata = {
                'source_file': Path(csv_file).name,
                'region': self._extract_region_from_filename(Path(csv_file).name),
                'lat': df_valid['lat'].values if 'lat' in df_valid.columns else None,
                'lon': df_valid['lon'].values if 'lon' in df_valid.columns else None
            }
            
            all_features.append(features)
            all_targets.append(targets)
            all_metadata.append(metadata)
            
            print(f"  {Path(csv_file).name}: {len(df_valid):,} valid pixels")
        
        if not all_features:
            raise ValueError("No valid training data found in any CSV file")
        
        # Combine all data
        combined_features = np.vstack(all_features)
        combined_targets = np.concatenate(all_targets)
        
        # Sample if too large
        if self.max_samples and len(combined_features) > self.max_samples:
            indices = np.random.choice(len(combined_features), self.max_samples, replace=False)
            combined_features = combined_features[indices]
            combined_targets = combined_targets[indices]
            print(f"🎲 Randomly sampled {self.max_samples:,} pixels from {len(combined_features):,} total")
        
        return combined_features, combined_targets, all_metadata
    
    def _extract_features(self, df):
        """Extract Google Embedding features only (A00-A63)"""
        # Google Embedding v1 (64 bands): A00-A63 only
        feature_cols = [f'A{i:02d}' for i in range(64)]
        
        # Check available columns
        available_cols = [col for col in feature_cols if col in df.columns]
        missing_cols = [col for col in feature_cols if col not in df.columns]
        
        if missing_cols:
            print(f"⚠️  Missing Google Embedding columns: {missing_cols}")
            
        if not available_cols:
            raise ValueError("No Google Embedding columns (A00-A63) found in data")
        
        if len(available_cols) != 64:
            print(f"⚠️  Expected 64 Google Embedding bands, found {len(available_cols)}")
        
        print(f"Using {len(available_cols)} Google Embedding features (A00-A63)")
        
        # Extract and clean features
        features = df[available_cols].values
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features.astype(np.float32)
    
    def _extract_region_from_filename(self, filename):
        """Extract region name from CSV filename"""
        if 'dchm_04hf3' in filename:
            return 'kochi'
        elif 'dchm_05LE4' in filename:
            return 'hyogo'  
        elif 'dchm_09gd4' in filename:
            return 'tochigi'
        else:
            return 'unknown'
    
    def _preprocess_features(self):
        """Advanced feature preprocessing"""
        print("🔧 Preprocessing features...")
        
        # Robust scaling
        self.scaler = QuantileTransformer(output_distribution='normal', random_state=42)
        self.features = self.scaler.fit_transform(self.features).astype(np.float32)
        
        # Feature selection based on variance
        feature_vars = np.var(self.features, axis=0)
        self.selected_features = feature_vars > 0.01  # Remove low-variance features
        self.features = self.features[:, self.selected_features]
        
        # Ensure targets are float32
        self.targets = self.targets.astype(np.float32)
        
        n_selected = np.sum(self.selected_features)
        n_total = len(self.selected_features)
        print(f"✅ Selected {n_selected}/{n_total} features after variance filtering")
        
        # Clean up memory
        gc.collect()
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.features[idx]).float(), torch.tensor(self.targets[idx]).float()


def train_gedi_mlp(dataset, model, device, epochs=200, batch_size=1024, 
                   learning_rate=0.001, output_dir="chm_outputs"):
    """Advanced training for GEDI pixel MLP"""
    
    # Train-validation split
    train_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
    
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    
    print(f"📊 Training samples: {len(train_dataset):,}")
    print(f"📊 Validation samples: {len(val_dataset):,}")
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Advanced optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
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
    
    print(f"🚀 Starting GEDI MLP training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_features, batch_targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            
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
                batch_targets = batch_targets.to(device)
                
                predictions = model(batch_features)
                loss = weighted_huber_loss(predictions, batch_targets)
                val_loss += loss.item()
                
                all_val_preds.extend(predictions.cpu().numpy())
                all_val_targets.extend(batch_targets.cpu().numpy())
        
        val_loss /= len(val_loader)
        
        # Calculate metrics
        val_r2 = r2_score(all_val_targets, all_val_preds)
        val_rmse = np.sqrt(mean_squared_error(all_val_targets, all_val_preds))
        val_mae = mean_absolute_error(all_val_targets, all_val_preds)
        
        # Record metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_r2_scores.append(val_r2)
        
        # Print progress
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, "
              f"Val R² = {val_r2:.4f}, RMSE = {val_rmse:.2f}m, MAE = {val_mae:.2f}m, LR = {current_lr:.6f}")
        
        # Early stopping and best model saving
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            patience_counter = 0
            
            # Save best model
            model_filename = f'{output_dir}/gedi_pixel_mlp_{dataset.scenario}_{dataset.band_selection}_best.pth'
            os.makedirs(output_dir, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_r2': val_r2,
                'val_loss': val_loss,
                'val_rmse': val_rmse,
                'val_mae': val_mae,
                'scaler': dataset.scaler,
                'selected_features': dataset.selected_features,
                'scenario': dataset.scenario,
                'band_selection': dataset.band_selection
            }, model_filename)
            
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"⏹️  Early stopping triggered after {epoch+1} epochs")
            break
    
    return train_losses, val_losses, val_r2_scores, best_val_r2


def main():
    parser = argparse.ArgumentParser(description='GEDI Pixel MLP training for Scenario 4')
    parser.add_argument('--csv-dir', default='chm_outputs/', help='Directory containing GEDI CSV files')
    parser.add_argument('--output-dir', default='chm_outputs/gedi_pixel_mlp_scenario4/', help='Output directory')
    parser.add_argument('--scenario', default='scenario4', help='Scenario name (scenario4 = no filter)')
    parser.add_argument('--band-selection', default='embedding', 
                       help='Band selection: embedding (Google Embedding A00-A63 only)')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=1024, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--max-samples', type=int, default=None, help='Max samples (None = use all)')
    
    args = parser.parse_args()
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("🚀 GEDI Pixel MLP Training - Scenario 4 (No Filter)")
    print(f"📅 Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🖥️  Device: {device}")
    print(f"📁 CSV directory: {args.csv_dir}")
    print(f"📊 Output directory: {args.output_dir}")
    print(f"🎯 Scenario: {args.scenario}")
    print(f"🎵 Band selection: {args.band_selection}")
    
    try:
        # Load dataset
        print(f"📂 Loading GEDI pixel dataset...")
        dataset = GEDIPixelDataset(
            csv_dir=args.csv_dir,
            scenario=args.scenario,
            band_selection=args.band_selection,
            max_samples=args.max_samples
        )
        
        # Create model
        print("🧠 Creating GEDI MLP model...")
        input_dim = np.sum(dataset.selected_features)
        model = AdvancedGEDIMLP(
            input_dim=input_dim,
            hidden_dims=[1024, 512, 256, 128, 64],
            dropout_rate=0.4
        ).to(device)
        
        print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"📊 Input features: {input_dim}")
        
        # Train model
        print("🚀 Starting GEDI pixel training...")
        train_losses, val_losses, val_r2_scores, best_val_r2 = train_gedi_mlp(
            dataset=dataset,
            model=model,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            output_dir=args.output_dir
        )
        
        print(f"\n🎉 GEDI MLP training completed!")
        print(f"🎯 Scenario: {args.scenario}")
        print(f"🎵 Band selection: {args.band_selection}")
        print(f"🏆 Best validation R²: {best_val_r2:.4f}")
        
        # Save training results
        results = {
            'scenario': args.scenario,
            'band_selection': args.band_selection,
            'best_val_r2': float(best_val_r2),
            'train_losses': [float(x) for x in train_losses],
            'val_losses': [float(x) for x in val_losses],
            'val_r2_scores': [float(x) for x in val_r2_scores],
            'total_samples': int(len(dataset)),
            'input_features': int(input_dim),
            'model_path': f'{args.output_dir}/gedi_pixel_mlp_{args.scenario}_{args.band_selection}_best.pth'
        }
        
        results_filename = f'gedi_pixel_mlp_{args.scenario}_{args.band_selection}_results.json'
        with open(os.path.join(args.output_dir, results_filename), 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"📁 Results saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
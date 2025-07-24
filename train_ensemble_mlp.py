#!/usr/bin/env python3
"""
Training script for ensemble MLP combining GEDI and production MLP models
Supports both:
- Scenario 2: Reference + GEDI U-Net Ensemble (Original)
- Scenario 5: Reference + GEDI Pixel MLP Ensemble (New)
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import json
import pickle as _pickle
from tqdm import tqdm
from datetime import datetime
import logging

# Import band utilities
from utils.band_utils import extract_bands_by_name, find_band_by_name, check_patch_compatibility

# Import ensemble models
from models.ensemble_mlp import create_ensemble_model

# Import existing models for prediction generation
sys.path.append('.')

class EnsembleDataset(Dataset):
    """Dataset for ensemble training using existing model predictions"""
    
    def __init__(self, patch_dir: str, gedi_model_path: str, mlp_model_path: str, 
                 reference_tif_path: str, patch_pattern: str = "*05LE4*",
                 max_samples_per_patch: int = 50000, gedi_model_type: str = 'unet',
                 band_selection: str = 'all'):
        
        self.patch_dir = Path(patch_dir)
        self.gedi_model_path = gedi_model_path
        self.mlp_model_path = mlp_model_path
        self.reference_tif_path = reference_tif_path
        self.patch_pattern = patch_pattern
        self.max_samples_per_patch = max_samples_per_patch
        self.gedi_model_type = gedi_model_type
        self.band_selection = band_selection
        
        # Load models
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gedi_model, self.gedi_scaler, self.gedi_feature_selector = self._load_gedi_model()
        self.mlp_model, self.mlp_scaler, self.mlp_feature_selector = self._load_mlp_model()
        
        # Find patches and load data
        self.patch_files = self._find_patches()
        print(f"Found {len(self.patch_files)} patches for ensemble training")
        
        # Generate ensemble training data
        self.gedi_predictions, self.mlp_predictions, self.reference_targets = self._generate_training_data()
        print(f"Generated {len(self.gedi_predictions)} ensemble training samples")
        
    def _find_patches(self):
        """Find available enhanced patches"""
        pattern = str(self.patch_dir / self.patch_pattern)
        files = glob.glob(pattern)
        return sorted(files)
    
    def _load_model(self, model_path, model_name, model_type):
        """Load a trained model."""
        print(f"Loading {model_name} model ({model_type}) from: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"{model_name} model not found: {model_path}")

        try:
            # Default to weights_only=True for security
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        except _pickle.UnpicklingError:
            # If it fails, it might be due to the QuantileTransformer
            print("⚠️ Weights-only load failed. Falling back to legacy loading.")
            print("   This is likely due to a pickled scikit-learn object.")
            print("   Only proceed if you trust the source of this checkpoint.")
            
            # Allow the specific QuantileTransformer global
            torch.serialization.add_safe_globals([QuantileTransformer])
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        if model_type == 'mlp':
            # Determine if this is a GEDI pixel model or reference model
            if 'gedi_pixel' in model_path.lower() or 'scenario4' in model_path.lower():
                from train_gedi_pixel_mlp_scenario4 import AdvancedGEDIMLP
                input_dim = checkpoint.get('input_features', 64)  # GEDI pixel models use 64 embedding features
                model = AdvancedGEDIMLP(input_dim=input_dim).to(self.device)
            else:
                from train_production_mlp import AdvancedReferenceHeightMLP
                input_dim = checkpoint.get('input_features', np.sum(checkpoint['selected_features']))
                model = AdvancedReferenceHeightMLP(input_dim=input_dim).to(self.device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            scaler = checkpoint.get('scaler')
            feature_selector = checkpoint.get('selected_features')
        elif model_type == 'unet':
            from models.trainers.shift_aware_trainer import ShiftAwareUNet
            
            # Determine input channels from band selection
            if self.band_selection == 'embedding':
                # For embedding, we need to check what was actually saved in the checkpoint
                # as it depends on what bands were selected during training
                if isinstance(checkpoint, dict) and 'enc1.0.weight' in checkpoint:
                    in_channels = checkpoint['enc1.0.weight'].shape[1]  # Use actual saved dimensions
                    print(f"Detected {in_channels} input channels from saved GEDI U-Net checkpoint")
                else:
                    in_channels = 64  # Default for pure embedding (A00-A63)
            elif self.band_selection == 'original':
                in_channels = 30  # Original 30-band satellite data
            else:
                # For 'all' or other modes, try to infer from checkpoint
                # Look for the first conv layer to determine input channels
                if isinstance(checkpoint, dict) and 'enc1.0.weight' in checkpoint:
                    in_channels = checkpoint['enc1.0.weight'].shape[1]
                else:
                    in_channels = 30  # Default fallback
            
            print(f"Creating U-Net model with {in_channels} input channels for band_selection='{self.band_selection}'")
            model = ShiftAwareUNet(in_channels=in_channels).to(self.device)
            model.load_state_dict(checkpoint)
            scaler = None
            feature_selector = None
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        model.eval()
        print(f"✅ {model_name} model loaded successfully.")
        return model, scaler, feature_selector

    def _load_gedi_model(self):
        return self._load_model(self.gedi_model_path, "GEDI", self.gedi_model_type)

    def _load_mlp_model(self):
        return self._load_model(self.mlp_model_path, "MLP", 'mlp')
    
    def _generate_training_data(self):
        """Generate ensemble training data by running both models on patches"""
        
        all_gedi_preds = []
        all_mlp_preds = []
        all_targets = []
        
        for patch_file in tqdm(self.patch_files, desc="Generating ensemble data"):
            try:
                if not check_patch_compatibility(patch_file, "reference"):
                    continue
                
                satellite_features, reference_band = extract_bands_by_name(patch_file, supervision_mode="reference", band_selection=self.band_selection)
                satellite_features = np.nan_to_num(satellite_features.astype(np.float32))
                
                valid_ref_mask = (~np.isnan(reference_band)) & (reference_band > 0) & (reference_band <= 100)
                
                if np.sum(valid_ref_mask) < 10:
                    continue
                
                ref_y, ref_x = np.where(valid_ref_mask)
                
                if len(ref_y) > self.max_samples_per_patch:
                    indices = np.random.choice(len(ref_y), self.max_samples_per_patch, replace=False)
                    ref_y, ref_x = ref_y[indices], ref_x[indices]
                
                pixel_features = satellite_features[:, ref_y, ref_x].T
                reference_targets = reference_band[ref_y, ref_x]
                
                with torch.no_grad():
                    if self.gedi_model_type == 'unet':
                        patch_tensor = torch.FloatTensor(satellite_features).unsqueeze(0).to(self.device)
                        gedi_patch_pred = self.gedi_model(patch_tensor).squeeze().cpu().numpy()
                        gedi_preds = gedi_patch_pred[ref_y, ref_x]
                    else: # mlp
                        gedi_features = pixel_features.copy()
                        if self.gedi_scaler:
                            gedi_features = self.gedi_scaler.transform(gedi_features)
                        if self.gedi_feature_selector is not None:
                            gedi_features = gedi_features[:, self.gedi_feature_selector]
                        gedi_tensor = torch.FloatTensor(gedi_features).to(self.device)
                        gedi_preds = self.gedi_model(gedi_tensor).cpu().numpy()

                    mlp_features = pixel_features.copy()
                    if self.mlp_scaler:
                        mlp_features = self.mlp_scaler.transform(mlp_features)
                    if self.mlp_feature_selector is not None:
                        mlp_features = mlp_features[:, self.mlp_feature_selector]
                    
                    mlp_tensor = torch.FloatTensor(mlp_features).to(self.device)
                    mlp_preds = self.mlp_model(mlp_tensor).cpu().numpy()
                
                all_gedi_preds.extend(gedi_preds)
                all_mlp_preds.extend(mlp_preds)
                all_targets.extend(reference_targets)
                
            except Exception as e:
                print(f"❌ Error processing {patch_file}: {e}")
                continue
        
        return np.array(all_gedi_preds), np.array(all_mlp_preds), np.array(all_targets)
    
    def __len__(self):
        return len(self.gedi_predictions)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor([self.gedi_predictions[idx]]),
            torch.FloatTensor([self.mlp_predictions[idx]]),
            torch.FloatTensor([self.reference_targets[idx]])
        )


def train_ensemble_model(dataset, model, device, epochs=100, batch_size=1024, learning_rate=0.001):
    """Train ensemble MLP model"""
    
    # Split data
    train_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
    
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"Training: {len(train_dataset)} samples, Validation: {len(val_dataset)} samples")
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    criterion = nn.MSELoss()
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    val_r2_scores = []
    
    print(f"Starting ensemble training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for gedi_pred, mlp_pred, target in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            gedi_pred = gedi_pred.squeeze().to(device)
            mlp_pred = mlp_pred.squeeze().to(device)
            target = target.squeeze().to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            ensemble_pred = model(gedi_pred, mlp_pred)
            loss = criterion(ensemble_pred, target)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for gedi_pred, mlp_pred, target in val_loader:
                gedi_pred = gedi_pred.squeeze().to(device)
                mlp_pred = mlp_pred.squeeze().to(device)
                target = target.squeeze().to(device)
                
                ensemble_pred = model(gedi_pred, mlp_pred)
                loss = criterion(ensemble_pred, target)
                val_loss += loss.item()
                
                all_preds.extend(ensemble_pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_r2 = r2_score(all_targets, all_preds)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Record metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_r2_scores.append(val_r2)
        
        print(f"Epoch {epoch+1:3d}: Train={train_loss:.4f}, Val={val_loss:.4f}, R²={val_r2:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'val_r2': val_r2,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_r2_scores': val_r2_scores
            }
    
    return best_model_state, val_r2


def main():
    parser = argparse.ArgumentParser(description='Train ensemble MLP for Scenario 2')
    parser.add_argument('--gedi-model-path', default='chm_outputs/gedi_pixel_mlp_scenario4/gedi_pixel_mlp_scenario4_embedding_best.pth',
                       help='Path to trained GEDI model (supports both U-Net and MLP models)')
    parser.add_argument('--reference-model-path', default='chm_outputs/production_mlp_reference_embedding_best.pth',
                       help='Path to production MLP model (Google Embedding Scenario 1)')
    parser.add_argument('--patch-dir', default='chm_outputs/enhanced_patches/',
                       help='Directory containing enhanced patches')
    parser.add_argument('--patch-pattern', default='*05LE4*', help='Pattern to match patch files')
    parser.add_argument('--reference-height-path', default='downloads/dchm_05LE4.tif',
                       help='Reference height TIF file')
    parser.add_argument('--output-dir', default='chm_outputs/gedi_scenario5_ensemble',
                       help='Output directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=1024, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--gedi-model-type', default='unet', choices=['unet', 'mlp'], help='Type of GEDI model (unet or mlp)')
    parser.add_argument('--model-type', default='simple', choices=['simple', 'advanced', 'adaptive'],
                       help='Type of ensemble model')
    parser.add_argument('--band-selection', type=str, default='all',
                       choices=['all', 'embedding', 'original', 'auxiliary'],
                       help='Band selection: all, embedding (A00-A63), original (30-band), auxiliary')
    
    args = parser.parse_args()
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("🚀 Ensemble MLP Training - Scenario 5 (Reference + GEDI Pixel)")
    print(f"📅 Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🖥️  Device: {device}")
    print(f"🧠 Model type: {args.model_type}")
    print(f"📁 Output: {args.output_dir}")
    
    try:
        # Create dataset
        print("📂 Creating ensemble dataset...")
        dataset = EnsembleDataset(
            patch_dir=args.patch_dir,
            gedi_model_path=args.gedi_model_path,
            mlp_model_path=args.reference_model_path,
            reference_tif_path=args.reference_height_path,
            patch_pattern=args.patch_pattern,
            band_selection=args.band_selection,
            gedi_model_type=args.gedi_model_type
        )
        
        # Create ensemble model
        print(f"🧠 Creating {args.model_type} ensemble model...")
        model = create_ensemble_model(args.model_type).to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"📊 Model parameters: {total_params:,}")
        
        # Train model
        print("🚀 Starting ensemble training...")
        best_model_state, best_r2 = train_ensemble_model(
            dataset=dataset,
            model=model,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
        
        # Save results
        model_path = os.path.join(args.output_dir, 'ensemble_mlp_best.pth')
        torch.save(best_model_state, model_path)
        
        # Save training info
        training_info = {
            'scenario': 'Scenario 5: Reference + GEDI Pixel Ensemble',
            'ensemble_type': args.model_type,
            'best_val_r2': float(best_r2),
            'total_samples': len(dataset),
            'gedi_model_path': args.gedi_model_path,
            'gedi_model_type': args.gedi_model_type,
            'reference_model_path': args.reference_model_path,
            'band_selection': args.band_selection,
            'training_date': datetime.now().isoformat()
        }
        
        info_path = os.path.join(args.output_dir, 'training_info.json')
        with open(info_path, 'w') as f:
            json.dump(training_info, f, indent=2)
        
        print(f"\n🎉 Ensemble training completed!")
        print(f"🏆 Best validation R²: {best_r2:.4f}")
        print(f"📁 Model saved: {model_path}")
        
        # Print model weights if available
        if hasattr(model, 'get_weights'):
            weights = model.get_weights()
            print(f"📊 Learned weights: {weights}")
        elif hasattr(model, 'get_prediction_weights'):
            weights = model.get_prediction_weights()
            print(f"📊 Learned weights: {weights}")
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
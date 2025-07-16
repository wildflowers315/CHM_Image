#!/usr/bin/env python3
"""
Scenario 3: Direct Ensemble Approach for Tochigi Region
Instead of fine-tuning, combine existing models with weighted ensemble
"""

import os
import sys
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

# Import existing models
from train_production_mlp import AdvancedReferenceHeightMLP
from data.multi_patch import load_multi_patch_gedi_data

class TochigEnsembleDataset(Dataset):
    """Dataset for ensemble training on Tochigi region"""
    
    def __init__(self, patch_dir):
        self.patch_dir = Path(patch_dir)
        self.samples = []
        
        # Find Tochigi patches
        pattern = str(self.patch_dir / "*09gd4*bandNum31*.tif")
        patch_files = glob.glob(pattern)
        
        print(f"Found {len(patch_files)} Tochigi patches")
        
        for patch_file in tqdm(patch_files, desc="Loading GEDI data"):
            try:
                with rasterio.open(patch_file) as src:
                    data = src.read().astype(np.float32)
                    
                    if data.shape[0] >= 31:
                        # Satellite features (30 bands)
                        satellite_features = data[:30]
                        
                        # GEDI targets (band 31)
                        gedi_targets = data[30]
                        
                        # Reference height (band 32 if available)
                        reference_height = data[31] if data.shape[0] > 31 else None
                        
                        # Extract valid pixels
                        valid_mask = (gedi_targets > 0) & (gedi_targets <= 100)
                        
                        if np.sum(valid_mask) > 0:
                            h, w = gedi_targets.shape
                            
                            for i in range(h):
                                for j in range(w):
                                    if valid_mask[i, j]:
                                        # Extract pixel features
                                        pixel_features = satellite_features[:, i, j]
                                        
                                        # Handle NaN values
                                        if not np.any(np.isnan(pixel_features)):
                                            sample = {
                                                'satellite_features': pixel_features,
                                                'gedi_target': gedi_targets[i, j],
                                                'reference_height': reference_height[i, j] if reference_height is not None else 0.0,
                                                'has_reference': reference_height is not None
                                            }
                                            self.samples.append(sample)
                            
            except Exception as e:
                print(f"Error loading {patch_file}: {e}")
                continue
        
        print(f"Loaded {len(self.samples)} valid GEDI samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        features = torch.FloatTensor(sample['satellite_features'])
        gedi_target = torch.FloatTensor([sample['gedi_target']])
        reference_height = torch.FloatTensor([sample['reference_height']])
        has_reference = torch.FloatTensor([float(sample['has_reference'])])
        
        return features, gedi_target, reference_height, has_reference

class AdaptiveEnsemble(nn.Module):
    """Adaptive ensemble that combines reference MLP with GEDI knowledge"""
    
    def __init__(self, reference_model_path, device):
        super().__init__()
        
        # Load reference MLP
        self.reference_mlp = AdvancedReferenceHeightMLP(
            input_dim=30, 
            hidden_dims=[1024, 512, 256, 128, 64], 
            dropout_rate=0.4, 
            use_residuals=True
        )
        
        # Load pretrained reference model
        checkpoint = torch.load(reference_model_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                self.reference_mlp.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.reference_mlp.load_state_dict(checkpoint)
        else:
            self.reference_mlp.load_state_dict(checkpoint)
        
        # Freeze reference model
        for param in self.reference_mlp.parameters():
            param.requires_grad = False
        
        # GEDI adaptation layers
        self.gedi_adapter = nn.Sequential(
            nn.Linear(30, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.ReLU()
        )
        
        # Ensemble weight predictor
        self.weight_predictor = nn.Sequential(
            nn.Linear(30, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        # Get reference prediction
        with torch.no_grad():
            ref_pred = self.reference_mlp(x)
        
        # Get GEDI adaptation
        gedi_pred = self.gedi_adapter(x)
        
        # Get ensemble weights
        weights = self.weight_predictor(x)
        
        # Combine predictions
        ensemble_pred = weights[:, 0:1] * ref_pred + weights[:, 1:2] * gedi_pred
        
        return ensemble_pred, ref_pred, gedi_pred, weights

def train_ensemble(model, train_loader, val_loader, device, epochs=50, learning_rate=0.001):
    """Train adaptive ensemble"""
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    val_r2_scores = []
    best_val_r2 = -float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for features, gedi_targets, ref_heights, has_ref in train_loader:
            features = features.to(device)
            gedi_targets = gedi_targets.to(device)
            
            optimizer.zero_grad()
            
            ensemble_pred, ref_pred, gedi_pred, weights = model(features)
            
            # Loss function
            loss = criterion(ensemble_pred, gedi_targets)
            
            # Regularization to prevent overfitting
            reg_loss = 0.01 * torch.mean(torch.abs(weights[:, 1] - 0.5))  # Encourage balanced weights
            loss += reg_loss
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for features, gedi_targets, ref_heights, has_ref in val_loader:
                features = features.to(device)
                gedi_targets = gedi_targets.to(device)
                
                ensemble_pred, ref_pred, gedi_pred, weights = model(features)
                
                loss = criterion(ensemble_pred, gedi_targets)
                val_loss += loss.item()
                
                all_preds.extend(ensemble_pred.cpu().numpy().flatten())
                all_targets.extend(gedi_targets.cpu().numpy().flatten())
        
        val_loss /= len(val_loader)
        
        # Calculate R²
        if len(all_targets) > 0:
            from sklearn.metrics import r2_score
            val_r2 = r2_score(all_targets, all_preds)
        else:
            val_r2 = -float('inf')
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_r2_scores.append(val_r2)
        
        scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val R² = {val_r2:.4f}, LR = {current_lr:.6f}")
        
        # Save best model
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_r2': val_r2,
                'val_loss': val_loss,
                'ensemble_type': 'adaptive_tochigi',
                'target_region': 'tochigi'
            }, 'chm_outputs/scenario3_tochigi_ensemble_direct/adaptive_ensemble_best.pth')
    
    return train_losses, val_losses, val_r2_scores, best_val_r2

def main():
    print("🔧 Scenario 3: Direct Ensemble Approach for Tochigi Region")
    print(f"📅 Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = 'chm_outputs/scenario3_tochigi_ensemble_direct'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🖥️  Device: {device}")
    
    try:
        # Load dataset
        print("📂 Loading Tochigi region dataset...")
        dataset = TochigEnsembleDataset('chm_outputs/enhanced_patches/')
        
        if len(dataset) == 0:
            print("❌ No valid data found")
            return
        
        # Split data
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
        
        print(f"🔄 Training samples: {len(train_dataset)}")
        print(f"🔄 Validation samples: {len(val_dataset)}")
        
        # Create ensemble model
        model = AdaptiveEnsemble(
            reference_model_path='chm_outputs/production_mlp_best.pth',
            device=device
        ).to(device)
        
        print("🚀 Starting ensemble training...")
        train_losses, val_losses, val_r2_scores, best_val_r2 = train_ensemble(
            model, train_loader, val_loader, device, epochs=50, learning_rate=0.001
        )
        
        # Save results
        results = {
            'best_val_r2': float(best_val_r2),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_r2_scores': val_r2_scores,
            'approach': 'direct_ensemble',
            'target_region': 'tochigi',
            'training_completed': True
        }
        
        results_path = os.path.join(output_dir, 'ensemble_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Ensemble training completed!")
        print(f"📊 Best validation R²: {best_val_r2:.4f}")
        print(f"💾 Results saved to: {results_path}")
        
    except Exception as e:
        print(f"❌ Error during ensemble training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
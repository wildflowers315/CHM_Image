#!/usr/bin/env python3
"""
Ensemble prediction pipeline for Scenario 2
Combines GEDI shift-aware U-Net and production MLP predictions
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import rasterio
from pathlib import Path
import glob
from tqdm import tqdm
import json
from datetime import datetime
import logging

# Import ensemble models
from models.ensemble_mlp import create_ensemble_model
from models.trainers.shift_aware_trainer import ShiftAwareUNet

def setup_logging(output_dir):
    """Setup logging configuration"""
    log_file = os.path.join(output_dir, 'prediction.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

class EnsemblePredictor:
    """Ensemble prediction using trained GEDI and MLP models"""
    
    def __init__(self, ensemble_model_path, gedi_model_path, mlp_model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load ensemble model
        self.ensemble_model = self._load_ensemble_model(ensemble_model_path)
        
        # Load individual models
        self.gedi_model = self._load_gedi_model(gedi_model_path)
        self.mlp_model = self._load_mlp_model(mlp_model_path)
        
        print(f"✅ All models loaded successfully on {self.device}")
        
    def _load_ensemble_model(self, model_path):
        """Load trained ensemble model"""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        ensemble_model = create_ensemble_model(model_type='simple', input_dim=2)
        ensemble_model.load_state_dict(checkpoint['model_state_dict'])
        ensemble_model.to(self.device)
        ensemble_model.eval()
        
        # Print learned weights for inspection
        weights = checkpoint.get('learned_weights', {})
        print(f"📊 Ensemble learned weights: {weights}")
        
        return ensemble_model
        
    def _load_gedi_model(self, model_path):
        """Load GEDI shift-aware U-Net model"""
        model = ShiftAwareUNet(in_channels=30).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint)
        model.eval()
        return model
        
    def _load_mlp_model(self, model_path):
        """Load production MLP model"""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Import the correct production MLP class
        sys.path.append('.')
        from train_production_mlp import AdvancedReferenceHeightMLP
        
        # Create production MLP with matching architecture
        model = AdvancedReferenceHeightMLP(
            input_dim=30,
            hidden_dims=[1024, 512, 256, 128, 64],
            dropout_rate=0.4,
            use_residuals=True
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        # Also load scaler and feature selector if available
        self.mlp_scaler = checkpoint.get('scaler')
        self.mlp_feature_selector = checkpoint.get('selected_features')
        
        return model
        
    def predict_patch(self, patch_data):
        """
        Generate ensemble prediction for a single patch
        
        Args:
            patch_data: numpy array of shape (32, 256, 256) for enhanced patches
                       or (31, 256, 256) for regular patches
        
        Returns:
            numpy array of shape (256, 256) with ensemble predictions
        """
        # Extract satellite features (first 30 bands)
        satellite_features = patch_data[:30].astype(np.float32)
        
        # Handle NaN values (same as training)
        for band_idx in range(30):
            band_data = satellite_features[band_idx]
            if np.isnan(band_data).any():
                if band_idx in [23, 24]:  # GLO30 slope/aspect
                    replacement_value = 0.0
                else:
                    valid_pixels = band_data[~np.isnan(band_data)]
                    replacement_value = np.median(valid_pixels) if len(valid_pixels) > 0 else 0.0
                satellite_features[band_idx] = np.nan_to_num(band_data, nan=replacement_value)
        
        # Generate GEDI predictions (full patch)
        patch_tensor = torch.FloatTensor(satellite_features).unsqueeze(0).to(self.device)
        with torch.no_grad():
            gedi_pred_patch = self.gedi_model(patch_tensor).squeeze().cpu().numpy()
        
        # Generate MLP predictions (pixel-wise)
        h, w = satellite_features.shape[1], satellite_features.shape[2]
        mlp_predictions = np.zeros((h, w))
        
        # Process in chunks for memory efficiency
        chunk_size = 10000
        for i in range(0, h, 64):  # Process 64x64 blocks
            for j in range(0, w, 64):
                end_i = min(i + 64, h)
                end_j = min(j + 64, w)
                
                # Extract pixel features for this block
                block_features = satellite_features[:, i:end_i, j:end_j]
                pixel_features = block_features.reshape(30, -1).T  # (n_pixels, 30)
                
                # Apply MLP preprocessing (scaler and feature selection)
                mlp_features = pixel_features.copy()
                if self.mlp_scaler is not None:
                    mlp_features = self.mlp_scaler.transform(mlp_features)
                if self.mlp_feature_selector is not None:
                    mlp_features = mlp_features[:, self.mlp_feature_selector]
                
                # MLP prediction
                with torch.no_grad():
                    mlp_tensor = torch.FloatTensor(mlp_features).to(self.device)
                    mlp_block_pred = self.mlp_model(mlp_tensor).cpu().numpy().flatten()
                
                # Reshape back to block
                mlp_predictions[i:end_i, j:end_j] = mlp_block_pred.reshape(end_i - i, end_j - j)
        
        # Combine predictions using ensemble model
        ensemble_predictions = np.zeros((h, w))
        for i in range(h):
            for j in range(w):
                # Create separate tensors for GEDI and MLP predictions
                gedi_val = torch.FloatTensor([gedi_pred_patch[i, j]]).to(self.device)
                mlp_val = torch.FloatTensor([mlp_predictions[i, j]]).to(self.device)
                
                with torch.no_grad():
                    ensemble_pred = self.ensemble_model(gedi_val, mlp_val).cpu().numpy()[0]
                    ensemble_predictions[i, j] = ensemble_pred
        
        return ensemble_predictions
        
    def predict_region(self, patch_dir, output_dir, region_name, patch_pattern="*"):
        """
        Generate predictions for all patches in a region
        
        Args:
            patch_dir: Directory containing patch files
            output_dir: Directory to save predictions
            region_name: Name of the region (e.g., 'kochi', 'tochigi')
            patch_pattern: Pattern to match patch files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Find patch files
        patch_files = glob.glob(os.path.join(patch_dir, f"*{patch_pattern}*.tif"))
        patch_files.sort()
        
        if not patch_files:
            raise ValueError(f"No patch files found in {patch_dir} with pattern *{patch_pattern}*")
        
        print(f"🔍 Found {len(patch_files)} patches for {region_name}")
        
        prediction_info = {
            'region': region_name,
            'total_patches': len(patch_files),
            'successful_predictions': 0,
            'failed_predictions': 0,
            'start_time': datetime.now().isoformat()
        }
        
        for patch_file in tqdm(patch_files, desc=f"Predicting {region_name}"):
            try:
                # Load patch
                with rasterio.open(patch_file) as src:
                    patch_data = src.read()
                    profile = src.profile
                
                # Generate prediction
                ensemble_pred = self.predict_patch(patch_data)
                
                # Save prediction
                patch_name = os.path.basename(patch_file)
                pred_name = f"ensemble_{region_name}_{patch_name}"
                pred_path = os.path.join(output_dir, pred_name)
                
                # Update profile for single band output
                profile.update({
                    'count': 1,
                    'dtype': rasterio.float32,
                    'compress': 'lzw'
                })
                
                with rasterio.open(pred_path, 'w', **profile) as dst:
                    dst.write(ensemble_pred.astype(np.float32), 1)
                
                prediction_info['successful_predictions'] += 1
                
            except Exception as e:
                print(f"❌ Error processing {patch_file}: {str(e)}")
                prediction_info['failed_predictions'] += 1
        
        prediction_info['end_time'] = datetime.now().isoformat()
        
        # Save prediction info
        info_path = os.path.join(output_dir, f"{region_name}_prediction_info.json")
        with open(info_path, 'w') as f:
            json.dump(prediction_info, f, indent=2)
        
        print(f"✅ {region_name} prediction completed: {prediction_info['successful_predictions']}/{prediction_info['total_patches']} successful")
        
        return prediction_info

def main():
    parser = argparse.ArgumentParser(description='Ensemble prediction for Scenario 2')
    parser.add_argument('--ensemble-model', required=True, help='Path to trained ensemble model')
    parser.add_argument('--gedi-model', required=True, help='Path to GEDI shift-aware U-Net model')
    parser.add_argument('--mlp-model', required=True, help='Path to production MLP model')
    parser.add_argument('--region', required=True, choices=['kochi', 'tochigi', 'both'], 
                        help='Target region for prediction')
    parser.add_argument('--patch-dir', required=True, help='Directory containing patch files')
    parser.add_argument('--output-dir', required=True, help='Directory to save predictions')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='Device to use')
    
    args = parser.parse_args()
    
    # Setup logging
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(args.output_dir)
    
    logger.info(f"🚀 Starting ensemble prediction for Scenario 2")
    logger.info(f"📅 Start time: {datetime.now()}")
    logger.info(f"🖥️  Device: {args.device}")
    logger.info(f"🎯 Target region: {args.region}")
    
    # Initialize predictor
    predictor = EnsemblePredictor(
        ensemble_model_path=args.ensemble_model,
        gedi_model_path=args.gedi_model,
        mlp_model_path=args.mlp_model,
        device=args.device
    )
    
    # Region-specific configurations
    region_configs = {
        'kochi': {'pattern': '04hf3', 'patch_dir': 'chm_outputs/'},
        'tochigi': {'pattern': '09gd4', 'patch_dir': 'chm_outputs/'}
    }
    
    # Run predictions
    regions_to_process = ['kochi', 'tochigi'] if args.region == 'both' else [args.region]
    
    for region in regions_to_process:
        config = region_configs[region]
        logger.info(f"🔮 Processing {region} region...")
        
        prediction_info = predictor.predict_region(
            patch_dir=args.patch_dir,
            output_dir=os.path.join(args.output_dir, region),
            region_name=region,
            patch_pattern=config['pattern']
        )
        
        logger.info(f"✅ {region} completed: {prediction_info['successful_predictions']}/{prediction_info['total_patches']} successful")
    
    logger.info(f"🎉 Ensemble prediction completed at {datetime.now()}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
MLP-based prediction pipeline for cross-region testing
Applies trained MLP model to generate height predictions for any region
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from pathlib import Path
import glob
from tqdm import tqdm
import pickle
from sklearn.preprocessing import QuantileTransformer
import json
from datetime import datetime
import logging

class AdvancedReferenceHeightMLP(nn.Module):
    """Advanced MLP with residual connections and attention - same as training"""
    
    def __init__(self, input_dim=30, hidden_dims=[1024, 512, 256, 128, 64], 
                 dropout_rate=0.4, use_residuals=True):
        super().__init__()
        
        self.use_residuals = use_residuals
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.norms.append(nn.BatchNorm1d(hidden_dim))
            dropout_p = dropout_rate * (1 - i / len(hidden_dims))
            self.dropouts.append(nn.Dropout(dropout_p))
            prev_dim = hidden_dim
        
        self.output_norm = nn.BatchNorm1d(prev_dim)
        self.output = nn.Linear(prev_dim, 1)
        
        # Feature attention mechanism
        self.feature_attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim),
            nn.Sigmoid()
        )
    
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
            
            if residual is not None:
                x = x + residual
        
        x = self.output_norm(x)
        x = self.output(x)
        
        return x.squeeze(-1)


class MLPPredictor:
    """MLP-based height prediction for cross-region testing"""
    
    def __init__(self, model_path: str, device: str = 'auto'):
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'auto' else 'cpu')
        self.model = None
        self.scaler = None
        self.selected_features = None
        
        self.load_model(model_path)
        
    def load_model(self, model_path: str):
        """Load trained MLP model and preprocessing components"""
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print(f"📂 Loading MLP model from: {model_path}")
        
        # Load with weights_only=False to handle scikit-learn objects
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Extract model parameters
        input_dim = checkpoint.get('input_features', 30)
        self.scaler = checkpoint.get('scaler')
        self.selected_features = checkpoint.get('selected_features')
        
        # Create model
        self.model = AdvancedReferenceHeightMLP(
            input_dim=input_dim,
            hidden_dims=[1024, 512, 256, 128, 64],
            dropout_rate=0.4
        ).to(self.device)
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✅ Model loaded successfully")
        print(f"   Input features: {input_dim}")
        print(f"   Selected features: {np.sum(self.selected_features) if self.selected_features is not None else 'All'}")
        print(f"   Device: {self.device}")
        
    def predict_patch(self, patch_file: str, output_file: str = None):
        """Generate height predictions for a single patch"""
        
        print(f"🔍 Processing patch: {os.path.basename(patch_file)}")
        
        # Load patch data
        with rasterio.open(patch_file) as src:
            patch_data = src.read()
            transform = src.transform
            crs = src.crs
            height, width = src.height, src.width
        
        # Extract satellite bands (first 30 bands) - consistent for all patch types
        # Enhanced patches: 32 bands (30 satellite + GEDI + reference) -> use first 30 
        # Original patches: 30-31 bands -> use first 30 satellite bands
        if patch_data.shape[0] >= 30:
            satellite_features = patch_data[:30]  # Always use first 30 satellite bands
        else:
            raise ValueError(f"Patch must have at least 30 bands, got {patch_data.shape[0]}")
        
        # Handle NaN values
        satellite_features = np.nan_to_num(satellite_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Reshape for prediction
        pixels = satellite_features.reshape(satellite_features.shape[0], -1).T  # (H*W, 30_bands)
        
        # Apply preprocessing
        if self.scaler is not None:
            pixels = self.scaler.transform(pixels.astype(np.float32))
        
        # Apply feature selection
        if self.selected_features is not None:
            pixels = pixels[:, self.selected_features]
        
        # Predict in batches
        batch_size = 10000
        predictions = []
        
        self.model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, len(pixels), batch_size), desc="Predicting"):
                batch_pixels = pixels[i:i+batch_size]
                batch_tensor = torch.from_numpy(batch_pixels).float().to(self.device)
                batch_pred = self.model(batch_tensor)
                predictions.extend(batch_pred.cpu().numpy())
        
        # Reshape predictions back to image format
        predictions = np.array(predictions).reshape(height, width)
        
        # Clip to reasonable height range
        predictions = np.clip(predictions, 0, 100)
        
        # Save prediction
        if output_file is None:
            output_file = patch_file.replace('.tif', '_mlp_prediction.tif')
        
        with rasterio.open(
            output_file,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=np.float32,
            crs=crs,
            transform=transform,
            compress='lzw'
        ) as dst:
            dst.write(predictions.astype(np.float32), 1)
        
        print(f"✅ Prediction saved: {output_file}")
        
        # Generate statistics
        valid_pred = predictions[predictions > 0]
        stats = {
            'file': os.path.basename(patch_file),
            'output': os.path.basename(output_file),
            'total_pixels': predictions.size,
            'valid_pixels': len(valid_pred),
            'coverage': len(valid_pred) / predictions.size * 100,
            'height_range': [float(np.min(valid_pred)), float(np.max(valid_pred))] if len(valid_pred) > 0 else [0, 0],
            'mean_height': float(np.mean(valid_pred)) if len(valid_pred) > 0 else 0,
            'std_height': float(np.std(valid_pred)) if len(valid_pred) > 0 else 0
        }
        
        return output_file, stats
    
    def predict_region(self, patch_dir: str, output_dir: str, patch_pattern: str = "*.tif"):
        """Generate predictions for all patches in a region"""
        
        patch_dir = Path(patch_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Find all patches
        patch_files = list(patch_dir.glob(patch_pattern))
        
        if not patch_files:
            print(f"❌ No patches found in {patch_dir} with pattern {patch_pattern}")
            return
        
        print(f"🌍 Processing {len(patch_files)} patches for region prediction")
        
        all_stats = []
        successful_predictions = []
        
        for patch_file in patch_files:
            try:
                output_file = output_dir / f"{patch_file.stem}_mlp_prediction.tif"
                pred_file, stats = self.predict_patch(str(patch_file), str(output_file))
                
                all_stats.append(stats)
                successful_predictions.append(pred_file)
                
            except Exception as e:
                print(f"❌ Error processing {patch_file}: {e}")
                continue
        
        # Save region summary
        region_summary = {
            'prediction_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_path': 'production_mlp_best.pth',
            'total_patches': len(patch_files),
            'successful_predictions': len(successful_predictions),
            'failed_predictions': len(patch_files) - len(successful_predictions),
            'patch_statistics': all_stats
        }
        
        summary_file = output_dir / 'region_prediction_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(region_summary, f, indent=2)
        
        print(f"\n📊 Region Prediction Summary:")
        print(f"   Total patches: {len(patch_files)}")
        print(f"   Successful: {len(successful_predictions)}")
        print(f"   Failed: {len(patch_files) - len(successful_predictions)}")
        print(f"   Summary saved: {summary_file}")
        
        # Calculate overall statistics
        if all_stats:
            total_pixels = sum(s['total_pixels'] for s in all_stats)
            valid_pixels = sum(s['valid_pixels'] for s in all_stats)
            heights = [s['mean_height'] for s in all_stats if s['mean_height'] > 0]
            
            print(f"\n📈 Overall Statistics:")
            print(f"   Total pixels: {total_pixels:,}")
            print(f"   Valid pixels: {valid_pixels:,} ({valid_pixels/total_pixels*100:.1f}%)")
            if heights:
                print(f"   Mean height: {np.mean(heights):.2f} ± {np.std(heights):.2f} m")
                print(f"   Height range: {np.min(heights):.2f} - {np.max(heights):.2f} m")
        
        return successful_predictions


def main():
    parser = argparse.ArgumentParser(description='MLP-based height prediction for cross-region testing')
    parser.add_argument('--model-path', default='chm_outputs/production_mlp_best.pth', 
                       help='Path to trained MLP model')
    parser.add_argument('--patch-dir', help='Directory containing patches to predict')
    parser.add_argument('--output-dir', required=True, help='Output directory for predictions')
    parser.add_argument('--patch-pattern', default='*.tif', help='Pattern to match patch files')
    parser.add_argument('--single-patch', help='Predict single patch file instead of directory')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'], 
                       help='Device to use for prediction')
    
    args = parser.parse_args()
    
    print("🚀 MLP Cross-Region Height Prediction")
    print(f"📅 Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🧠 Model: {args.model_path}")
    print(f"📂 Input: {args.patch_dir}")
    print(f"📁 Output: {args.output_dir}")
    
    try:
        # Initialize predictor
        predictor = MLPPredictor(args.model_path, args.device)
        
        if args.single_patch:
            # Single patch prediction
            if not args.patch_dir:
                args.patch_dir = os.path.dirname(args.single_patch)
            
            output_file = os.path.join(args.output_dir, 
                                     os.path.basename(args.single_patch).replace('.tif', '_mlp_prediction.tif'))
            os.makedirs(args.output_dir, exist_ok=True)
            
            pred_file, stats = predictor.predict_patch(args.single_patch, output_file)
            
            print(f"\n📊 Prediction Statistics:")
            print(f"   Coverage: {stats['coverage']:.1f}%")
            print(f"   Height range: {stats['height_range'][0]:.2f} - {stats['height_range'][1]:.2f} m")
            print(f"   Mean height: {stats['mean_height']:.2f} ± {stats['std_height']:.2f} m")
            
        else:
            # Region prediction
            if not args.patch_dir:
                raise ValueError("--patch-dir is required for region prediction")
            
            successful_predictions = predictor.predict_region(
                args.patch_dir, 
                args.output_dir, 
                args.patch_pattern
            )
            
            print(f"\n✅ Region prediction completed!")
            print(f"📁 Predictions saved in: {args.output_dir}")
            print(f"🎯 Successful predictions: {len(successful_predictions)}")
    
    except Exception as e:
        print(f"❌ Prediction failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
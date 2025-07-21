#!/usr/bin/env python3
"""
GEDI-only prediction pipeline for Scenario 1.5
Uses only the shift-aware U-Net GEDI model without reference MLP or ensemble combination
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

# Import GEDI model
from models.trainers.shift_aware_trainer import ShiftAwareUNet

# Import band utilities
from utils.band_utils import extract_bands_by_name, find_satellite_bands, check_patch_compatibility

def setup_logging(output_dir):
    """Setup logging configuration"""
    log_file = os.path.join(output_dir, 'gedi_only_prediction.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

class GEDIOnlyPredictor:
    """GEDI-only prediction using trained shift-aware U-Net model"""
    
    def __init__(self, gedi_model_path, device='cuda', band_selection='embedding'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.band_selection = band_selection
        
        # Load GEDI model
        self.gedi_model = self._load_gedi_model(gedi_model_path)
        
        print(f"✅ GEDI model loaded successfully on {self.device}")
        
    def _load_gedi_model(self, model_path):
        """Load shift-aware U-Net GEDI model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"GEDI model not found: {model_path}")
        
        # Load checkpoint to determine input channels
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Determine input channels from the first layer
        first_layer_key = next(iter(checkpoint.keys()))
        if 'conv1.weight' in checkpoint:
            input_channels = checkpoint['conv1.weight'].shape[1]
        elif 'encoder.0.weight' in checkpoint:
            input_channels = checkpoint['encoder.0.weight'].shape[1]
        else:
            # Try to find the first convolutional layer
            for key, tensor in checkpoint.items():
                if 'weight' in key and len(tensor.shape) == 4:
                    input_channels = tensor.shape[1]
                    break
            else:
                # Default based on band selection
                input_channels = 64 if self.band_selection == 'embedding' else 30
        
        print(f"Detected {input_channels} input channels from GEDI model checkpoint")
        
        # Create model with correct input channels
        model = ShiftAwareUNet(in_channels=input_channels)
        model.load_state_dict(checkpoint)
        model = model.to(self.device)
        model.eval()
        
        print(f"Creating GEDI U-Net model with {input_channels} input channels for band_selection='{self.band_selection}'")
        
        return model
    
    def predict_patch(self, patch_path):
        """Generate GEDI-only prediction for a single patch"""
        try:
            with rasterio.open(patch_path) as src:
                # Read all bands
                patch_data = src.read()  # Shape: (bands, height, width)
                transform = src.transform
                crs = src.crs
                
            # Extract relevant bands based on selection
            if self.band_selection == 'embedding':
                # Use Google Embedding bands (A00-A63) - first 64 bands for both 69 and 70 band patches
                if patch_data.shape[0] >= 64:
                    # Extract first 64 bands (Google Embedding A00-A63)
                    features = patch_data[:64]
                else:
                    print(f"⚠️  Warning: Patch {patch_path} has only {patch_data.shape[0]} bands, need at least 64 for embedding")
                    return None
            elif self.band_selection == 'all':
                # Use all available bands
                features = patch_data
            else:
                # Use first 30 bands (original satellite data)
                features = patch_data[:30]
            
            # Convert to tensor and add batch dimension
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            
            # Generate GEDI prediction
            with torch.no_grad():
                gedi_pred = self.gedi_model(features_tensor)
                gedi_pred = gedi_pred.squeeze().cpu().numpy()
            
            return {
                'prediction': gedi_pred,
                'transform': transform,
                'crs': crs,
                'shape': gedi_pred.shape
            }
            
        except Exception as e:
            print(f"❌ Error processing {patch_path}: {e}")
            return None
    
    def predict_region(self, patch_dir, region_pattern, output_dir):
        """Generate GEDI-only predictions for all patches in a region"""
        # Find patches matching the pattern - include both 69 and 70 band patches
        if region_pattern == 'kochi':
            pattern = "*04hf3*embedding*"
        elif region_pattern == 'hyogo':
            pattern = "*05LE4*embedding*"
        elif region_pattern == 'tochigi':
            pattern = "*09gd4*embedding*"
        else:
            pattern = region_pattern
        
        patch_files = glob.glob(os.path.join(patch_dir, pattern))
        patch_files.sort()
        
        if not patch_files:
            print(f"⚠️  No patches found for pattern: {pattern}")
            return 0
        
        print(f"🔍 Found {len(patch_files)} patches for {region_pattern}")
        
        # Create output directory for region
        region_output_dir = os.path.join(output_dir, region_pattern)
        os.makedirs(region_output_dir, exist_ok=True)
        
        successful_predictions = 0
        
        for patch_file in tqdm(patch_files, desc=f"Predicting {region_pattern}"):
            patch_name = os.path.basename(patch_file)
            output_name = patch_name.replace('.tif', '_gedi_only_prediction.tif')
            output_path = os.path.join(region_output_dir, output_name)
            
            # Skip if prediction already exists
            if os.path.exists(output_path):
                successful_predictions += 1
                continue
            
            # Generate prediction
            result = self.predict_patch(patch_file)
            
            if result is not None:
                # Save prediction as GeoTIFF
                with rasterio.open(
                    output_path,
                    'w',
                    driver='GTiff',
                    height=result['shape'][0],
                    width=result['shape'][1],
                    count=1,
                    dtype=rasterio.float32,
                    crs=result['crs'],
                    transform=result['transform'],
                    compress='lzw'
                ) as dst:
                    dst.write(result['prediction'].astype(np.float32), 1)
                
                successful_predictions += 1
        
        print(f"✅ {region_pattern} prediction completed: {successful_predictions}/{len(patch_files)} successful")
        return successful_predictions

def main():
    parser = argparse.ArgumentParser(description='GEDI-only predictions for Scenario 1.5')
    parser.add_argument('--gedi-model', required=True, help='Path to GEDI shift-aware U-Net model')
    parser.add_argument('--region', required=True, choices=['hyogo', 'kochi', 'tochigi', 'all'], 
                       help='Region to predict')
    parser.add_argument('--patch-dir', required=True, help='Directory containing patch files')
    parser.add_argument('--output-dir', required=True, help='Directory to save GEDI-only predictions')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--band-selection', type=str, default='embedding',
                       choices=['all', 'embedding', 'original'],
                       help='Band selection strategy')
    
    args = parser.parse_args()
    
    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(args.output_dir)
    
    print(f"🚀 Starting GEDI-only predictions for Scenario 1.5")
    print(f"📊 GEDI Model: {args.gedi_model}")
    print(f"🌍 Region: {args.region}")
    print(f"📁 Output: {args.output_dir}")
    
    # Initialize predictor
    try:
        predictor = GEDIOnlyPredictor(
            gedi_model_path=args.gedi_model,
            device=args.device,
            band_selection=args.band_selection
        )
    except Exception as e:
        print(f"❌ Failed to initialize predictor: {e}")
        return 1
    
    # Generate predictions
    if args.region == 'all':
        regions = ['kochi', 'hyogo', 'tochigi']
    else:
        regions = [args.region]
    
    total_successful = 0
    for region in regions:
        print(f"\n🌍 Processing region: {region}")
        successful = predictor.predict_region(args.patch_dir, region, args.output_dir)
        total_successful += successful
    
    print(f"\n✅ GEDI-only predictions completed!")
    print(f"📊 Total successful predictions: {total_successful}")
    print(f"📁 Results saved in: {args.output_dir}")

if __name__ == "__main__":
    sys.exit(main())
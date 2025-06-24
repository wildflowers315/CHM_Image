#!/usr/bin/env python3
"""
Mosaic Utilities Module

Provides comprehensive mosaicking functionality for canopy height predictions.
Handles both partial and complete coverage scenarios with automatic patch detection.

Features:
- Automatic patch type detection (30 vs 31 bands)
- Multi-model prediction support
- Comprehensive coverage analysis
- Spatial merging and quality assessment
"""

import numpy as np
import torch
import torch.nn as nn
import rasterio
from rasterio.merge import merge
from pathlib import Path
import json
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def load_model_for_prediction(model_path, input_channels, device):
    """
    Load trained model for prediction
    
    Args:
        model_path: Path to trained model
        input_channels: Number of input channels (30 for features only)
        device: PyTorch device
        
    Returns:
        Loaded model in evaluation mode
    """
    from models.trainers.shift_aware_trainer import ShiftAwareUNet
    
    model = ShiftAwareUNet(in_channels=input_channels).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model

def preprocess_patch_for_prediction(patch_path):
    """
    Preprocess patch for prediction with automatic band detection
    
    Args:
        patch_path: Path to patch TIF file
        
    Returns:
        Tuple of (features_tensor, metadata, patch_type)
    """
    with rasterio.open(patch_path) as src:
        data = src.read().astype(np.float32)
        
        # Handle different band counts
        if data.shape[0] == 31:
            features = data[:-1]  # Exclude GEDI labels
            patch_type = "labeled"
        elif data.shape[0] == 30:
            features = data  # Use all bands
            patch_type = "unlabeled"
        else:
            raise ValueError(f"Unexpected band count: {data.shape[0]}")
        
        # Apply same normalization as training
        features = np.nan_to_num(features, nan=0.0, posinf=100.0, neginf=-100.0)
        
        # Normalize features to reasonable range
        for i in range(features.shape[0]):
            band = features[i]
            if np.std(band) > 0:
                band_norm = (band - np.mean(band)) / (np.std(band) + 1e-8)
                features[i] = np.clip(band_norm, -5.0, 5.0)
            else:
                features[i] = np.zeros_like(band)
        
        return torch.FloatTensor(features).unsqueeze(0), src.meta, patch_type

def predict_single_patch(model, patch_path, device):
    """
    Generate prediction for a single patch
    
    Args:
        model: Trained model
        patch_path: Path to patch file
        device: PyTorch device
        
    Returns:
        Tuple of (prediction_array, metadata, patch_type)
    """
    try:
        features, meta, patch_type = preprocess_patch_for_prediction(patch_path)
        features = features.to(device)
        
        with torch.no_grad():
            prediction = model(features)
            prediction_np = prediction.squeeze().cpu().numpy()
        
        # Clip to reasonable height range
        prediction_np = np.clip(prediction_np, 0.0, 100.0)
        
        # Update metadata for output
        output_meta = meta.copy()
        output_meta.update({
            'count': 1,
            'dtype': 'float32',
            'compress': 'lzw'
        })
        
        return prediction_np, output_meta, patch_type
        
    except Exception as e:
        print(f"⚠️  Error predicting {patch_path}: {e}")
        return None, None, None

def find_all_patches():
    """
    Find all available patches with automatic type detection
    
    Returns:
        Dictionary with patch statistics and file lists
    """
    patch_files_30 = list(Path("chm_outputs").glob("*bandNum30*.tif"))
    patch_files_31 = list(Path("chm_outputs").glob("*bandNum31*.tif"))
    
    return {
        'patches_30_band': patch_files_30,
        'patches_31_band': patch_files_31,
        'all_patches': patch_files_30 + patch_files_31,
        'count_30_band': len(patch_files_30),
        'count_31_band': len(patch_files_31),
        'total_count': len(patch_files_30) + len(patch_files_31)
    }

def create_comprehensive_mosaic(model_path, output_name="comprehensive_height_mosaic.tif"):
    """
    Create comprehensive prediction mosaic using all available patches
    
    Args:
        model_path: Path to trained model
        output_name: Output mosaic filename
        
    Returns:
        Dictionary with mosaic statistics and metadata
    """
    print("🗺️  Creating Comprehensive Canopy Height Mosaic")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Find all patches
    patch_info = find_all_patches()
    all_patches = patch_info['all_patches']
    
    print(f"📁 Found patches:")
    print(f"   🔹 30-band patches (unlabeled): {patch_info['count_30_band']}")
    print(f"   🔹 31-band patches (labeled): {patch_info['count_31_band']}")
    print(f"   📊 Total patches: {patch_info['total_count']}")
    
    if patch_info['total_count'] == 0:
        raise ValueError("No patch files found in chm_outputs/")
    
    # Load model
    print(f"🤖 Loading model: {model_path}")
    model = load_model_for_prediction(model_path, input_channels=30, device=device)
    
    # Create output directory
    output_dir = Path("chm_outputs/predictions")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate individual predictions
    prediction_files = []
    successful_predictions = 0
    labeled_patches = 0
    unlabeled_patches = 0
    
    print(f"🔄 Generating predictions for {len(all_patches)} patches...")
    
    for patch_file in tqdm(all_patches, desc="Predicting patches"):
        try:
            prediction, meta, patch_type = predict_single_patch(model, patch_file, device)
            
            if prediction is not None:
                # Save individual prediction
                patch_name = patch_file.stem
                output_file = output_dir / f"{patch_name}_prediction.tif"
                
                with rasterio.open(output_file, 'w', **meta) as dst:
                    dst.write(prediction, 1)
                
                prediction_files.append(str(output_file))
                successful_predictions += 1
                
                if patch_type == "labeled":
                    labeled_patches += 1
                else:
                    unlabeled_patches += 1
                
        except Exception as e:
            print(f"⚠️  Failed to predict {patch_file}: {e}")
    
    print(f"✅ Generated {successful_predictions}/{len(all_patches)} predictions")
    print(f"   🔹 From labeled patches: {labeled_patches}")
    print(f"   🔹 From unlabeled patches: {unlabeled_patches}")
    
    if successful_predictions == 0:
        raise ValueError("No successful predictions generated")
    
    # Create mosaic
    print("🧩 Creating mosaic from individual predictions...")
    
    # Open all prediction files
    src_files = []
    for pred_file in prediction_files:
        src = rasterio.open(pred_file)
        src_files.append(src)
    
    # Create mosaic
    print("🔄 Merging patches into comprehensive mosaic...")
    mosaic, out_trans = merge(src_files, method='first')
    
    # Close source files
    for src in src_files:
        src.close()
    
    # Update metadata for mosaic
    mosaic_meta = src_files[0].meta.copy()
    mosaic_meta.update({
        "height": mosaic.shape[1],
        "width": mosaic.shape[2], 
        "transform": out_trans,
        "compress": "lzw"
    })
    
    # Save mosaic
    print(f"💾 Saving mosaic: {output_name}")
    with rasterio.open(output_name, 'w', **mosaic_meta) as dst:
        dst.write(mosaic)
    
    # Generate statistics
    valid_pixels = mosaic[mosaic > 0]
    
    if len(valid_pixels) > 0:
        stats = {
            'mosaic_shape': mosaic.shape,
            'total_pixels': mosaic.size,
            'valid_pixels': len(valid_pixels),
            'coverage_percent': len(valid_pixels) / mosaic.size * 100,
            'height_stats': {
                'min': float(valid_pixels.min()),
                'max': float(valid_pixels.max()),
                'mean': float(valid_pixels.mean()),
                'median': float(np.median(valid_pixels)),
                'std': float(valid_pixels.std())
            },
            'height_distribution': {}
        }
        
        # Height distribution
        bins = [0, 5, 10, 20, 30, 50, 100]
        for i in range(len(bins)-1):
            count = int(np.sum((valid_pixels >= bins[i]) & (valid_pixels < bins[i+1])))
            pct = count / len(valid_pixels) * 100
            stats['height_distribution'][f'{bins[i]}-{bins[i+1]}m'] = {
                'count': count,
                'percentage': pct
            }
        
        # Tall trees analysis
        tall_trees = valid_pixels[valid_pixels > 30]
        stats['tall_trees'] = {
            'count': len(tall_trees),
            'percentage': len(tall_trees) / len(valid_pixels) * 100 if len(valid_pixels) > 0 else 0,
            'max_height': float(tall_trees.max()) if len(tall_trees) > 0 else 0
        }
        
        print(f"\n📊 Mosaic Statistics:")
        print(f"   🗺️  Dimensions: {stats['mosaic_shape'][1]}×{stats['mosaic_shape'][2]} pixels")
        print(f"   📏 Valid pixels: {stats['valid_pixels']:,} ({stats['coverage_percent']:.1f}%)")
        print(f"   📈 Height range: {stats['height_stats']['min']:.2f} - {stats['height_stats']['max']:.2f} m")
        print(f"   📊 Mean height: {stats['height_stats']['mean']:.2f} m")
    else:
        stats = {}
    
    # Save metadata
    metadata = {
        'model_used': os.path.basename(model_path),
        'total_patches_processed': successful_predictions,
        'labeled_patches': labeled_patches,
        'unlabeled_patches': unlabeled_patches,
        'mosaic_file': output_name,
        'individual_predictions_dir': str(output_dir),
        'statistics': stats,
        'creation_timestamp': '2025-06-24'
    }
    
    metadata_file = output_name.replace('.tif', '_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n🎉 Comprehensive mosaic creation complete!")
    print(f"📁 Mosaic: {output_name}")
    print(f"📁 Metadata: {metadata_file}")
    print(f"📁 Individual predictions: {output_dir}/")
    
    return metadata

def compare_mosaics(mosaic1_path, mosaic2_path):
    """
    Compare two mosaics and generate comparison statistics
    
    Args:
        mosaic1_path: Path to first mosaic
        mosaic2_path: Path to second mosaic
        
    Returns:
        Dictionary with comparison metrics
    """
    print(f"🆚 Comparing mosaics:")
    print(f"   📁 Mosaic 1: {mosaic1_path}")
    print(f"   📁 Mosaic 2: {mosaic2_path}")
    
    try:
        with rasterio.open(mosaic1_path) as src1:
            data1 = src1.read(1)
        
        with rasterio.open(mosaic2_path) as src2:
            data2 = src2.read(1)
        
        valid1 = data1[data1 > 0]
        valid2 = data2[data2 > 0]
        
        comparison = {
            'mosaic1_valid_pixels': len(valid1),
            'mosaic2_valid_pixels': len(valid2),
            'coverage_improvement': len(valid2) - len(valid1),
            'coverage_improvement_pct': ((len(valid2) - len(valid1)) / len(valid1) * 100) if len(valid1) > 0 else 0,
            'mosaic1_mean_height': float(valid1.mean()) if len(valid1) > 0 else 0,
            'mosaic2_mean_height': float(valid2.mean()) if len(valid2) > 0 else 0
        }
        
        print(f"📊 Comparison Results:")
        print(f"   Valid pixels: {comparison['mosaic1_valid_pixels']:,} → {comparison['mosaic2_valid_pixels']:,}")
        print(f"   Improvement: +{comparison['coverage_improvement']:,} pixels ({comparison['coverage_improvement_pct']:.1f}%)")
        print(f"   Mean height: {comparison['mosaic1_mean_height']:.2f}m → {comparison['mosaic2_mean_height']:.2f}m")
        
        return comparison
        
    except Exception as e:
        print(f"❌ Error comparing mosaics: {e}")
        return {}

if __name__ == "__main__":
    print("🧪 Testing Mosaic Utilities")
    
    # Test finding patches
    patch_info = find_all_patches()
    print(f"Found {patch_info['total_count']} total patches")
    
    # Test with a model if available
    model_path = "chm_outputs/models/shift_aware/shift_aware_unet_r2.pth"
    if os.path.exists(model_path):
        metadata = create_comprehensive_mosaic(model_path)
        print("✅ Mosaic creation test successful!")
    else:
        print("ℹ️  No trained model found for testing")
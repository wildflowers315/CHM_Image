#!/usr/bin/env python3
"""
Evaluate ensemble cross-region predictions for Scenario 2
Compare against Scenario 1 baseline performance
"""

import os
import sys
import argparse
import numpy as np
import rasterio
import glob
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import json
from datetime import datetime
import pandas as pd
from tqdm import tqdm

def load_reference_data(ref_tif_path, target_bounds=None):
    """Load reference height data"""
    with rasterio.open(ref_tif_path) as src:
        if target_bounds:
            # Crop to target bounds if specified
            window = rasterio.windows.from_bounds(*target_bounds, src.transform)
            ref_data = src.read(1, window=window)
            transform = src.window_transform(window)
        else:
            ref_data = src.read(1)
            transform = src.transform
        
        profile = src.profile
        profile.update({'transform': transform})
        
    return ref_data, profile

def evaluate_predictions(pred_dir, ref_tif_path, region_name, output_dir):
    """
    Evaluate ensemble predictions against reference data
    
    Args:
        pred_dir: Directory containing prediction TIF files
        ref_tif_path: Path to reference height TIF
        region_name: Name of region (kochi, tochigi)
        output_dir: Directory to save evaluation results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load reference data
    print(f"📊 Loading reference data for {region_name}...")
    ref_data, ref_profile = load_reference_data(ref_tif_path)
    
    # Find prediction files
    pred_files = glob.glob(os.path.join(pred_dir, f"ensemble_{region_name}_*.tif"))
    pred_files.sort()
    
    if not pred_files:
        print(f"❌ No prediction files found for {region_name}")
        return None
    
    print(f"🔍 Found {len(pred_files)} prediction files for {region_name}")
    
    # Collect all predictions and reference values
    all_predictions = []
    all_references = []
    patch_results = []
    
    for pred_file in tqdm(pred_files, desc=f"Evaluating {region_name}"):
        try:
            # Load prediction
            with rasterio.open(pred_file) as src:
                pred_data = src.read(1)
                pred_transform = src.transform
                pred_bounds = src.bounds
            
            # Find corresponding reference region
            # This is a simplified approach - in practice, you'd need proper spatial alignment
            ref_window = rasterio.windows.from_bounds(*pred_bounds, ref_profile['transform'])
            ref_crop = ref_data[
                max(0, int(ref_window.row_off)):min(ref_data.shape[0], int(ref_window.row_off + ref_window.height)),
                max(0, int(ref_window.col_off)):min(ref_data.shape[1], int(ref_window.col_off + ref_window.width))
            ]
            
            # Resize reference to match prediction if needed
            if ref_crop.shape != pred_data.shape:
                from skimage.transform import resize
                ref_crop = resize(ref_crop, pred_data.shape, preserve_range=True)
            
            # Find valid pixels (reference > 0 and < 100, prediction not NaN)
            valid_mask = (
                (~np.isnan(ref_crop)) & 
                (~np.isnan(pred_data)) & 
                (ref_crop > 0) & 
                (ref_crop <= 100) &
                (pred_data > -50) & 
                (pred_data <= 100)
            )
            
            if np.sum(valid_mask) > 10:  # Minimum valid pixels
                ref_valid = ref_crop[valid_mask]
                pred_valid = pred_data[valid_mask]
                
                # Calculate metrics for this patch
                patch_r2 = r2_score(ref_valid, pred_valid)
                patch_rmse = np.sqrt(mean_squared_error(ref_valid, pred_valid))
                patch_mae = mean_absolute_error(ref_valid, pred_valid)
                
                patch_results.append({
                    'patch_file': os.path.basename(pred_file),
                    'valid_pixels': len(ref_valid),
                    'r2': patch_r2,
                    'rmse': patch_rmse,
                    'mae': patch_mae,
                    'ref_mean': np.mean(ref_valid),
                    'pred_mean': np.mean(pred_valid),
                    'ref_std': np.std(ref_valid),
                    'pred_std': np.std(pred_valid)
                })
                
                # Add to overall collection
                all_references.extend(ref_valid)
                all_predictions.extend(pred_valid)
                
        except Exception as e:
            print(f"❌ Error processing {pred_file}: {str(e)}")
    
    if not all_predictions:
        print(f"❌ No valid predictions collected for {region_name}")
        return None
    
    # Calculate overall metrics
    all_references = np.array(all_references)
    all_predictions = np.array(all_predictions)
    
    overall_r2 = r2_score(all_references, all_predictions)
    overall_rmse = np.sqrt(mean_squared_error(all_references, all_predictions))
    overall_mae = mean_absolute_error(all_references, all_predictions)
    
    # Calculate correlation
    correlation = np.corrcoef(all_references, all_predictions)[0, 1]
    
    # Create results dictionary
    results = {
        'region': region_name,
        'evaluation_date': datetime.now().isoformat(),
        'total_patches': len(pred_files),
        'successful_patches': len(patch_results),
        'total_valid_pixels': len(all_references),
        'overall_metrics': {
            'r2': float(overall_r2),
            'rmse': float(overall_rmse),
            'mae': float(overall_mae),
            'correlation': float(correlation)
        },
        'data_statistics': {
            'reference_mean': float(np.mean(all_references)),
            'reference_std': float(np.std(all_references)),
            'prediction_mean': float(np.mean(all_predictions)),
            'prediction_std': float(np.std(all_predictions)),
            'reference_range': [float(np.min(all_references)), float(np.max(all_references))],
            'prediction_range': [float(np.min(all_predictions)), float(np.max(all_predictions))]
        },
        'patch_results': patch_results
    }
    
    # Save detailed results
    results_file = os.path.join(output_dir, f"{region_name}_evaluation_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create summary plot
    plt.figure(figsize=(12, 8))
    
    # Scatter plot
    plt.subplot(2, 2, 1)
    plt.scatter(all_references, all_predictions, alpha=0.1, s=1)
    plt.plot([0, 100], [0, 100], 'r--', alpha=0.8)
    plt.xlabel('Reference Height (m)')
    plt.ylabel('Predicted Height (m)')
    plt.title(f'{region_name.title()} - Scatter Plot\nR² = {overall_r2:.4f}')
    plt.grid(True, alpha=0.3)
    
    # Histogram of residuals
    plt.subplot(2, 2, 2)
    residuals = all_predictions - all_references
    plt.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Residuals (Predicted - Reference)')
    plt.ylabel('Frequency')
    plt.title(f'Residuals Distribution\nMAE = {overall_mae:.2f}m')
    plt.grid(True, alpha=0.3)
    
    # Height distribution comparison
    plt.subplot(2, 2, 3)
    plt.hist(all_references, bins=30, alpha=0.7, label='Reference', density=True)
    plt.hist(all_predictions, bins=30, alpha=0.7, label='Predicted', density=True)
    plt.xlabel('Height (m)')
    plt.ylabel('Density')
    plt.title('Height Distribution Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Patch-wise R² distribution
    plt.subplot(2, 2, 4)
    patch_r2_values = [p['r2'] for p in patch_results]
    plt.hist(patch_r2_values, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Patch R²')
    plt.ylabel('Frequency')
    plt.title(f'Patch-wise R² Distribution\nMean = {np.mean(patch_r2_values):.3f}')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{region_name}_evaluation_plot.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary
    print(f"\n📊 {region_name.title()} Evaluation Results:")
    print(f"  📈 Overall R²: {overall_r2:.4f}")
    print(f"  📏 RMSE: {overall_rmse:.2f}m")
    print(f"  📐 MAE: {overall_mae:.2f}m")
    print(f"  🔗 Correlation: {correlation:.4f}")
    print(f"  📊 Valid pixels: {len(all_references):,}")
    print(f"  📦 Successful patches: {len(patch_results)}/{len(pred_files)}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate ensemble cross-region predictions')
    parser.add_argument('--pred-dir', required=True, help='Directory containing prediction files')
    parser.add_argument('--output-dir', required=True, help='Directory to save evaluation results')
    parser.add_argument('--region', choices=['kochi', 'tochigi', 'both'], default='both',
                        help='Region to evaluate')
    
    args = parser.parse_args()
    
    # Region-specific reference files
    reference_files = {
        'kochi': 'downloads/dchm_04hf3.tif',
        'tochigi': 'downloads/dchm_09gd4.tif'
    }
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"🚀 Starting Ensemble Cross-Region Evaluation - Scenario 2")
    print(f"📅 Start time: {datetime.now()}")
    
    # Determine regions to evaluate
    regions_to_evaluate = ['kochi', 'tochigi'] if args.region == 'both' else [args.region]
    
    all_results = {}
    
    for region in regions_to_evaluate:
        print(f"\n🔍 Evaluating {region} region...")
        
        region_pred_dir = os.path.join(args.pred_dir, region)
        region_output_dir = os.path.join(args.output_dir, region)
        
        if not os.path.exists(region_pred_dir):
            print(f"❌ Prediction directory not found: {region_pred_dir}")
            continue
        
        if not os.path.exists(reference_files[region]):
            print(f"❌ Reference file not found: {reference_files[region]}")
            continue
        
        results = evaluate_predictions(
            pred_dir=region_pred_dir,
            ref_tif_path=reference_files[region],
            region_name=region,
            output_dir=region_output_dir
        )
        
        if results:
            all_results[region] = results
    
    # Create summary comparison
    if len(all_results) > 1:
        print(f"\n📊 Cross-Region Summary:")
        print(f"{'Region':<10} {'R²':<10} {'RMSE':<10} {'MAE':<10} {'Pixels':<12}")
        print("-" * 50)
        
        for region, results in all_results.items():
            metrics = results['overall_metrics']
            pixels = results['total_valid_pixels']
            print(f"{region:<10} {metrics['r2']:<10.4f} {metrics['rmse']:<10.2f} {metrics['mae']:<10.2f} {pixels:<12,}")
    
    # Save combined results
    combined_results = {
        'evaluation_date': datetime.now().isoformat(),
        'scenario': 'Scenario 2 - Ensemble (GEDI + MLP)',
        'regions': all_results
    }
    
    combined_file = os.path.join(args.output_dir, 'combined_evaluation_results.json')
    with open(combined_file, 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    print(f"\n✅ Evaluation completed!")
    print(f"📄 Results saved to: {args.output_dir}")
    print(f"📊 Combined results: {combined_file}")
    
    # Compare with Scenario 1 baseline
    print(f"\n🔍 Scenario 1 baseline: R² < 0.0 (around -0.04)")
    for region, results in all_results.items():
        r2 = results['overall_metrics']['r2']
        improvement = "✅ IMPROVED" if r2 > 0.0 else "❌ Still poor"
        print(f"  {region}: R² = {r2:.4f} ({improvement})")

if __name__ == "__main__":
    main()
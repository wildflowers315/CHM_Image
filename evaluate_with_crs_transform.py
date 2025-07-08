#!/usr/bin/env python3
"""
CRS-aware evaluation script for MLP cross-region predictions.
Handles coordinate system transformations properly.
"""

import os
import glob
import rasterio
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.transform import from_bounds
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import argparse
from pathlib import Path

def evaluate_with_crs_transform(pred_dir: str, ref_tif: str, region_name: str, output_dir: str):
    """
    Evaluate predictions with proper CRS transformation.
    
    Strategy:
    1. Reproject predictions to match reference CRS and grid
    2. Sample overlapping areas
    3. Calculate metrics on valid pixels
    """
    
    print(f'🔄 CRS-aware evaluation for {region_name}')
    
    # Load reference info
    with rasterio.open(ref_tif) as ref_src:
        ref_crs = ref_src.crs
        ref_bounds = ref_src.bounds
        ref_transform = ref_src.transform
        ref_shape = ref_src.shape
        ref_nodata = ref_src.nodata or -99999.0
        
    print(f'Reference TIF:')
    print(f'  CRS: {ref_crs}')
    print(f'  Shape: {ref_shape}')
    print(f'  Bounds: {ref_bounds}')
    print(f'  NoData: {ref_nodata}')
    
    # Find prediction files
    pred_files = glob.glob(os.path.join(pred_dir, '*_prediction.tif'))
    print(f'Found {len(pred_files)} prediction files')
    
    all_pred_vals = []
    all_ref_vals = []
    processed_files = 0
    
    for pred_file in pred_files[:10]:  # Process first 10 files to avoid memory issues
        try:
            print(f'\\nProcessing: {os.path.basename(pred_file)}')
            
            # Load prediction
            with rasterio.open(pred_file) as pred_src:
                pred_data = pred_src.read(1)
                pred_crs = pred_src.crs
                pred_transform = pred_src.transform
                pred_bounds = pred_src.bounds
                
            print(f'  Pred CRS: {pred_crs}')
            print(f'  Pred bounds: {pred_bounds}')
            print(f'  Pred shape: {pred_data.shape}')
            print(f'  Pred range: {np.nanmin(pred_data):.2f} - {np.nanmax(pred_data):.2f}m')
            
            # Transform prediction bounds to reference CRS
            pred_bounds_in_ref = transform_bounds(pred_crs, ref_crs, *pred_bounds)
            print(f'  Pred bounds in ref CRS: {pred_bounds_in_ref}')
            
            # Check if there's spatial overlap
            overlap = (pred_bounds_in_ref[0] < ref_bounds[2] and pred_bounds_in_ref[2] > ref_bounds[0] and
                      pred_bounds_in_ref[1] < ref_bounds[3] and pred_bounds_in_ref[3] > ref_bounds[1])
            
            if not overlap:
                print(f'  No spatial overlap, skipping')
                continue
                
            # Calculate overlap bounds
            overlap_bounds = (
                max(pred_bounds_in_ref[0], ref_bounds[0]),
                max(pred_bounds_in_ref[1], ref_bounds[1]),
                min(pred_bounds_in_ref[2], ref_bounds[2]),
                min(pred_bounds_in_ref[3], ref_bounds[3])
            )
            print(f'  Overlap bounds: {overlap_bounds}')
            
            # Calculate reasonable resolution for overlap area
            overlap_width = overlap_bounds[2] - overlap_bounds[0]
            overlap_height = overlap_bounds[3] - overlap_bounds[1]
            
            # Use 100m resolution for manageable memory usage
            target_res = 100.0
            target_width = int(overlap_width / target_res)
            target_height = int(overlap_height / target_res)
            
            if target_width < 10 or target_height < 10:
                print(f'  Overlap area too small ({target_width}x{target_height}), skipping')
                continue
                
            print(f'  Target grid: {target_width}x{target_height} at {target_res}m resolution')
            
            # Create target transform
            target_transform = from_bounds(*overlap_bounds, target_width, target_height)
            
            # Reproject prediction to target grid
            pred_reproj = np.full((target_height, target_width), np.nan, dtype=np.float32)
            reproject(
                source=pred_data,
                destination=pred_reproj,
                src_transform=pred_transform,
                src_crs=pred_crs,
                dst_transform=target_transform,
                dst_crs=ref_crs,
                resampling=Resampling.bilinear
            )
            
            # Reproject reference to target grid
            ref_reproj = np.full((target_height, target_width), np.nan, dtype=np.float32)
            with rasterio.open(ref_tif) as ref_src:
                reproject(
                    source=rasterio.band(ref_src, 1),
                    destination=ref_reproj,
                    src_transform=ref_src.transform,
                    src_crs=ref_src.crs,
                    dst_transform=target_transform,
                    dst_crs=ref_crs,
                    resampling=Resampling.bilinear
                )
            
            # Clean data and find valid pixels
            valid_pred = (~np.isnan(pred_reproj)) & (pred_reproj > 0) & (pred_reproj < 100)
            valid_ref = (~np.isnan(ref_reproj)) & (ref_reproj != ref_nodata) & (ref_reproj > 0) & (ref_reproj < 100)
            valid_both = valid_pred & valid_ref
            
            valid_count = valid_both.sum()
            print(f'  Valid pixels: {valid_count}')
            
            if valid_count > 50:  # Require at least 50 valid pixels
                pred_vals = pred_reproj[valid_both]
                ref_vals = ref_reproj[valid_both]
                
                all_pred_vals.extend(pred_vals)
                all_ref_vals.extend(ref_vals)
                processed_files += 1
                
                # Calculate patch-level metrics
                if len(ref_vals) > 1:
                    patch_r2 = r2_score(ref_vals, pred_vals)
                    patch_rmse = np.sqrt(mean_squared_error(ref_vals, pred_vals))
                    print(f'  Patch R²: {patch_r2:.4f}, RMSE: {patch_rmse:.2f}m')
                else:
                    print(f'  Insufficient variation for R² calculation')
            else:
                print(f'  Insufficient valid overlap ({valid_count} pixels)')
                
        except Exception as e:
            print(f'  Error: {e}')
            continue
    
    print(f'\\nProcessed {processed_files}/{len(pred_files)} files')
    
    if len(all_pred_vals) > 10:
        all_pred_vals = np.array(all_pred_vals)
        all_ref_vals = np.array(all_ref_vals)
        
        # Calculate overall metrics
        r2 = r2_score(all_ref_vals, all_pred_vals)
        rmse = np.sqrt(mean_squared_error(all_ref_vals, all_pred_vals))
        mae = mean_absolute_error(all_ref_vals, all_pred_vals)
        bias = np.mean(all_pred_vals - all_ref_vals)
        relative_rmse = rmse / np.mean(all_ref_vals) * 100
        
        results = {
            'region': region_name,
            'evaluation_method': 'crs_transform_100m_grid',
            'n_pixels': len(all_pred_vals),
            'n_files_processed': processed_files,
            'n_files_available': len(pred_files),
            'r2_score': float(r2),
            'rmse': float(rmse),
            'mae': float(mae),
            'bias': float(bias),
            'relative_rmse_percent': float(relative_rmse),
            'pred_mean': float(np.mean(all_pred_vals)),
            'pred_std': float(np.std(all_pred_vals)),
            'pred_min': float(np.min(all_pred_vals)),
            'pred_max': float(np.max(all_pred_vals)),
            'ref_mean': float(np.mean(all_ref_vals)),
            'ref_std': float(np.std(all_ref_vals)),
            'ref_min': float(np.min(all_ref_vals)),
            'ref_max': float(np.max(all_ref_vals)),
            'pred_crs': str(pred_crs),
            'ref_crs': str(ref_crs)
        }
        
        print(f'\\n📈 CRS-Transformed Results for {region_name}:')
        print(f'   R² Score: {r2:.4f}')
        print(f'   RMSE: {rmse:.2f}m ({relative_rmse:.1f}%)')
        print(f'   MAE: {mae:.2f}m')
        print(f'   Bias: {bias:.2f}m')
        print(f'   Pixels: {len(all_pred_vals):,}')
        print(f'   Files: {processed_files}/{len(pred_files)}')
        print(f'   Pred: {np.min(all_pred_vals):.1f} - {np.max(all_pred_vals):.1f}m (μ={np.mean(all_pred_vals):.1f})')
        print(f'   Ref:  {np.min(all_ref_vals):.1f} - {np.max(all_ref_vals):.1f}m (μ={np.mean(all_ref_vals):.1f})')
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'{region_name}_crs_evaluation.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'💾 Results saved to: {output_file}')
        
        return results
    else:
        print(f'❌ Insufficient valid pixels for evaluation ({len(all_pred_vals)})')
        return None

def main():
    parser = argparse.ArgumentParser(description='CRS-aware MLP evaluation')
    parser.add_argument('--pred-dir', required=True, help='Prediction directory')
    parser.add_argument('--ref-tif', required=True, help='Reference TIF file')
    parser.add_argument('--region-name', required=True, help='Region name')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    
    args = parser.parse_args()
    
    print("🔄 CRS-Aware MLP Cross-Region Evaluation")
    print(f"📁 Predictions: {args.pred_dir}")
    print(f"🏔️  Reference: {args.ref_tif}")
    print(f"🌍 Region: {args.region_name}")
    print(f"📊 Output: {args.output_dir}")
    print()
    
    result = evaluate_with_crs_transform(args.pred_dir, args.ref_tif, args.region_name, args.output_dir)
    
    if result:
        print(f"✅ CRS-aware evaluation completed for {args.region_name}")
        print(f"🎯 R² = {result['r2_score']:.4f}")
    else:
        print(f"❌ Evaluation failed for {args.region_name}")

if __name__ == "__main__":
    main()
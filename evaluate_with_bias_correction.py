#!/usr/bin/env python3
"""
Bias-corrected evaluation to test the systematic scaling hypothesis.
Apply 2.5x correction factor and re-evaluate.
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

def evaluate_with_bias_correction(pred_dir: str, ref_tif: str, region_name: str, output_dir: str, 
                                 correction_factor: float = 2.5):
    """
    Evaluate predictions with bias correction applied.
    """
    
    print(f'🔧 Bias-Corrected Evaluation for {region_name} (Factor: {correction_factor:.1f}x)')
    
    # Load reference info
    with rasterio.open(ref_tif) as ref_src:
        ref_crs = ref_src.crs
        ref_bounds = ref_src.bounds
        ref_transform = ref_src.transform
        ref_shape = ref_src.shape
        ref_nodata = ref_src.nodata or -99999.0
    
    print(f'Reference: {ref_crs}, Shape: {ref_shape}')
    
    # Find prediction files
    pred_files = glob.glob(os.path.join(pred_dir, '*_prediction.tif'))
    print(f'Found {len(pred_files)} prediction files')
    
    all_pred_vals = []
    all_ref_vals = []
    all_pred_corrected = []
    processed_files = 0
    
    for pred_file in pred_files[:10]:  # Process first 10 files
        try:
            print(f'\\nProcessing: {os.path.basename(pred_file)}')
            
            # Load prediction
            with rasterio.open(pred_file) as pred_src:
                pred_data = pred_src.read(1)
                pred_crs = pred_src.crs
                pred_transform = pred_src.transform
                pred_bounds = pred_src.bounds
            
            # Apply bias correction
            pred_corrected = pred_data / correction_factor
            
            print(f'  Original range: {np.nanmin(pred_data):.1f} - {np.nanmax(pred_data):.1f}m')
            print(f'  Corrected range: {np.nanmin(pred_corrected):.1f} - {np.nanmax(pred_corrected):.1f}m')
            
            # Transform prediction bounds to reference CRS
            pred_bounds_in_ref = transform_bounds(pred_crs, ref_crs, *pred_bounds)
            
            # Check spatial overlap
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
            
            # Use 100m resolution for manageable memory
            target_res = 100.0
            overlap_width = overlap_bounds[2] - overlap_bounds[0]
            overlap_height = overlap_bounds[3] - overlap_bounds[1]
            target_width = int(overlap_width / target_res)
            target_height = int(overlap_height / target_res)
            
            if target_width < 10 or target_height < 10:
                print(f'  Overlap area too small, skipping')
                continue
            
            print(f'  Target grid: {target_width}x{target_height} at {target_res}m')
            
            # Create target transform
            target_transform = from_bounds(*overlap_bounds, target_width, target_height)
            
            # Reproject original prediction to target grid
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
            
            # Reproject corrected prediction to target grid
            pred_corrected_reproj = np.full((target_height, target_width), np.nan, dtype=np.float32)
            reproject(
                source=pred_corrected,
                destination=pred_corrected_reproj,
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
            
            # Find valid pixels
            valid_pred = (~np.isnan(pred_reproj)) & (pred_reproj > 0) & (pred_reproj < 100)
            valid_ref = (~np.isnan(ref_reproj)) & (ref_reproj != ref_nodata) & (ref_reproj > 0) & (ref_reproj < 100)
            valid_both = valid_pred & valid_ref
            
            valid_count = valid_both.sum()
            print(f'  Valid pixels: {valid_count}')
            
            if valid_count > 50:
                pred_vals = pred_reproj[valid_both]
                pred_corrected_vals = pred_corrected_reproj[valid_both] 
                ref_vals = ref_reproj[valid_both]
                
                all_pred_vals.extend(pred_vals)
                all_pred_corrected.extend(pred_corrected_vals)
                all_ref_vals.extend(ref_vals)
                processed_files += 1
                
                # Calculate patch-level metrics
                if len(ref_vals) > 1:
                    r2_orig = r2_score(ref_vals, pred_vals)
                    r2_corr = r2_score(ref_vals, pred_corrected_vals)
                    print(f'  Original R²: {r2_orig:.4f}')
                    print(f'  Corrected R²: {r2_corr:.4f} (Δ={r2_corr-r2_orig:+.4f})')
            else:
                print(f'  Insufficient valid overlap ({valid_count} pixels)')
                
        except Exception as e:
            print(f'  Error: {e}')
            continue
    
    print(f'\\nProcessed {processed_files}/{len(pred_files)} files')
    
    if len(all_pred_vals) > 10:
        all_pred_vals = np.array(all_pred_vals)
        all_pred_corrected = np.array(all_pred_corrected)
        all_ref_vals = np.array(all_ref_vals)
        
        # Calculate metrics for original predictions
        r2_orig = r2_score(all_ref_vals, all_pred_vals)
        rmse_orig = np.sqrt(mean_squared_error(all_ref_vals, all_pred_vals))
        bias_orig = np.mean(all_pred_vals - all_ref_vals)
        
        # Calculate metrics for corrected predictions
        r2_corr = r2_score(all_ref_vals, all_pred_corrected)
        rmse_corr = np.sqrt(mean_squared_error(all_ref_vals, all_pred_corrected))
        bias_corr = np.mean(all_pred_corrected - all_ref_vals)
        
        results = {
            'region': region_name,
            'correction_factor': correction_factor,
            'n_pixels': len(all_pred_vals),
            'n_files_processed': processed_files,
            'original': {
                'r2_score': float(r2_orig),
                'rmse': float(rmse_orig),
                'bias': float(bias_orig),
                'pred_mean': float(np.mean(all_pred_vals)),
                'pred_std': float(np.std(all_pred_vals))
            },
            'corrected': {
                'r2_score': float(r2_corr),
                'rmse': float(rmse_corr),
                'bias': float(bias_corr),
                'pred_mean': float(np.mean(all_pred_corrected)),
                'pred_std': float(np.std(all_pred_corrected))
            },
            'reference': {
                'ref_mean': float(np.mean(all_ref_vals)),
                'ref_std': float(np.std(all_ref_vals)),
                'ref_min': float(np.min(all_ref_vals)),
                'ref_max': float(np.max(all_ref_vals))
            },
            'improvement': {
                'r2_improvement': float(r2_corr - r2_orig),
                'rmse_improvement': float(rmse_orig - rmse_corr),
                'bias_reduction': float(abs(bias_orig) - abs(bias_corr))
            }
        }
        
        print(f'\\n📈 BIAS CORRECTION RESULTS for {region_name}:')
        print(f'   Correction Factor: {correction_factor:.1f}x')
        print(f'   Pixels Evaluated: {len(all_pred_vals):,}')
        print()
        print(f'   ORIGINAL PREDICTIONS:')
        print(f'     R²: {r2_orig:.4f}')
        print(f'     RMSE: {rmse_orig:.2f}m')
        print(f'     Bias: {bias_orig:.2f}m')
        print(f'     Mean: {np.mean(all_pred_vals):.1f}m')
        print()
        print(f'   CORRECTED PREDICTIONS:')
        print(f'     R²: {r2_corr:.4f} (Δ={r2_corr-r2_orig:+.4f})')
        print(f'     RMSE: {rmse_corr:.2f}m (Δ={rmse_corr-rmse_orig:+.2f})')
        print(f'     Bias: {bias_corr:.2f}m (Δ={bias_corr-bias_orig:+.2f})')
        print(f'     Mean: {np.mean(all_pred_corrected):.1f}m')
        print()
        print(f'   REFERENCE:')
        print(f'     Mean: {np.mean(all_ref_vals):.1f}m')
        print(f'     Range: {np.min(all_ref_vals):.1f} - {np.max(all_ref_vals):.1f}m')
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'{region_name}_bias_corrected_evaluation.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'💾 Results saved to: {output_file}')
        
        return results
    else:
        print(f'❌ Insufficient valid pixels for evaluation ({len(all_pred_vals)})')
        return None

def main():
    parser = argparse.ArgumentParser(description='Bias-corrected MLP evaluation')
    parser.add_argument('--pred-dir', required=True, help='Prediction directory')
    parser.add_argument('--ref-tif', required=True, help='Reference TIF file')
    parser.add_argument('--region-name', required=True, help='Region name')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--correction-factor', type=float, default=2.5, help='Bias correction factor')
    
    args = parser.parse_args()
    
    print("🔧 BIAS-CORRECTED MLP EVALUATION")
    print(f"📁 Predictions: {args.pred_dir}")
    print(f"🏔️  Reference: {args.ref_tif}")
    print(f"🌍 Region: {args.region_name}")
    print(f"🔧 Correction Factor: {args.correction_factor:.1f}x")
    print(f"📊 Output: {args.output_dir}")
    print()
    
    result = evaluate_with_bias_correction(args.pred_dir, args.ref_tif, args.region_name, 
                                         args.output_dir, args.correction_factor)
    
    if result:
        print(f"✅ Bias-corrected evaluation completed for {args.region_name}")
        print(f"🎯 Corrected R² = {result['corrected']['r2_score']:.4f}")
        print(f"📈 R² Improvement = {result['improvement']['r2_improvement']:+.4f}")
    else:
        print(f"❌ Evaluation failed for {args.region_name}")

if __name__ == "__main__":
    main()
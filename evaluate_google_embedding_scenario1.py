#!/usr/bin/env python3
"""
Google Embedding Scenario 1 Evaluation Script
Compare Google Embedding predictions with reference data and original 30-band MLP predictions
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
from rasterio.transform import from_bounds
from pathlib import Path
import glob
import json
import gc
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import partial

def process_prediction_file(args):
    """Helper function for parallel processing of prediction files"""
    ref_path, pred_file, bias_correction_factor, region_id, apply_bias_correction = args
    
    try:
        ref_aligned, pred_aligned = GoogleEmbeddingEvaluator.align_prediction_with_reference(ref_path, pred_file)
        
        if ref_aligned is not None and pred_aligned is not None:
            # Apply bias correction if requested
            if apply_bias_correction and bias_correction_factor != 1.0:
                pred_aligned = pred_aligned / bias_correction_factor
            
            print(f"    Processed {pred_file}: {len(ref_aligned)} pixels")
            return ref_aligned, pred_aligned
        else:
            print(f"    Failed to align {pred_file} - insufficient valid pixels")
            return None, None
    except Exception as e:
        print(f"    Error processing {pred_file}: {e}")
        return None, None

class GoogleEmbeddingEvaluator:
    """Evaluator for Google Embedding Scenario 1 vs Original 30-band MLP"""
    
    def __init__(self, downloads_dir="downloads", output_dir="chm_outputs/google_embedding_evaluation"):
        self.downloads_dir = downloads_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Region mapping
        self.regions = {
            '04hf3': 'Kochi',
            '05LE4': 'Hyogo', 
            '09gd4': 'Tochigi'
        }
        
        # Bias correction factors from original 30-band MLP
        self.bias_correction = {
            '04hf3': 2.5,   # Kochi
            '09gd4': 3.7,   # Tochigi  
            '05LE4': 1.0    # Hyogo (training region)
        }
        
    def find_prediction_files(self, predictions_dir, region_id, prediction_type="embedding", max_patches=3):
        """Find prediction files for a region with dynamic directory detection"""
        print(f"Looking for {prediction_type} predictions in region {region_id}")
        
        pred_files = []
        
        # Search in root directory first
        patterns = [
            f"*{region_id}*embedding*prediction*.tif",  # Google Embedding
            f"*{region_id}*gedi_only_prediction*.tif",  # GEDI-only predictions
            f"*{region_id}*mlp_prediction*.tif",        # Original MLP
            f"*{region_id}*pred*.tif"                   # General
        ]
        
        for pattern in patterns:
            full_pattern = os.path.join(predictions_dir, pattern)
            pred_files.extend(glob.glob(full_pattern))
        
        # Dynamic subdirectory detection
        region_name_map = {
            '04hf3': 'kochi',
            '05LE4': 'hyogo', 
            '09gd4': 'tochigi'
        }
        
        region_name = region_name_map.get(region_id, '')
        
        # Try different possible subdirectory structures
        possible_subdirs = [
            region_name,                    # Scenario 2A: kochi, hyogo, tochigi
            f"{region_id}_{region_name}",   # Scenario 1: 04hf3_kochi, 05LE4_hyogo, 09gd4_tochigi
            region_id                       # Direct region ID
        ]
        
        subdir_found = None
        for subdir in possible_subdirs:
            subdir_path = os.path.join(predictions_dir, subdir)
            if os.path.exists(subdir_path):
                subdir_found = subdir
                break
        
        if subdir_found:
            subdir = subdir_found
            
            subdir_path = os.path.join(predictions_dir, subdir)
            print(subdir_path)
            if os.path.exists(subdir_path):
                # Simple patterns that match any prediction file in the region subdirectory
                simple_patterns = [
                    "*gedi_only_prediction.tif",  # GEDI-only prediction files (Scenario 1.5)
                    "*mlp_prediction.tif",        # Original Scenario 1 MLP files
                    "*prediction*.tif",           # General prediction files
                    "ensemble_*.tif",             # Scenario 2A ensemble files 
                    "*ensemble*.tif",             # Alternative ensemble pattern
                    "*.tif"                       # Final fallback - any tif file
                ]
                for pattern in simple_patterns:
                    full_pattern = os.path.join(subdir_path, pattern)
                    files_found = glob.glob(full_pattern)
                    if files_found:
                        print(f"  Found {len(files_found)} files with pattern {pattern}")
                        pred_files.extend(files_found)
                        break  # Stop after first successful pattern to avoid duplicates
        
        # Remove duplicates and sort
        pred_files = list(set(pred_files))
        pred_files.sort()
        
        result = pred_files[:max_patches] if pred_files else []
        print(f"  Returning {len(result)} files for processing")
        return result
    
    @staticmethod
    def align_prediction_with_reference(ref_path, pred_path, max_pixels=100000):
        """Align prediction with reference data using proper spatial alignment"""
        try:
            # Import the raster utils
            from raster_utils import load_and_align_rasters
            
            # Use the same alignment logic as the existing evaluation system
            pred_data, ref_data, transform, forest_mask = load_and_align_rasters(
                pred_path, ref_path, None, None  # No forest mask, no output dir
            )
            
            # Use the same filtering logic as height_analysis_utils.py
            # Remove NaN values first
            valid_mask = ~(np.isnan(ref_data) | np.isnan(pred_data))
            
            if np.sum(valid_mask) < 100:
                return None, None
                
            ref_valid = ref_data[valid_mask]
            pred_valid = pred_data[valid_mask]
            
            # Apply outlier filtering with 3-sigma rule (same as height_analysis_utils.py)
            ref_mean, ref_std = np.mean(ref_valid), np.std(ref_valid)
            pred_mean, pred_std = np.mean(pred_valid), np.std(pred_valid)
            
            outlier_mask = (
                (np.abs(ref_valid - ref_mean) < 3 * ref_std) &
                (np.abs(pred_valid - pred_mean) < 3 * pred_std) &
                (ref_valid > 0) & (pred_valid >= 0) &  # Exclude zero reference heights
                (ref_valid <= 100) & (pred_valid <= 100)  # Remove unrealistic heights
            )
            
            print(f"    Valid pixels: initial={np.sum(valid_mask)}, after_filtering={np.sum(outlier_mask)}")
            
            if np.sum(outlier_mask) > 100:
                ref_clean = ref_valid[outlier_mask]
                pred_clean = pred_valid[outlier_mask]
                
                # Sample if too many pixels
                if len(ref_clean) > max_pixels:
                    indices = np.random.choice(len(ref_clean), max_pixels, replace=False)
                    ref_clean = ref_clean[indices]
                    pred_clean = pred_clean[indices]
                
                return ref_clean, pred_clean
            
            return None, None
            
        except Exception as e:
            print(f"Error aligning {pred_path}: {e}")
            return None, None
    
    def calculate_metrics(self, ref_data, pred_data):
        """Calculate evaluation metrics"""
        if len(ref_data) < 10:
            return None
        
        try:
            r2 = r2_score(ref_data, pred_data)
            rmse = np.sqrt(mean_squared_error(ref_data, pred_data))
            mae = mean_absolute_error(ref_data, pred_data)
            bias = np.mean(pred_data - ref_data)
            
            # Additional metrics
            corr_coef, p_value = stats.pearsonr(ref_data, pred_data)
            
            return {
                'r2': r2,
                'rmse': rmse,
                'mae': mae,
                'bias': bias,
                'correlation': corr_coef,
                'p_value': p_value,
                'n_samples': len(ref_data),
                'ref_mean': np.mean(ref_data),
                'pred_mean': np.mean(pred_data),
                'ref_std': np.std(ref_data),
                'pred_std': np.std(pred_data)
            }
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            return None
    
    def evaluate_scenario(self, scenario_name, predictions_dir, apply_bias_correction=True, prediction_type="embedding", max_patches=3):
        """Evaluate a scenario across all regions"""
        results = []
        aggregated_data = {}
        
        for region_id, region_name in self.regions.items():
            print(f"Evaluating {scenario_name} - {region_name} ({region_id})...")
            
            # Find reference file
            ref_path = f"{self.downloads_dir}/dchm_{region_id}.tif"
            if not os.path.exists(ref_path):
                print(f"Reference file not found: {ref_path}")
                continue
            
            # Find prediction files
            pred_files = self.find_prediction_files(predictions_dir, region_id, prediction_type, max_patches)
            if not pred_files:
                print(f"No prediction files found for {region_id}")
                continue
            
            print(f"  Processing {len(pred_files)} prediction files with {min(cpu_count(), len(pred_files))} cores...")
            
            # Aggregate data across patches using parallel processing
            aggregated_ref = []
            aggregated_pred = []
            
            # Get bias correction factor
            bias_correction_factor = self.bias_correction.get(region_id, 1.0)
            
            # Prepare arguments for parallel processing
            args_list = [
                (ref_path, pred_file, bias_correction_factor, region_id, apply_bias_correction)
                for pred_file in pred_files
            ]
            
            # Use a memory-efficient parallel processing iterator
            num_cores = min(cpu_count(), len(pred_files))
            with Pool(num_cores) as pool:
                for ref_aligned, pred_aligned in pool.imap_unordered(process_prediction_file, args_list):
                    if ref_aligned is not None and pred_aligned is not None:
                        aggregated_ref.extend(ref_aligned)
                        aggregated_pred.extend(pred_aligned)
                        # Clean up intermediate results immediately
                        del ref_aligned, pred_aligned
                        gc.collect()

            print(f"  Collected {len(aggregated_ref)} reference pixels and {len(aggregated_pred)} prediction pixels")
            
            # Calculate metrics for aggregated data (lower threshold like analysis utils)
            if len(aggregated_ref) > 50:
                aggregated_ref = np.array(aggregated_ref)
                aggregated_pred = np.array(aggregated_pred)
                
                metrics = self.calculate_metrics(aggregated_ref, aggregated_pred)
                if metrics:
                    metrics.update({
                        'scenario': scenario_name,
                        'region': region_name,
                        'region_id': region_id,
                        'bias_correction_applied': apply_bias_correction,
                        'bias_correction_factor': self.bias_correction.get(region_id, 1.0) if apply_bias_correction else 1.0
                    })
                    results.append(metrics)
                    
                    # Store for plotting
                    aggregated_data[region_id] = {
                        'ref': aggregated_ref,
                        'pred': aggregated_pred,
                        'metrics': metrics
                    }
                    
                    print(f"  {region_name}: R² = {metrics['r2']:.3f}, RMSE = {metrics['rmse']:.2f}m, N = {metrics['n_samples']:,}")
                    # Clean up memory for the next region
                    del aggregated_ref, aggregated_pred
                    gc.collect()
                    
        return results, aggregated_data
    
    def create_correlation_plot(self, data_dict, scenario_name, output_file):
        """Create 3-panel correlation plot"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        regions = ['04hf3', '05LE4', '09gd4']
        region_names = ['Kochi', 'Hyogo', 'Tochigi']
        
        for i, (region_id, region_name) in enumerate(zip(regions, region_names)):
            ax = axes[i]
            
            if region_id in data_dict:
                ref_data = data_dict[region_id]['ref']
                pred_data = data_dict[region_id]['pred']
                metrics = data_dict[region_id]['metrics']
                
                # Create hexbin plot
                hb = ax.hexbin(ref_data, pred_data, gridsize=30, cmap='viridis', 
                              mincnt=1, alpha=0.8)
                
                # Add 1:1 line
                min_val = min(np.min(ref_data), np.min(pred_data))
                max_val = max(np.max(ref_data), np.max(pred_data))
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
                       label='1:1 Line')
                
                # Add regression line
                if len(ref_data) > 1:
                    coeffs = np.polyfit(ref_data, pred_data, 1)
                    poly_line = np.poly1d(coeffs)
                    x_line = np.linspace(min_val, max_val, 100)
                    ax.plot(x_line, poly_line(x_line), 'g-', linewidth=2,
                           label=f'Regression (y = {coeffs[0]:.3f}x + {coeffs[1]:.3f})')
                
                # Add metrics text
                metrics_text = (
                    f"R² = {metrics['r2']:.3f}\n"
                    f"RMSE = {metrics['rmse']:.2f} m\n"
                    f"Bias = {metrics['bias']:.2f} m\n"
                    f"N = {metrics['n_samples']:,}\n"
                    f"Corr = {metrics['correlation']:.3f}"
                )
                
                if metrics['bias_correction_applied']:
                    metrics_text += f"\nBias Corr: {metrics['bias_correction_factor']:.1f}x"
                
                ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
                       verticalalignment='top', fontsize=11,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                # Add colorbar for first subplot
                if i == 0:
                    cb = plt.colorbar(hb, ax=ax)
                    cb.set_label('Point Density', fontsize=12)
                
            else:
                ax.text(0.5, 0.5, f'No data for {region_name}', transform=ax.transAxes,
                       ha='center', va='center', fontsize=14)
            
            # Set labels and title
            ax.set_xlabel('Reference Height (m)', fontsize=12)
            ax.set_ylabel('Predicted Height (m)', fontsize=12)
            ax.set_title(f'{region_name} Region', fontsize=14, fontweight='bold')
            ax.set_aspect('equal', adjustable='box')
            ax.grid(True, alpha=0.3)
            if region_id in data_dict:
                ax.legend(loc='lower right', fontsize=10)
        
        # Overall title
        fig.suptitle(f'{scenario_name}: Reference vs Predicted Height',
                    fontsize=16, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved correlation plot: {output_file}")
    
    def create_comparison_table(self, all_results, output_file):
        """Create comparison table between scenarios"""
        df = pd.DataFrame(all_results)
        
        # Create summary table
        summary = df.groupby('scenario').agg({
            'r2': ['mean', 'std', 'min', 'max'],
            'rmse': ['mean', 'std', 'min', 'max'], 
            'bias': ['mean', 'std'],
            'n_samples': 'sum'
        }).round(3)
        
        # Save to CSV
        csv_file = output_file.replace('.png', '.csv')
        summary.to_csv(csv_file)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # R² comparison
        r2_data = df.pivot(index='region', columns='scenario', values='r2')
        sns.heatmap(r2_data, annot=True, fmt='.3f', cmap='viridis', ax=ax1)
        ax1.set_title('R² Performance Comparison')
        ax1.set_xlabel('Scenario')
        ax1.set_ylabel('Region')
        
        # RMSE comparison
        rmse_data = df.pivot(index='region', columns='scenario', values='rmse')
        sns.heatmap(rmse_data, annot=True, fmt='.2f', cmap='viridis_r', ax=ax2)
        ax2.set_title('RMSE Performance Comparison (m)')
        ax2.set_xlabel('Scenario')
        ax2.set_ylabel('Region')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved comparison table: {output_file}")
        print(f"Saved summary data: {csv_file}")
        
        return summary


def main():
    parser = argparse.ArgumentParser(description='Evaluate Google Embedding Scenario 1')
    parser.add_argument('--google-embedding-dir', 
                       default='chm_outputs/google_embedding_scenario1_predictions',
                       help='Google Embedding predictions directory')
    parser.add_argument('--original-mlp-dir',
                       default='chm_outputs/cross_region_predictions',  
                       help='Original 30-band MLP predictions directory')
    parser.add_argument('--downloads-dir', default='downloads', 
                       help='Directory containing reference TIF files')
    parser.add_argument('--output-dir', default='chm_outputs/google_embedding_evaluation',
                       help='Output directory for evaluation results')
    parser.add_argument('--no-bias-correction', action='store_true',
                       help='Disable bias correction')
    parser.add_argument('--max-patches', type=int, default=3,
                       help='Maximum number of patches to process per region (default: 3)')
    
    args = parser.parse_args()
    
    print("🚀 Google Embedding Scenario 1 Evaluation")
    print(f"📅 Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📂 Google Embedding dir: {args.google_embedding_dir}")
    print(f"📂 Original MLP dir: {args.original_mlp_dir}")
    print(f"📊 Output dir: {args.output_dir}")
    
    # Initialize evaluator
    evaluator = GoogleEmbeddingEvaluator(args.downloads_dir, args.output_dir)
    
    # Evaluate Google Embedding Scenario 1
    print("\n=== Google Embedding Scenario 1 Evaluation ===")
    embedding_results, embedding_data = evaluator.evaluate_scenario(
        "Google Embedding (64-band)", 
        args.google_embedding_dir,
        apply_bias_correction=not args.no_bias_correction,
        prediction_type="embedding",
        max_patches=args.max_patches
    )
    
        # Create correlation plots
    if embedding_data:
        evaluator.create_correlation_plot(
            embedding_data,
            "Google Embedding Scenario 1 (64-band)",
            f"{args.output_dir}/google_embedding_correlation.png"
        )
        del embedding_data
        gc.collect()
    
    # Evaluate Original 30-band MLP (if available and provided)
    original_results, original_data = [], {}
    if args.original_mlp_dir and args.original_mlp_dir.strip() and os.path.exists(args.original_mlp_dir):
        print("\n=== Original 30-band MLP Evaluation ===")
        original_results, original_data = evaluator.evaluate_scenario(
            "Original Satellite (30-band)",
            args.original_mlp_dir,
            apply_bias_correction=not args.no_bias_correction,
            prediction_type="mlp",
            max_patches=args.max_patches
        )
    elif args.original_mlp_dir and args.original_mlp_dir.strip():
        print(f"\n⚠️  Original MLP directory not found: {args.original_mlp_dir}")
    else:
        print("\n📊 Evaluating single scenario only (no comparison baseline provided)")
    

    if original_data:
        evaluator.create_correlation_plot(
            original_data,
            "Original Satellite MLP (30-band)",
            f"{args.output_dir}/original_satellite_correlation.png"
        )
        del original_data
        gc.collect()
    
    # Create comparison analysis
    all_results = embedding_results + original_results
    if all_results:
        summary = evaluator.create_comparison_table(
            all_results,
            f"{args.output_dir}/scenario_comparison.png"
        )
        
        # Save detailed results
        results_file = f"{args.output_dir}/detailed_evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print(f"\n📄 Detailed results saved: {results_file}")
        
        # Print summary
        print("\n📊 EVALUATION SUMMARY:")
        print("=" * 50)
        
        df = pd.DataFrame(all_results)
        for scenario in df['scenario'].unique():
            scenario_data = df[df['scenario'] == scenario]
            avg_r2 = scenario_data['r2'].mean()
            avg_rmse = scenario_data['rmse'].mean()
            avg_bias = scenario_data['bias'].mean()
            total_samples = scenario_data['n_samples'].sum()
            
            print(f"\n🎯 {scenario}:")
            print(f"   Average R²: {avg_r2:.3f}")
            print(f"   Average RMSE: {avg_rmse:.2f} m")
            print(f"   Average Bias: {avg_bias:.2f} m")
            print(f"   Total Samples: {total_samples:,}")
        
        # Performance comparison
        if len(df['scenario'].unique()) > 1:
            embedding_avg_r2 = df[df['scenario'].str.contains('Google')]['r2'].mean()
            original_avg_r2 = df[df['scenario'].str.contains('Original')]['r2'].mean()
            
            if not np.isnan(embedding_avg_r2) and not np.isnan(original_avg_r2):
                improvement = embedding_avg_r2 - original_avg_r2
                print(f"\n🚀 PERFORMANCE IMPROVEMENT:")
                print(f"   Google Embedding R²: {embedding_avg_r2:.3f}")
                print(f"   Original Satellite R²: {original_avg_r2:.3f}")
                print(f"   Improvement: {improvement:+.3f} ({improvement/original_avg_r2*100:+.1f}%)")
        
        print(f"\n✅ Evaluation completed successfully!")
        print(f"📁 Results saved in: {args.output_dir}")
    
    else:
        print("❌ No evaluation results generated - check prediction directories")


if __name__ == "__main__":
    main()
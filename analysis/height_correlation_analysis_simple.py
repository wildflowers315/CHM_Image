#!/usr/bin/env python3
"""
Simple Height Correlation Analysis - Test Version

Quick test with multiple patches per region to verify implementation works correctly.
Uses the new height_analysis_utils module for reusable components.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import utilities
import sys
sys.path.append('/home/WUR/ishik001/CHM_Image')
from analysis.height_analysis_utils import HeightCorrelationAnalyzer

# Set up matplotlib for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SimpleHeightCorrelationAnalyzer(HeightCorrelationAnalyzer):
    """Simple analyzer for testing height data correlations with multiple patches"""
    
    def __init__(self, downloads_dir, patches_dir, output_dir):
        super().__init__(downloads_dir, patches_dir, output_dir)
        
        # Results storage
        self.results = []
        
        print(f"Output directory: {output_dir}")
        print(f"Region mapping: {self.regions}")
    
    def get_sample_patches(self, region_id, max_patches=3):
        """Get multiple sample patches per region that contain GEDI data"""
        return self.get_optimal_gedi_patches(region_id, max_patches)
    
    
    
    
    
    def analyze_region_simple(self, region_id):
        """Analyze height correlations for multiple patches in a region, aggregating data"""
        region_name = self.regions[region_id]
        print(f"\n=== Simple Analysis: {region_name} ({region_id}) ===")
        
        ref_path = os.path.join(self.downloads_dir, f"{region_id}.tif")
        if not os.path.exists(ref_path):
            print(f"Warning: Reference file not found: {ref_path}")
            return
            
        sample_patches = self.get_sample_patches(region_id, max_patches=3)
        if not sample_patches:
            print(f"No GEDI patches found for {region_id}")
            return
            
        height_data_aggregated = {}
        
        for patch_path, patch_num in sample_patches:
            print(f"\nProcessing patch {patch_num} for {region_name}...")
            height_bands_info, _ = self.extract_height_bands_info(patch_path)
            if not height_bands_info:
                print(f"No height bands found in patch {patch_num}")
                continue
                
            for band_name, band_info in height_bands_info.items():
                print(f"Collecting data from {band_name}...")
                if band_name == 'GEDI':
                    ref_aligned, aux_aligned = self.align_gedi_points_with_reference(
                        ref_path, patch_path, band_info['index']
                    )
                else:
                    ref_aligned, aux_aligned = self.align_patch_with_reference(
                        ref_path, patch_path, band_info['index']
                    )
                if ref_aligned is None or aux_aligned is None:
                    print(f"Failed to align {band_name} data")
                    continue
                    
                valid_mask = (~np.isnan(aux_aligned) & (aux_aligned > 0) & 
                              ~np.isnan(ref_aligned) & (ref_aligned > 0))
                min_threshold = 5 if band_name == 'GEDI' else 50
                
                if np.sum(valid_mask) < min_threshold:
                    print(f"Insufficient {band_name} data in patch {patch_num}")
                    continue
                    
                if band_name not in height_data_aggregated:
                    height_data_aggregated[band_name] = {'ref_data': [], 'aux_data': []}
                
                height_data_aggregated[band_name]['ref_data'].append(ref_aligned[valid_mask])
                height_data_aggregated[band_name]['aux_data'].append(aux_aligned[valid_mask])

        print(f"\nCreating aggregated plots for {region_name}...")
        for band_name, data in height_data_aggregated.items():
            if not data['ref_data']:
                continue
                
            combined_ref = np.concatenate(data['ref_data'])
            combined_aux = np.concatenate(data['aux_data'])
            
            metrics = self.calculate_metrics(combined_ref, combined_aux, 
                                           min_samples=5 if band_name == 'GEDI' else 50)
            if not metrics:
                print(f"Insufficient aggregated data for {band_name} metrics calculation")
                continue
                
            output_file = os.path.join(self.output_dir, 
                                     f"{region_name.lower()}_{band_name.lower()}_aggregated.png")
            self.create_correlation_plot(combined_ref, combined_aux, metrics, 
                                       f"{region_name} (Aggregated)", band_name, output_file)
            
            result = {
                'region': region_name,
                'height_source': band_name,
                'r2': metrics['r2'],
                'rmse': metrics['rmse'],
                'bias': metrics['bias'],
                'n_samples': metrics['n_samples'],
                'ref_clean': metrics['ref_clean'],
                'aux_clean': metrics['aux_clean']
            }
            self.results.append(result)
            
            print(f"{band_name} aggregated metrics: R²={metrics['r2']:.3f}, "
                  f"RMSE={metrics['rmse']:.2f}m, Bias={metrics['bias']:.2f}m, "
                  f"N={metrics['n_samples']:,}")

    def create_combined_plot(self):
        """Create a combined plot with all correlations in a 3x5 grid"""
        if not self.results:
            print("No results available for combined plot")
            return
            
        results_dict = {}
        for result in self.results:
            region = result['region']
            height_source = result['height_source']
            if region not in results_dict:
                results_dict[region] = {}
            results_dict[region][height_source] = result
            
        height_sources = ['GEDI', 'Pauls2024', 'Tolan2024', 'Lang2022', 'Potapov2021']
        regions = ['Kochi', 'Hyogo', 'Tochigi']
        
        fig, axes = plt.subplots(3, 5, figsize=(25, 15))
        fig.suptitle('Height Correlation Analysis: Reference vs Auxiliary Height Sources (Aggregated)', 
                     fontsize=20, fontweight='bold')
        
        for i, region in enumerate(regions):
            for j, height_source in enumerate(height_sources):
                ax = axes[i, j]
                ax.set_title(f'{region} - {height_source}', fontsize=14, fontweight='bold')
                
                if region in results_dict and height_source in results_dict[region]:
                    result = results_dict[region][height_source]
                    ref_clean = result['ref_clean']
                    aux_clean = result['aux_clean']
                    
                    hb = ax.hexbin(ref_clean, aux_clean, gridsize=20, cmap='viridis', mincnt=1, alpha=0.8)
                    
                    min_val = min(np.min(ref_clean), np.min(aux_clean))
                    max_val = max(np.max(ref_clean), np.max(aux_clean))
                    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
                    
                    metrics_text = (f"R² = {result['r2']:.3f}\n"
                                    f"RMSE = {result['rmse']:.1f}m\n"
                                    f"N = {result['n_samples']:,}")
                    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, 
                           verticalalignment='top', fontsize=10, 
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                else:
                    ax.text(0.5, 0.5, 'No data available', transform=ax.transAxes, 
                           ha='center', va='center', fontsize=12)
                
                if i == 2:
                    ax.set_xlabel('Reference Height (m)', fontsize=12)
                if j == 0:
                    ax.set_ylabel('Auxiliary Height (m)', fontsize=12)
                
                ax.grid(True, alpha=0.3)
                ax.set_aspect('equal', adjustable='box')

        combined_output = os.path.join(self.output_dir, 'height_correlation_combined_aggregated.png')
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(combined_output, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved combined correlation plot: {combined_output}")
    
    def run_simple_analysis(self):
        """Run simple correlation analysis for all regions"""
        print("Starting Simple Height Correlation Analysis")
        print("=" * 60)
        print("Testing with multiple patches per region...")
        
        for region_id in self.regions.keys():
            self.analyze_region_simple(region_id)
        
        # Save results summary
        if self.results:
            results_df = pd.DataFrame(self.results)
            results_file = os.path.join(self.output_dir, 'height_correlation_simple_test.csv')
            results_df.to_csv(results_file, index=False)
            print(f"\nResults saved to: {results_file}")
            
            # Create combined plot
            print("\nCreating combined correlation plot...")
            self.create_combined_plot()
            
            # Print summary
            print("\n" + "=" * 60)
            print("SIMPLE TEST CORRELATION ANALYSIS SUMMARY")
            print("=" * 60)
            
            for _, row in results_df.iterrows():
                print(f"{row['region']} - {row['height_source']}:")
                print(f"  Patch: {row['patch_file']}")
                print(f"  R² = {row['r2']:.3f}")
                print(f"  RMSE = {row['rmse']:.2f} m")
                print(f"  Bias = {row['bias']:.2f} m")
                print(f"  Sample size = {row['n_samples']:,}")
                print()
        else:
            print("No results generated")

def main():
    """Main execution function"""
    # Define paths
    downloads_dir = "/home/WUR/ishik001/CHM_Image/downloads"
    patches_dir = "/home/WUR/ishik001/CHM_Image/chm_outputs" 
    output_dir = "/home/WUR/ishik001/CHM_Image/chm_outputs/plot_analysis_simple_test"
    
    # Create analyzer and run simple analysis
    analyzer = SimpleHeightCorrelationAnalyzer(downloads_dir, patches_dir, output_dir)
    analyzer.run_simple_analysis()

if __name__ == "__main__":
    main()
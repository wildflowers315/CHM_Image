#!/usr/bin/env python3
"""
Height Correlation Analysis - Full Version

Aggregates data from ALL available patches with GEDI data to produce
a comprehensive correlation analysis for each region.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import the centralized analyzer
import sys
sys.path.append('/home/WUR/ishik001/CHM_Image')
from analysis.height_analysis_utils import HeightCorrelationAnalyzer

# Set up matplotlib for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class FullHeightCorrelationAnalyzer(HeightCorrelationAnalyzer):
    """Analyzer for full regional height correlation, inheriting from the utility class."""
    
    def __init__(self, downloads_dir, patches_dir, output_dir):
        super().__init__(downloads_dir, patches_dir, output_dir)
        self.results = []
        print(f"Output directory for full analysis: {output_dir}")

    def analyze_region_full(self, region_id):
        """Analyze height correlations for all relevant patches in a region."""
        region_name = self.regions[region_id]
        print(f"\n{'='*20} Full Analysis: {region_name} ({region_id}) {'='*20}")
        
        ref_path = os.path.join(self.downloads_dir, f"{region_id}.tif")
        if not os.path.exists(ref_path):
            print(f"Warning: Reference file not found: {ref_path}")
            return

        # Use the new function to get all patches with GEDI data
        all_gedi_patches = self.get_all_patches_with_gedi(region_id)
        if not all_gedi_patches:
            print(f"No patches with GEDI data found for {region_id}. Skipping region.")
            return

        height_data_aggregated = {}

        for patch_path, patch_num in all_gedi_patches:
            print(f"-- Processing patch {patch_num} for {region_name} --")
            height_bands_info, _ = self.extract_height_bands_info(patch_path)
            if not height_bands_info:
                continue

            for band_name, band_info in height_bands_info.items():
                if band_name == 'GEDI':
                    ref_aligned, aux_aligned = self.align_gedi_points_with_reference(
                        ref_path, patch_path, band_info['index']
                    )
                else:
                    ref_aligned, aux_aligned = self.align_patch_with_reference(
                        ref_path, patch_path, band_info['index']
                    )

                if ref_aligned is None or aux_aligned is None:
                    continue

                valid_mask = (~np.isnan(ref_aligned) & (ref_aligned > 0) &
                              ~np.isnan(aux_aligned) & (aux_aligned > 0))
                
                if np.sum(valid_mask) == 0:
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

            metrics = self.calculate_metrics(combined_ref, combined_aux)
            if not metrics:
                continue

            output_file = os.path.join(self.output_dir, f"{region_name.lower()}_{band_name.lower()}_full_aggregated.png")
            self.create_correlation_plot(combined_ref, combined_aux, metrics,
                                       f"{region_name} (Full Aggregation)", band_name, output_file)

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
            print(f"{band_name} aggregated metrics: R²={metrics['r2']:.3f}, RMSE={metrics['rmse']:.2f}m, N={metrics['n_samples']:,}")

    def create_combined_plot_full(self):
        """Create a combined plot for the full analysis."""
        if not self.results:
            print("No results for combined plot.")
            return

        results_dict = {res['region']: {} for res in self.results}
        for res in self.results:
            results_dict[res['region']][res['height_source']] = res

        height_sources = ['GEDI', 'Pauls2024', 'Tolan2024', 'Lang2022', 'Potapov2021']
        regions = ['Kochi', 'Hyogo', 'Tochigi']

        fig, axes = plt.subplots(3, 5, figsize=(25, 15))
        fig.suptitle('Full Height Correlation Analysis (All Patches)', fontsize=20, fontweight='bold')

        for i, region in enumerate(regions):
            for j, height_source in enumerate(height_sources):
                ax = axes[i, j]
                ax.set_title(f'{region} - {height_source}', fontsize=14, fontweight='bold')

                if region in results_dict and height_source in results_dict[region]:
                    result = results_dict[region][height_source]
                    ax.hexbin(result['ref_clean'], result['aux_clean'], gridsize=20, cmap='viridis', mincnt=1)
                    min_val = min(np.min(result['ref_clean']), np.min(result['aux_clean']))
                    max_val = max(np.max(result['ref_clean']), np.max(result['aux_clean']))
                    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
                    metrics_text = f"R²={result['r2']:.3f}\nRMSE={result['rmse']:.1f}m\nN={result['n_samples']:,}"
                    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, va='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                else:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center')

                if i == 2: ax.set_xlabel('Reference Height (m)')
                if j == 0: ax.set_ylabel('Auxiliary Height (m)')
                ax.grid(True, alpha=0.3)
                ax.set_aspect('equal', adjustable='box')

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(os.path.join(self.output_dir, 'height_correlation_combined_full.png'), dpi=300)
        plt.close()
        print("Saved full combined correlation plot.")

    def run_full_analysis(self):
        """Run the full correlation analysis for all regions."""
        for region_id in self.regions.keys():
            self.analyze_region_full(region_id)
        
        if self.results:
            pd.DataFrame(self.results).to_csv(os.path.join(self.output_dir, 'height_correlation_full_summary.csv'), index=False)
            print("\nFull analysis summary saved.")
            self.create_combined_plot_full()
        else:
            print("No results generated from the full analysis.")

def main():
    """Main execution function."""
    downloads_dir = "/home/WUR/ishik001/CHM_Image/downloads"
    patches_dir = "/home/WUR/ishik001/CHM_Image/chm_outputs"
    output_dir = "/home/WUR/ishik001/CHM_Image/chm_outputs/plot_analysis_full"
    
    analyzer = FullHeightCorrelationAnalyzer(downloads_dir, patches_dir, output_dir)
    analyzer.run_full_analysis()

if __name__ == "__main__":
    main()

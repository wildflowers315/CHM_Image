#!/usr/bin/env python3
"""
Height Correlation Analysis Utilities

Reusable components for height correlation analysis between reference data
and auxiliary height sources (GEDI, canopy height products).
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Import existing utilities
import sys
sys.path.append('/home/WUR/ishik001/CHM_Image')
from raster_utils import load_and_align_rasters
import pyproj
from rasterio.warp import transform

class HeightCorrelationAnalyzer:
    """Utility class for height correlation analysis"""
    
    def __init__(self, downloads_dir, patches_dir, output_dir):
        self.downloads_dir = downloads_dir
        self.patches_dir = patches_dir
        self.output_dir = output_dir
        
        # Corrected region mapping
        self.regions = {
            'dchm_04hf3': 'Kochi',    # Fixed: 04hf3 is Kochi
            'dchm_05LE4': 'Hyogo',   # Fixed: 05LE4 is Hyogo
            'dchm_09gd4': 'Tochigi'
        }
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def get_optimal_gedi_patches(self, region_id, max_patches=3):
        """Get patches with highest GEDI point density for a region"""
        # Optimal patches based on GEDI analysis (corrected region mapping)
        gedi_patches = {
            'dchm_04hf3': [0],  # Kochi - patch 0 has 130 GEDI pixels
            'dchm_05LE4': [33, 27, 13],  # Hyogo - top 3 patches (1118, 977, 973 pixels)
            'dchm_09gd4': [42, 49, 12]  # Tochigi - top 3 patches (128, 102, 89 pixels)
        }
        
        if region_id not in gedi_patches:
            print(f"No GEDI patches defined for {region_id}")
            return []
            
        # Find available patches with GEDI data
        selected_patches = []
        for patch_num in gedi_patches[region_id][:max_patches]:
            pattern = os.path.join(self.patches_dir, f"{region_id}_embedding_bandNum70_*patch{patch_num:04d}.tif")
            patch_files = glob.glob(pattern)
            
            if patch_files:
                selected_patches.append((patch_files[0], patch_num))
                print(f"Found GEDI patch for {region_id}: {os.path.basename(patch_files[0])} (patch {patch_num})")
        
        return selected_patches

    def get_all_patches_with_gedi(self, region_id):
        """Finds all patches for a region that contain a valid GEDI band."""
        pattern = os.path.join(self.patches_dir, f"{region_id}_embedding_bandNum70_*.tif")
        all_patches = glob.glob(pattern)
        
        gedi_patches_found = []
        print(f"Scanning {len(all_patches)} patches for GEDI data in {region_id}...")

        for patch_path in all_patches:
            try:
                height_bands_info, _ = self.extract_height_bands_info(patch_path)
                if 'GEDI' in height_bands_info:
                    patch_num = int(patch_path.split('patch')[-1].split('.')[0])
                    gedi_patches_found.append((patch_path, patch_num))
            except Exception as e:
                print(f"Could not process {os.path.basename(patch_path)}: {e}")

        print(f"Found {len(gedi_patches_found)} patches with a valid GEDI band.")
        return gedi_patches_found
    
    def extract_height_bands_info(self, patch_path):
        """Extract height band information from patch"""
        height_band_names = {
            'rh': 'GEDI',
            'ch_pauls2024': 'Pauls2024',
            'ch_tolan2024': 'Tolan2024', 
            'ch_lang2022': 'Lang2022',
            'ch_potapov2021': 'Potapov2021'
        }
        
        height_bands_info = {}
        forest_mask_index = None
        
        with rasterio.open(patch_path) as src:
            band_descriptions = [src.descriptions[i] for i in range(src.count)]
            
            for i, desc in enumerate(band_descriptions):
                if desc and desc in height_band_names:
                    height_bands_info[height_band_names[desc]] = {
                        'index': i,
                        'name': desc
                    }
                elif desc == 'forest_mask':
                    forest_mask_index = i
        
        return height_bands_info, forest_mask_index

    def align_patch_with_reference(self, ref_path, patch_path, height_band_index):
        """Align patch height band with reference data"""
        try:
            # Create temporary single-band raster for the height band
            temp_path = f"/tmp/temp_height_band_{os.getpid()}_{height_band_index}.tif"
            
            with rasterio.open(patch_path) as src:
                # Read the specific height band
                height_data = src.read(height_band_index + 1)  # rasterio uses 1-based indexing
                
                # Write temporary single-band raster
                profile = src.profile.copy()
                profile.update({
                    'count': 1,
                    'dtype': rasterio.float32
                })
                
                with rasterio.open(temp_path, 'w', **profile) as dst:
                    dst.write(height_data.astype(rasterio.float32), 1)
            
            # Use proven alignment function
            height_aligned, ref_aligned, transform, mask = load_and_align_rasters(
                temp_path, ref_path, forest_mask_path=None, output_dir=None
            )
            
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return ref_aligned, height_aligned
            
        except Exception as e:
            print(f"Error aligning patch data: {e}")
            return None, None

    def align_gedi_points_with_reference(self, ref_path, patch_path, gedi_band_index):
        """Extracts GEDI points, reprojects them, and samples the reference raster."""
        try:
            with rasterio.open(patch_path) as patch_src, rasterio.open(ref_path) as ref_src:
                patch_band = patch_src.read(gedi_band_index + 1)
                patch_transform = patch_src.transform
                patch_crs = pyproj.CRS(patch_src.crs)
                ref_crs = pyproj.CRS(ref_src.crs)

                rows, cols = np.where((patch_band > 0) & ~np.isnan(patch_band))
                if len(rows) == 0:
                    return None, None

                xs, ys = rasterio.transform.xy(patch_transform, rows, cols)
                
                transformer = pyproj.Transformer.from_crs(patch_crs, ref_crs, always_xy=True)
                ref_xs, ref_ys = transformer.transform(xs, ys)

                ref_values = np.array(list(ref_src.sample(zip(ref_xs, ref_ys))))[:, 0]
                patch_values = patch_band[rows, cols]

                return ref_values, patch_values

        except Exception as e:
            print(f"Error during GEDI point alignment: {e}")
            return None, None
    
    def calculate_metrics(self, ref_data, aux_data, min_samples=50):
        """Calculate correlation metrics with outlier filtering"""
        # Remove NaN values
        valid_mask = ~(np.isnan(ref_data) | np.isnan(aux_data))
        
        if np.sum(valid_mask) < min_samples:
            return None
            
        ref_valid = ref_data[valid_mask]
        aux_valid = aux_data[valid_mask]
        
        # Remove extreme outliers and zero reference heights
        ref_mean, ref_std = np.mean(ref_valid), np.std(ref_valid)
        aux_mean, aux_std = np.mean(aux_valid), np.std(aux_valid)
        
        outlier_mask = (
            (np.abs(ref_valid - ref_mean) < 3 * ref_std) &
            (np.abs(aux_valid - aux_mean) < 3 * aux_std) &
            (ref_valid > 0) & (aux_valid >= 0) &  # Exclude zero reference heights
            (ref_valid <= 100) & (aux_valid <= 100)  # Remove unrealistic heights
        )
        
        if np.sum(outlier_mask) < min_samples:
            return None
            
        ref_clean = ref_valid[outlier_mask]
        aux_clean = aux_valid[outlier_mask]
        
        # Calculate metrics
        r2 = r2_score(ref_clean, aux_clean)
        rmse = np.sqrt(mean_squared_error(ref_clean, aux_clean))
        bias = np.mean(aux_clean - ref_clean)
        
        return {
            'r2': r2,
            'rmse': rmse,
            'bias': bias,
            'n_samples': len(ref_clean),
            'ref_mean': np.mean(ref_clean),
            'aux_mean': np.mean(aux_clean),
            'ref_std': np.std(ref_clean),
            'aux_std': np.std(aux_clean),
            'ref_clean': ref_clean,
            'aux_clean': aux_clean
        }
    
    def create_correlation_plot(self, ref_data, aux_data, metrics, region_name, 
                              aux_name, output_file, plot_style='hexbin'):
        """Create correlation scatter plot with hexbin density visualization"""
        if metrics is None:
            print(f"No metrics available for {region_name} - {aux_name}")
            return
        
        # Use clean data from metrics calculation
        ref_clean = metrics['ref_clean']
        aux_clean = metrics['aux_clean']
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if plot_style == 'hexbin':
            # Hexagonal binning plot for density-based coloring
            hb = ax.hexbin(ref_clean, aux_clean, gridsize=30, cmap='viridis', mincnt=1, alpha=0.8)
            
            # Add colorbar
            cb = plt.colorbar(hb, ax=ax)
            cb.set_label('Point Density', fontsize=14)
        else:
            # Simple scatter plot
            ax.scatter(ref_clean, aux_clean, alpha=0.6, s=20, color='steelblue')
        
        # Add 1:1 reference line
        min_val = min(np.min(ref_clean), np.min(aux_clean))
        max_val = max(np.max(ref_clean), np.max(aux_clean))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=3, 
                label='1:1 Reference Line')
        
        # Add regression line
        if len(ref_clean) > 1:
            coeffs = np.polyfit(ref_clean, aux_clean, 1)
            poly_line = np.poly1d(coeffs)
            ax.plot(ref_clean, poly_line(ref_clean), 'g-', linewidth=2, 
                   label=f'Regression Line (y = {coeffs[0]:.3f}x + {coeffs[1]:.3f})')
        
        # Set labels and title
        ax.set_xlabel('Reference Height (m)', fontsize=14)
        ax.set_ylabel(f'{aux_name} Height (m)', fontsize=14)
        ax.set_title(f'{region_name}: Reference vs {aux_name} Height', 
                    fontsize=16, fontweight='bold')
        
        # Add metrics text
        metrics_text = (
            f"R² = {metrics['r2']:.3f}\n"
            f"RMSE = {metrics['rmse']:.2f} m\n"
            f"Bias = {metrics['bias']:.2f} m\n"
            f"N = {metrics['n_samples']:,}"
        )
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, 
               verticalalignment='top', fontsize=12, 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add legend with improved styling
        legend = ax.legend(loc='lower right', fontsize=12)
        legend.get_frame().set_alpha(0.8)
        legend.get_frame().set_facecolor('white')
        
        # Set equal aspect ratio and grid
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)
        
        # Save plot
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved correlation plot: {output_file}")
    
    def process_height_band(self, ref_path, patch_path, band_name, band_info, 
                           region_name, patch_num):
        """Process a single height band and return results"""
        print(f"Processing {band_name}...")
        
        # Align data
        ref_aligned, aux_aligned = self.align_patch_with_reference(
            ref_path, patch_path, band_info['index']
        )
        
        if ref_aligned is None or aux_aligned is None:
            print(f"Failed to align {band_name} data")
            return None
        
        # Apply data filtering based on band type
        if band_name == 'GEDI':
            # GEDI data - only analyze pixels with valid GEDI values and non-zero reference
            valid_mask = (~np.isnan(aux_aligned) & (aux_aligned > 0) & 
                         ~np.isnan(ref_aligned) & (ref_aligned > 0))
            min_threshold = 5  # Very low threshold for sparse GEDI data
        else:
            # Canopy height data - use all valid pixels excluding zero reference heights
            valid_mask = (~np.isnan(ref_aligned) & ~np.isnan(aux_aligned) & 
                         (ref_aligned > 0))
            min_threshold = 50
        
        print(f"Found {np.sum(valid_mask)} valid {band_name} pixels")
        
        if np.sum(valid_mask) < min_threshold:
            print(f"Insufficient {band_name} data: {np.sum(valid_mask)} valid pixels")
            return None
        
        ref_data = ref_aligned[valid_mask]
        aux_data = aux_aligned[valid_mask]
        
        # Calculate metrics (includes outlier filtering)
        metrics = self.calculate_metrics(ref_data, aux_data, min_samples=min_threshold)
        
        if not metrics:
            print(f"Insufficient data for {band_name} metrics calculation after outlier filtering")
            return None
        
        # Create plot with patch number
        output_file = os.path.join(self.output_dir, 
                                 f"{region_name.lower()}_{band_name.lower()}_patch{patch_num:02d}.png")
        self.create_correlation_plot(ref_data, aux_data, metrics, 
                                   f"{region_name} (Patch {patch_num})", band_name, output_file)
        
        # Store results
        result = {
            'region': region_name,
            'height_source': band_name,
            'region_id': patch_path.split('_')[0].split('/')[-1],  # Extract region_id from path
            'patch_number': patch_num,
            'patch_file': os.path.basename(patch_path),
            'r2': metrics['r2'],
            'rmse': metrics['rmse'],
            'bias': metrics['bias'],
            'n_samples': metrics['n_samples'],
            'ref_mean': metrics['ref_mean'],
            'aux_mean': metrics['aux_mean'],
            'ref_std': metrics['ref_std'],
            'aux_std': metrics['aux_std']
        }
        
        print(f"{band_name} metrics: R²={metrics['r2']:.3f}, "
              f"RMSE={metrics['rmse']:.2f}m, Bias={metrics['bias']:.2f}m, "
              f"N={metrics['n_samples']:,}")
        
        return result
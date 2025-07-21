#!/usr/bin/env python3
"""
Simplified Prediction Visualization with RGB and Reference Height
Creates row-layout visualizations: RGB + Reference + 5 Scenarios per region
"""

import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import random

# Try to import Earth Engine with graceful fallback
try:
    import ee
    import requests
    EE_AVAILABLE = True
    print("✅ Earth Engine import successful")
except ImportError as e:
    print(f"⚠️  Earth Engine not available: {e}")
    print("📊 Continuing without RGB composite generation...")
    EE_AVAILABLE = False
    ee = None
    requests = None

class SimplifiedPredictionVisualizer:
    """Create simplified row-layout prediction visualizations"""
    
    def __init__(self, output_dir: str = "chm_outputs/simplified_prediction_visualizations", vis_scale: float = 1.0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.vis_scale = vis_scale
        
        # Initialize Earth Engine (with error handling)
        if EE_AVAILABLE:
            try:
                ee.Initialize(project='my-project-423921')
                self.ee_initialized = True
                print("✅ Earth Engine initialized successfully")
            except Exception as e:
                print(f"⚠️  Earth Engine initialization failed: {e}")
                print("RGB composites will be skipped")
                self.ee_initialized = False
        else:
            self.ee_initialized = False
            print("📊 Earth Engine not available - RGB composites will be skipped")
        
        self.regions = {
            'kochi': {
                'id': '04hf3', 
                'name': 'Kochi Forest Region',
                'aoi_path': 'downloads/dchm_04hf3.geojson',
                'reference_path': 'downloads/dchm_04hf3.tif'
            },
            'hyogo': {
                'id': '05LE4',
                'name': 'Hyogo Forest Region',
                'aoi_path': 'downloads/dchm_05LE4.geojson', 
                'reference_path': 'downloads/dchm_05LE4.tif'
            },
            'tochigi': {
                'id': '09gd4',
                'name': 'Tochigi Forest Region',
                'aoi_path': 'downloads/dchm_09gd4.geojson',
                'reference_path': 'downloads/dchm_09gd4.tif'
            }
        }
        
        # Default scenario configuration (can be modified)
        self.scenarios = {
            'scenario1': {
                'name': 'Scenario 1',
                'subtitle': '(Reference MLP)',
                'path_template': 'chm_outputs/google_embedding_scenario1_predictions/{region}/',
                'performance': 'R²=0.87*'
            },
            'scenario1_5': {
                'name': 'Scenario 1.5', 
                'subtitle': '(GEDI-only)',
                'path_template': 'chm_outputs/scenario1_5_gedi_only_predictions/{region}/',
                'performance': 'R²=-7.75'
            },
            'scenario2a': {
                'name': 'Scenario 2A',
                'subtitle': '(Ensemble)', 
                'path_template': 'chm_outputs/google_embedding_scenario2a_predictions/{region}/',
                'performance': 'R²=0.78'
            },
            'scenario3a': {
                'name': 'Scenario 3A',
                'subtitle': '(From-scratch)',
                'path_template': 'chm_outputs/google_embedding_scenario3a_predictions/{region}/',
                'performance': 'R²=-1.96'
            },
            'scenario3b': {
                'name': 'Scenario 3B',
                'subtitle': '(Fine-tuned)',
                'path_template': 'chm_outputs/google_embedding_scenario3b_predictions/{region}/',
                'performance': 'R²=-1.94'
            }
        }
        
        # Set random seed for consistent patch selection across regions
        self.random_seed = 42
        random.seed(self.random_seed)
    
    def get_single_prediction_file(self, scenario: str, region: str, patch_index: int = None) -> Optional[str]:
        """Get single prediction file per scenario/region with optional patch index selection"""
        if scenario not in self.scenarios:
            return None
            
        pred_dir = self.scenarios[scenario]['path_template'].format(region=region)
        
        if not os.path.exists(pred_dir):
            print(f"⚠️  Directory not found: {pred_dir}")
            return None
        
        # Search patterns in preference order
        patterns = [
            f"{pred_dir}*gedi_only_prediction*.tif",   # GEDI-only files (Scenario 1.5)
            f"{pred_dir}*prediction*.tif",             # Standard prediction files
            f"{pred_dir}*ensemble*.tif",               # Ensemble files  
            f"{pred_dir}*.tif"                         # Any TIF file
        ]
        
        all_files = []
        for pattern in patterns:
            files = glob.glob(pattern)
            if files:
                all_files.extend(files)
                break  # Stop after first successful pattern to avoid duplicates
        
        if not all_files:
            print(f"❌ No prediction file found for {scenario}/{region}")
            return None
        
        # Sort files for consistency
        all_files = sorted(all_files)
        
        # Select file based on patch index (if provided) or random selection
        if patch_index is not None and patch_index < len(all_files):
            selected_file = all_files[patch_index]
        else:
            # Use consistent random selection across scenarios for same region
            random.seed(self.random_seed + hash(region))  # Region-specific but consistent seed
            selected_file = random.choice(all_files)
        
        print(f"✅ Found {scenario}/{region}: {os.path.basename(selected_file)}")
        return selected_file
    
    def generate_rgb_composite(self, region: str, pred_path: str) -> Optional[str]:
        """Generate Sentinel-2 RGB composite (if EE available)"""
        if not self.ee_initialized:
            print(f"⚠️  Skipping RGB composite for {region} - Earth Engine not available")
            return None
            
        # Create output path based on prediction file name
        pred_basename = os.path.splitext(os.path.basename(pred_path))[0]
        output_path = self.output_dir / f"{region}_{pred_basename}_rgb_composite.tif"
        
        if output_path.exists():
            print(f"RGB composite exists: {output_path}")
            return str(output_path)
        
        try:
            # Extract AOI from prediction patch using utils function
            import importlib.util
            spec = importlib.util.spec_from_file_location("utils_module", "./utils.py")
            utils_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(utils_module)
            
            # Create temporary GeoJSON from prediction TIF
            temp_geojson = utils_module.geotiff_to_geojson(pred_path)
            
            # Load the generated GeoJSON as AOI
            with open(temp_geojson, 'r') as f:
                aoi_dict = json.load(f)
            aoi = ee.Geometry(aoi_dict['features'][0]['geometry'])
            
            print(f"    Using AOI from prediction patch: {os.path.basename(pred_path)}")
            
            # Clean up temp file
            os.remove(temp_geojson)
            
            # Create cloud-masked RGB composite
            collection = (
                ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
                .filterDate('2022-01-01', '2022-12-31')
                .filterBounds(aoi)
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                .map(self._mask_s2_clouds)
            )
            
            rgb_composite = collection.select(['B4', 'B3', 'B2']).median().clip(aoi)
            
            # Download
            download_params = {
                'bands': ['B4', 'B3', 'B2'],
                'region': aoi,
                'scale': 10,
                'crs': 'EPSG:4326',
                'format': 'GEO_TIFF'
            }
            
            url = rgb_composite.getDownloadUrl(download_params)
            response = requests.get(url)
            response.raise_for_status()
            
            with open(output_path, 'wb') as fd:
                fd.write(response.content)
            
            print(f"✅ RGB composite downloaded: {output_path}")
            return str(output_path)
            
        except Exception as e:
            print(f"❌ Failed to generate RGB composite for {region}: {e}")
            return None
    
    @staticmethod
    def _mask_s2_clouds(image):
        """Cloud masking for Sentinel-2"""
        qa = image.select('QA60')
        cloud_bit_mask = 1 << 10
        cirrus_bit_mask = 1 << 11
        
        mask = (
            qa.bitwiseAnd(cloud_bit_mask).eq(0)
            .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
        )
        
        return image.updateMask(mask).divide(10000)
    
    def load_and_align_rasters_for_visualization(self, pred_path: str, ref_path: str) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Load and align prediction and reference data using raster_utils for memory efficiency"""
        try:
            # Import the raster utils (same as evaluation script)
            from raster_utils import load_and_align_rasters
            
            # Use the same alignment logic as the existing evaluation system
            pred_data, ref_data, transform, _ = load_and_align_rasters(
                pred_path, ref_path, None, None  # No forest mask, no output dir
            )
            
            # Filter valid data
            valid_mask = ~(np.isnan(ref_data) | np.isnan(pred_data))
            if np.sum(valid_mask) < 100:
                return None, None, None
            
            # Apply basic filtering (same as evaluation script)
            ref_data = np.where((ref_data > 0) & (ref_data <= 100), ref_data, np.nan)
            pred_data = np.where((pred_data >= 0) & (pred_data <= 100), pred_data, np.nan)
            
            metadata = {
                'transform': transform,
                'shape': pred_data.shape
            }
            
            print(f"    Aligned data shape: {pred_data.shape}, valid pixels: {np.sum(~np.isnan(pred_data))}")
            
            return pred_data, ref_data, metadata
            
        except Exception as e:
            print(f"Error aligning rasters: {e}")
            return None, None, None
    
    def load_and_process_raster(self, file_path: str, is_rgb: bool = False, ref_path: str = None) -> Tuple[np.ndarray, dict]:
        """Load and process raster data with memory management"""
        if not is_rgb and ref_path:
            # For height data, use alignment with reference for memory efficiency
            pred_data, _, metadata = self.load_and_align_rasters_for_visualization(file_path, ref_path)
            if pred_data is not None:
                return pred_data, metadata
            else:
                return None, None
        
        # For RGB or single files, load directly but with decimation for large files
        with rasterio.open(file_path) as src:
            height, width = src.height, src.width
            
            # Decimate large files to prevent memory issues
            if height * width > 10000000:  # ~10M pixels
                decimation = max(2, int((height * width / 5000000) ** 0.5))  # Target ~5M pixels
                print(f"    Decimating {os.path.basename(file_path)} by {decimation}x for memory efficiency")
                
                if is_rgb:
                    data = src.read([1, 2, 3], 
                                  out_shape=(3, height // decimation, width // decimation),
                                  resampling=rasterio.enums.Resampling.bilinear)
                    data = np.moveaxis(data, 0, -1)  # Move channels to last axis
                    # Enhance RGB for visualization
                    data = np.clip(data, 0.02, 0.3)
                    data = (data - 0.02) / (0.3 - 0.02)  # Normalize to 0-1
                else:
                    data = src.read(1, 
                                  out_shape=(height // decimation, width // decimation),
                                  resampling=rasterio.enums.Resampling.bilinear)
                    # Filter valid heights
                    data = np.where((data > 0) & (data < 100) & ~np.isnan(data), data, np.nan)
            else:
                if is_rgb:
                    data = src.read([1, 2, 3])
                    data = np.moveaxis(data, 0, -1)
                    data = np.clip(data, 0.02, 0.3)
                    data = (data - 0.02) / (0.3 - 0.02)
                else:
                    data = src.read(1)
                    data = np.where((data > 0) & (data < 100) & ~np.isnan(data), data, np.nan)
            
            metadata = {
                'bounds': src.bounds,
                'transform': src.transform,
                'crs': src.crs,
                'shape': data.shape
            }
            
            return data, metadata
    
    def create_region_visualization(self, region: str, selected_scenarios: List[str] = None, patch_index: int = None) -> str:
        """Create row visualization for a region"""
        if selected_scenarios is None:
            selected_scenarios = list(self.scenarios.keys())
        
        print(f"\n🎨 Creating visualization for {self.regions[region]['name']}")
        print(f"📊 Selected scenarios: {', '.join(selected_scenarios)}")
        if patch_index is not None:
            print(f"🎯 Using patch index: {patch_index}")
        else:
            print(f"🎲 Using random patch selection (seed: {self.random_seed})")
        if self.vis_scale != 1.0:
            print(f"📏 Prediction scaling factor: {self.vis_scale}x")
        
        # Get reference path
        ref_path = self.regions[region]['reference_path']
        
        # Get the first prediction file to use as spatial template
        template_pred_path = None
        for scenario_key in selected_scenarios:
            pred_path = self.get_single_prediction_file(scenario_key, region, patch_index)
            if pred_path and os.path.exists(pred_path):
                template_pred_path = pred_path
                print(f"📐 Using {os.path.basename(pred_path)} as spatial template")
                break
        
        if not template_pred_path:
            print(f"❌ No prediction files found for {region}")
            return None
        
        # Load template prediction and reference aligned to it for consistent shapes
        print(f"🔄 Aligning all data to prediction template...")
        template_pred_data, reference_data, template_metadata = self.load_and_align_rasters_for_visualization(
            template_pred_path, ref_path
        )
        
        if template_pred_data is None or reference_data is None:
            print(f"❌ Failed to align template data for {region}")
            return None
        
        # Calculate height range from both reference and prediction data for optimal legend scale
        ref_valid = reference_data[~np.isnan(reference_data)]
        pred_valid = template_pred_data[~np.isnan(template_pred_data)]
        
        # Combine reference and prediction data to get comprehensive range
        combined_valid = []
        if len(ref_valid) > 0:
            combined_valid.extend(ref_valid)
        if len(pred_valid) > 0:
            # Apply vis_scale to predictions for range calculation
            scaled_pred_valid = pred_valid * self.vis_scale
            combined_valid.extend(scaled_pred_valid)
        
        if len(combined_valid) > 100:
            combined_valid = np.array(combined_valid)
            height_vmin = max(0, np.percentile(combined_valid, 1))
            height_vmax = min(50, np.percentile(combined_valid, 99))
            print(f"📊 Combined height range (ref + scaled pred): {height_vmin:.1f} - {height_vmax:.1f}m")
        elif len(ref_valid) > 100:
            # Fallback to reference only if insufficient combined data
            height_vmin = max(0, np.percentile(ref_valid, 1))
            height_vmax = min(50, np.percentile(ref_valid, 99))
            print(f"📊 Reference height range: {height_vmin:.1f} - {height_vmax:.1f}m")
        else:
            # Final fallback to default range
            height_vmin, height_vmax = 0, 40
            print(f"📊 Using default height range: {height_vmin:.1f} - {height_vmax:.1f}m")
        
        # Determine layout: Skip RGB if not available, Reference + N scenarios
        include_rgb = self.ee_initialized
        n_panels = (1 if include_rgb else 0) + 1 + len(selected_scenarios)  # [RGB] + Reference + scenarios
        
        # Create figure with optimized layout for consistent image sizes
        fig_width = 4 * n_panels + 1  # Extra space for rightmost colorbar
        fig, axes = plt.subplots(1, n_panels, figsize=(fig_width, 6))
        if n_panels == 1:
            axes = [axes]
        
        panel_idx = 0
        
        # Panel 1: RGB Composite (only if available)
        if include_rgb:
            rgb_path = self.generate_rgb_composite(region, template_pred_path)
            
            if rgb_path and os.path.exists(rgb_path):
                try:
                    rgb_data, rgb_metadata = self.load_and_process_raster(rgb_path, is_rgb=True)
                    
                    # Ensure RGB data shape consistency (RGB is HxWx3, we need WxH for extent)
                    if len(rgb_data.shape) == 3:
                        rgb_height, rgb_width = rgb_data.shape[0], rgb_data.shape[1]
                    else:
                        rgb_height, rgb_width = rgb_data.shape
                    
                    # Display RGB with same extent format as height data (256x256 equivalent)
                    axes[panel_idx].imshow(rgb_data, extent=[0, 256, 0, 256], aspect='equal')
                    axes[panel_idx].set_title('RGB\nSentinel-2', fontweight='bold')
                    print(f"    RGB shape: {rgb_data.shape}, displayed as 256×256")
                    
                except Exception as e:
                    print(f"⚠️  Could not display RGB: {e}")
                    axes[panel_idx].text(0.5, 0.5, 'RGB\nError', 
                                       ha='center', va='center', transform=axes[panel_idx].transAxes,
                                       fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
                    axes[panel_idx].set_title('RGB\nSentinel-2', fontweight='bold')
            else:
                axes[panel_idx].text(0.5, 0.5, 'RGB\nUnavailable', 
                                   ha='center', va='center', transform=axes[panel_idx].transAxes,
                                   fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
                axes[panel_idx].set_title('RGB\nSentinel-2', fontweight='bold')
                axes[panel_idx].set_facecolor('lightgray')
            
            axes[panel_idx].set_xticks([])
            axes[panel_idx].set_yticks([])
            panel_idx += 1
        
        # Panel 2: Reference Height (using pre-aligned reference data)
        im_ref = axes[panel_idx].imshow(
            reference_data,
            cmap='plasma',
            vmin=height_vmin,
            vmax=height_vmax,
            extent=[0, 256, 0, 256],  # Fixed 256×256 display extent
            aspect='equal'
        )
        
        # Store reference image for shared colorbar (added at the end)
        shared_colorbar_image = im_ref
        
        axes[panel_idx].set_title('Reference\n(Truth)', fontweight='bold')
        axes[panel_idx].set_xticks([])
        axes[panel_idx].set_yticks([])
        panel_idx += 1
        
        # Panels 3+: Prediction Scenarios (load each aligned to same template)
        for scenario_key in selected_scenarios:
            if panel_idx >= len(axes):
                break
                
            scenario_info = self.scenarios[scenario_key]
            pred_path = self.get_single_prediction_file(scenario_key, region, patch_index)
            
            if pred_path and os.path.exists(pred_path):
                try:
                    # Load this prediction aligned to the template
                    pred_data, _, pred_metadata = self.load_and_align_rasters_for_visualization(pred_path, ref_path)
                    
                    if pred_data is not None:
                        # Apply visualization scaling to prediction data
                        scaled_pred_data = pred_data * self.vis_scale
                        
                        # Use same colormap as reference for consistency
                        axes[panel_idx].imshow(
                            scaled_pred_data,
                            cmap='plasma',  # Same as reference
                            vmin=height_vmin,
                            vmax=height_vmax,
                            extent=[0, 256, 0, 256],  # Fixed 256×256 display extent
                            aspect='equal'
                        )
                        
                        # Title with performance and scaling info
                        if self.vis_scale != 1.0:
                            scale_info = f"\n(scaled {self.vis_scale}x)"
                            title = f"{scenario_info['name']}\n{scenario_info['subtitle']}\n{scenario_info['performance']}{scale_info}"
                        else:
                            title = f"{scenario_info['name']}\n{scenario_info['subtitle']}\n{scenario_info['performance']}"
                        axes[panel_idx].set_title(title, fontweight='bold', fontsize=9)
                    else:
                        print(f"⚠️  Failed to align {scenario_key}")
                        axes[panel_idx].text(0.5, 0.5, f'{scenario_info["name"]}\nAlignment\nFailed',
                                           ha='center', va='center', transform=axes[panel_idx].transAxes,
                                           fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
                        title = f"{scenario_info['name']}\n{scenario_info['subtitle']}"
                        axes[panel_idx].set_title(title, fontweight='bold', fontsize=10)
                    
                except Exception as e:
                    print(f"⚠️  Could not load {scenario_key}: {e}")
                    axes[panel_idx].text(0.5, 0.5, f'{scenario_info["name"]}\nError',
                                       ha='center', va='center', transform=axes[panel_idx].transAxes,
                                       fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
                    title = f"{scenario_info['name']}\n{scenario_info['subtitle']}"
                    axes[panel_idx].set_title(title, fontweight='bold', fontsize=10)
            else:
                axes[panel_idx].text(0.5, 0.5, f'{scenario_info["name"]}\nUnavailable',
                                   ha='center', va='center', transform=axes[panel_idx].transAxes,
                                   fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
                title = f"{scenario_info['name']}\n{scenario_info['subtitle']}"
                axes[panel_idx].set_title(title, fontweight='bold', fontsize=10)
            
            axes[panel_idx].set_xticks([])
            axes[panel_idx].set_yticks([])
            panel_idx += 1
        
        # Add shared colorbar at the rightmost position
        # Create space for colorbar by adjusting subplot parameters
        plt.subplots_adjust(right=0.85, top=0.85)
        
        # Add shared colorbar spanning the rightmost area
        cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.6])  # [left, bottom, width, height]
        cbar = plt.colorbar(shared_colorbar_image, cax=cbar_ax)
        cbar.set_label('Height (m)', rotation=270, labelpad=20, fontsize=12)
        cbar.ax.tick_params(labelsize=10)
        
        # Overall title
        fig.suptitle(f'{self.regions[region]["name"]}: Multi-Scenario Canopy Height Predictions',
                    fontsize=16, fontweight='bold', y=0.95)
        
        # Apply tight layout to image panels only (not colorbar)
        plt.tight_layout(rect=[0, 0, 0.85, 0.95])
        
        # Generate descriptive filename with patch index and plot count
        n_scenarios = len(selected_scenarios)
        if patch_index is not None:
            patch_info = f"patch{patch_index}"
        else:
            patch_info = f"seed{self.random_seed}"
        
        if self.vis_scale != 1.0:
            scale_info = f"_scale{self.vis_scale}x"
        else:
            scale_info = ""
        
        filename = f"{region}_{n_scenarios}scenarios_{patch_info}{scale_info}_predictions.png"
        output_path = self.output_dir / filename
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Visualization saved: {output_path}")
        return str(output_path)
    
    def create_all_visualizations(self, selected_scenarios: List[str] = None, patch_index: int = None) -> Dict[str, str]:
        """Create visualizations for all regions"""
        results = {}
        
        if selected_scenarios is None:
            selected_scenarios = list(self.scenarios.keys())
        
        print("🎨 Creating Simplified Multi-Scenario Prediction Visualizations")
        print(f"📊 Scenarios: {', '.join(selected_scenarios)}")
        if patch_index is not None:
            print(f"🎯 Patch Index: {patch_index} (consistent across all regions)")
        else:
            print(f"🎲 Random Selection: seed {self.random_seed} (consistent per region)")
        print("=" * 70)
        
        for region in self.regions.keys():
            results[region] = self.create_region_visualization(region, selected_scenarios, patch_index)
        
        print(f"\n✅ All visualizations completed!")
        print(f"📁 Results saved in: {self.output_dir}")
        
        return results


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create simplified prediction visualizations')
    parser.add_argument('--scenarios', nargs='+', 
                       choices=['scenario1', 'scenario1_5', 'scenario2a', 'scenario3a', 'scenario3b'],
                       default=['scenario1', 'scenario1_5', 'scenario2a', 'scenario3a', 'scenario3b'],
                       help='Scenarios to include in visualization')
    parser.add_argument('--output-dir', default='chm_outputs/simplified_prediction_visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--patch-index', type=int, default=None,
                       help='Specific patch index to use (0-based). If not specified, random selection is used')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for consistent patch selection (default: 42)')
    parser.add_argument('--vis-scale', type=float, default=1.0,
                       help='Scaling factor for prediction visualization (default: 1.0)')
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = SimplifiedPredictionVisualizer(args.output_dir, args.vis_scale)
    
    # Override random seed if specified
    if args.random_seed != 42:
        visualizer.random_seed = args.random_seed
        random.seed(args.random_seed)
        print(f"🎲 Using custom random seed: {args.random_seed}")
    
    # Generate visualizations
    results = visualizer.create_all_visualizations(args.scenarios, args.patch_index)
    
    # Summary
    print(f"\n📋 VISUALIZATION SUMMARY:")
    print("=" * 50)
    for region, path in results.items():
        if path:
            print(f"✅ {region.title()}: {path}")
        else:
            print(f"❌ {region.title()}: Failed")
    
    print(f"\n🎉 Simplified Prediction Visualization Pipeline Completed!")


if __name__ == "__main__":
    main()
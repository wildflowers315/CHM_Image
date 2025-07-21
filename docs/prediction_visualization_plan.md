# Prediction Visualization Plan: RGB Overlay with Sentinel-2

## Executive Summary

This document outlines a comprehensive plan for visualizing canopy height predictions from different scenarios (Google Embedding Scenarios 1, 1.5, 2A, 3A, 3B) overlaid with Sentinel-2 RGB imagery. The visualization will help assess prediction quality, spatial patterns, and regional differences across the three study regions (Kochi, Hyogo, Tochigi).

## Objectives

1. **Spatial Context**: Provide RGB visual context for understanding prediction quality
2. **Cross-Scenario Comparison**: Compare prediction patterns across different modeling approaches
3. **Regional Analysis**: Assess how predictions vary across different forest ecosystems
4. **Quality Assessment**: Identify areas of good/poor prediction performance
5. **Scientific Communication**: Create publication-ready visualizations for research dissemination

## Data Requirements

### Prediction Data Sources
```
Available Prediction Results:
├── Scenario 1: chm_outputs/google_embedding_scenario1_predictions/
│   ├── kochi/     # 04hf3 region predictions  
│   ├── hyogo/     # 05LE4 region predictions
│   └── tochigi/   # 09gd4 region predictions
├── Scenario 1.5: chm_outputs/scenario1_5_gedi_only_predictions/
│   ├── kochi/     # GEDI-only predictions
│   ├── hyogo/     # GEDI-only predictions  
│   └── tochigi/   # GEDI-only predictions
├── Scenario 2A: chm_outputs/google_embedding_scenario2a_predictions/
│   ├── kochi/     # Ensemble predictions
│   ├── hyogo/     # Ensemble predictions
│   └── tochigi/   # Ensemble predictions
├── Scenario 3A: chm_outputs/google_embedding_scenario3a_predictions/
│   ├── kochi/     # From-scratch GEDI + fixed ensemble
│   ├── hyogo/     # From-scratch GEDI + fixed ensemble
│   └── tochigi/   # From-scratch GEDI + fixed ensemble
└── Scenario 3B: chm_outputs/google_embedding_scenario3b_predictions/
    ├── kochi/     # Fine-tuned GEDI + fixed ensemble
    ├── hyogo/     # Fine-tuned GEDI + fixed ensemble
    └── tochigi/   # Fine-tuned GEDI + fixed ensemble
```

### Reference Data Sources
```
Reference Height Data:
├── downloads/dchm_04hf3.tif    # Kochi reference heights
├── downloads/dchm_05LE4.tif    # Hyogo reference heights  
└── downloads/dchm_09gd4.tif    # Tochigi reference heights

AOI Boundary Data:
├── downloads/dchm_04hf3.geojson # Kochi boundary
├── downloads/dchm_05LE4.geojson # Hyogo boundary
└── downloads/dchm_09gd4.geojson # Tochigi boundary
```

## Visualization Architecture

### 1. Sentinel-2 RGB Base Layer Generation

**Objective**: Create cloud-free RGB composites for each region as visualization base layers.

#### Technical Approach
```python
# Leverage existing sentinel2_source.py with modifications for RGB focus
def get_sentinel2_rgb_composite(
    aoi_geojson_path: str,
    year: int = 2022,
    start_date: str = "01-01", 
    end_date: str = "12-31",
    cloud_threshold: int = 20,
    target_resolution: int = 10,
    patch_size: int = 256
) -> ee.Image:
    """Generate cloud-free RGB composite for visualization"""
    
    # Load AOI from GeoJSON
    with open(aoi_geojson_path, 'r') as f:
        aoi_dict = json.load(f)
    aoi = ee.Geometry(aoi_dict['features'][0]['geometry'])
    
    # Get Sentinel-2 collection with cloud masking
    collection = (
        ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
        .filterDate(f'{year}-{start_date}', f'{year}-{end_date}')
        .filterBounds(aoi)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_threshold))
        .map(mask_s2_clouds)
    )
    
    # Create RGB composite
    rgb_composite = collection.select(['B4', 'B3', 'B2']).median()
    
    return rgb_composite.clip(aoi)

def mask_s2_clouds(image):
    """Enhanced cloud masking for better RGB visualization"""
    qa = image.select('QA60')
    
    # Bits 10 and 11 are clouds and cirrus
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    
    # Clear conditions mask
    mask = (
        qa.bitwiseAnd(cloud_bit_mask).eq(0)
        .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    )
    
    return image.updateMask(mask).divide(10000)
```

#### RGB Composite Parameters
```python
RGB_VISUALIZATION_PARAMS = {
    'year': 2022,  # Match prediction data year
    'cloud_threshold': 15,  # Stricter cloud filtering for visualization
    'target_resolution': 10,  # Match prediction resolution
    'bands': ['B4', 'B3', 'B2'],  # Red, Green, Blue
    'vis_params': {
        'min': 0.0,
        'max': 0.3,
        'gamma': 1.2  # Enhance contrast for forest visualization
    }
}
```

### 2. Multi-Panel Prediction Overlay System

#### Panel Layout Design
```
Visualization Layout (6 panels per region):
┌─────────────────────────────────────────────────────────────────┐
│                    Region Name (e.g., "Kochi Forest Region")    │
├─────────────┬─────────────┬─────────────┬─────────────┬─────────────┬─────────────┤
│ Reference   │ Scenario 1  │ Scenario    │ Scenario 2A │ Scenario 3A │ Scenario 3B │
│ Height      │ (Ref MLP)   │ 1.5 (GEDI)  │ (Ensemble)  │ (From-      │ (Fine-      │
│ (Truth)     │ R²=0.87*    │ R²=-7.75    │ R²=0.78     │ scratch)    │ tuned)      │
│             │             │             │             │ R²=-1.96    │ R²=-1.94    │
└─────────────┴─────────────┴─────────────┴─────────────┴─────────────┴─────────────┘

* Training region performance; cross-region varies
```

#### Color Scheme Strategy
```python
HEIGHT_COLORMAP_PARAMS = {
    'colormap': 'viridis',  # Good for scientific visualization
    'min_height': 0,
    'max_height': 40,  # Based on reference height distribution
    'transparency': 0.7,  # Allow RGB base to show through
    'bad_prediction_color': 'red',  # Highlight problematic areas
    'no_data_color': 'transparent'
}

RGB_BASE_PARAMS = {
    'min': [0.02, 0.02, 0.02],  # Darker base for contrast
    'max': [0.25, 0.25, 0.25], 
    'gamma': [0.9, 1.1, 1.0],  # Enhance green channel for forests
}
```

### 3. Implementation Structure

#### Core Visualization Script: `create_prediction_rgb_overlays.py`

```python
#!/usr/bin/env python3
"""
Prediction Visualization with Sentinel-2 RGB Overlays
Creates comprehensive visualizations of canopy height predictions overlaid on RGB imagery
"""

import os
import ee
import json
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import geopandas as gpd
from pathlib import Path
import requests
from typing import Dict, List, Tuple, Optional

class PredictionRGBVisualizer:
    """Main class for creating prediction overlays with RGB imagery"""
    
    def __init__(self, output_dir: str = "chm_outputs/prediction_visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Earth Engine
        try:
            ee.Initialize()
        except Exception:
            ee.Authenticate()
            ee.Initialize()
        
        # Define region mapping
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
        
        # Define scenarios with performance metrics
        self.scenarios = {
            'reference': {
                'name': 'Reference Height',
                'description': 'Ground Truth Data',
                'path_pattern': 'downloads/dchm_{region_id}.tif',
                'performance': 'Truth',
                'color_scheme': 'plasma'
            },
            'scenario1': {
                'name': 'Scenario 1 (Reference MLP)',
                'description': 'Google Embedding MLP',
                'path_pattern': 'chm_outputs/google_embedding_scenario1_predictions/{region}/',
                'performance': {'train': 0.8734, 'cross_region': -1.68},
                'color_scheme': 'viridis'
            },
            'scenario1_5': {
                'name': 'Scenario 1.5 (GEDI-only)', 
                'description': 'Pure GEDI Model',
                'path_pattern': 'chm_outputs/scenario1_5_gedi_only_predictions/{region}/',
                'performance': {'average': -7.746},
                'color_scheme': 'Reds'  # Red for poor performance
            },
            'scenario2a': {
                'name': 'Scenario 2A (Ensemble)',
                'description': 'GEDI + Reference Ensemble', 
                'path_pattern': 'chm_outputs/google_embedding_scenario2a_predictions/{region}/',
                'performance': {'train': 0.7844, 'cross_region': -1.95},
                'color_scheme': 'viridis'
            },
            'scenario3a': {
                'name': 'Scenario 3A (From-scratch)',
                'description': 'Target GEDI + Fixed Ensemble',
                'path_pattern': 'chm_outputs/google_embedding_scenario3a_predictions/{region}/',
                'performance': {'average': -1.955},
                'color_scheme': 'viridis'
            },
            'scenario3b': {
                'name': 'Scenario 3B (Fine-tuned)',
                'description': 'Fine-tuned GEDI + Fixed Ensemble',
                'path_pattern': 'chm_outputs/google_embedding_scenario3b_predictions/{region}/',
                'performance': {'average': -1.944},
                'color_scheme': 'viridis'
            }
        }

    def generate_rgb_composite(self, region: str) -> str:
        """Generate Sentinel-2 RGB composite for region"""
        region_info = self.regions[region]
        output_path = self.output_dir / f"{region}_sentinel2_rgb_composite.tif"
        
        if output_path.exists():
            print(f"RGB composite already exists: {output_path}")
            return str(output_path)
        
        # Load AOI
        with open(region_info['aoi_path'], 'r') as f:
            aoi_dict = json.load(f)
        aoi = ee.Geometry(aoi_dict['features'][0]['geometry'])
        
        # Create Sentinel-2 RGB composite
        collection = (
            ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
            .filterDate('2022-01-01', '2022-12-31')
            .filterBounds(aoi)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 15))
            .map(self._mask_s2_clouds)
        )
        
        rgb_composite = collection.select(['B4', 'B3', 'B2']).median().clip(aoi)
        
        # Download RGB composite
        download_params = {
            'bands': ['B4', 'B3', 'B2'],
            'region': aoi,
            'scale': 10,
            'crs': 'EPSG:4326',
            'format': 'GEO_TIFF'
        }
        
        try:
            url = rgb_composite.getDownloadUrl(download_params)
            response = requests.get(url)
            response.raise_for_status()
            
            with open(output_path, 'wb') as fd:
                fd.write(response.content)
            
            print(f"RGB composite downloaded: {output_path}")
            return str(output_path)
            
        except Exception as e:
            print(f"Error downloading RGB composite for {region}: {e}")
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

    def find_prediction_files(self, scenario: str, region: str) -> List[str]:
        """Find prediction files for a scenario and region"""
        if scenario == 'reference':
            return [self.regions[region]['reference_path']]
        
        pattern = self.scenarios[scenario]['path_pattern'].format(
            region=region,
            region_id=self.regions[region]['id']
        )
        
        # Use glob to find actual files
        import glob
        if os.path.isdir(pattern):
            # Look for TIF files in directory
            pred_files = glob.glob(os.path.join(pattern, "*.tif"))
            pred_files.extend(glob.glob(os.path.join(pattern, "*prediction*.tif")))
            pred_files.extend(glob.glob(os.path.join(pattern, "*ensemble*.tif")))
        else:
            pred_files = glob.glob(pattern)
        
        # Return top 3 files for consistency
        return sorted(list(set(pred_files)))[:3]

    def create_region_visualization(self, region: str) -> str:
        """Create comprehensive visualization for a region"""
        print(f"\n📊 Creating visualization for {self.regions[region]['name']}")
        
        # Generate RGB base layer
        rgb_path = self.generate_rgb_composite(region)
        if not rgb_path:
            print(f"❌ Failed to generate RGB composite for {region}")
            return None
        
        # Load RGB data
        with rasterio.open(rgb_path) as src:
            rgb_data = src.read([1, 2, 3])  # R, G, B
            transform = src.transform
            bounds = src.bounds
            crs = src.crs
        
        # Create 6-panel visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Color normalization for height data
        height_norm = plt.Normalize(vmin=0, vmax=40)
        
        panel_idx = 0
        for scenario_key, scenario_info in self.scenarios.items():
            if panel_idx >= 6:  # Limit to 6 panels
                break
                
            ax = axes[panel_idx]
            
            # Display RGB base layer
            rgb_display = np.moveaxis(rgb_data, 0, -1)  # Move channels to last axis
            rgb_display = np.clip(rgb_display, 0.02, 0.25)  # Enhance contrast
            rgb_display = (rgb_display - 0.02) / (0.25 - 0.02)  # Normalize to 0-1
            
            ax.imshow(rgb_display, extent=[bounds.left, bounds.right, bounds.bottom, bounds.top])
            
            # Overlay predictions if available
            pred_files = self.find_prediction_files(scenario_key, region)
            
            if pred_files:
                # Use first available prediction file
                pred_path = pred_files[0]
                
                try:
                    with rasterio.open(pred_path) as pred_src:
                        pred_data = pred_src.read(1)
                        
                        # Mask invalid values
                        pred_data = np.where(
                            (pred_data > 0) & (pred_data < 100) & ~np.isnan(pred_data),
                            pred_data,
                            np.nan
                        )
                        
                        # Create overlay
                        im = ax.imshow(
                            pred_data,
                            extent=[bounds.left, bounds.right, bounds.bottom, bounds.top], 
                            alpha=0.6,
                            cmap=scenario_info['color_scheme'],
                            norm=height_norm
                        )
                        
                        # Add colorbar for first panel
                        if panel_idx == 0:
                            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                            cbar.set_label('Height (m)', rotation=270, labelpad=20)
                
                except Exception as e:
                    print(f"⚠️  Could not load predictions for {scenario_key}/{region}: {e}")
                    # Show RGB only
                    pass
            
            # Formatting
            ax.set_title(f"{scenario_info['name']}\n{scenario_info['description']}", 
                        fontsize=11, fontweight='bold')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.grid(True, alpha=0.3)
            
            panel_idx += 1
        
        # Hide unused panels
        for idx in range(panel_idx, len(axes)):
            axes[idx].set_visible(False)
        
        # Overall title
        fig.suptitle(f'{self.regions[region]["name"]}: Canopy Height Predictions on Sentinel-2 RGB',
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        # Save visualization
        output_path = self.output_dir / f"{region}_prediction_rgb_overlay.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Visualization saved: {output_path}")
        return str(output_path)

    def create_all_visualizations(self) -> Dict[str, str]:
        """Create visualizations for all regions"""
        results = {}
        
        print("🌍 Creating Prediction RGB Overlays for All Regions")
        print("=" * 60)
        
        for region in self.regions.keys():
            results[region] = self.create_region_visualization(region)
        
        print("\n✅ All visualizations completed!")
        print(f"📁 Results saved in: {self.output_dir}")
        
        return results
```

#### Batch Processing Script: `sbatch/create_prediction_visualizations.sh`

```bash
#!/bin/bash
#SBATCH --job-name=pred_vis
#SBATCH --output=logs/%j_prediction_visualizations.txt  
#SBATCH --error=logs/%j_prediction_visualizations_error.txt
#SBATCH --time=0-2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=main

# Create output directory
mkdir -p logs chm_outputs/prediction_visualizations

# Activate environment
source chm_env/bin/activate

echo "🎨 Starting Prediction RGB Visualization Pipeline at $(date)"
echo "📊 Creating overlays for all scenarios and regions"

# Run visualization script
python create_prediction_rgb_overlays.py

echo "✅ Prediction visualization pipeline completed at $(date)"
echo "📁 Results saved in: chm_outputs/prediction_visualizations/"

# Show generated files
echo ""
echo "📊 Generated Visualizations:"
ls -la chm_outputs/prediction_visualizations/*.png

echo ""  
echo "🌍 RGB Composites:"
ls -la chm_outputs/prediction_visualizations/*_rgb_composite.tif

echo "🎉 Prediction RGB Visualization Pipeline Completed!"
```

### 4. Advanced Visualization Features

#### Interactive Web Visualization (Optional Enhancement)

```python
# Interactive visualization using Folium
def create_interactive_visualization(self, region: str) -> str:
    """Create interactive web map with prediction overlays"""
    import folium
    from folium import plugins
    
    # Load region boundary
    with open(self.regions[region]['aoi_path'], 'r') as f:
        aoi_dict = json.load(f)
    
    # Create base map
    center_lat = aoi_dict['features'][0]['properties'].get('center_lat', 35.0)
    center_lon = aoi_dict['features'][0]['properties'].get('center_lon', 135.0)
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
    
    # Add scenario layers with control
    for scenario_key, scenario_info in self.scenarios.items():
        pred_files = self.find_prediction_files(scenario_key, region)
        
        if pred_files:
            # Create raster overlay for each scenario
            # Note: This requires additional processing to convert TIF to web-friendly format
            pass
    
    # Save interactive map
    output_path = self.output_dir / f"{region}_interactive_predictions.html"
    m.save(str(output_path))
    
    return str(output_path)
```

#### Statistical Overlay Enhancement

```python
def add_performance_annotations(self, ax, scenario: str, region: str) -> None:
    """Add performance metrics as text annotations"""
    scenario_info = self.scenarios[scenario]
    
    # Get performance metrics
    if 'performance' in scenario_info:
        perf = scenario_info['performance']
        
        if isinstance(perf, dict):
            if region == 'hyogo':  # Training region
                metric_text = f"R² = {perf.get('train', 'N/A'):.3f}"
            else:  # Cross-region
                metric_text = f"R² = {perf.get('cross_region', perf.get('average', 'N/A')):.3f}"
        else:
            metric_text = f"Performance: {perf}"
        
        # Add text box with performance
        ax.text(0.05, 0.95, metric_text, transform=ax.transAxes,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
               fontsize=10, fontweight='bold', verticalalignment='top')
```

## Implementation Status - ✅ COMPLETED

### ✅ **PRODUCTION IMPLEMENTATION COMPLETED**

**Alternative Implementation**: `create_simplified_prediction_visualizations.py`
- **Status**: ✅ **FULLY OPERATIONAL** 
- **Architecture**: Simplified row-layout design instead of 6-panel grid
- **Key Features**: RGB + Reference + Multi-Scenario predictions with optimized layout
- **Documentation**: See `docs/simplified_prediction_visualization_implementation.md`

### Phase 1: Infrastructure Setup ✅ **COMPLETED**
- [x] **Planning**: Complete visualization plan document ✅
- [x] **Code Development**: Created `create_simplified_prediction_visualizations.py` ✅
- [x] **Earth Engine**: SSL authentication and RGB composite generation resolved ✅
- [x] **Testing**: RGB generation validated for all regions ✅

### Phase 2: Core Visualization ✅ **COMPLETED**
- [x] **RGB Composites**: Patch-based Sentinel-2 RGB generation implemented ✅
- [x] **Prediction Loading**: Automatic prediction file discovery and loading ✅
- [x] **Overlay System**: Memory-efficient raster alignment system ✅
- [x] **Multi-panel Layout**: Row-layout visualization with consistent sizing ✅

### Phase 3: Enhancement and Analysis ✅ **COMPLETED**
- [x] **Performance Annotations**: R² values and scaling factors in plot titles ✅
- [x] **Color Optimization**: Shared 'plasma' colormap with adaptive height range ✅
- [x] **Prediction Scaling**: `--vis-scale` parameter for enhanced visualization ✅
- [x] **Batch Processing**: SLURM integration with `sbatch/create_simplified_visualizations.sh` ✅

### Phase 4: Validation and Documentation ✅ **COMPLETED**
- [x] **Quality Check**: All visualizations validated for accuracy and consistency ✅
- [x] **Documentation**: Complete implementation guide created ✅
- [x] **Technical Optimization**: Equal image sizes, rightmost legend, memory efficiency ✅
- [x] **Publication Prep**: High-resolution PNG output ready for research use ✅

## Expected Outputs

### Static Visualizations
```
chm_outputs/prediction_visualizations/
├── kochi_prediction_rgb_overlay.png        # 6-panel comparison for Kochi
├── hyogo_prediction_rgb_overlay.png        # 6-panel comparison for Hyogo  
├── tochigi_prediction_rgb_overlay.png      # 6-panel comparison for Tochigi
├── kochi_sentinel2_rgb_composite.tif       # RGB base layer for Kochi
├── hyogo_sentinel2_rgb_composite.tif       # RGB base layer for Hyogo
└── tochigi_sentinel2_rgb_composite.tif     # RGB base layer for Tochigi
```

### Interactive Visualizations (Optional)
```
chm_outputs/prediction_visualizations/interactive/
├── kochi_interactive_predictions.html      # Interactive web map for Kochi
├── hyogo_interactive_predictions.html      # Interactive web map for Hyogo
└── tochigi_interactive_predictions.html    # Interactive web map for Tochigi
```

## Scientific Value

### Spatial Pattern Analysis
1. **Forest Structure Correlation**: Assess how predictions correlate with visible forest structure in RGB imagery
2. **Edge Effects**: Identify prediction quality at forest-non-forest boundaries
3. **Topographic Influence**: Analyze how terrain affects prediction accuracy
4. **Scenario Comparison**: Visually compare spatial patterns across modeling approaches

### Quality Assessment Insights
1. **GEDI-only Failure Visualization**: Show why Scenario 1.5 fails through spatial patterns
2. **Ensemble Benefits**: Demonstrate how ensemble approaches improve spatial continuity  
3. **Regional Adaptation**: Visualize improvements from target region fine-tuning (Scenario 3)
4. **Cross-region Generalization**: Compare prediction quality across different forest regions

### Publication-Ready Outputs
- **High-resolution** visualizations suitable for scientific journals
- **Standardized color schemes** for consistent comparison
- **Performance annotations** for quantitative context
- **RGB context** for intuitive interpretation by broad scientific audience

## Conclusion

This visualization plan provides comprehensive spatial context for understanding canopy height prediction performance across all implemented scenarios. By overlaying predictions on Sentinel-2 RGB imagery, researchers can:

1. **Assess prediction quality** in visual context of actual forest structure
2. **Compare scenarios** through standardized 6-panel layouts  
3. **Identify spatial patterns** in model performance and failure modes
4. **Communicate results** effectively to scientific and broader audiences
5. **Guide future development** through spatial analysis of model behavior

The implementation leverages existing infrastructure (`sentinel2_source.py`) while providing new capabilities for comprehensive prediction visualization and analysis.
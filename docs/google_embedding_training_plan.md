# Google Embedding Training Plan

## Executive Summary

This document outlines the comprehensive training plan for incorporating Google Embedding v1 (64 bands) with auxiliary canopy height data (4 products) and forest mask into the CHM modeling system. The plan leverages the highly successful **MLP-based reference height training** (R² = 0.5026) as the foundation and extends it to evaluate Google Embedding data effectiveness.

## New Dataset Overview

### ✅ **Google Embedding Dataset** - 🌍 **COMPLETED EXTRACTION**
- **Total Patches**: 189 patches across all three regions
- **Core Data**: Google Embedding v1 (64 bands) + auxiliary height data (4-5 bands) + forest mask
- **Patch Dimensions**: 256×256 pixels at 10m resolution (2.56km × 2.56km)
- **Year**: 2022 data
- **Data Quality**: Pre-normalized values in [-1, 1] range

#### **Regional Distribution**
| Region | Area ID | Patches | Band Count | File Pattern |
|--------|---------|---------|------------|-------------|
| **Kochi** | dchm_04hf3 | 63 | 69-70 | `*04hf3*embedding*bandNum{69,70}*` |
| **Hyogo** | dchm_05LE4 | 63 | 70 | `*05LE4*embedding*bandNum70*` |
| **Tochigi** | dchm_09gd4 | 63 | 69-70 | `*09gd4*embedding*bandNum70*` |

#### **Band Composition Analysis**
```
Google Embedding Dataset Structure:
├── Google Embedding v1: 64 bands (core satellite data) , which trained by following dataset (not bandnames).
│   ├── Sentinel-1 (SAR): Multi-temporal backscatter
│   ├── Sentinel-2 (Optical): Multi-temporal reflectance
│   ├── DEM: Digital elevation model
│   ├── ALOS2: Additional SAR data
│   ├── ERA5: Climate variables
│   └── Land Cover: Classification data
├── Auxiliary Height Data: 4-5 bands
│   ├── ch_potapov2021: Global forest canopy height
│   ├── ch_lang2022: Deep learning canopy height
│   ├── ch_tolan2024: Advanced canopy height product
│   ├── ch_pauls2024: Recent canopy height estimates
│   └── rh: GEDI height data (sparse, not all patches)
└── Forest Mask: 1 band
    └── forest_mask: Binary forest classification (1:forest, 0:non-forest)
```

## Training Strategy: Two-Scenario Comparison Framework

Following the proven methodology from `docs/reference_height_training_plan.md`, this plan implements a focused two-scenario experimental framework to evaluate Google Embedding v1 effectiveness compared to the original 30-band satellite data approach.

### **Scenario 1: Google Embedding Reference-Only Training** - ✅ **COMPLETED**
**Objective**: Evaluate Google Embedding v1 performance using 64 embedding bands with reference height supervision
- **Input Features**: 64 Google Embedding bands only
- **Architecture**: Advanced MLP (same as successful reference height training - `AdvancedReferenceHeightMLP`)
- **Training Data**: Reference height TIF supervision (dense coverage)
- **Training Region**: Hyogo (05LE4) - 63 patches
- **Validation**: Direct cross-region application to Kochi (04hf3) and Tochigi (09gd4)
- **Key Question**: Does Google Embedding v1 outperform original 30-band satellite data?
- **Status**: ✅ Successfully completed with outstanding performance in the training region, but poor generalization.
- **Results**: Training R² = 0.8734 (significant improvement over original 30-band MLP R² = 0.5026). However, the cross-region R² is -1.68 without bias correction, indicating the model does not generalize well to other regions without calibration.
- **Model**: `chm_outputs/production_mlp_reference_embedding_best.pth`
- **Predictions**: `chm_outputs/google_embedding_scenario1_predictions/{kochi,hyogo,tochigi}/`

### **Scenario 2A: Google Embedding + GEDI Spatial U-Net Ensemble**
**Objective**: Evaluate ensemble combining Google Embedding reference training with GEDI spatial U-Net (following original satellite 2A approach)
- **Component Models**: 
  - **GEDI Model**: Spatial U-Net with shift-aware loss trained on sparse GEDI rh data using 64 embedding bands
  - **Reference Model**: MLP trained on reference height TIF using 64 embedding bands
- **Ensemble Architecture**: MLP combining both model outputs (same as existing `ensemble_mlp.py`)
- **Training Data**: Hyogo patches with both reference height TIF and GEDI supervision
- **Validation**: Direct cross-region application (no fine-tuning)
- **Key Question**: Can spatial U-Net with Google Embedding succeed where original satellite U-Net failed?

#### **🔍 Learning from Original Satellite 2A Failure**
The original 30-band satellite Scenario 2A failed with:
- **Training R²**: 0.1611 (GEDI ignored: weight -0.0013, MLP: 0.7512)
- **Cross-Region**: Kochi R² = -8.58, Tochigi R² = -7.95 (200x worse than MLP)
- **Root Cause**: Spatial U-Net incompatible with sparse GEDI supervision

**Google Embedding Hypothesis**: The 64-band embedding may provide richer spatial context that enables U-Net to better handle sparse GEDI supervision through improved feature representation.

### **Scenario 2B: Google Embedding + GEDI Pixel-Level MLP Ensemble**
**Objective**: Evaluate ensemble combining Google Embedding reference training with GEDI pixel-level MLP (following original satellite 2B approach)
- **Component Models**: 
  - **GEDI Model**: MLP trained on sparse GEDI rh data using 64 embedding bands (pixel-level approach)
  - **Reference Model**: MLP trained on reference height TIF using 64 embedding bands
- **Ensemble Architecture**: MLP combining both model outputs (dual-MLP ensemble)
- **Training Data**: Hyogo patches with both reference height TIF and GEDI supervision
- **Validation**: Direct cross-region application (no fine-tuning)
- **Key Question**: Can pixel-level GEDI MLP with Google Embedding succeed where original satellite MLP failed?

#### **🔍 Learning from Original Satellite 2B Failure**
The original 30-band satellite Scenario 2B failed with:
- **Cross-Region**: Kochi R² = -5.14, Tochigi R² = -9.95 (worse than Scenario 1)
- **Root Cause**: Sparse GEDI supervision insufficient even with pixel-level approach

**Google Embedding Hypothesis**: The 64-band embedding may provide more robust pixel-level features that enable effective GEDI MLP training despite supervision sparsity, leading to successful dual-MLP ensemble performance.

## Performance Comparison Framework

### **Baseline Comparison with Existing 30-Band MLP Results**
Based on `docs/reference_height_training_plan.md`, we will compare against both the **Production MLP** and **Ensemble MLP** results:

| Approach | Input Features | Training R² | Cross-Region R² | Status | Prediction Path |
|----------|---------------|-------------|-----------------|--------|------------------|
| **Original Satellite MLP** | 30 bands | 0.5026 | -26.58 (no bias correction) | ✅ Production Ready | `chm_outputs/cross_region_predictions/` |
| **Original Satellite Ensemble (2A)** | 30 bands × 2 models | 0.1611 | -8.58 to -7.95 | ❌ Failed | `chm_outputs/scenario2_cross_region_predictions/` |
| **Original Satellite Ensemble (2B)** | 30 bands × 2 models | N/A | -5.14 to -9.95 | ❌ Failed | `chm_outputs/scenario2b_cross_region_predictions/` |
| **Google Embedding Scenario 1** | 64 bands | 0.8734 | -1.68 (no bias correction) | ✅ Completed | `chm_outputs/google_embedding_scenario1_predictions/` |
| **Google Embedding Scenario 2A** | 64 bands × 2 models (U-Net+MLP) | R² > 0.5500 | Expected > 0.3 | 🔄 Planned | `chm_outputs/google_embedding_scenario2a_predictions/` |
| **Google Embedding Scenario 2B** | 64 bands × 2 models (MLP+MLP) | R² > 0.5500 | Expected > 0.3 | 🔄 Planned | `chm_outputs/google_embedding_scenario2b_predictions/` |

### **Existing 30-Band MLP Results (Production Ready vs Failed Ensembles)**
From `reference_height_training_plan.md`:

#### **✅ Production MLP (Scenario 1 - Reference-Only)**
- **Model**: `chm_outputs/production_mlp_best.pth`
- **Architecture**: AdvancedReferenceHeightMLP with 30 satellite features
- **Performance**: Training R² = 0.5026. The non-bias-corrected cross-region R² is -26.58, indicating poor generalization without calibration.
- **Cross-Region Predictions**: 
  - Kochi: `chm_outputs/cross_region_predictions/04hf3_kochi/*_mlp_prediction.tif`
  - Tochigi: `chm_outputs/cross_region_predictions/09gd4_tochigi/*_mlp_prediction.tif`
  - Hyogo: `chm_outputs/cross_region_predictions/05LE4_hyogo/*_mlp_prediction.tif`
- **Bias Correction**: With region-specific factors (Kochi: 2.5x, Tochigi: 3.7x), the model is production-ready.

#### **❌ Failed Ensemble Approaches (Scenario 2A & 2B)**
- **Scenario 2A**: GEDI Spatial U-Net + Reference MLP Ensemble
  - **Training R²**: 0.1611 (GEDI ignored: weight -0.0013, MLP: 0.7512)
  - **Cross-Region**: Kochi R² = -8.58, Tochigi R² = -7.95 (200x worse than MLP)
  - **Root Cause**: Spatial U-Net incompatible with sparse GEDI supervision
- **Scenario 2B**: GEDI Pixel-Level MLP + Reference MLP Ensemble  
  - **Cross-Region**: Kochi R² = -5.14, Tochigi R² = -9.95 (worse than Scenario 1)
  - **Root Cause**: Sparse GEDI supervision insufficient even with pixel-level approach

#### **🔍 Key Insights for Google Embedding Scenario 2**
- **Ensemble Challenge**: Both 30-band satellite ensemble approaches failed
- **GEDI Supervision Limitation**: Sparse GEDI data (<0.3% coverage) problematic for both spatial and pixel-level models
- **Google Embedding Opportunity**: 64-band embedding may provide richer features to overcome GEDI sparsity
- **Architecture Consideration**: Pixel-level MLP ensemble preferred over spatial U-Net ensemble

### **Cross-Region Evaluation**
- **Training Region**: Hyogo (05LE4) - 63 patches with Google Embedding data
- **Validation Regions**: Kochi (04hf3) and Tochigi (09gd4) - 126 patches total
- **Evaluation Metrics**: R², RMSE, Bias, Cross-region generalization
- **Bias Correction**: Apply the same proven region-specific correction factors (2.5x for Kochi, 3.7x for Tochigi)
- **Visualization**: Heatmap plots following `analysis/height_analysis_utils.py` approach

## Technical Implementation

### **Leveraging Existing MLP Architecture**
This plan leverages the existing `AdvancedReferenceHeightMLP` architecture from `train_production_mlp.py` with input dimension modifications:

```python
# Scenario 1: Google Embedding Reference Training
# Use existing AdvancedReferenceHeightMLP with input_dim=64
model = AdvancedReferenceHeightMLP(
    input_dim=64,  # Google Embedding bands
    hidden_dims=[1024, 512, 256, 128, 64],
    dropout_rate=0.4,
    use_residuals=True,
    feature_attention=True
)

# Scenario 2: Google Embedding GEDI Training
# Same architecture but trained on GEDI pixels
gedi_model = AdvancedReferenceHeightMLP(
    input_dim=64,  # Google Embedding bands
    hidden_dims=[1024, 512, 256, 128, 64],
    dropout_rate=0.4,
    use_residuals=True,
    feature_attention=True
)
```

### **Data Processing Pipeline**
### **Memory-Efficient Evaluation**
To handle large-scale evaluation across multiple regions and scenarios without running into memory limitations, the evaluation scripts (`evaluate_google_embedding_scenario1.py`) implement a memory management strategy:

- **Garbage Collection**: Python's `gc` module is used to explicitly trigger garbage collection.
- **Per-Region Cleanup**: After processing each region, large data variables (e.g., aggregated pixel arrays) are deleted using `del` and `gc.collect()` is called to free up memory.
- **Per-Scenario Cleanup**: Similarly, after a scenario's results are plotted, the corresponding data is cleared from memory before the next scenario begins.

This ensures that memory usage remains stable throughout the evaluation process, even when `--max-patches` is set to a high value.
```python
def load_google_embedding_data(patch_path, include_aux_bands=False):
    """Load Google Embedding data with optional auxiliary bands"""
    with rasterio.open(patch_path) as src:
        data = src.read()
        band_descriptions = src.descriptions
        
        # Find Google Embedding bands (A00 to A63)
        embedding_indices = []
        for i, desc in enumerate(band_descriptions):
            if desc and desc.startswith('A') and len(desc) == 3:
                # Check if it's A00 to A63 format
                try:
                    band_num = int(desc[1:])
                    if 0 <= band_num <= 63:
                        embedding_indices.append(i)
                except ValueError:
                    continue
        
        # Sort by band number to ensure A00, A01, ..., A63 order
        embedding_indices.sort(key=lambda i: int(band_descriptions[i][1:]))
        selected_indices = embedding_indices
        
        # Optional: Add auxiliary bands (via argument)
        if include_aux_bands:
            aux_height_indices = []
            forest_mask_index = None
            
            height_bands = ['ch_potapov2021', 'ch_lang2022', 'ch_tolan2024', 'ch_pauls2024']
            for i, desc in enumerate(band_descriptions):
                if desc in height_bands:
                    aux_height_indices.append(i)
                elif desc == 'forest_mask':
                    forest_mask_index = i
            
            selected_indices = embedding_indices + aux_height_indices
            if forest_mask_index:
                selected_indices.append(forest_mask_index)
        
        return data[selected_indices]

def get_embedding_band_indices(band_descriptions):
    """Get indices of Google Embedding bands A00-A63"""
    embedding_indices = []
    for i, desc in enumerate(band_descriptions):
        if desc and desc.startswith('A') and len(desc) == 3:
            try:
                band_num = int(desc[1:])
                if 0 <= band_num <= 63:
                    embedding_indices.append((i, band_num))
            except ValueError:
                continue
    
    # Sort by band number and return indices
    embedding_indices.sort(key=lambda x: x[1])
    return [idx for idx, _ in embedding_indices]
```

## Implementation Scripts

### **Scenario 1: Google Embedding Reference Training**
Using existing `train_production_mlp.py` with modifications for Google Embedding bands:

```bash
# Scenario 1: Google Embedding Reference-Only Training
python train_production_mlp.py \
  --patch-dir chm_outputs/ \
  --patch-pattern "*05LE4*embedding*" \
  --supervision-mode reference_only \
  --input-bands 64 \
  --band-selection embedding \
  --reference-height-path downloads/dchm_05LE4.tif \
  --output-dir chm_outputs/google_embedding_scenario1/ \
  --epochs 100 \
  --batch-size 32 \
  --learning-rate 0.001
```

### **Scenario 2A: Google Embedding + GEDI Spatial U-Net Ensemble Training**
Following the original satellite 2A approach with spatial U-Net:

```bash
# Step 1: Train GEDI Spatial U-Net using Google Embedding
python train_predict_map.py \
  --patch-dir chm_outputs/ \
  --patch-pattern "*05LE4*embedding*" \
  --model shift_aware_unet \
  --shift-radius 2 \
  --supervision-mode gedi_only \
  --input-bands 64 \
  --band-selection embedding \
  --output-dir chm_outputs/google_embedding_gedi_unet/ \
  --epochs 100 \
  --batch-size 4 \
  --learning-rate 0.0001

# Step 2: Train Ensemble combining GEDI U-Net + Reference MLP
python train_ensemble_mlp.py \
  --gedi-model-path chm_outputs/google_embedding_gedi_unet/best_model.pth \
  --reference-model-path chm_outputs/google_embedding_scenario1/best_model.pth \
  --patch-dir chm_outputs/ \
  --patch-pattern "*05LE4*embedding*" \
  --reference-height-path downloads/dchm_05LE4.tif \
  --output-dir chm_outputs/google_embedding_ensemble_2a/ \
  --epochs 100 \
  --learning-rate 0.001
```

### **Scenario 2B: Google Embedding + GEDI Pixel-Level MLP Ensemble Training**
Following the original satellite 2B approach with pixel-level MLP:

```bash
# Step 1: Train GEDI Pixel-Level MLP using Google Embedding
python train_production_mlp.py \
  --patch-dir chm_outputs/ \
  --patch-pattern "*05LE4*embedding*" \
  --supervision-mode gedi_only \
  --input-bands 64 \
  --band-selection embedding \
  --output-dir chm_outputs/google_embedding_gedi_mlp/ \
  --epochs 100 \
  --batch-size 32 \
  --learning-rate 0.001

# Step 2: Train Dual-MLP Ensemble combining both MLPs
python train_ensemble_mlp.py \
  --gedi-model-path chm_outputs/google_embedding_gedi_mlp/best_model.pth \
  --reference-model-path chm_outputs/google_embedding_scenario1/best_model.pth \
  --patch-dir chm_outputs/ \
  --patch-pattern "*05LE4*embedding*" \
  --reference-height-path downloads/dchm_05LE4.tif \
  --output-dir chm_outputs/google_embedding_ensemble_2b/ \
  --epochs 100 \
  --learning-rate 0.001
```

### **Cross-Region Prediction Scripts**
Using existing prediction pipelines:

```bash
# Scenario 1: Google Embedding Reference-Only Predictions
python predict_mlp_cross_region.py \
  --model-path chm_outputs/google_embedding_scenario1/best_model.pth \
  --patch-dir chm_outputs/ \
  --regions "04hf3,09gd4" \
  --band-selection embedding \
  --output-dir chm_outputs/google_embedding_scenario1_predictions/

# Scenario 2A: Google Embedding Spatial U-Net Ensemble Predictions
python predict_ensemble.py \
  --gedi-model chm_outputs/google_embedding_gedi_unet/best_model.pth \
  --reference-model chm_outputs/google_embedding_scenario1/best_model.pth \
  --ensemble-mlp chm_outputs/google_embedding_ensemble_2a/best_model.pth \
  --patch-dir chm_outputs/ \
  --regions "04hf3,09gd4" \
  --band-selection embedding \
  --output-dir chm_outputs/google_embedding_scenario2a_predictions/

# Scenario 2B: Google Embedding Dual-MLP Ensemble Predictions
python predict_ensemble.py \
  --gedi-model chm_outputs/google_embedding_gedi_mlp/best_model.pth \
  --reference-model chm_outputs/google_embedding_scenario1/best_model.pth \
  --ensemble-mlp chm_outputs/google_embedding_ensemble_2b/best_model.pth \
  --patch-dir chm_outputs/ \
  --regions "04hf3,09gd4" \
  --band-selection embedding \
  --output-dir chm_outputs/google_embedding_scenario2b_predictions/
```

### **Evaluation and Comparison Scripts**
Using existing evaluation framework:

```bash
# Evaluate all scenarios with bias correction
python evaluate_with_bias_correction.py \
  --scenario1-results chm_outputs/google_embedding_scenario1_predictions/ \
  --scenario2a-results chm_outputs/google_embedding_scenario2a_predictions/ \
  --scenario2b-results chm_outputs/google_embedding_scenario2b_predictions/ \
  --original-results chm_outputs/production_mlp_results/ \
  --output-dir chm_outputs/google_embedding_evaluation/

# Create comparison analysis
python comprehensive_comparison.py \
  --google-embedding-results chm_outputs/google_embedding_evaluation/ \
  --original-satellite-results chm_outputs/production_mlp_results/ \
  --output-dir chm_outputs/google_vs_original_comparison/
```

## Visualization and Analysis

### **Universal Correlation Analysis**
Following the proven `analysis/height_analysis_utils.py` approach with enhanced multi-patch aggregation for both Google Embedding and original satellite data:

```python
class UniversalModelAnalyzer(HeightCorrelationAnalyzer):
    def __init__(self, downloads_dir, patches_dir, output_dir):
        super().__init__(downloads_dir, patches_dir, output_dir)
        
    def analyze_model_predictions(self, scenario_name, predictions_dir, region_list, data_type="embedding"):
        """Analyze model predictions vs reference data - works for both embedding and original satellite data"""
        results = []
        
        for region_id in region_list:
            region_name = self.regions[region_id]
            ref_path = f"{self.downloads_dir}/dchm_{region_id}.tif"
            
            # Get top 3 prediction patches for this region
            pred_patches = self.get_top_prediction_patches(predictions_dir, region_id, max_patches=3)
            
            aggregated_ref = []
            aggregated_pred = []
            
            for pred_file in pred_patches:
                # Align prediction with reference
                ref_aligned, pred_aligned = self.align_prediction_with_reference(ref_path, pred_file)
                
                if ref_aligned is not None and pred_aligned is not None:
                    
                    # Filter valid pixels
                    valid_mask = (~np.isnan(ref_aligned) & ~np.isnan(pred_aligned) & 
                                 (ref_aligned > 0) & (pred_aligned > 0))
                    
                    if np.sum(valid_mask) > 100:
                        aggregated_ref.extend(ref_aligned[valid_mask])
                        aggregated_pred.extend(pred_aligned[valid_mask])
            
            # Calculate metrics for aggregated data
            if len(aggregated_ref) > 500:
                metrics = self.calculate_metrics(np.array(aggregated_ref), np.array(aggregated_pred))
                if metrics:
                    results.append({
                        'scenario': scenario_name,
                        'data_type': data_type,
                        'region': region_name,
                        'region_id': region_id,
                        'r2': metrics['r2'],
                        'rmse': metrics['rmse'],
                        'bias': metrics['bias'],
                        'n_samples': metrics['n_samples'],
                        'ref_clean': metrics['ref_clean'],
                        'pred_clean': metrics['aux_clean']
                    })
        
        return results
    
    def get_top_prediction_patches(self, predictions_dir, region_id, max_patches=3):
        """Get top prediction patches for a region - works for both data types"""
        # Pattern matching for both embedding and original satellite predictions
        patterns = [
            os.path.join(predictions_dir, f"*{region_id}*embedding*prediction*.tif"),  # Embedding predictions
            os.path.join(predictions_dir, f"*{region_id}*mlp_prediction*.tif"),       # Original 30-band MLP predictions
            os.path.join(predictions_dir, f"*{region_id}*pred*.tif")                  # General prediction pattern
        ]
        
        pred_files = []
        for pattern in patterns:
            pred_files.extend(glob.glob(pattern))
        
        # Remove duplicates and sort
        pred_files = list(set(pred_files))
        pred_files.sort()
        
        # Return top patches (up to max_patches)
        return pred_files[:max_patches] if pred_files else []
        
    def create_aggregated_correlation_plot(self, results_data, scenario_name, output_file):
        """Create 3-column correlation plot for all regions"""
        if not results_data:
            print(f"No results data for {scenario_name}")
            return
        
        # Create 3-column subplot (one per region)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        regions = ['Kochi', 'Hyogo', 'Tochigi']
        
        for i, region in enumerate(regions):
            ax = axes[i]
            
            # Find data for this region
            region_data = next((r for r in results_data if r['region'] == region), None)
            
            if region_data:
                ref_clean = region_data['ref_clean']
                pred_clean = region_data['pred_clean']
                
                # Create hexbin plot with density coloring
                hb = ax.hexbin(ref_clean, pred_clean, gridsize=30, cmap='viridis', 
                              mincnt=1, alpha=0.8)
                
                # Add 1:1 reference line
                min_val = min(np.min(ref_clean), np.min(pred_clean))
                max_val = max(np.max(ref_clean), np.max(pred_clean))
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
                       label='1:1 Line')
                
                # Add regression line
                if len(ref_clean) > 1:
                    coeffs = np.polyfit(ref_clean, pred_clean, 1)
                    poly_line = np.poly1d(coeffs)
                    ax.plot(ref_clean, poly_line(ref_clean), 'g-', linewidth=2, 
                           label=f'Regression (y = {coeffs[0]:.3f}x + {coeffs[1]:.3f})')
                
                # Add metrics text
                metrics_text = (
                    f"R² = {region_data['r2']:.3f}\n"
                    f"RMSE = {region_data['rmse']:.2f} m\n"
                    f"Bias = {region_data['bias']:.2f} m\n"
                    f"N = {region_data['n_samples']:,}"
                )
                ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, 
                       verticalalignment='top', fontsize=11, 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                # Add colorbar for first subplot only
                if i == 0:
                    cb = plt.colorbar(hb, ax=ax)
                    cb.set_label('Point Density', fontsize=12)
                
            else:
                ax.text(0.5, 0.5, f'No data for {region}', transform=ax.transAxes, 
                       ha='center', va='center', fontsize=14)
            
            # Set labels and title
            ax.set_xlabel('Reference Height (m)', fontsize=12)
            ax.set_ylabel('Predicted Height (m)', fontsize=12)
            ax.set_title(f'{region} Region', fontsize=14, fontweight='bold')
            ax.set_aspect('equal', adjustable='box')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='lower right', fontsize=10)
        
        # Overall title
        fig.suptitle(f'{scenario_name}: Reference vs Predicted Height', 
                    fontsize=16, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved aggregated correlation plot: {output_file}")
```

### **Comprehensive Visualization Pipeline**
```python
def create_comprehensive_model_visualizations():
    """Create visualization plots for both Google Embedding and Original Satellite data"""
    analyzer = UniversalModelAnalyzer(
        downloads_dir="downloads",
        patches_dir="chm_outputs", 
        output_dir="chm_outputs/model_comparison_visualizations"
    )
    
    # Analyze Google Embedding scenarios
    embedding_scenario1_results = analyzer.analyze_model_predictions(
        "Google Embedding Scenario 1", 
        "chm_outputs/google_embedding_scenario1_predictions",
        ['dchm_04hf3', 'dchm_05LE4', 'dchm_09gd4'],
        data_type="embedding"
    )
    
    embedding_scenario2a_results = analyzer.analyze_model_predictions(
        "Google Embedding Scenario 2A", 
        "chm_outputs/google_embedding_scenario2a_predictions",
        ['dchm_04hf3', 'dchm_05LE4', 'dchm_09gd4'],
        data_type="embedding"
    )
    
    embedding_scenario2b_results = analyzer.analyze_model_predictions(
        "Google Embedding Scenario 2B", 
        "chm_outputs/google_embedding_scenario2b_predictions",
        ['dchm_04hf3', 'dchm_05LE4', 'dchm_09gd4'],
        data_type="embedding"
    )
    
    # Analyze Original Satellite data scenarios (for comparison)
    original_satellite_results = analyzer.analyze_model_predictions(
        "Original Satellite (30-band MLP)", 
        "chm_outputs/cross_region_predictions",  # Existing 30-band MLP predictions
        ['dchm_04hf3', 'dchm_05LE4', 'dchm_09gd4'],
        data_type="satellite"
    )
    
    # Analyze Original Satellite Ensemble failures (for comparison)  
    original_ensemble_2a_results = analyzer.analyze_model_predictions(
        "Original Satellite Ensemble 2A (Failed)", 
        "chm_outputs/scenario2_cross_region_predictions",  # Failed ensemble predictions
        ['dchm_04hf3', 'dchm_05LE4', 'dchm_09gd4'],
        data_type="satellite"
    )
    
    original_ensemble_2b_results = analyzer.analyze_model_predictions(
        "Original Satellite Ensemble 2B (Failed)", 
        "chm_outputs/scenario2b_cross_region_predictions",  # Failed ensemble predictions
        ['dchm_04hf3', 'dchm_05LE4', 'dchm_09gd4'],
        data_type="satellite"
    )
    
    # Create aggregated correlation plots for Google Embedding
    analyzer.create_aggregated_correlation_plot(
        embedding_scenario1_results, 
        "Google Embedding Scenario 1 (Reference-Only)",
        "chm_outputs/model_comparison_visualizations/google_embedding_scenario1_correlation.png"
    )
    
    analyzer.create_aggregated_correlation_plot(
        embedding_scenario2a_results, 
        "Google Embedding Scenario 2A (Reference + GEDI U-Net Ensemble)",
        "chm_outputs/model_comparison_visualizations/google_embedding_scenario2a_correlation.png"
    )
    
    analyzer.create_aggregated_correlation_plot(
        embedding_scenario2b_results, 
        "Google Embedding Scenario 2B (Reference + GEDI MLP Ensemble)",
        "chm_outputs/model_comparison_visualizations/google_embedding_scenario2b_correlation.png"
    )
    
    # Create correlation plot for Original Satellite data
    analyzer.create_aggregated_correlation_plot(
        original_satellite_results, 
        "Original Satellite Data (30-band MLP)",
        "chm_outputs/model_comparison_visualizations/original_satellite_correlation.png"
    )
    
    # Create correlation plots for Original Satellite Ensemble failures (for comparison)
    analyzer.create_aggregated_correlation_plot(
        original_ensemble_2a_results, 
        "Original Satellite Ensemble 2A (Failed)",
        "chm_outputs/model_comparison_visualizations/original_ensemble_2a_correlation.png"
    )
    
    analyzer.create_aggregated_correlation_plot(
        original_ensemble_2b_results, 
        "Original Satellite Ensemble 2B (Failed)",
        "chm_outputs/model_comparison_visualizations/original_ensemble_2b_correlation.png"
    )
    
    # Create comprehensive comparison heatmap including all approaches
    all_results = (embedding_scenario1_results + embedding_scenario2a_results + embedding_scenario2b_results + 
                  original_satellite_results + original_ensemble_2a_results + original_ensemble_2b_results)
    create_comprehensive_comparison_heatmap(all_results, 
        "chm_outputs/model_comparison_visualizations/comprehensive_comparison_heatmap.png")
```

### **Comprehensive Comparison Heatmap**
```python
def create_comprehensive_comparison_heatmap(all_results, output_file):
    """Create heatmap comparing Google Embedding and Original Satellite data across regions"""
    import pandas as pd
    import seaborn as sns
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Create combined scenario + data_type labels for better visualization
    df['method'] = df['scenario'] + ' (' + df['data_type'] + ')'
    
    # Create pivot tables for R² and RMSE
    pivot_r2 = df.pivot(index='region', columns='method', values='r2')
    pivot_rmse = df.pivot(index='region', columns='method', values='rmse')
    
    # Create dual heatmaps
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    
    # R² heatmap
    sns.heatmap(pivot_r2, annot=True, fmt='.3f', cmap='viridis', ax=ax1)
    ax1.set_title('R² Performance: Google Embedding vs Original Satellite Data')
    ax1.set_xlabel('Training Method')
    ax1.set_ylabel('Region')
    ax1.tick_params(axis='x', rotation=45)
    
    # RMSE heatmap
    sns.heatmap(pivot_rmse, annot=True, fmt='.2f', cmap='viridis_r', ax=ax2)
    ax2.set_title('RMSE Performance: Google Embedding vs Original Satellite Data')
    ax2.set_xlabel('Training Method')
    ax2.set_ylabel('Region')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comprehensive comparison heatmap: {output_file}")

def create_performance_comparison_table(all_results, output_file):
    """Create detailed performance comparison table"""
    import pandas as pd
    
    df = pd.DataFrame(all_results)
    
    # Create summary table
    summary_table = df.groupby(['scenario', 'data_type']).agg({
        'r2': ['mean', 'std'],
        'rmse': ['mean', 'std'],
        'bias': ['mean', 'std'],
        'n_samples': 'sum'
    }).round(3)
    
    # Save to CSV
    summary_table.to_csv(output_file.replace('.png', '.csv'))
    
    # Create visualization table
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Convert to display format
    display_data = []
    for (scenario, data_type), group in df.groupby(['scenario', 'data_type']):
        display_data.append([
            f"{scenario} ({data_type})",
            f"{group['r2'].mean():.3f} ± {group['r2'].std():.3f}",
            f"{group['rmse'].mean():.2f} ± {group['rmse'].std():.2f}",
            f"{group['bias'].mean():.2f} ± {group['bias'].std():.2f}",
            f"{group['n_samples'].sum():,}"
        ])
    
    table = ax.table(cellText=display_data,
                    colLabels=['Method', 'R² (mean ± std)', 'RMSE (mean ± std)', 'Bias (mean ± std)', 'Total Samples'],
                    cellLoc='center',
                    loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    plt.title('Performance Comparison: Google Embedding vs Original Satellite Data', 
              fontsize=14, fontweight='bold', pad=20)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved performance comparison table: {output_file}")
```

## Expected Outcomes

### **Performance Results and Hypotheses**

#### **✅ Confirmed Results - Scenario 1**
1. **Google Embedding Scenario 1**: R² = 0.8734 ✅ **OUTSTANDING** (73% improvement over original 30-band satellite MLP R² = 0.5026)
   - **Training Performance**: 63,009 samples, 64 input features, excellent convergence
   - **Cross-Region Transfer**: Under evaluation with bias correction needed
   - **Key Achievement**: Demonstrates Google Embedding v1 effectiveness for canopy height prediction

#### **🔄 Future Hypotheses - Scenarios 2A & 2B**
2. **Google Embedding Scenario 2A**: R² > 0.5500 (spatial U-Net ensemble with GEDI provides additional signal)
3. **Google Embedding Scenario 2B**: R² > 0.5500 (dual-MLP ensemble with GEDI provides additional signal)
4. **Ensemble Redemption**: Google Embedding may succeed where original satellite ensemble failed due to richer feature representation
5. **2A vs 2B Comparison**: Determine which ensemble approach (spatial vs pixel-level) works better with Google Embedding

### **Cross-Region Generalization**
- **Kochi (04hf3)**: Expected R² > 0.4 with bias correction
- **Tochigi (09gd4)**: Expected R² > 0.4 with bias correction
- **Bias Correction**: Apply proven region-specific factors (2.5x for Kochi, 3.7x for Tochigi)

### **Analysis Focus**
- **Google Embedding Effectiveness**: Quantify 64-band embedding performance vs 30-band satellite data
- **Feature Importance**: Identify most predictive embedding components (A00-A63)
- **GEDI Integration**: Assess GEDI ensemble contribution with Google Embedding base
- **Cross-Region Robustness**: Evaluate generalization across Japanese forest regions

## File Organization

### **Leveraging Existing Code Structure**
No new Python files needed - using existing proven components:
```
Existing Code Usage:
├── train_production_mlp.py            # Modified for --band-selection embedding
├── predict_mlp_cross_region.py        # Modified for Google Embedding bands
├── train_ensemble_mlp.py              # Existing ensemble training
├── predict_ensemble.py                # Existing ensemble prediction
├── evaluate_with_bias_correction.py   # Existing evaluation framework
├── models/
│   └── ensemble_mlp.py                # Existing ensemble architecture
├── analysis/
│   └── height_analysis_utils.py       # Existing correlation analysis
└── sbatch/
    ├── train_google_embedding_scenario1.sh  # NEW - Scenario 1 training
    ├── train_google_embedding_scenario2.sh  # NEW - Scenario 2 training
    └── evaluate_google_embedding.sh         # NEW - Comprehensive evaluation
```

### **Output Structure**
```
chm_outputs/
├── google_embedding_scenario1/         # Scenario 1: Reference-only results
├── google_embedding_gedi_unet/         # Scenario 2A: GEDI U-Net component
├── google_embedding_gedi_mlp/          # Scenario 2B: GEDI MLP component
├── google_embedding_ensemble_2a/       # Scenario 2A: U-Net+MLP ensemble results
├── google_embedding_ensemble_2b/       # Scenario 2B: MLP+MLP ensemble results
├── google_embedding_scenario1_predictions/  # Scenario 1 cross-region predictions
├── google_embedding_scenario2a_predictions/ # Scenario 2A cross-region predictions
├── google_embedding_scenario2b_predictions/ # Scenario 2B cross-region predictions
├── google_embedding_evaluation/        # Comprehensive evaluation
├── google_vs_original_comparison/      # Comparison analysis
└── google_embedding_visualizations/    # Heatmaps and correlation plots
```



## Success Metrics

### **Technical Success**
- [x] **Step 1**: Google Embedding Scenario 1 successfully trained on Hyogo region ✅
- [x] **Step 2**: Cross-region predictions generated for Scenario 1 ✅
- [x] **Step 3**: Comprehensive evaluation framework operational ✅
- [x] **Step 4**: Initial correlation analysis and evaluation completed ✅
- [ ] **Step 5**: Scenario 2A & 2B training and evaluation 🔄 In progress

### **Performance Success**
- [x] **Step 1**: Google Embedding Scenario 1 > Original Satellite MLP ✅ **OUTSTANDING** (R² = 0.8734 vs 0.5026 = +73% improvement)
- [ ] **Step 2**: Google Embedding Scenario 2A > Original Satellite Ensemble 2A (R² > 0.0 vs R² -8.58 to -7.95) 🔄 Next
- [ ] **Step 3**: Google Embedding Scenario 2B > Original Satellite Ensemble 2B (R² > 0.0 vs R² -5.14 to -9.95) 🔄 Next
- [ ] **Step 4**: Ensemble Redemption: Successful GEDI integration using Google Embedding 🔄 Next
- [ ] **Step 5**: Determine optimal ensemble approach: 2A (spatial) vs 2B (pixel-level) 🔄 Next

### **Scientific Success**
- [ ] Quantified Google Embedding v1 effectiveness for canopy height prediction
- [ ] Demonstrated auxiliary height data contribution to prediction accuracy
- [ ] Identified optimal feature combination for cross-region generalization
- [ ] Published comprehensive comparison framework for embedding-based approaches

## Implementation Timeline

### **Week 1-2: Code Modifications** - ✅ **COMPLETED**
- [x] **Infrastructure**: Modify `train_production_mlp.py` to support `--band-selection embedding` ✅
- [x] **Infrastructure**: Update data loading to handle A00-A63 band names ✅
- [x] **Infrastructure**: Modify `predict_mlp_cross_region.py` for Google Embedding bands ✅
- [x] **Infrastructure**: Create batch processing scripts for both scenarios ✅

### **Week 3-4: Training and Prediction** - ✅ **SCENARIO 1 COMPLETED**
- [x] **Scenario 1**: Train Google Embedding reference-only model on Hyogo region ✅ (R² = 0.8734)
- [x] **Prediction**: Run cross-region predictions for Scenario 1 ✅
- [x] **Evaluation**: Initial evaluation completed, bias correction analysis in progress ✅
- [ ] **Scenario 2**: Train Google Embedding GEDI MLP + ensemble model 🔄 Next priority

### **Week 5-6: Analysis and Visualization**
- [ ] **Analysis**: Implement Google Embedding correlation analysis
- [ ] **Visualization**: Create heatmap comparisons (Scenario 1 vs Scenario 2 vs Original)
- [ ] **Comparison**: Comprehensive comparison with original 30-band satellite data
- [ ] **Documentation**: Performance analysis and recommendations

### **Week 7-8: Production Integration**
- [ ] **Integration**: Best-performing scenario integration with existing pipeline
- [ ] **Documentation**: Complete implementation guide and user documentation
- [ ] **Deployment**: Production-ready scripts and batch processing capabilities
- [ ] **Validation**: Final validation and performance verification

## Conclusion

This Google Embedding training plan leverages the proven MLP-based approach (R² = 0.5026) and two-scenario framework from `docs/reference_height_training_plan.md` to systematically evaluate Google Embedding v1 effectiveness using existing code infrastructure.

### **Key Advantages**
1. **Proven Methodology**: Follows successful reference height training framework
2. **Existing Code Leverage**: Minimal new development - uses existing `train_production_mlp.py` and ensemble code
3. **Robust Band Selection**: Uses A00-A63 band names for reliable Google Embedding extraction
4. **Cross-Region Validation**: Proven evaluation framework with bias correction
5. **Production Integration**: Seamless integration with existing successful pipeline

### **Expected Impact**
- **Performance**: Significant improvement over original 30-band satellite data (R² > 0.5026)
- **Generalization**: Enhanced cross-region performance with Google Embedding v1
- **Scientific**: Quantified Google Embedding v1 effectiveness for forest applications
- **Operational**: Production-ready embedding-based canopy height prediction system

### **Implementation Efficiency**
- **Minimal Code Changes**: Only need to modify existing scripts for A00-A63 band selection
- **Auxiliary Band Support**: Code can accommodate auxiliary bands via `--include-aux-bands` argument
- **Proven Architecture**: Uses successful `AdvancedReferenceHeightMLP` with 64-band input
- **Existing Infrastructure**: Leverages all existing evaluation, visualization, and batch processing code

This approach positions the system for substantial performance improvements while maintaining the robust foundation established by the successful MLP-based reference height training methodology with minimal development overhead.
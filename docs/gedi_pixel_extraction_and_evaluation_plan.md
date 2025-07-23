# GEDI Pixel Extraction and Evaluation Plan

## Executive Summary

This document outlines the comprehensive workflow for GEDI pixel-level data extraction using Google Embedding v1 satellite data, reference height integration, and the planned evaluation framework for texture-enhanced canopy height modeling. This approach extends beyond patch-based modeling to pixel-level GEDI analysis across three Japanese forest regions.

## 🎯 **GEDI Pixel Extraction Workflow**

### **Phase 1: GEDI Data Extraction with Google Embedding Bands** ✅ **COMPLETED**

#### **Script**: `extract_gedi_with_embedding.py`

**Objective**: Extract GEDI footprint data with co-located Google Embedding v1 satellite features and auxiliary data as CSV for machine learning training.

**Key Features**:
- **Input Data**: Google Embedding v1 (64 bands) + canopy height products (4 bands) + GLCM texture features + forest mask
- **GEDI Integration**: Samples satellite data at GEDI footprint locations with spatial precision
- **Flexible Configuration**: Supports multiple texture metrics, forest mask types, and buffer zones
- **Earth Engine Export**: Batch export to Google Drive for large-scale processing

#### **Data Composition**:
```
GEDI Pixel Dataset Structure:
├── Google Embedding v1: 64 bands (A00-A63)
│   ├── Sentinel-1/2: Multi-temporal backscatter/reflectance
│   ├── DEM: Digital elevation model
│   ├── ALOS2: Additional SAR data
│   ├── ERA5: Climate variables
│   └── Land Cover: Classification data
├── Canopy Height Products: 4 bands
│   ├── ch_potapov2021: Global forest canopy height
│   ├── ch_lang2022: Deep learning canopy height
│   ├── ch_tolan2024: Advanced canopy height product
│   └── ch_pauls2024: Recent canopy height estimates
├── GLCM Texture Features: Variable bands (depends on --select-band)
│   ├── Single metric: 2 bands (mean_X, median_X)
│   └── All metrics: 14 bands (mean/median for 7 texture types)
├── Forest Mask: 1 band
│   └── forest_mask: NDVI-based forest classification
└── GEDI Data: 1 band
    └── rh: Height quantile (rh98 default)
```

#### **Usage Examples**:
```bash
# Single texture metric (recommended for manageable dataset size)
python extract_gedi_with_embedding.py \
    --aoi downloads/dchm_09gd4.geojson \
    --year 2022 \
    --buffer 5000 \
    --select-band contrast \
    --scale 10 \
    --window-size 3

# All texture metrics (research-focused, large dataset)
python extract_gedi_with_embedding.py \
    --aoi downloads/dchm_04hf3.geojson \
    --year 2022 \
    --buffer 5000 \
    --select-band All \
    --scale 10 \
    --window-size 3
```

#### **Output**: 
- **CSV files** exported to Google Drive `GEE_exports/` folder
- **Naming convention**: `gedi_embedding_{area}_{mask_type}_b{bands}_{year}_{buffer}m_scale{scale}m_w{window}_{texture}.csv`
- **Example**: `gedi_embedding_dchm_04hf3_NDVI30_b83_2022_5000m_scale10m_w3_contrast.csv`

### **Phase 2: Reference Height Integration** ✅ **COMPLETED**

#### **Script**: `analysis/add_reference_heights.py`

**Objective**: Add reference height values from original TIF files to GEDI CSV data for correlation analysis and model validation.

**Key Features**:
- **Optimized Sampling**: Vectorized raster sampling (~32,000 points/second vs 0.05 points/second previously)
- **Automatic Area Detection**: Extracts area codes from filenames (dchm_04hf3, dchm_05LE4, dchm_09gd4)
- **CRS Handling**: Proper coordinate transformation between WGS84 (CSV) and local projections (TIF)
- **Quality Control**: Comprehensive statistics on sampling success and height ranges

#### **Performance Improvement**:
```
Sampling Performance Comparison:
├── Previous Method: 21 seconds/point × 5,611 points = 33 hours
└── Optimized Method: 0.00003 seconds/point × 5,611 points = 0.2 seconds
    └── Improvement: 99.95% faster
```

#### **Usage**:
```bash
python analysis/add_reference_heights.py \
    --csv chm_outputs/gedi_embedding_dchm_04hf3_DW_b83_2022_1m_scale10m_w3_All.csv
```

#### **Output**:
- **Enhanced CSV**: Original CSV + `reference_height` column
- **Quality Report**: Sampling statistics, height ranges, and data validation
- **Example**: 5,608/5,611 points sampled (99.9% success), heights 0.00-53.59m, mean 16.23m

## 🔬 **Planned Evaluation Framework**

### **Phase 3: Height Correlation Analysis** 🔄 **PLANNED**

#### **Objective**: Evaluate correlations between reference heights and various height products/GEDI measurements.

**Analysis Components**:

1. **Multi-Height Product Correlation**:
   ```python
   height_columns = [
       'reference_height',  # Ground truth from TIF
       'rh',               # GEDI height quantile
       'ch_potapov2021',   # Global canopy height
       'ch_lang2022',      # Deep learning height
       'ch_tolan2024',     # Advanced height product
       'ch_pauls2024'      # Recent height estimates
   ]
   ```

2. **Correlation Matrix Analysis**:
   - **Pearson correlation** between all height products
   - **Regional comparison** across Kochi, Hyogo, Tochigi
   - **Scatter plot matrices** with regression lines
   - **Error analysis** (RMSE, MAE, bias) for each height product vs reference

3. **Height Product Performance Ranking**:
   ```python
   # Expected performance hierarchy (hypothesis)
   performance_ranking = [
       'ch_tolan2024',     # Most recent, advanced methods
       'ch_lang2022',      # Deep learning approach
       'rh',               # GEDI direct measurement
       'ch_pauls2024',     # Recent estimates
       'ch_potapov2021'    # Global, older methodology
   ]
   ```

### **Phase 4: Texture-Based Enhancement Analysis** 🔄 **PLANNED**

#### **Objective**: Evaluate GLCM texture features for enhancing GEDI height predictions and filtering low-quality data.

**Texture Feature Analysis**:

1. **Available Texture Metrics**:
   ```python
   texture_metrics = {
       'mean_asm': 'Angular Second Moment (uniformity)',
       'mean_contrast': 'Local contrast variation', 
       'mean_corr': 'Pixel correlation',
       'mean_var': 'Variance (intensity spread)',
       'mean_idm': 'Inverse Difference Moment (homogeneity)',
       'mean_savg': 'Sum Average',
       'mean_ent': 'Entropy (randomness)',
       'median_*': 'Median equivalents of above metrics'
   }
   ```

2. **Quality Filtering Investigation**:
   - **Correlation Enhancement**: Which texture metrics correlate with reference-GEDI agreement?
   - **Outlier Detection**: Use texture features to identify problematic GEDI footprints
   - **Forest Structure**: Relate texture patterns to forest heterogeneity/homogeneity

3. **Filter Development**:
   ```python
   # Hypothesis: Homogeneous areas (high IDM) = better GEDI accuracy
   quality_filters = {
       'scenario_4': 'no_filter',  # Baseline: use all GEDI points
       'scenario_5': 'texture_enhanced',  # Filter based on texture analysis
   }
   ```

### **Phase 5: Multi-Region MLP Training** 🔄 **PLANNED**

#### **Objective**: Train MLP models using GEDI pixel data with texture-based filtering across all three regions.

**Training Scenarios**:

1. **Scenario 4: No Filter Approach**
   ```python
   # Use all available GEDI points
   training_config = {
       'input_features': 64,  # Google Embedding only
       'architecture': 'AdvancedReferenceHeightMLP',
       'data_filter': None,
       'regions': ['dchm_04hf3', 'dchm_05LE4', 'dchm_09gd4'],
       'target': 'rh'  # GEDI height quantile
   }
   ```

2. **Scenario 5: Texture-Enhanced Filtering**
   ```python
   # Apply optimal texture-based filters from Phase 4
   training_config = {
       'input_features': 64 + N_texture_features,
       'architecture': 'AdvancedReferenceHeightMLP', 
       'data_filter': 'optimal_texture_threshold',
       'regions': ['dchm_04hf3', 'dchm_05LE4', 'dchm_09gd4'],
       'target': 'rh'
   }
   ```

3. **Cross-Region Validation**:
   - **Leave-one-region-out**: Train on 2 regions, test on 1
   - **Regional Adaptation**: Fine-tuning approaches for target regions
   - **Performance Comparison**: Scenarios 4 vs 5 vs existing patch-based approaches

### **Phase 6: Image Patch Prediction** 🔄 **PLANNED**

#### **Objective**: Apply trained GEDI models to predict canopy heights across image patches and compare with existing approaches.

**Prediction Pipeline**:

1. **Model Application**:
   ```bash
   # Apply Scenario 4 (no filter) model
   python predict_gedi_mlp.py \
       --model-path models/gedi_scenario4_best.pth \
       --patch-dir chm_outputs/ \
       --regions "04hf3,05LE4,09gd4" \
       --output-dir gedi_scenario4_predictions/

   # Apply Scenario 5 (texture-enhanced) model
   python predict_gedi_mlp.py \
       --model-path models/gedi_scenario5_best.pth \
       --patch-dir chm_outputs/ \
       --regions "04hf3,05LE4,09gd4" \
       --output-dir gedi_scenario5_predictions/
   ```

2. **Prediction Quality Assessment**:
   - **Reference Correlation**: R², RMSE, MAE vs ground truth TIF
   - **Spatial Consistency**: Edge effects, patch boundaries
   - **Height Range**: Realistic forest height distributions

### **Phase 7: Comprehensive Evaluation and Visualization** 🔄 **PLANNED**

#### **Objective**: Compare all approaches using standardized evaluation framework based on `evaluate_google_embedding_scenario1.py`.

**Evaluation Matrix**:

| Approach | Data Type | Training Method | Expected R² | Status |
|----------|-----------|-----------------|-------------|--------|
| **Original 30-band MLP** | 30 satellite bands | Patch-based reference | 0.5026 | ✅ Production |
| **Google Embedding Scenario 1** | 64 embedding bands | Patch-based reference | 0.8734 | ✅ Completed |
| **Google Embedding Scenario 1.5** | 64 embedding bands | GEDI-only (patch) | -7.746 | ✅ Failed baseline |
| **Google Embedding Scenario 2A** | 64 embedding bands | GEDI + Reference ensemble | 0.7844 | ✅ Completed |
| **GEDI Scenario 4** | 64 embedding bands | GEDI pixel-level (no filter) | TBD | 🔄 Planned |
| **GEDI Scenario 5** | 64 embedding + texture | GEDI pixel-level (filtered) | TBD | 🔄 Planned |

**Visualization Framework**:

1. **Multi-Scenario Comparison Plots**:
   ```python
   scenarios = [
       'original_30band',
       'google_embedding_scenario1', 
       'google_embedding_scenario2a',
       'gedi_scenario4_no_filter',
       'gedi_scenario5_texture_enhanced'
   ]
   
   # Create RGB + Reference + All Predictions visualization
   create_multi_scenario_visualization(
       patch_index=12,
       scenarios=scenarios,
       output_path='comprehensive_comparison.png'
   )
   ```

2. **Performance Heatmaps**:
   - **R² Performance**: All scenarios × all regions
   - **RMSE Analysis**: Error patterns across approaches
   - **Improvement Tracking**: Relative performance gains

3. **Correlation Analysis**:
   - **Reference vs Predicted**: Hexbin density plots
   - **Cross-Region Stability**: Generalization assessment  
   - **Statistical Significance**: P-values, confidence intervals

## 📊 **Expected Outcomes and Hypotheses**

### **Height Product Correlation Results** (Predicted):
```python
expected_correlations = {
    'reference_vs_ch_tolan2024': 0.85,    # Best modern algorithm
    'reference_vs_ch_lang2022': 0.80,     # Deep learning advantage
    'reference_vs_rh': 0.75,              # GEDI direct measurement
    'reference_vs_ch_pauls2024': 0.70,    # Recent methodology
    'reference_vs_ch_potapov2021': 0.65   # Global baseline
}
```

### **Texture Enhancement Impact** (Hypothesis):
- **Quality Filtering**: 15-25% improvement in GEDI prediction accuracy
- **Optimal Metrics**: IDM (homogeneity) and contrast most predictive
- **Regional Variation**: Different texture patterns across Kochi/Hyogo/Tochigi

### **GEDI MLP Performance** (Projected):
```python
performance_projections = {
    'gedi_scenario4_no_filter': {
        'training_r2': 0.45,
        'cross_region_r2': 0.25,
        'advantage': 'Large training dataset'
    },
    'gedi_scenario5_texture_enhanced': {
        'training_r2': 0.60,
        'cross_region_r2': 0.40,
        'advantage': 'Quality-filtered training data'
    }
}
```

## 🗂️ **File Organization and Outputs**

### **Data Structure**:
```
CHM_Image/
├── extract_gedi_with_embedding.py          # Phase 1: GEDI extraction
├── analysis/
│   └── add_reference_heights.py            # Phase 2: Reference integration
├── docs/
│   └── gedi_pixel_extraction_and_evaluation_plan.md  # This document
├── chm_outputs/
│   ├── gedi_embedding_*.csv                # GEDI pixel datasets
│   ├── *_with_reference.csv               # Reference-enhanced datasets
│   ├── gedi_scenario4_predictions/        # No-filter predictions
│   ├── gedi_scenario5_predictions/        # Texture-enhanced predictions
│   └── gedi_comprehensive_evaluation/      # Final evaluation results
└── downloads/
    ├── dchm_04hf3.tif                      # Kochi reference
    ├── dchm_05LE4.tif                      # Hyogo reference  
    └── dchm_09gd4.tif                      # Tochigi reference
```

### **Key Output Files**:
```
Expected Deliverables:
├── GEDI Datasets (Phase 1-2):
│   ├── gedi_embedding_dchm_04hf3_*_with_reference.csv  # Kochi dataset
│   ├── gedi_embedding_dchm_05LE4_*_with_reference.csv  # Hyogo dataset
│   └── gedi_embedding_dchm_09gd4_*_with_reference.csv  # Tochigi dataset
├── Analysis Results (Phase 3-4):
│   ├── height_correlation_matrix.png       # Multi-height correlation
│   ├── texture_enhancement_analysis.png    # Texture filtering optimization
│   └── quality_filter_recommendations.json # Optimal filtering thresholds
├── Models (Phase 5):
│   ├── gedi_scenario4_no_filter_best.pth   # No-filter MLP model
│   └── gedi_scenario5_texture_best.pth     # Texture-enhanced MLP model
├── Predictions (Phase 6):
│   ├── gedi_scenario4_predictions/         # No-filter patch predictions
│   └── gedi_scenario5_predictions/         # Texture-enhanced predictions
└── Evaluation (Phase 7):
    ├── comprehensive_scenario_comparison.png  # All-scenario visualization
    ├── gedi_vs_patch_performance_heatmap.png # Performance comparison matrix
    └── final_evaluation_report.json          # Comprehensive results summary
```

## 🎯 **Implementation Timeline**

### **Completed ✅**:
- **Phase 1**: GEDI extraction infrastructure (`extract_gedi_with_embedding.py`)
- **Phase 2**: Reference height integration (`add_reference_heights.py`)
- **Infrastructure**: Optimized raster sampling in `utils/spatial_utils.py`

### **Next Priorities 🔄**:

1. **Week 1-2**: Height Correlation Analysis (Phase 3)
   - Implement correlation matrix analysis
   - Generate height product comparison plots
   - Identify best-performing height products

2. **Week 3-4**: Texture Enhancement Analysis (Phase 4) 
   - Analyze texture metrics vs GEDI accuracy
   - Develop optimal filtering thresholds
   - Create texture-based quality assessment

3. **Week 5-6**: GEDI MLP Training (Phase 5)
   - Train Scenario 4 (no filter) models
   - Train Scenario 5 (texture-enhanced) models
   - Cross-region validation and optimization

4. **Week 7-8**: Prediction and Evaluation (Phases 6-7)
   - Generate patch-level predictions
   - Comprehensive evaluation vs existing approaches
   - Final visualization and documentation

## 🔬 **Scientific Contributions**

This workflow advances canopy height modeling through:

1. **Pixel-Level GEDI Integration**: First comprehensive pixel-level GEDI analysis with Google Embedding data
2. **Texture-Enhanced Filtering**: Novel application of GLCM texture features for GEDI data quality assessment
3. **Multi-Height Product Evaluation**: Systematic comparison of global canopy height products vs ground truth
4. **Cross-Region Validation**: Robust evaluation across diverse Japanese forest ecosystems
5. **Scalable Framework**: Production-ready pipeline for operational canopy height mapping

The expected outcome is a significant advancement in remote sensing-based forest height estimation, with practical applications for forest monitoring, carbon assessment, and biodiversity conservation across regional scales.

## 📝 **Usage Notes**

### **Memory Management**:
- Use `max_patches=3` for large-scale evaluation to prevent memory overflow
- Implement garbage collection (`gc.collect()`) after processing each region
- Consider batch processing for very large datasets

### **Quality Control**:
- Verify sampling success rates (>95% recommended)
- Check height range validity (0-60m typical for temperate forests)
- Monitor texture metric distributions for outlier detection

### **Performance Optimization**:
- Use `--select-band contrast` for balanced performance vs dataset size
- Apply `--buffer 5000` for adequate spatial context
- Set `--scale 10` for optimal GEDI-satellite alignment

This comprehensive framework provides a robust foundation for advancing pixel-level canopy height modeling using state-of-the-art satellite data and machine learning approaches.
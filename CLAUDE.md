# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### activate chm_env python environment
source chm_env/bin/activate 
- **IMPORTANT**: Always activate the Python environment using `source chm_env/bin/activate` before running any Python code

## Reference Documentation

### Project Planning and Status
- **Main Training Plan**: `docs/reference_height_training_plan.md` - Comprehensive 3-scenario comparison framework
- **Scenario 3 Implementation**: `docs/scenario3_implementation_plan.md` - Detailed plan for target region GEDI adaptation
- **Height Correlation Analysis**: `docs/height_correlation_analysis_plan.md` - Plan and initial results for auxiliary height data analysis.
- **Slurm Instructions**: `docs/slurm_instruction.md` - HPC usage guidelines for Annuna server

### Key Documentation Files
- **Training Plan**: Complete implementation guide for all scenarios with performance metrics
- **Scenario 3 Plan**: Focused implementation plan for Tochigi region GEDI fine-tuning
- **HPC Guidelines**: Slurm commands, sinteractive usage, and batch processing tips

## File Organization Guidelines

- length of python file should be below 500~800 lines because longer scripts are hard to read. If each file getting longer, we can consider to split them into several modules.

### Development Files
- **Temporary/Debug Files**: Always place in `tmp/` directory for debug scripts, experimental code, and temporary utilities
- **Legacy Files**: Move deprecated documentation and old scripts to `old/` directory  
- **Production Code**: Keep in root or appropriate module directories (utils/, models/, data/, etc.)
- **sbach file**: keep necessary sbach .sh file in `sbatch` directory, other temporal sbatch file should go to `tmp` directory.

### Sbatch Logging Tips
- For sbatch scripts, logs should be starting from job id, then I can easily find latest one.

### Module Structure
- **utils/**: Core utilities (spatial_utils.py for mosaicking and spatial processing)
- **models/**: Model architectures and training components
  - `models/trainers/`: Model-specific trainer classes
  - `models/losses/`: Loss function implementations
  - `models/ensemble_mlp.py`: Ensemble MLP architecture for combining models
- **data/**: Data processing pipeline components
- **training/**: Modular training system components (future expansion)

## HPC and Performance

- You can use GPU under HPC environments for speed up training and prediction.

## Data Input Options

### Google Embedding v1 Support
- **Google Embedding v1**: Annual satellite data embedding with 64 bands representing multi-modal data
- **Data Source**: `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`
- **Characteristics**: 
  - 64 bands per year (Sentinel-1, Sentinel-2, DEM, ALOS2, GEDI, ERA5 climate, land cover)
  - 10m resolution
  - Value range: -1 to 1 (pre-normalized, no additional normalization required)
  - Yearly product (no temporal dimension)
- **Usage**: Use `--embedding-only` flag to process only Google Embedding data
- **Implementation**: `get_google_embedding_data()` function in `chm_main.py`

### ✅ **Extracted Embedding Patch Dataset** - 📊 **COMPLETED**
- **Total Patches**: 189 patches across all three regions
- **Dataset Size**: 1.38 GB (7.3 MB per patch average)
- **Patch Dimensions**: 256×256 pixels at 10m resolution (2.56km × 2.56km)
- **Band Count**: 69-70 bands per patch (64 embedding + 5-6 additional bands)
- **Data Type**: Float32, pre-normalized values in [-1, 1] range
- **Year**: 2022 data
- **Quality**: All patches validated as properly normalized

#### **Regional Distribution**
| Region | Area ID | Patches | Avg Bands | File Size | Value Range | 
|--------|---------|---------|-----------|-----------|-------------|
| **Hyogo** | dchm_04hf3 | 63 | 69 | 7.2 MB | 0.025 to 0.293 |
| **Kochi** | dchm_05LE4 | 63 | 70 | 7.3 MB | 0.004 to 0.293 |
| **Tochigi** | dchm_09gd4 | 63 | 70 | 7.2 MB | -0.094 to 0.207 |

#### **Band Composition**
- **Core Embedding**: 64 bands (Google Embedding v1)
- **Additional Bands**: 5-6 bands (canopy height, forest mask, GEDI data)
- **Total**: 69-70 bands per patch depending on available auxiliary data

#### **Data Quality Verification**
- ✅ All values within expected [-1, 1] range
- ✅ Consistent 256×256 pixel dimensions
- ✅ Proper Float32 data type
- ✅ Valid CRS (EPSG:4326)
- ✅ Complete coverage across all three study regions

#### **Usage for Training**
- **Ready for ML**: Pre-normalized, no additional preprocessing required
- **Patch Format**: Compatible with existing PyTorch/TensorFlow workflows
- **File Location**: `chm_outputs/*embedding*scale10*.tif`
- **Recommended Use**: Direct input to CNN/MLP models for canopy height prediction

## Project Completion Status

### ✅ **COMPLETED SCENARIOS**

#### **Scenario 1: Reference-Only Training** - ✅ **FULLY COMPLETED**
- **Status**: Production-ready MLP model with bias correction
- **Performance**: R² = 0.5026 (6.7x improvement over U-Net)
- **Cross-Region**: 161 patches, 10.55M pixels, 100% success rate
- **Key Files**: `chm_outputs/production_mlp_best.pth`, `predict_mlp_cross_region.py`
- **Bias Correction**: Region-specific factors (Kochi: 2.5x, Tochigi: 3.7x)

#### **Scenario 2A: Reference + GEDI Training (Spatial U-Net)** - ❌ **FAILED**
- **Status**: Completed but failed due to poor GEDI model performance
- **Results**: Kochi R² = -8.58, Tochigi R² = -7.95 (200x worse than Scenario 1)
- **Root Cause**: Sparse GEDI supervision incompatible with spatial U-Net architecture
- **Key Files**: `train_ensemble_mlp.py`, `predict_ensemble.py`, `models/ensemble_mlp.py`
- **Lesson**: Spatial models require dense supervision; pixel-level models suit sparse data

### ❌ **FAILED SCENARIOS**

#### **Scenario 2B: Pixel-Level GEDI Training** - ❌ **FAILED**
- **Status**: Completed but failed with poor performance
- **Results**: Kochi R² = -5.14, Tochigi R² = -9.95 (worse than Scenario 1)
- **Root Cause**: Sparse GEDI supervision insufficient even with pixel-level approach
- **Key Files**: `train_production_mlp.py`, `predict_ensemble.py`, `evaluate_ensemble_cross_region.py`
- **Lesson**: Both spatial (2A) and pixel-level (2B) GEDI approaches fail with sparse supervision

#### **Scenario 2C: Shift-Aware Pixel Training** - 💡 **FUTURE CONCEPT**
- **Approach**: Extract surrounding pixels (1-3 radius) from GEDI points
- **Loss**: Calculate losses with different shifts, choose minimum per patch
- **Purpose**: Compensate for GEDI geolocation uncertainty at pixel level
- **Status**: Research concept for future discussion

### 🎯 **PRODUCTION-READY COMPONENTS**

#### **Training Scripts**
- `train_production_mlp.py` - Advanced MLP training with reference height supervision
- `train_ensemble_mlp.py` - Ensemble training combining multiple models
- `preprocess_reference_bands.py` - Enhanced patches preprocessing

#### **Prediction Scripts**
- `predict_mlp_cross_region.py` - Multi-region MLP inference pipeline
- `predict_ensemble.py` - Ensemble prediction for cross-region deployment
- `evaluate_with_crs_transform.py` - CRS-aware evaluation with bias correction

#### **Evaluation Scripts**
- `evaluate_ensemble_cross_region.py` - Ensemble performance evaluation
- `evaluate_with_bias_correction.py` - Systematic bias correction testing

#### **Batch Processing**
- `sbatch/train_ensemble_scenario2.sh` - Ensemble training job
- `sbatch/predict_ensemble_cross_region.sh` - Cross-region prediction job
- `sbatch/evaluate_ensemble_when_ready.sh` - Ensemble evaluation job

### 📊 **Key Performance Metrics**

| Approach | Training R² | Cross-Region R² | Status |
|----------|-------------|-----------------|--------|
| **U-Net (Scenario 1)** | 0.074 | N/A | ❌ Deprecated |
| **MLP (Scenario 1)** | 0.5026 | +0.012 (bias-corrected) | ✅ Production |
| **Ensemble (Scenario 2A)** | 0.1611 | -8.58 to -7.95 | ❌ Failed |
| **Dual-MLP (Scenario 2B)** | N/A | -5.14 to -9.95 | ❌ Failed |

### 🔧 **Implementation Guidelines**

#### **For Scenario 3 Implementation** - 🔄 **CURRENT FOCUS**
1. Fine-tune pre-trained GEDI models on Tochigi region data (30 patches)
2. Test both spatial U-Net and pixel-level MLP adaptation approaches
3. Train dual-track ensembles with adapted GEDI models + Reference MLP
4. Evaluate target region adaptation effectiveness vs failed Scenario 2 results
5. **Detailed Plan**: See `docs/scenario3_implementation_plan.md`

#### **Bias Correction Application**
```python
# Apply region-specific bias correction
correction_factors = {
    'kochi': 2.5,      # 41.4m → 16.5m
    'tochigi': 3.7,    # 61.7m → 16.7m  
    'hyogo': 1.0       # Training region
}
corrected_prediction = original_prediction / correction_factors[region]
```
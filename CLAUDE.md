# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### activate chm_env python environment
source chm_env/bin/activate 
- **IMPORTANT**: Always activate the Python environment using `source chm_env/bin/activate` before running any Python code

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

### 🔄 **PROPOSED SCENARIOS**

#### **Scenario 2B: Pixel-Level GEDI Training** - 🔄 **PROPOSED**
- **Approach**: Train GEDI MLP on sparse GEDI rh data (pixel-level)
- **Ensemble**: Combine GEDI MLP + Reference MLP (dual-MLP ensemble)
- **Expected**: R² > 0.3 for GEDI MLP, R² > 0.5 for ensemble
- **Key Innovation**: Both models use same architecture but different supervision sources

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
| **Dual-MLP (Scenario 2B)** | TBD | TBD | 🔄 Proposed |

### 🔧 **Implementation Guidelines**

#### **For Scenario 2B Implementation**
1. Modify `train_production_mlp.py` to support `--supervision-mode gedi_only`
2. Extract GEDI pixels from enhanced patches for training
3. Train dual-MLP ensemble combining GEDI MLP + Reference MLP
4. Evaluate cross-region performance with CRS-aware evaluation

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
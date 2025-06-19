# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### activate chm_env python environment
source chm_env/bin/activate 

### Testing
```bash
# Run all tests
python -m pytest tests/

# Run specific test module
python -m pytest tests/test_3d_unet.py

# Run tests with verbose output
python -m pytest tests/ -v
```

### Main Workflow Execution

#### Traditional 4-Step Pipeline (GEE-based)
```bash
# Complete 4-step workflow
python run_main.py --aoi_path downloads/dchm_09gd4.geojson --year 2022 --start-date 01-01 --end-date 12-31 --eval_tif_path downloads/dchm_09gd4.tif --use-patches --patch-size 2560 --patch-overlap 0.0 --model 3d_unet --steps data_preparation height_analysis train_predict evaluate

# Individual workflow steps
python run_main.py --steps data_preparation  # GEE data collection and processing
python run_main.py --steps height_analysis   # Height data combination and analysis
python run_main.py --steps train_predict     # Model training and prediction generation
python run_main.py --steps evaluate          # Comprehensive evaluation with PDF reports

# Temporal mode (Paul's 2025 methodology)
python run_main.py --temporal-mode --monthly-composite median --steps data_preparation
```

#### Unified Patch-Based Training System
```bash
# Train and predict with unified system (all models use same patch TIF input)
python train_predict_map.py --patch-path "chm_outputs/patch.tif" --model [rf|mlp|2d_unet|3d_unet] --output-dir chm_outputs/results

# Examples:
# Non-temporal Random Forest
python train_predict_map.py --patch-path "chm_outputs/dchm_09gd4_bandNum31_scale10_patch0000.tif" --model rf --output-dir chm_outputs/rf_results

# Temporal 3D U-Net (Paul's 2025 methodology)
python train_predict_map.py --patch-path "chm_outputs/dchm_09gd4_temporal_bandNum196_scale10_patch0000.tif" --model 3d_unet --output-dir chm_outputs/3d_unet_results --generate-prediction
```

### Google Earth Engine Setup
```bash
# Required before first use
earthengine authenticate
```

## Architecture Overview

### Core Processing Pipeline
This is a **unified temporal canopy height modeling system** that supports both temporal and non-temporal approaches:

1. **Data Sources**: Sentinel-1 SAR, Sentinel-2 optical, ALOS-2 SAR, GEDI LiDAR, DEM
2. **Temporal Processing**: 12-month time series (Paul's 2025) vs median composites (Paul's 2024)
3. **Patch-based Architecture**: 2.56km × 2.56km patches (256×256 pixels at 10m resolution)
4. **Model Types**: Random Forest, MLP, 2D U-Net, 3D U-Net (unified patch-based training)
5. **Training Approaches**: Traditional (CSV-based) and Unified (patch TIF-based)

### Key Components

#### Data Processing (`/data/`)
- **`image_patches.py`**: 3D patch creation and management with configurable overlap
- **`normalization.py`**: Band-specific normalization strategies for multi-modal data
- **`large_area.py`**: Scalable processing for regional/global applications
- **`patch_loader.py`**: Efficient data loading for training and inference

#### Models (`/models/`)
- **`3d_unet.py`**: Primary architecture with temporal 3D convolutions
- **`unet_3d.py`**: Alternative 3D U-Net implementation
- **`train_predict_map.py`**: Contains Height2DUNet for non-temporal processing
- Input formats:
  - 3D U-Net: `(batch, channels, time, height, width)` - handles 12-month sequences
  - 2D U-Net: `(batch, channels, height, width)` - non-temporal spatial processing
  - RF/MLP: Sparse GEDI pixels extracted from patches
- Output: Pixel-wise height predictions with modified Huber loss

#### Configuration (`/config/`)
- **`resolution_config.py`**: Multi-resolution support (10m/20m/30m)
- Patch dimensions automatically scale with resolution
- Consistent 2.56km physical patch size across resolutions

### Data Flow Architecture

1. **Google Earth Engine Collection** (`chm_main.py`):
   - Multi-band satellite data collection
   - Temporal compositing and cloud masking
   - GEDI point data extraction for training

2. **Unified Training System** (`train_predict_map.py`):
   - Automatic temporal/non-temporal mode detection
   - Unified patch-based training for all 4 model types
   - Sparse GEDI pixel extraction for RF/MLP from patch TIFs
   - Model training with shift-aware supervision
   - Full prediction map generation for all models
   - Intelligent 3D U-Net fallback to temporal averaging

3. **Evaluation Pipeline** (`evaluate_predictions.py`, `save_evaluation_pdf.py`):
   - Comprehensive accuracy assessment
   - Height-stratified analysis (critical for tall tree performance)
   - Feature importance and spatial distribution analysis

### Critical Architecture Decisions

#### 2025 Methodology vs 2024 (Unified Support)
- **2024 Approach**: 2D models with median composites (~31 bands)
  - Random Forest, MLP, 2D U-Net with non-temporal data
  - Single composite per band type
- **2025 Approach**: Temporal modeling with 12-month time series (~196 bands)
  - 3D U-Net with temporal 3D convolutions
  - Random Forest, MLP with temporal feature engineering
  - Multi-year training (2019-2022)
- **Key Improvement**: Better handling of tall trees (>30m) with reduced underestimation
- **Unified Framework**: Both approaches supported through automatic detection

#### Temporal Modeling
- **Input Formats**: 
  - Temporal: 12-month time series per band (S1: 24, S2: 132, ALOS2: 24, others: ~16)
  - Non-temporal: Median composites per band type (~31 bands total)
- **Processing**:
  - 3D Convolutions: Process temporal, spatial, and spectral dimensions
  - 2D Convolutions: Process spatial and spectral dimensions only
- **Automatic Detection**: Based on band naming patterns (_M01-_M12 suffixes)
- **Shift-aware Supervision**: Corrects for GEDI geolocation uncertainties

#### Multi-Modal Data Fusion
- **Optical**: Sentinel-2 (10 bands, phenology tracking)
- **SAR**: Sentinel-1 + ALOS-2 (structure information, cloud-independent)
- **Topographic**: DEM derivatives (elevation, slope, aspect)
- **Reference**: GEDI L2A (LiDAR heights for training/validation)

## Development Guidelines

### Model Development
- **Temporal models** must handle temporal dimensions: `(B, C, T, H, W)`
- **Non-temporal models** handle spatial dimensions: `(B, C, H, W)`
- Use `config/resolution_config.py` for consistent patch sizing
- Follow the unified patch-based approach in `train_predict_map.py`
- Support both temporal and non-temporal data automatically
- Implement sparse GEDI pixel extraction for traditional models

### Data Processing
- Always use `data/normalization.py` functions for band-specific scaling
- Implement forest masking using multiple strategies (NDVI, WorldCover, etc.)
- Patch processing must handle edge cases and boundary conditions
- Support automatic 256x256 patch cropping for dimension consistency
- Handle both temporal and non-temporal patch formats
- Extract sparse GEDI pixels from patches for traditional model training

### Testing
- Each new module requires corresponding test in `tests/`
- Test both individual components and integration workflows
- Include data validation and error handling tests

### Configuration Management
- Use `config/resolution_config.py` for scale-dependent parameters
- Model parameters are defined in respective training scripts
- GEE parameters (dates, cloud thresholds) configured in `run_main.py`

## Expected Output Structure
```
chm_outputs/
├── models/                    # Trained model artifacts
├── predictions/               # Height prediction GeoTIFFs
├── training_data/             # Processed CSV files with features (traditional)
├── patches/                   # Patch TIF files for unified training
├── evaluation/                # PDF reports and metrics
├── comparison/                # Model comparison results
│   ├── rf_temporal/          # RF with temporal data
│   ├── rf_non_temporal/      # RF with non-temporal data
│   ├── mlp_temporal/         # MLP with temporal data
│   ├── mlp_non_temporal/     # MLP with non-temporal data
│   ├── 2d_unet/              # 2D U-Net (non-temporal)
│   └── 3d_unet/              # 3D U-Net (temporal)
└── unified_*/                 # Unified training results
```

## Dependencies Note
This project requires Google Earth Engine authentication and PyTorch for U-Net model training. The system supports both high-memory (3D temporal) and standard-memory (2D non-temporal) processing modes. The unified training system automatically adapts to available memory and data formats.

## Model Performance Rankings
Based on comprehensive testing:
1. **MLP (temporal)**: R² = 0.391, RMSE = 5.95m ⭐⭐
2. **RF (non-temporal)**: R² = 0.175, RMSE = 6.92m ⭐
3. **U-Net models**: Require additional training/tuning for optimal performance

## Key Features
- **Unified Patch-Based Training**: All models use same patch TIF input
- **Automatic Mode Detection**: Temporal vs non-temporal based on band patterns
- **Sparse GEDI Supervision**: <0.3% pixel coverage handled efficiently
- **Intelligent Fallbacks**: 3D U-Net falls back to temporal averaging when needed
- **Full Prediction Maps**: All models generate complete spatial predictions
- **Model Comparison Framework**: Systematic evaluation across architectures
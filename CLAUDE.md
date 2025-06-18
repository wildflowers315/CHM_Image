# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

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
```bash
# Complete 4-step workflow
python run_main.py data_preparation height_analysis train_predict evaluate

# Individual workflow steps
python run_main.py data_preparation  # GEE data collection and processing
python run_main.py height_analysis   # Height data combination and analysis
python run_main.py train_predict     # Model training and prediction generation
python run_main.py evaluate          # Comprehensive evaluation with PDF reports

# Training and prediction for specific models
python train_predict_map.py  # Local model training and inference
```

### Google Earth Engine Setup
```bash
# Required before first use
earthengine authenticate
```

## Architecture Overview

### Core Processing Pipeline
This is a **3D temporal canopy height modeling system** that processes multi-modal satellite data:

1. **Data Sources**: Sentinel-1 SAR, Sentinel-2 optical, ALOS-2 SAR, GEDI LiDAR, DEM
2. **Temporal Processing**: 12-month time series with 3D convolutions
3. **Patch-based Architecture**: 2.56km × 2.56km patches (256×256 pixels at 10m resolution)
4. **Model Types**: Random Forest, MLP, 3D U-Net (primary focus)

### Key Components

#### Data Processing (`/data/`)
- **`image_patches.py`**: 3D patch creation and management with configurable overlap
- **`normalization.py`**: Band-specific normalization strategies for multi-modal data
- **`large_area.py`**: Scalable processing for regional/global applications
- **`patch_loader.py`**: Efficient data loading for training and inference

#### Models (`/models/`)
- **`3d_unet.py`**: Primary architecture with temporal 3D convolutions
- **`unet_3d.py`**: Alternative 3D U-Net implementation
- Input: `(batch, channels, time, height, width)` - handles 12-month sequences
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

2. **Local Processing** (`train_predict_map.py`):
   - 3D patch creation with temporal dimensions
   - Model training with shift-aware supervision
   - Large-area prediction with patch stitching

3. **Evaluation Pipeline** (`evaluate_predictions.py`, `save_evaluation_pdf.py`):
   - Comprehensive accuracy assessment
   - Height-stratified analysis (critical for tall tree performance)
   - Feature importance and spatial distribution analysis

### Critical Architecture Decisions

#### 2025 Methodology vs 2024
- **2024**: 2D U-Net with median composites, single-year training
- **2025**: 3D U-Net with temporal modeling, multi-year training (2019-2022)
- **Key Improvement**: Better handling of tall trees (>30m) with reduced underestimation

#### Temporal Modeling
- **Input**: 12-month time series per band
- **3D Convolutions**: Process temporal, spatial, and spectral dimensions
- **Shift-aware Supervision**: Corrects for GEDI geolocation uncertainties

#### Multi-Modal Data Fusion
- **Optical**: Sentinel-2 (10 bands, phenology tracking)
- **SAR**: Sentinel-1 + ALOS-2 (structure information, cloud-independent)
- **Topographic**: DEM derivatives (elevation, slope, aspect)
- **Reference**: GEDI L2A (LiDAR heights for training/validation)

## Development Guidelines

### Model Development
- All new models must handle temporal dimensions: `(B, C, T, H, W)`
- Use `config/resolution_config.py` for consistent patch sizing
- Follow the 3D U-Net architecture pattern in `models/3d_unet.py`

### Data Processing
- Always use `data/normalization.py` functions for band-specific scaling
- Implement forest masking using multiple strategies (NDVI, WorldCover, etc.)
- Patch processing must handle edge cases and boundary conditions

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
├── models/           # Trained model artifacts
├── predictions/      # Height prediction GeoTIFFs
├── training_data/    # Processed CSV files with features
└── evaluation/       # PDF reports and metrics
```

## Dependencies Note
This project requires Google Earth Engine authentication and PyTorch for 3D model training. The system is designed for high-memory environments due to 3D temporal processing requirements.
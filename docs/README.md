# CHM Documentation

This directory contains comprehensive documentation for the Canopy Height Modeling (CHM) project, particularly focusing on the June 2025 implementation of 2D model training with GEDI filtering.

## Document Overview

### Core Implementation Guides

#### 📊 [2D Model Training Results](2d_model_training_results.md)
Comprehensive analysis of the June 2025 2D model training implementation on Annuna HPC cluster:
- Random Forest, MLP, and 2D U-Net model performance
- Training results with GEDI filtering
- Complete prediction generation for 27 patches
- Performance metrics and analysis

#### 🔧 [GEDI Filtering Implementation](gedi_filtering_implementation.md)
Technical documentation for the GEDI sample filtering system:
- Quality control implementation
- Mode-based filtering (training vs prediction)
- API documentation and usage examples
- Testing and validation procedures

#### 🏗️ [HPC Workflow Guide](hpc_workflow_guide.md)
Practical guide for running models on Annuna HPC cluster:
- SLURM job submission and monitoring
- Interactive development workflow
- Resource allocation guidelines
- Troubleshooting and best practices

### Existing Documentation

#### 📋 [SLURM Instructions](slurm_instruction.md)
Basic SLURM usage for Annuna cluster operations

#### 📁 [Results Archive](results/)
Historical documentation and implementation summaries from various development phases

## Quick Start

### For New Users
1. Read [HPC Workflow Guide](hpc_workflow_guide.md) for cluster setup
2. Review [GEDI Filtering Implementation](gedi_filtering_implementation.md) for data quality control
3. Check [2D Model Training Results](2d_model_training_results.md) for expected outcomes

### For Developers
1. Follow file organization guidelines in `/CLAUDE.md`
2. Use test scripts in `tmp/` directory for validation
3. Reference implementation details in technical docs

### For Production Use
1. Use HPC workflow scripts: `run_2d_training.sh`, `run_2d_prediction.sh`
2. Monitor jobs with `tmp/monitor_training.py`
3. Follow resource allocation guidelines for optimal performance

## Implementation Summary

### ✅ Completed Features (June 2025)
- **GEDI Sample Filtering**: Configurable minimum sample thresholds
- **Multi-Model Training**: RF, MLP, 2D U-Net unified pipeline
- **HPC Integration**: SLURM job scripts and monitoring
- **Prediction Generation**: Complete spatial coverage (27 patches)
- **Quality Control**: Comprehensive validation and error handling

### 🎯 Key Results
- **Random Forest**: Best performing model (R² = 0.074, RMSE = 10.2m)
- **Prediction Coverage**: 100% spatial coverage (27/27 patches)
- **GEDI Filtering**: Successfully applied (9 patches filtered during training)
- **HPC Performance**: Efficient processing on 8-core nodes

### 📁 Output Structure
```
chm_outputs/
├── models/                    # Trained model artifacts
│   ├── rf/                   # Random Forest models
│   ├── mlp/                  # MLP models  
│   └── 2d_unet/              # 2D U-Net models
├── predictions/              # Height prediction GeoTIFFs
│   └── rf_predictions/       # RF prediction patches (27 files)
├── comparison/               # Model comparison results
│   └── rf_non_temporal/      # RF non-temporal results
└── patches/                  # Original patch TIF files (31 bands)
```

## Next Steps

### Immediate Actions
1. **Production Deployment**: Use Random Forest model for operational predictions
2. **Model Improvement**: Tune 2D U-Net hyperparameters
3. **Spatial Mosaicking**: Combine prediction patches into continuous maps

### Future Enhancements
1. **Temporal Integration**: Incorporate time series data (Paul's 2025 methodology)
2. **Advanced Models**: Attention mechanisms and transformer architectures
3. **Ensemble Methods**: Combine RF and MLP predictions
4. **Cross-validation**: Patch-based validation strategies

## Contact and Support

For technical questions or implementation details:
- Review relevant documentation in this directory
- Check test scripts in `tmp/` directory
- Consult HPC workflow guide for cluster-specific issues
- Reference CLAUDE.md for development guidelines

---
*Last Updated: June 24, 2025*  
*Project: CHM Image Processing*  
*Environment: Annuna HPC Cluster*
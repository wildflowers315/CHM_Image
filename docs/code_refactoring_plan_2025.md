# Code Refactoring Plan 2025

## Executive Summary

This document outlines the comprehensive refactoring of the CHM_Image codebase to address code organization issues, eliminate temporary/debug file clutter, and create a maintainable modular architecture.

## Problem Analysis

### Current Issues
1. **File Proliferation**: 40+ debug/test/temporary files scattered throughout root directory
2. **Monolithic Architecture**: `train_predict_map.py` grew to 3,179 lines with multiple responsibilities
3. **Unclear File Purpose**: Many files with similar names (debug_*.py, predict_*.py, train_*.py)
4. **Maintenance Complexity**: Difficult to identify which files are active vs. deprecated
5. **Import Confusion**: Circular dependencies and unclear module relationships

### Impact Assessment
- **Developer Experience**: Difficult to navigate and understand codebase
- **Code Maintenance**: Hard to identify which files to update
- **Testing**: Unclear which components need testing
- **Documentation**: Scattered functionality across many files

## Refactoring Strategy

### Phase 1: File Organization and Cleanup ✅ COMPLETED

#### 1.1 Create Organizational Directories
```bash
mkdir -p tmp/ old/ utils/ models/losses models/trainers training/core training/data training/workflows
```

#### 1.2 Move Temporary and Debug Files
**Moved to `tmp/`:**
- All `debug_*.py` files (debug_temporal_training.py, debug_rgb_issue.py, etc.)
- Temporary prediction scripts (predict_manual.py, predict_simple.py, etc.)
- Test/experimental scripts (train_simple.py, quick_demo_workflow.py, etc.)
- Development utilities (create_spatial_mosaic.py, compare_results.py, etc.)

**Moved to `old/`:**
- Legacy documentation (WORKFLOW_SUMMARY.md, RGB_NORMALIZATION_FIX_SUMMARY.md, etc.)
- Deprecated training scripts (train_temporal_3d_unet.py)
- Old workflow summaries and temporary documentation

#### 1.3 Rename Key Files
- `predict_with_mosaic.py` → `predict.py` (primary prediction tool)
- `enhanced_spatial_merger.py` → `utils/spatial_utils.py`

### Phase 2: Modular Architecture Implementation ✅ COMPLETED

#### 2.1 Core Utilities Structure
```
utils/
├── __init__.py              # Export EnhancedSpatialMerger
└── spatial_utils.py         # Spatial processing and mosaicking
```

#### 2.2 Model Architecture Structure  
```
models/
├── losses/
│   └── __init__.py          # Loss functions for training
└── trainers/
    ├── __init__.py          # Export all trainers
    ├── rf_trainer.py        # Random Forest trainer
    ├── mlp_trainer.py       # MLP trainer  
    ├── unet_2d_trainer.py   # 2D U-Net trainer
    └── unet_3d_trainer.py   # 3D U-Net trainer
```

#### 2.3 Training System Structure
```
training/
├── core/                    # Base training infrastructure
├── data/                    # Data handling and augmentation
├── models/                  # Model-specific components
└── workflows/               # High-level training workflows
```

### Phase 3: Import System Modernization ✅ COMPLETED

#### 3.1 Updated Import Statements
- `train_predict_map.py`: Updated to use `from utils.spatial_utils import EnhancedSpatialMerger`
- Established proper module hierarchy with `__init__.py` files
- Removed dependencies on temporary/debug files

#### 3.2 Backward Compatibility
- All existing workflows continue to function
- Primary entry points (`train_predict_map.py`, `run_main.py`) unchanged
- CLI arguments and API remain stable

### Phase 4: Documentation and Guidelines Updates 🔄 IN PROGRESS

#### 4.1 Documentation Updates
- **CLAUDE.md**: Update with new file organization guidelines
- **implementation_plan_2025.md**: Add refactoring completion details
- **New file**: `code_refactoring_plan_2025.md` (this document)

#### 4.2 Development Guidelines
- Clear file naming conventions
- Module organization principles
- Import best practices

## File Classification System

### Active Production Files
**Core Pipeline:**
- `run_main.py` - Main workflow orchestration
- `train_predict_map.py` - Unified training and prediction system
- `predict.py` - Primary prediction tool with spatial mosaicking
- `chm_main.py` - Google Earth Engine data collection

**Data Processing:**
- `data/` package - Image patches, normalization, multi-patch support
- `utils/spatial_utils.py` - Enhanced spatial processing

**Model Components:**
- `models/3d_unet.py` - 3D U-Net architecture
- `dl_models.py` - MLP and traditional models
- `models/trainers/` - Modular training components

**Evaluation:**
- `evaluate_predictions.py` - Metrics calculation
- `evaluate_temporal_results.py` - Comprehensive PDF evaluation
- `save_evaluation_pdf.py` - Report generation

### Temporary/Development Files (tmp/)
**Debug Scripts:**
- `debug_temporal_training.py` - Training diagnostics
- `debug_rgb_issue.py` - RGB processing investigation  
- `debug_patch_data.py` - Patch data validation

**Experimental Predictions:**
- `predict_manual.py` - Manual prediction experiments
- `predict_simple.py` - Simplified prediction testing
- `generate_prediction_map.py` - Alternative prediction approaches

**Training Experiments:**
- `train_simple.py` - Simplified training tests
- `quick_demo_workflow.py` - Demo/testing workflows

### Legacy Files (old/)
**Documentation:**
- `WORKFLOW_SUMMARY.md` - Previous workflow documentation
- `RGB_NORMALIZATION_FIX_SUMMARY.md` - Legacy bug fix notes
- `TEMPORAL_RESULTS_SUMMARY.md` - Previous result analysis

**Deprecated Code:**
- `train_temporal_3d_unet.py` - Superseded by unified system

## Benefits Achieved

### 1. Code Organization
- **Clear Structure**: Logical separation of concerns
- **Reduced Clutter**: 40+ files organized into purpose-driven directories
- **Easy Navigation**: Developers can quickly locate relevant code

### 2. Maintainability 
- **Modular Design**: Components can be updated independently
- **Clear Dependencies**: Explicit import relationships
- **Version Control**: Easier to track changes to specific components

### 3. Development Efficiency
- **Faster Onboarding**: New developers can understand structure quickly
- **Focused Development**: Clear separation between production and experimental code
- **Better Testing**: Modular components are easier to test

### 4. Production Stability
- **Backward Compatibility**: All existing workflows preserved
- **Clear Active Files**: Production code clearly separated from experiments
- **Reduced Risk**: Less chance of accidentally modifying experimental code

## Implementation Status

### ✅ Completed
- File organization and cleanup (tmp/, old/ directories)
- Modular architecture implementation (utils/, models/trainers/)
- Import system modernization
- Enhanced spatial merger integration
- Backward compatibility preservation

### 🔄 In Progress  
- Documentation updates (CLAUDE.md, implementation_plan_2025.md)
- Development guidelines establishment

### 📋 Future Considerations
- Automated testing for modular components
- CI/CD pipeline integration
- Performance optimization of modular structure
- Further extraction of training components from train_predict_map.py

## Directory Structure Summary

```
CHM_Image/
├── tmp/                     # Temporary and debug files
│   ├── debug_*.py          # Debug scripts
│   ├── predict_*.py        # Experimental prediction scripts
│   └── train_*.py          # Training experiments
├── old/                     # Legacy documentation and deprecated code
│   ├── *.md                # Old documentation files
│   └── *.py                # Deprecated scripts
├── utils/                   # Core utilities
│   ├── __init__.py         
│   └── spatial_utils.py    # Spatial processing and mosaicking
├── models/                  # Model architectures and training
│   ├── losses/             # Loss functions
│   ├── trainers/           # Model-specific trainers
│   ├── 3d_unet.py         # 3D U-Net architecture
│   └── unet_3d.py         # Alternative 3D U-Net
├── training/                # Modular training system (future expansion)
│   ├── core/               # Base training infrastructure
│   ├── data/               # Data handling and augmentation  
│   ├── models/             # Model-specific components
│   └── workflows/          # High-level training workflows
├── data/                    # Data processing pipeline
├── config/                  # Configuration management
├── tests/                   # Unit and integration tests
├── docs/                    # Documentation
├── run_main.py             # Main workflow orchestration
├── train_predict_map.py    # Unified training and prediction
├── predict.py              # Primary prediction tool
└── chm_main.py            # GEE data collection
```

This refactoring provides a solid foundation for continued development while maintaining all existing functionality and improving code maintainability significantly.
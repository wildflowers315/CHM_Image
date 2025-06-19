# Modular Training System Usage Guide

## Overview

The new modular training system provides a clean, organized alternative to the large `train_predict_map.py` file. It offers the same functionality with better code organization and easier maintenance.

## Quick Start

### 1. Using the Fixed Current System (Immediate Solution)

The original `train_predict_map.py` has been **fixed** for the negative stride tensor error. You can use it right away:

```bash
# Your command should now work without the tensor error
python train_predict_map.py \
    --patch-path "../drive/MyDrive/GEE_exports" \
    --model 2d_unet \
    --output-dir chm_outputs/2d_unet/predictions \
    --use-enhanced-training \
    --augment \
    --validation-split 0.2 \
    --early-stopping-patience 10
```

### 2. Using the New Modular System (Cleaner Architecture)

Use the new `train_modular.py` entry point for the same functionality:

```bash
# Same training with cleaner modular architecture
python train_modular.py \
    --patch-path "../drive/MyDrive/GEE_exports" \
    --model 2d_unet \
    --output-dir chm_outputs \
    --augment \
    --validation-split 0.2 \
    --early-stopping-patience 10 \
    --generate-prediction
```

## Model Types Supported

All four model types are supported in both systems:

### Random Forest
```bash
python train_modular.py \
    --patch-path "path/to/patches" \
    --model rf \
    --n-estimators 200 \
    --max-depth 20
```

### MLP (Multi-Layer Perceptron)
```bash
python train_modular.py \
    --patch-path "path/to/patches" \
    --model mlp \
    --hidden-sizes 512 256 128 \
    --dropout-rate 0.3 \
    --epochs 100
```

### 2D U-Net (Non-temporal)
```bash
python train_modular.py \
    --patch-path "path/to/patches" \
    --model 2d_unet \
    --batch-size 8 \
    --epochs 50 \
    --augment \
    --base-channels 64
```

### 3D U-Net (Temporal)
```bash
python train_modular.py \
    --patch-path "path/to/temporal/patches" \
    --model 3d_unet \
    --batch-size 4 \
    --epochs 30 \
    --base-channels 32
```

## Key Features

### 1. Automatic Device Detection
```bash
# Automatically uses CUDA if available, otherwise CPU
python train_modular.py --model 2d_unet --device auto

# Force specific device
python train_modular.py --model 2d_unet --device cuda
python train_modular.py --model 2d_unet --device cpu
```

### 2. Data Augmentation (Fixed for Negative Stride Issue)
```bash
# 12x augmentation (3 flips × 4 rotations)
python train_modular.py \
    --model 2d_unet \
    --augment \
    --augment-factor 12
```

### 3. Early Stopping
```bash
# Stop training if validation doesn't improve for 15 epochs
python train_modular.py \
    --model 2d_unet \
    --early-stopping-patience 15
```

### 4. Validation Split
```bash
# Use 30% of data for validation
python train_modular.py \
    --model 2d_unet \
    --validation-split 0.3
```

## Migration Guide

### From Old train_predict_map.py to Modular System

| Old Command | New Modular Command |
|-------------|---------------------|
| `--use-enhanced-training` | Enabled by default |
| `--checkpoint-freq 5` | Built into trainers |
| `--resume-from checkpoint.pth` | Built into trainers |

### Directory Structure

The modular system creates organized output directories:

```
chm_outputs/
├── rf/
│   ├── models/          # Trained models
│   ├── predictions/     # Prediction maps  
│   ├── logs/           # Training logs
│   └── training_results.json
├── mlp/
├── 2d_unet/
└── 3d_unet/
```

## Advanced Usage

### 1. Custom Training Parameters
```python
# Python API usage
from training.workflows.unified_trainer import UnifiedTrainer
from training.core.callbacks import TrainingLogger

logger = TrainingLogger()
trainer = UnifiedTrainer('2d_unet', 'outputs', logger=logger)

results = trainer.train(
    'path/to/patches',
    augment=True,
    validation_split=0.2,
    batch_size=8,
    epochs=50
)
```

### 2. Batch Processing Multiple Areas
```bash
# Process multiple patch directories
for area in area1 area2 area3; do
    python train_modular.py \
        --patch-path "data/$area" \
        --model 2d_unet \
        --output-dir "results/$area"
done
```

## Debugging and Troubleshooting

### 1. Check Patch Validity
```python
from training.data.preprocessing import PatchPreprocessor

preprocessor = PatchPreprocessor()
summary = preprocessor.batch_validate_patches(patch_files)
print(f"Valid patches: {summary['valid_patches']}")
```

### 2. Monitor Training Progress
All training progress is logged to `{output_dir}/{model}/training.log`

### 3. Common Issues and Solutions

**Issue**: "No valid GEDI samples found"
**Solution**: Check patch files have valid height data in the last band

**Issue**: "CUDA out of memory"  
**Solution**: Reduce batch size or use CPU: `--device cpu --batch-size 4`

**Issue**: "Negative stride tensor error"
**Solution**: Fixed in both systems - update to latest code

## Performance Comparison

| System | Code Lines | Maintainability | Features |
|--------|------------|-----------------|----------|
| Original `train_predict_map.py` | 3,179 | Difficult | Complete |
| Modular `train_modular.py` | ~200 + modules | Easy | Complete + Enhanced |

## Next Steps

1. **Immediate**: Use fixed `train_predict_map.py` for your current work
2. **Migration**: Switch to `train_modular.py` for new projects  
3. **Development**: Use modular architecture for adding new features

The modular system is fully backward compatible and provides the same functionality with better organization.
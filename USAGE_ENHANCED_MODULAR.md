# Enhanced Modular Training System Usage Guide

## 🚀 New Features Implemented

Your `train_modular.py` now supports:

1. **✅ Multi-patch training and prediction** in folders
2. **✅ Prediction-only mode** with pretrained weights  
3. **✅ Automatic model discovery** for prediction mode
4. **✅ Aggregated map generation** with overlap handling
5. **✅ Flexible file pattern matching**
6. **✅ Enhanced error handling and guidance**

## 📋 Quick Usage Examples

### 1. Train on All Patches in Folder
```bash
python train_modular.py \
    --patch-path "/path/to/patches/" \
    --model 2d_unet \
    --mode train_predict \
    --aggregate-predictions
```

### 2. Prediction-Only with Pretrained Model
```bash
# Auto-find existing model
python train_modular.py \
    --patch-path "/path/to/patches/" \
    --model 2d_unet \
    --mode predict \
    --aggregate-predictions

# Or specify model explicitly
python train_modular.py \
    --patch-path "/path/to/patches/" \
    --model 2d_unet \
    --mode predict \
    --pretrained-model "chm_outputs/2d_unet/final_model.pt" \
    --aggregate-predictions
```

### 3. Train Only (No Prediction)
```bash
python train_modular.py \
    --patch-path "/path/to/patches/" \
    --model 2d_unet \
    --mode train \
    --augment \
    --epochs 100
```

### 4. Specific File Pattern
```bash
python train_modular.py \
    --patch-path "/path/to/data/" \
    --patch-pattern "*temporal*patch*.tif" \
    --model 3d_unet \
    --mode train_predict
```

### 5. Your Specific Use Case
```bash
# For your Google Drive exports folder
python train_modular.py \
    --patch-path "chm_outputs/" \
    --patch-pattern "*.tif" \
    --model 2d_unet \
    --mode predict \
    --aggregate-predictions
```

## 🎯 Command Line Arguments

### Core Arguments
- `--patch-path`: Path to patch file or directory
- `--model`: Model type (rf, mlp, 2d_unet, 3d_unet)
- `--mode`: Operation mode (train, predict, train_predict)

### Multi-Patch Options
- `--multi-patch`: Explicitly enable multi-patch (auto-enabled for directories)
- `--patch-pattern`: File pattern for matching patches (default: "*.tif")
- `--aggregate-predictions`: Merge patch predictions into single map
- `--overlap-method`: How to handle overlaps (first, mean, max)

### Prediction Options
- `--pretrained-model`: Path to pretrained model file
- (Auto-discovery searches common locations if not specified)

### Training Options (all existing ones still work)
- `--augment`, `--validation-split`, `--early-stopping-patience`
- `--batch-size`, `--epochs`, `--learning-rate`
- Model-specific parameters

## 🔍 Model Auto-Discovery

When using `--mode predict` without `--pretrained-model`, the system searches:

```
{output_dir}/{model}/predictions/final_model.pt
{output_dir}/{model}/predictions/final_model.pth  
{output_dir}/{model}/final_model.pt
{output_dir}/{model}/final_model.pth
chm_outputs/{model}/predictions/final_model.pt
chm_outputs/{model}/predictions/final_model.pth
chm_outputs/{model}/final_model.pt
chm_outputs/{model}/final_model.pth
```

## 📁 File Organization

The enhanced system creates organized outputs:

```
chm_outputs/
├── {model}/
│   ├── predictions/
│   │   ├── final_model.pt              # Trained model
│   │   ├── prediction_patch0001.tif    # Individual predictions
│   │   ├── prediction_patch0002.tif
│   │   ├── ...
│   │   ├── merged_prediction_map.tif   # Aggregated map
│   │   └── training_metrics.json
│   └── patch_list.txt                  # Generated patch list
```

## 🔄 Workflow Examples

### Complete Multi-Patch Workflow
```bash
# 1. Train on all patches in folder
python train_modular.py \
    --patch-path "data/patches/" \
    --model 2d_unet \
    --mode train \
    --augment \
    --validation-split 0.2

# 2. Generate predictions with trained model
python train_modular.py \
    --patch-path "data/new_patches/" \
    --model 2d_unet \
    --mode predict \
    --aggregate-predictions

# 3. Or do both in one step
python train_modular.py \
    --patch-path "data/patches/" \
    --model 2d_unet \
    --mode train_predict \
    --aggregate-predictions
```

### Different Model Types
```bash
# Random Forest with temporal data
python train_modular.py \
    --patch-path "data/temporal_patches/" \
    --patch-pattern "*temporal*.tif" \
    --model rf \
    --mode train_predict

# 3D U-Net with temporal data
python train_modular.py \
    --patch-path "data/temporal_patches/" \
    --model 3d_unet \
    --mode train_predict \
    --batch-size 4 \
    --epochs 30
```

## 🚨 Error Handling

The enhanced system provides helpful error messages:

- **No patches found**: Check `--patch-pattern` and directory path
- **No pretrained model**: Use `--pretrained-model` or train first  
- **Invalid patch files**: Ensures files are valid GeoTIFF format
- **Multi-patch detection**: Automatically enables multi-patch mode

## 🔧 Troubleshooting

### Common Issues

1. **"No files matching pattern found"**
   ```bash
   # Check what files exist
   ls /path/to/patches/
   
   # Adjust pattern
   --patch-pattern "*.tiff"  # or *.TIF, *patch*.tif, etc.
   ```

2. **"No pretrained model found"**
   ```bash
   # Specify model explicitly
   --pretrained-model "path/to/your/final_model.pt"
   
   # Or train first
   --mode train
   ```

3. **Path issues**
   ```bash
   # Use absolute paths
   --patch-path "/full/path/to/patches/"
   
   # Or relative from current directory
   --patch-path "./chm_outputs/"
   ```

## 🎯 Recommended Workflows

### For Your Use Case
```bash
# 1. Train on local patches
python train_modular.py \
    --patch-path "chm_outputs/" \
    --model 2d_unet \
    --mode train \
    --augment \
    --epochs 50

# 2. Generate aggregated prediction map
python train_modular.py \
    --patch-path "chm_outputs/" \
    --model 2d_unet \
    --mode predict \
    --aggregate-predictions \
    --overlap-method mean
```

This enhanced system gives you full control over multi-patch processing while maintaining the clean interface you designed!
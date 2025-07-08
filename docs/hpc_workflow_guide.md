# HPC Workflow Guide

## Overview

This guide provides comprehensive instructions for running 2D canopy height model training and prediction on the Annuna HPC cluster using SLURM job scheduling.

## Environment Setup

### Prerequisites

```bash
# activate chm_env for python before implementation
source chm_env/bin/activate
```

### Module Loading
```bash
# Note: Lmod warnings for gcc and shared modules can be ignored
# These are system-level warnings that don't affect Python execution
+------------------------------------------------------------------+
|    Do you get Lmod warnings upon login? Make sure you do not     |
|    load any modules from your .bashrc!                            |
+------------------------------------------------------------------+
```

## Interactive Development

### Starting Interactive Session
```bash
# For testing and development (2 hours, 4 CPUs, 8GB RAM)
sinteractive -c 4 --mem 8000M --time=0-2:00:00

# For GPU work (if needed)
sinteractive -p gpu --gres=gpu:1 --constraint='nvidia&A100'

# Activate Python environment after session starts
source chm_env/bin/activate
```

### Interactive Testing Commands
```bash
# Test environment
python -c "import numpy, pandas, sklearn, rasterio; print('✅ Key packages available')"

# Quick RF training test
python train_predict_map.py \
    --patch-path "chm_outputs/dchm_09gd4_bandNum31_scale10_patch0000.tif" \
    --model rf \
    --output-dir "test_results" \
    --min-gedi-samples 10 \
    --verbose

# Exit interactive session
exit
```

## Production Job Submission

### Job Script Templates

#### 1. Complete 2D Model Training (`run_2d_training.sh`)
```bash
#!/bin/bash

#SBATCH --job-name=2d_model_training
#SBATCH --output=logs/%j_2d_training.txt
#SBATCH --error=logs/%j_2d_training_error.txt
#SBATCH --time=0-4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G

# Create output directories
mkdir -p logs results_2d

# Activate environment
source chm_env/bin/activate

# Train all three models sequentially
python train_predict_map.py --patch-dir "chm_outputs/" --model rf --output-dir "results_2d/rf" --min-gedi-samples 10 --verbose --patch-pattern "*_bandNum31_*.tif" --generate-prediction --save-model

python train_predict_map.py --patch-dir "chm_outputs/" --model mlp --output-dir "results_2d/mlp" --min-gedi-samples 10 --verbose --patch-pattern "*_bandNum31_*.tif" --epochs 100 --learning-rate 0.001 --hidden-layers "128,64,32" --generate-prediction --save-model

python train_predict_map.py --patch-dir "chm_outputs/" --model 2d_unet --output-dir "results_2d/2d_unet" --min-gedi-samples 10 --verbose --patch-pattern "*_bandNum31_*.tif" --epochs 50 --learning-rate 0.001 --base-channels 32 --generate-prediction --save-model
```

#### 2. Prediction Generation (`run_2d_prediction.sh`)
```bash
#!/bin/bash

#SBATCH --job-name=2d_prediction
#SBATCH --output=logs/%j_2d_prediction.txt
#SBATCH --error=logs/%j_2d_prediction_error.txt
#SBATCH --time=0-2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G

# Create output directories
mkdir -p logs predictions_2d

# Activate environment
source chm_env/bin/activate

# Generate predictions using trained models
python train_predict_map.py \
    --patch-dir "chm_outputs/" \
    --model rf \
    --mode predict \
    --model-path "results_2d/rf/multi_patch_rf_model.pkl" \
    --output-dir "predictions_2d/rf" \
    --patch-pattern "*_bandNum31_*.tif" \
    --verbose
```

#### 3. Simple RF Test (`run_rf_simple.sh`)
```bash
#!/bin/bash

#SBATCH --job-name=rf_simple
#SBATCH --output=logs/%j rf_simple.txt
#SBATCH --error=logs/%j_rf_simple_error.txt
#SBATCH --time=0-1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G

# Activate environment
source chm_env/bin/activate

# Single patch RF training
python train_predict_map.py \
    --patch-path "chm_outputs/dchm_09gd4_bandNum31_scale10_patch0000.tif" \
    --model rf \
    --output-dir "results_rf_simple" \
    --min-gedi-samples 10 \
    --verbose \
    --generate-prediction \
    --save-model
```

### Job Submission Commands
```bash
# Make scripts executable
chmod +x run_2d_training.sh run_2d_prediction.sh run_rf_simple.sh

# Submit jobs
sbatch run_2d_training.sh        # Complete training pipeline
sbatch run_2d_prediction.sh      # Prediction generation
sbatch run_rf_simple.sh          # Simple test
```

## Job Monitoring

### SLURM Commands
```bash
# Check job status
squeue -u $USER

# Check all running jobs
squeue -a

# Cancel jobs
scancel -u $USER          # Cancel all user jobs
scancel <JOBID>           # Cancel specific job

# Job history
sacct -X -u $USER --starttime 2024-01-01 --endtime now

# Detailed job information
scontrol show job <JOBID>
```

### Log Monitoring
```bash
# Real-time log monitoring
tail -f logs/<JOBID>_2d_training.txt

# Check error logs
cat logs/<JOBID>_2d_training_error.txt

# Monitor progress
python tmp/monitor_training.py
```

### Progress Monitoring Script
```python
# tmp/monitor_training.py - Automated monitoring
def main():
    print("🔍 CHM Training Job Monitor")
    check_job_status()      # SLURM job status
    check_results()         # Training results
    check_logs()            # Recent log files
    check_rf_predictions()  # Prediction files
```

## Resource Allocation Guidelines

### Memory Requirements
```bash
# Light training (RF, MLP): 2-4GB per CPU
--mem-per-cpu=2G

# Heavy training (U-Net): 4-8GB per CPU
--mem-per-cpu=4G

# Large patch processing: 8-16GB per CPU
--mem-per-cpu=8G
```

### CPU Allocation
```bash
# Single model training: 4 CPUs
--cpus-per-task=4

# Multi-model pipeline: 8 CPUs
--cpus-per-task=8

# Prediction generation: 2-4 CPUs
--cpus-per-task=2
```

### Time Limits
```bash
# Interactive testing: 2 hours
--time=0-2:00:00

# Single model training: 1-2 hours
--time=0-1:00:00

# Complete pipeline: 4-6 hours
--time=0-4:00:00
```

## File Organization

### Input Data Structure
```
chm_outputs/
├── dchm_09gd4_bandNum31_scale10_patch0000.tif
├── dchm_09gd4_bandNum31_scale10_patch0001.tif
├── ... (27 total patches)
└── dchm_09gd4_bandNum31_scale10_patch0062.tif
```

### Output Structure
```
logs/                           # SLURM job logs
├── <JOBID>_2d_training.txt
├── <JOBID>_2d_training_error.txt
└── <JOBID>_rf_simple.txt

results_2d/                     # Training results
├── rf/
│   ├── multi_patch_rf_model.pkl
│   ├── multi_patch_training_metrics.json
│   └── patch_summary.csv
├── mlp/
│   └── multi_patch_mlp_model.pkl
└── 2d_unet/
    └── multi_patch_2d_unet_model.pth

predictions_2d/                 # Prediction outputs
├── rf/
│   ├── patch0000_prediction.tif
│   └── ... (27 prediction files)
└── mlp/
    └── ... (if MLP predictions generated)

tmp/                           # Temporary scripts
├── test_gedi_filtering.py
├── monitor_training.py
└── generate_rf_predictions.py
```

## Common Issues and Solutions

### 1. Environment Activation Issues
```bash
# Problem: ModuleNotFoundError for numpy
# Solution: Ensure proper environment activation
source chm_env/bin/activate
python -c "import numpy; print('NumPy available')"
```

### 2. Memory Issues
```bash
# Problem: Job killed due to memory limit
# Solution: Increase memory allocation
#SBATCH --mem-per-cpu=8G
```

### 3. Feature Dimension Mismatch
```bash
# Problem: Model expects different number of features
# Solution: Check band consistency between training and prediction
ls -la chm_outputs/*bandNum31*.tif | wc -l  # Count 31-band patches
```

### 4. No Valid GEDI Data
```bash
# Problem: All patches filtered out
# Solution: Lower GEDI threshold or check data quality
python train_predict_map.py --min-gedi-samples 5  # Lower threshold
```

## Performance Optimization

### Parallel Processing
```bash
# Multiple independent jobs
sbatch run_rf_training.sh &
sbatch run_mlp_training.sh &
sbatch run_unet_training.sh &
wait  # Wait for all jobs to complete
```

### Memory Optimization
```bash
# Process patches in batches
python train_predict_map.py \
    --patch-files "patch1.tif,patch2.tif,patch3.tif" \
    --model rf \
    --output-dir results
```

### Storage Management
```bash
# Clean up temporary files
rm -rf test_*_results/
rm -f logs/*_error_*.txt  # Remove empty error logs

# Compress large outputs
tar -czf predictions_backup.tar.gz predictions_2d/
```

## Best Practices

### 1. Testing Workflow
```bash
# Always test with small subset first
sbatch run_rf_simple.sh

# Check results before full pipeline
python tmp/monitor_training.py

# Submit full pipeline only after successful testing
sbatch run_2d_training.sh
```

### 2. Resource Management
```bash
# Monitor resource usage
squeue -u $USER -l  # Check allocated resources

# Adjust resource requests based on actual usageyo
sacct -j <JOBID> --format=JobID,MaxRSS,MaxVMSize,CPUTime
```

### 3. Data Backup
```bash
# Backup important results
cp -r results_2d/ /backup/location/
rsync -av predictions_2d/ /backup/predictions/
```

### 4. Documentation
```bash
# Keep detailed logs
echo "Training started: $(date)" >> training_log.txt
echo "Job ID: $SLURM_JOB_ID" >> training_log.txt
```

## Troubleshooting

### Job Failure Diagnosis
```bash
# Check exit codes
sacct -j <JOBID> --format=JobID,ExitCode,State

# Examine error logs
grep -i error logs/<JOBID>_2d_training_error.txt

# Check resource limits
sacct -j <JOBID> --format=JobID,MaxRSS,MaxVMSize,ReqMem
```

### Recovery Procedures
```bash
# Resume interrupted training
python train_predict_map.py --resume-from checkpoint.pth

# Reprocess failed patches
python train_predict_map.py --patch-files "failed_patch.tif"

# Clean restart
rm -rf results_2d/ && sbatch run_2d_training.sh
```

---
*Last Updated: June 24, 2025*  
*Environment: Annuna HPC Cluster*  
*SLURM Version: Compatible with current cluster configuration*
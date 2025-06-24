#!/bin/bash

#SBATCH --job-name=rf_simple
#SBATCH --output=logs/rf_simple_%j.txt
#SBATCH --error=logs/rf_simple_error_%j.txt
#SBATCH --time=0-1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G

# Create output directories
mkdir -p logs
mkdir -p results_rf_simple

echo "🌲 Starting Simple Random Forest Training"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

# Activate environment
source chm_env/bin/activate

# Train only Random Forest with focus on single consistent patches
python train_predict_map.py \
    --patch-path "chm_outputs/dchm_09gd4_bandNum31_scale10_patch0000.tif" \
    --model rf \
    --output-dir "results_rf_simple" \
    --min-gedi-samples 10 \
    --verbose \
    --generate-prediction \
    --save-model

echo "🎉 Simple RF training completed!"
echo "End time: $(date)"
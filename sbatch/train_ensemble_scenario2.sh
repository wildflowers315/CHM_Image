#!/bin/bash

#SBATCH --job-name=ensemble_s2
#SBATCH --output=chm_outputs/logs/ensemble_s2_%j.txt
#SBATCH --error=chm_outputs/logs/ensemble_s2_error_%j.txt
#SBATCH --time=0-2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --partition=gpu

echo "🚀 Starting Ensemble Training for Scenario 2"
echo "📅 Start time: $(date)"
echo "🖥️  Node: $(hostname)"
echo "🔧 GPU: $CUDA_VISIBLE_DEVICES"

# Create log directory
mkdir -p chm_outputs/logs

# Activate environment
source chm_env/bin/activate

echo "🧠 Training ensemble model combining GEDI shift-aware U-Net and production MLP..."

# Train ensemble model
python train_ensemble_mlp.py \
    --gedi-model-path chm_outputs/scenario2_gedi_shift_aware/shift_aware_unet_r2.pth \
    --reference-model-path chm_outputs/production_mlp_best.pth \
    --reference-height-path downloads/dchm_05LE4.tif \
    --output-dir chm_outputs/scenario2_ensemble \
    --patch-dir chm_outputs/enhanced_patches/ \
    --patch-pattern "*05LE4*" \
    --epochs 50 \
    --learning-rate 0.001 \
    --model-type simple

echo "✅ Ensemble training completed at $(date)"

# Check results
if [ -f "chm_outputs/scenario2_ensemble/simple_ensemble_best.pth" ]; then
    echo "✅ Ensemble model saved successfully"
    ls -la chm_outputs/scenario2_ensemble/
else
    echo "❌ Ensemble model not found"
    echo "Files in output directory:"
    ls -la chm_outputs/scenario2_ensemble/
fi
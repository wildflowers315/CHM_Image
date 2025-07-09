#!/bin/bash

#SBATCH --job-name=gedi_s2_train
#SBATCH --output=chm_outputs/logs/%j_gedi_s2_train.txt
#SBATCH --time=0-2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --error=chm_outputs/logs/%j_gedi_s2_train_error.txt

# Create log directory
mkdir -p chm_outputs/logs

echo "🚀 Starting GEDI Shift-Aware Training for Scenario 2"
echo "📅 Start time: $(date)"
echo "🖥️  Node: $(hostname)"
echo "🔧 GPU: $CUDA_VISIBLE_DEVICES"

# Activate environment
source chm_env/bin/activate

# Run GEDI training with NaN replacement
echo "🧠 Training GEDI model with NaN replacement strategy..."
python tmp/train_gedi_filtered.py

echo "✅ GEDI training completed at $(date)"

# Check if model was created
if [ -f "chm_outputs/scenario2_gedi_shift_aware/gedi_model_filtered.pth" ]; then
    echo "✅ GEDI model saved successfully"
    ls -la chm_outputs/scenario2_gedi_shift_aware/gedi_model_filtered.pth
else
    echo "❌ GEDI model not found"
    exit 1
fi
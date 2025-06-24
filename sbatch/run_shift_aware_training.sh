#!/bin/bash

#SBATCH --job-name=shift_aware_training
#SBATCH --output=logs/shift_aware_%j.txt
#SBATCH --error=logs/shift_aware_error_%j.txt
#SBATCH --time=0-3:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --partition=gpu

# Create output directory if it doesn't exist
mkdir -p logs

echo "🚀 SHIFT-AWARE U-NET TRAINING"
echo "=============================================================="
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo ""

echo "📋 Training Configuration:"
echo "  🎯 Model: shift_aware_unet (PRODUCTION)"
echo "  📊 Shift radius: 2 (optimal)"
echo "  📈 Epochs: 50"
echo "  🔄 Batch size: 2"
echo "  📚 Learning rate: 0.0001"
echo "  🗺️  Comprehensive mosaic: AUTO"
echo ""

# Activate environment
source chm_env/bin/activate

echo "🚀 Running shift-aware training..."

# Run shift-aware training with comprehensive mosaic generation
python train_predict_map.py \
  --patch-dir "chm_outputs/" \
  --model shift_aware_unet \
  --output-dir chm_outputs/results/shift_aware \
  --shift-radius 2 \
  --epochs 50 \
  --learning-rate 0.0001 \
  --batch-size 2 \
  --generate-prediction

echo ""
echo "📊 Checking training results..."
if [ $? -eq 0 ]; then
    echo "✅ Shift-aware training completed successfully!"
else
    echo "❌ Training failed with exit code: $?"
fi

echo ""
echo "📁 Output Summary:"
echo "   📂 Model directory: chm_outputs/results/shift_aware/"
echo "   📂 Predictions: chm_outputs/predictions/"
echo "   📂 Logs: logs/shift_aware_${SLURM_JOB_ID}.txt"

echo ""
echo "🏁 Training run completed!"
echo "End time: $(date)"
echo "=============================================================="
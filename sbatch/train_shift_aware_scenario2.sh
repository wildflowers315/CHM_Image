#!/bin/bash

#SBATCH --job-name=shift_aware_s2
#SBATCH --output=chm_outputs/logs/%j_shift_aware_s2.txt
#SBATCH --time=0-3:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --error=chm_outputs/logs/%j_shift_aware_s2_error.txt

# Create log directory
mkdir -p chm_outputs/logs

echo "🚀 Starting Shift-Aware U-Net Training for Scenario 2"
echo "📅 Start time: $(date)"
echo "🖥️  Node: $(hostname)"
echo "🔧 GPU: $CUDA_VISIBLE_DEVICES"

# Activate environment
source chm_env/bin/activate

# Train shift-aware U-Net on Hyogo patches
echo "🧠 Training GEDI shift-aware U-Net model..."
python train_predict_map.py \
  --patch-dir chm_outputs/ \
  --patch-pattern "*05LE4*" \
  --model shift_aware_unet \
  --shift-radius 2 \
  --output-dir chm_outputs/scenario2_gedi_shift_aware \
  --epochs 50 \
  --learning-rate 0.0001 \
  --batch-size 4 \
  --min-gedi-samples 5 \
  --save-model \
  --verbose

echo "✅ Shift-aware training completed at $(date)"

# Check if model was created
if [ -f "chm_outputs/scenario2_gedi_shift_aware/shift_aware_unet_model.pth" ]; then
    echo "✅ Shift-aware U-Net model saved successfully"
    ls -la chm_outputs/scenario2_gedi_shift_aware/shift_aware_unet_model.pth
else
    echo "❌ Shift-aware U-Net model not found"
    # List what files were created
    echo "Files in output directory:"
    ls -la chm_outputs/scenario2_gedi_shift_aware/
    exit 1
fi
#!/bin/bash

#SBATCH --job-name=gedi_s5_ensemble
#SBATCH --output=logs/%j_gedi_s5_ensemble.txt
#SBATCH --error=logs/%j_gedi_s5_ensemble_error.txt
#SBATCH --time=0-2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --partition=gpu_amd

echo "🚀 Starting GEDI Scenario 5 Ensemble Training"
echo "📅 Start time: $(date)"
echo "🖥️  Node: $(hostname)"
echo "🔧 GPU: $CUDA_VISIBLE_DEVICES"

# Create log directory
mkdir -p chm_outputs/logs

# Activate environment
source chm_env/bin/activate

echo "🧠 Training ensemble model combining GEDI pixel MLP (Scenario 4) and Google Embedding MLP (Scenario 1)..."

# Train ensemble model
python train_ensemble_mlp.py \
    --gedi-model-path chm_outputs/gedi_pixel_mlp_scenario4/gedi_pixel_mlp_scenario4_embedding_best.pth \
    --reference-model-path chm_outputs/production_mlp_reference_embedding_best.pth \
    --reference-height-path downloads/dchm_05LE4.tif \
    --output-dir chm_outputs/gedi_scenario5_ensemble \
    --patch-dir chm_outputs/ \
    --patch-pattern "*05LE4*embedding*" \
    --epochs 50 \
    --learning-rate 0.001 \
    --model-type simple \
    --gedi-model-type mlp \
    --band-selection embedding

echo "✅ Ensemble training completed at $(date)"

# Check results
if [ -f "chm_outputs/gedi_scenario5_ensemble/ensemble_mlp_best.pth" ]; then
    echo "✅ Ensemble model saved successfully"
    ls -la chm_outputs/gedi_scenario5_ensemble/
else
    echo "❌ Ensemble model not found"
    echo "Files in output directory:"
    ls -la chm_outputs/gedi_scenario5_ensemble/
fi
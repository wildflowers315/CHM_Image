#!/bin/bash

#SBATCH --job-name=google_emb_s2a_ensemble
#SBATCH --output=logs/%j_google_embedding_s2a_ensemble.txt
#SBATCH --time=0-2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=gpu_amd
#SBATCH --gpus=1
#SBATCH --error=logs/%j_google_embedding_s2a_ensemble_error.txt

# Create log directory
mkdir -p logs

echo "🚀 Starting Google Embedding Scenario 2A Ensemble Training"
echo "📅 Start time: $(date)"
echo "🖥️  Node: $(hostname)"
echo "🔧 GPU: $CUDA_VISIBLE_DEVICES"

# Activate environment
source chm_env/bin/activate

# Check prerequisite models
GEDI_MODEL="chm_outputs/google_embedding_scenario2a/gedi_unet_model/best_model.pth"
REFERENCE_MODEL="chm_outputs/production_mlp_reference_embedding_best.pth"

echo "🔍 Checking prerequisite models..."
if [ ! -f "$GEDI_MODEL" ]; then
    echo "❌ GEDI U-Net model not found: $GEDI_MODEL"
    echo "Please run GEDI training first (sbatch/run_google_embedding_scenario2a_training.sh)"
    exit 1
fi

if [ ! -f "$REFERENCE_MODEL" ]; then
    echo "❌ Reference MLP model not found: $REFERENCE_MODEL"
    echo "Please ensure Google Embedding Scenario 1 is completed"
    exit 1
fi

echo "✅ Found GEDI U-Net model: $GEDI_MODEL"
echo "✅ Found Reference MLP model: $REFERENCE_MODEL"

echo "🧠 Training ensemble model combining Google Embedding GEDI U-Net + Reference MLP..."

# Train ensemble model combining GEDI U-Net with Reference MLP (both using Google Embedding)
python train_ensemble_mlp.py \
    --gedi-model-path "$GEDI_MODEL" \
    --reference-model-path "$REFERENCE_MODEL" \
    --reference-height-path downloads/dchm_05LE4.tif \
    --output-dir chm_outputs/google_embedding_scenario2a_ensemble \
    --patch-dir chm_outputs/ \
    --patch-pattern "*05LE4*embedding*" \
    --band-selection embedding \
    --epochs 50 \
    --learning-rate 0.001 \
    --model-type simple

echo "✅ Google Embedding Scenario 2A ensemble training completed at $(date)"

# Check ensemble results
if [ -f "chm_outputs/google_embedding_scenario2a_ensemble/simple_ensemble_best.pth" ]; then
    echo "✅ Google Embedding Scenario 2A ensemble model saved successfully"
    echo "📁 Final ensemble model: chm_outputs/google_embedding_scenario2a_ensemble/simple_ensemble_best.pth"
    ls -la chm_outputs/google_embedding_scenario2a_ensemble/
    
    echo "🎯 Google Embedding Scenario 2A ensemble training completed successfully!"
    echo "📊 Ready for cross-region prediction and evaluation"
else
    echo "❌ Google Embedding Scenario 2A ensemble model not found"
    echo "Files in output directory:"
    ls -la chm_outputs/google_embedding_scenario2a_ensemble/
    exit 1
fi

echo "🎉 Google Embedding Scenario 2A (GEDI U-Net + Reference MLP Ensemble) completed at $(date)"
#!/bin/bash

#SBATCH --job-name=google_emb_s2a_pred
#SBATCH --output=logs/%j_google_embedding_s2a_prediction.out
#SBATCH --error=logs/%j_google_embedding_s2a_prediction.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Activate Python environment
source chm_env/bin/activate

echo "🚀 Starting Google Embedding Scenario 2A Cross-Region Prediction"
echo "📅 Start time: $(date)"
echo "🖥️  Node: $(hostname)"

# --- Configuration ---
ENSEMBLE_MODEL="chm_outputs/google_embedding_scenario2a/ensemble_model/ensemble_mlp_best.pth"
OUTPUT_DIR="chm_outputs/google_embedding_scenario2a_predictions"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# --- Step 1: Cross-Region Ensemble Prediction ---
echo "🧠 Step 1: Running cross-region ensemble prediction with Google Embedding..."

python predict_ensemble.py \
    --ensemble-model "${ENSEMBLE_MODEL}" \
    --gedi-model chm_outputs/google_embedding_scenario2a/gedi_unet_model/shift_aware_unet_r2.pth \
    --mlp-model chm_outputs/production_mlp_reference_embedding_best.pth \
    --region all \
    --patch-dir chm_outputs/ \
    --band-selection embedding \
    --output-dir "${OUTPUT_DIR}"

if [ $? -ne 0 ]; then
    echo "❌ Cross-region prediction failed. Exiting." >&2
    exit 1
fi

echo "✅ Cross-region prediction completed at $(date)"

# --- Step 2: Evaluate Predictions ---
echo "🧠 Step 2: Evaluating Google Embedding Scenario 2A predictions..."

python evaluate_ensemble_cross_region.py \
    --prediction-dir "${OUTPUT_DIR}" \
    --reference-heights downloads/ \
    --output-dir "${OUTPUT_DIR}/evaluation" \
    --band-selection embedding

if [ $? -ne 0 ]; then
    echo "❌ Evaluation failed. Exiting." >&2
    exit 1
fi

echo "✅ Evaluation completed at $(date)"

# --- Summary ---
echo "🎉 Google Embedding Scenario 2A Cross-Region Pipeline Completed!"
echo "📁 Predictions: ${OUTPUT_DIR}/"
echo "📊 Evaluation: ${OUTPUT_DIR}/evaluation/"
echo "⏰ Completed at: $(date)"

# List key results
echo "📋 Key Result Files:"
if [ -d "${OUTPUT_DIR}/evaluation" ]; then
    find "${OUTPUT_DIR}/evaluation" -name "*.json" -o -name "*.csv" | head -10
fi
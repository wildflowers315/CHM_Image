#!/bin/bash

#SBATCH --job-name=gedi_s5_ensemble_pred
#SBATCH --output=logs/%j_gedi_s5_ensemble_prediction.txt
#SBATCH --error=logs/%j_gedi_s5_ensemble_prediction_error.txt
#SBATCH --partition=gpu_amd
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Activate Python environment
source chm_env/bin/activate

echo "🚀 Starting GEDI Scenario 5 Ensemble Cross-Region Prediction"
echo "📅 Start time: $(date)"
echo "🖥️  Node: $(hostname)"

# --- Configuration ---
ENSEMBLE_MODEL="chm_outputs/gedi_scenario5_ensemble/ensemble_mlp_best.pth"
OUTPUT_DIR="chm_outputs/gedi_scenario5_predictions"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# --- Step 1: Cross-Region Ensemble Prediction ---
echo "🧠 Step 1: Running cross-region ensemble prediction with GEDI Scenario 5..."

python predict_ensemble.py \
    --ensemble-model "${ENSEMBLE_MODEL}" \
    --gedi-model chm_outputs/gedi_pixel_mlp_scenario4/gedi_pixel_mlp_scenario4_embedding_best.pth \
    --mlp-model chm_outputs/production_mlp_reference_embedding_best.pth \
    --region all \
    --patch-dir chm_outputs/ \
    --band-selection embedding \
    --gedi-model-type mlp \
    --output-dir "${OUTPUT_DIR}"

if [ $? -ne 0 ]; then
    echo "❌ Cross-region prediction failed. Exiting." >&2
    exit 1
fi

echo "✅ Cross-region prediction completed at $(date)"

#!/bin/bash
#SBATCH --job-name=pred_scenario3b
#SBATCH --output=logs/%j_predict_scenario3b.txt
#SBATCH --error=logs/%j_predict_scenario3b_error.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Activate Python environment
source chm_env/bin/activate

echo "🚀 Starting Scenario 3B cross-region predictions at $(date)"
echo "📊 Scenario 3B: GEDI fine-tuned + Fixed Ensemble"

# --- Configuration ---
GEDI_MODEL="chm_outputs/google_embedding_scenario3b/gedi_unet_model/shift_aware_unet_r2.pth"
REFERENCE_MODEL="chm_outputs/production_mlp_reference_embedding_best.pth"
ENSEMBLE_MLP="chm_outputs/google_embedding_scenario3b/ensemble_model/ensemble_mlp_best.pth"
OUTPUT_DIR="chm_outputs/google_embedding_scenario3b_predictions"

# Create output directories
mkdir -p "${OUTPUT_DIR}"/{kochi,hyogo,tochigi}

echo "📁 Model files:"
echo "   - GEDI U-Net: ${GEDI_MODEL}"
echo "   - Reference MLP: ${REFERENCE_MODEL}"
echo "   - Ensemble MLP: ${ENSEMBLE_MLP}"

# Generate cross-region predictions using existing predict_ensemble.py
for region in kochi hyogo tochigi; do
    echo "🌍 Generating predictions for ${region}..."
    python predict_ensemble.py \
      --ensemble-model "${ENSEMBLE_MLP}" \
      --gedi-model "${GEDI_MODEL}" \
      --mlp-model "${REFERENCE_MODEL}" \
      --region "${region}" \
      --patch-dir "chm_outputs/" \
      --band-selection embedding \
      --output-dir "${OUTPUT_DIR}"
done

echo "✅ Scenario 3B predictions completed at $(date)"
echo "📁 Predictions saved in: ${OUTPUT_DIR}"
echo "🌍 Regions: Kochi (04hf3), Hyogo (05LE4), Tochigi (09gd4)"
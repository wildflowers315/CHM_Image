#!/bin/bash
#SBATCH --job-name=google_embedding_s3a
#SBATCH --output=logs/%j_google_embedding_s3a.txt
#SBATCH --error=logs/%j_google_embedding_s3a_error.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Activate Python environment
source chm_env/bin/activate

# --- Configuration ---
PATCH_DIR="chm_outputs/"
PATCH_PATTERN="*09gd4*embedding*bandNum70*"  # Only patches with GEDI data (70 bands)
OUTPUT_ROOT="chm_outputs/google_embedding_scenario3a"

# --- Step 1: Train GEDI Spatial U-Net from Scratch on Tochigi ---
GEDI_UNET_DIR="${OUTPUT_ROOT}/gedi_unet_model"
GEDI_UNET_MODEL_PATH="${GEDI_UNET_DIR}/shift_aware_unet_r2.pth"

echo "--- Starting Step 1: Training GEDI Spatial U-Net from Scratch on Tochigi ---"
python train_predict_map.py \
  --patch-dir "${PATCH_DIR}" \
  --patch-pattern "${PATCH_PATTERN}" \
  --model shift_aware_unet \
  --shift-radius 2 \
  --supervision-mode gedi \
  --band-selection embedding \
  --output-dir "${GEDI_UNET_DIR}" \
  --epochs 50 \
  --learning-rate 0.0001 \
  --batch-size 4 \
  --min-gedi-samples 5 \
  --save-model \
  --verbose

if [ ! -f "${GEDI_UNET_MODEL_PATH}" ]; then
    echo "GEDI U-Net training failed. Exiting." >&2
    exit 1
fi
echo "--- GEDI U-Net training completed successfully. ---"

# --- Step 2: Copy Fixed Ensemble MLP (No Retraining) ---
FIXED_ENSEMBLE_PATH="chm_outputs/google_embedding_scenario2a/ensemble_model/ensemble_mlp_best.pth"
ENSEMBLE_DIR="${OUTPUT_ROOT}/ensemble_model"
mkdir -p "${ENSEMBLE_DIR}"

echo "--- Starting Step 2: Using Fixed Ensemble MLP from Scenario 2A ---"
cp "${FIXED_ENSEMBLE_PATH}" "${ENSEMBLE_DIR}/ensemble_mlp_best.pth"

if [ ! -f "${ENSEMBLE_DIR}/ensemble_mlp_best.pth" ]; then
    echo "Fixed ensemble copy failed. Exiting." >&2
    exit 1
fi
echo "--- Fixed ensemble setup completed successfully. ---"

echo "\n✅ Google Embedding Scenario 3A (from scratch + fixed ensemble) workflow finished."
echo "📁 Model outputs:"
echo "   - GEDI U-Net: ${GEDI_UNET_MODEL_PATH}"
echo "   - Fixed Ensemble: ${ENSEMBLE_DIR}/ensemble_mlp_best.pth"
echo "🎯 Ready for cross-region predictions with Scenario 3A models"
#!/bin/bash
#SBATCH --job-name=google_embedding_s3b
#SBATCH --output=logs/%j_google_embedding_s3b.txt
#SBATCH --error=logs/%j_google_embedding_s3b_error.txt
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
OUTPUT_ROOT="chm_outputs/google_embedding_scenario3b"

# --- Step 1: Fine-tune GEDI Spatial U-Net on Tochigi ---
GEDI_UNET_DIR="${OUTPUT_ROOT}/gedi_unet_model"
GEDI_UNET_MODEL_PATH="${GEDI_UNET_DIR}/shift_aware_unet_r2.pth"
PRETRAINED_GEDI_PATH="chm_outputs/google_embedding_scenario2a/gedi_unet_model/shift_aware_unet_r2.pth"

echo "--- Starting Step 1: Fine-tuning GEDI Spatial U-Net on Tochigi ---"
python train_predict_map.py \
  --patch-dir "${PATCH_DIR}" \
  --patch-pattern "${PATCH_PATTERN}" \
  --model shift_aware_unet \
  --shift-radius 2 \
  --supervision-mode gedi \
  --band-selection embedding \
  --pretrained-model-path "${PRETRAINED_GEDI_PATH}" \
  --output-dir "${GEDI_UNET_DIR}" \
  --epochs 50 \
  --learning-rate 0.0001 \
  --batch-size 4 \
  --min-gedi-samples 5 \
  --save-model \
  --verbose \
  --fine-tune-mode

if [ ! -f "${GEDI_UNET_MODEL_PATH}" ]; then
    echo "GEDI U-Net fine-tuning failed. Exiting." >&2
    exit 1
fi
echo "--- GEDI U-Net fine-tuning completed successfully. ---"

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

echo "\n✅ Google Embedding Scenario 3B (fine-tuned + fixed ensemble) workflow finished."
echo "📁 Model outputs:"
echo "   - GEDI U-Net: ${GEDI_UNET_MODEL_PATH}"
echo "   - Fixed Ensemble: ${ENSEMBLE_DIR}/ensemble_mlp_best.pth"
echo "🎯 Ready for cross-region predictions with Scenario 3B models"
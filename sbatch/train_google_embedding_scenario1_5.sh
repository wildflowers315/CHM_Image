#!/bin/bash
#SBATCH --job-name=google_embedding_s1_5
#SBATCH --output=logs/%j_google_embedding_s1_5.txt
#SBATCH --error=logs/%j_google_embedding_s1_5_error.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Activate Python environment
source chm_env/bin/activate

# --- Configuration ---
PATCH_DIR="chm_outputs/"
PATCH_PATTERN="*05LE4*embedding*"
REFERENCE_TIF="downloads/dchm_05LE4.tif"
OUTPUT_ROOT="chm_outputs/google_embedding_scenario2a"

# --- Step 1: Train GEDI Spatial U-Net with Google Embedding ---
GEDI_UNET_DIR="${OUTPUT_ROOT}/gedi_unet_model"
GEDI_UNET_MODEL_PATH="${GEDI_UNET_DIR}/shift_aware_unet_r2.pth"

echo "--- Starting Step 1: Training GEDI Spatial U-Net with Google Embedding ---"
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
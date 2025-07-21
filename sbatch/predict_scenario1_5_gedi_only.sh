#!/bin/bash
#SBATCH --job-name=scenario1_5_gedi_only
#SBATCH --output=logs/%j_scenario1_5_gedi_only.txt
#SBATCH --error=logs/%j_scenario1_5_gedi_only_error.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Activate Python environment
source chm_env/bin/activate

echo "🚀 Starting Scenario 1.5: GEDI-Only Predictions at $(date)"
echo "📊 Model: Hyogo-trained GEDI shift-aware U-Net (Google Embedding)"
echo "🎯 Testing pure GEDI performance without ensemble combination"

# --- Configuration ---
GEDI_MODEL="chm_outputs/google_embedding_scenario2a/gedi_unet_model/shift_aware_unet_r2.pth"
OUTPUT_DIR="chm_outputs/scenario1_5_gedi_only_predictions"

# Create output directories
mkdir -p "${OUTPUT_DIR}"/{kochi,hyogo,tochigi}

echo "📁 GEDI Model: ${GEDI_MODEL}"
echo "📁 Output Directory: ${OUTPUT_DIR}"

# Check if GEDI model exists
if [ ! -f "${GEDI_MODEL}" ]; then
    echo "❌ GEDI model not found: ${GEDI_MODEL}"
    exit 1
fi

# Generate GEDI-only predictions for each region
for region in kochi hyogo tochigi; do
    echo ""
    echo "🌍 Generating GEDI-only predictions for ${region}..."
    python predict_gedi_only.py \
      --gedi-model "${GEDI_MODEL}" \
      --region "${region}" \
      --patch-dir "chm_outputs/" \
      --output-dir "${OUTPUT_DIR}" \
      --band-selection embedding \
      --device cuda
    
    if [ $? -ne 0 ]; then
        echo "❌ Failed to generate predictions for ${region}"
        exit 1
    fi
done

echo ""
echo "✅ Scenario 1.5 GEDI-only predictions completed at $(date)"
echo "📁 Predictions saved in: ${OUTPUT_DIR}"
echo "🌍 Regions: Kochi, Hyogo, Tochigi"
echo "📊 Model: Pure GEDI shift-aware U-Net (no ensemble, no reference model)"
echo "🎯 Ready for evaluation against ensemble scenarios"
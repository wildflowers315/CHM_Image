#!/bin/bash

#SBATCH --job-name=gedi_pixel_s4_pred
#SBATCH --output=logs/%j_gedi_pixel_scenario4_predictions.txt
#SBATCH --error=logs/%j_gedi_pixel_scenario4_predictions_error.txt
#SBATCH --time=0-3:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --partition=gpu_amd

# Create output directories
mkdir -p logs
mkdir -p chm_outputs/gedi_pixel_scenario4_predictions/kochi
mkdir -p chm_outputs/gedi_pixel_scenario4_predictions/tochigi
mkdir -p chm_outputs/gedi_pixel_scenario4_predictions/hyogo

# Activate environment
source chm_env/bin/activate

# GEDI Pixel MLP Scenario 4: Cross-Region Prediction
# Using trained GEDI pixel model for patch-level prediction
echo "Starting GEDI Pixel MLP Scenario 4 cross-region prediction at $(date)"
echo "Using predict_mlp_cross_region.py with GEDI pixel-trained model"

# Check if model exists
MODEL_PATH="chm_outputs/gedi_pixel_mlp_scenario4/gedi_pixel_mlp_scenario4_embedding_best.pth"
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ GEDI Pixel MLP model file not found: $MODEL_PATH"
    echo "Please run train_gedi_pixel_mlp_scenario4.sh first"
    exit 1
fi

echo "✅ GEDI Pixel MLP model file found: $MODEL_PATH"

# Predict for Kochi region (dchm_04hf3)
echo "🌍 Predicting Kochi region (dchm_04hf3) with GEDI pixel model..."
python predict_mlp_cross_region.py \
    --model-path "$MODEL_PATH" \
    --patch-dir chm_outputs/ \
    --patch-pattern "*04hf3*embedding*" \
    --output-dir chm_outputs/gedi_pixel_scenario4_predictions/kochi/ \
    --device auto

# Predict for Tochigi region (dchm_09gd4)
echo "🌍 Predicting Tochigi region (dchm_09gd4) with GEDI pixel model..."
python predict_mlp_cross_region.py \
    --model-path "$MODEL_PATH" \
    --patch-dir chm_outputs/ \
    --patch-pattern "*09gd4*embedding*" \
    --output-dir chm_outputs/gedi_pixel_scenario4_predictions/tochigi/ \
    --device auto

# Predict for Hyogo region (dchm_05LE4) - for comparison
echo "🌍 Predicting Hyogo region (dchm_05LE4) with GEDI pixel model..."
python predict_mlp_cross_region.py \
    --model-path "$MODEL_PATH" \
    --patch-dir chm_outputs/ \
    --patch-pattern "*05LE4*embedding*" \
    --output-dir chm_outputs/gedi_pixel_scenario4_predictions/hyogo/ \
    --device auto

echo "GEDI Pixel MLP Scenario 4 cross-region prediction completed at $(date)"

# Count prediction files
echo "📊 GEDI Pixel Prediction Summary:"
echo "   Kochi predictions: $(find chm_outputs/gedi_pixel_scenario4_predictions/kochi/ -name '*_mlp_prediction.tif' 2>/dev/null | wc -l)"
echo "   Tochigi predictions: $(find chm_outputs/gedi_pixel_scenario4_predictions/tochigi/ -name '*_mlp_prediction.tif' 2>/dev/null | wc -l)"
echo "   Hyogo predictions: $(find chm_outputs/gedi_pixel_scenario4_predictions/hyogo/ -name '*_mlp_prediction.tif' 2>/dev/null | wc -l)"

# List some example prediction files
echo "📁 Example GEDI pixel prediction files:"
find chm_outputs/gedi_pixel_scenario4_predictions/ -name '*_mlp_prediction.tif' 2>/dev/null | head -5

# Display model info
echo "🧠 GEDI Pixel Model Information:"
python -c "
import torch
model_info = torch.load('$MODEL_PATH', map_location='cpu', weights_only=False)
print(f'Training R²: {model_info.get(\"val_r2\", \"N/A\"):.4f}')
print(f'Training Epoch: {model_info.get(\"epoch\", \"N/A\")}')
print(f'Scenario: {model_info.get(\"scenario\", \"N/A\")}')
print(f'Band Selection: {model_info.get(\"band_selection\", \"N/A\")}')
"
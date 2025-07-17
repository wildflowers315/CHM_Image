#!/bin/bash

#SBATCH --job-name=google_embed_s1_pred
#SBATCH --output=logs/%j_google_embedding_scenario1_predictions.txt
#SBATCH --error=logs/%j_google_embedding_scenario1_predictions_error.txt
#SBATCH --time=0-3:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --partition=gpu_amd

# Create output directory
mkdir -p logs

# Activate environment
source chm_env/bin/activate

# Google Embedding Scenario 1: Cross-Region Prediction
# Following the plan in docs/google_embedding_training_plan.md
echo "Starting Google Embedding Scenario 1 cross-region prediction at $(date)"
echo "Using existing predict_mlp_cross_region.py with Google Embedding model"

# Check if model exists
MODEL_PATH="chm_outputs/production_mlp_reference_embedding_best.pth"
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Model file not found: $MODEL_PATH"
    echo "Please run train_google_embedding_scenario1.sh first"
    exit 1
fi

echo "✅ Model file found: $MODEL_PATH"

# Predict for Kochi region (dchm_04hf3)
echo "🌍 Predicting Kochi region (dchm_04hf3)..."
python predict_mlp_cross_region.py \
    --model-path "$MODEL_PATH" \
    --patch-dir chm_outputs/ \
    --patch-pattern "*04hf3*embedding*" \
    --output-dir chm_outputs/google_embedding_scenario1_predictions/kochi/ \
    --device auto

# Predict for Tochigi region (dchm_09gd4)
echo "🌍 Predicting Tochigi region (dchm_09gd4)..."
python predict_mlp_cross_region.py \
    --model-path "$MODEL_PATH" \
    --patch-dir chm_outputs/ \
    --patch-pattern "*09gd4*embedding*" \
    --output-dir chm_outputs/google_embedding_scenario1_predictions/tochigi/ \
    --device auto

# Predict for Hyogo region (dchm_05LE4) - for comparison
echo "🌍 Predicting Hyogo region (dchm_05LE4) - training region..."
python predict_mlp_cross_region.py \
    --model-path "$MODEL_PATH" \
    --patch-dir chm_outputs/ \
    --patch-pattern "*05LE4*embedding*" \
    --output-dir chm_outputs/google_embedding_scenario1_predictions/hyogo/ \
    --device auto

echo "Google Embedding Scenario 1 cross-region prediction completed at $(date)"

# Count prediction files
echo "📊 Prediction Summary:"
echo "   Kochi predictions: $(find chm_outputs/google_embedding_scenario1_predictions/kochi/ -name '*_mlp_prediction.tif' | wc -l)"
echo "   Tochigi predictions: $(find chm_outputs/google_embedding_scenario1_predictions/tochigi/ -name '*_mlp_prediction.tif' | wc -l)"
echo "   Hyogo predictions: $(find chm_outputs/google_embedding_scenario1_predictions/hyogo/ -name '*_mlp_prediction.tif' | wc -l)"

# List some example prediction files
echo "📁 Example prediction files:"
find chm_outputs/google_embedding_scenario1_predictions/ -name '*_mlp_prediction.tif' | head -5
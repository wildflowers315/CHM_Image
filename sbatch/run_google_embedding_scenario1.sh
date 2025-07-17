#!/bin/bash

#SBATCH --job-name=google_embed_s1_full
#SBATCH --output=logs/%j_google_embedding_scenario1_full.txt
#SBATCH --error=logs/%j_google_embedding_scenario1_full_error.txt
#SBATCH --time=0-4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --partition=gpu_amd

# Google Embedding Scenario 1: Complete Pipeline
# Training + Cross-Region Prediction
echo "🚀 Starting Google Embedding Scenario 1 complete pipeline at $(date)"
echo "Following the plan in docs/google_embedding_training_plan.md"

# Create output directories
mkdir -p chm_outputs/google_embedding_scenario1
mkdir -p chm_outputs/google_embedding_scenario1_predictions

# Activate environment
source chm_env/bin/activate

# Step 1: Training
echo "==================== STEP 1: TRAINING ===================="
echo "📚 Training Google Embedding Scenario 1: Reference-Only"
echo "Using existing train_production_mlp.py with --band-selection embedding"

  python train_production_mlp.py \
    --patch-dir chm_outputs/ \
    --patch-pattern "*05LE4*embedding*" \
    --supervision-mode reference \
    --band-selection embedding \
    --reference-tif downloads/dchm_05LE4.tif \
    --output-dir chm_outputs/google_embedding_scenario1/ \
    --epochs 60 \
    --batch-size 512 \
    --learning-rate 0.001 \
    --max-samples 1000 \
    --augment-factor 3
    
# python train_production_mlp.py \
#   --patch-dir chm_outputs/ \
#   --patch-pattern "*05LE4*embedding*" \
#   --supervision-mode reference \
#   --band-selection embedding \
#   --reference-tif downloads/dchm_05LE4.tif \
#   --output-dir chm_outputs/google_embedding_scenario1/ \
#   --epochs 100 \
#   --batch-size 2048 \
#   --learning-rate 0.001 \
#   --max-samples 100000 \
#   --augment-factor 3

# Check training success
MODEL_PATH="chm_outputs/production_mlp_reference_embedding_best.pth"
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Training failed: Model file not found: $MODEL_PATH"
    exit 1
fi

echo "✅ Training completed successfully"
echo "💾 Model saved as: $MODEL_PATH"
ls -lh "$MODEL_PATH"

# Step 2: Cross-Region Prediction
echo "==================== STEP 2: CROSS-REGION PREDICTION ===================="
echo "🌍 Running cross-region predictions for all three regions"

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

# Predict for Hyogo region (dchm_05LE4) - training region for comparison
echo "🌍 Predicting Hyogo region (dchm_05LE4) - training region..."
python predict_mlp_cross_region.py \
    --model-path "$MODEL_PATH" \
    --patch-dir chm_outputs/ \
    --patch-pattern "*05LE4*embedding*" \
    --output-dir chm_outputs/google_embedding_scenario1_predictions/hyogo/ \
    --device auto

# Summary
echo "==================== SUMMARY ===================="
echo "🎉 Google Embedding Scenario 1 complete pipeline finished at $(date)"

# Count prediction files
echo "📊 Prediction Summary:"
KOCHI_COUNT=$(find chm_outputs/google_embedding_scenario1_predictions/kochi/ -name '*_mlp_prediction.tif' 2>/dev/null | wc -l)
TOCHIGI_COUNT=$(find chm_outputs/google_embedding_scenario1_predictions/tochigi/ -name '*_mlp_prediction.tif' 2>/dev/null | wc -l)
HYOGO_COUNT=$(find chm_outputs/google_embedding_scenario1_predictions/hyogo/ -name '*_mlp_prediction.tif' 2>/dev/null | wc -l)

echo "   Kochi predictions: $KOCHI_COUNT"
echo "   Tochigi predictions: $TOCHIGI_COUNT"
echo "   Hyogo predictions: $HYOGO_COUNT"
echo "   Total predictions: $((KOCHI_COUNT + TOCHIGI_COUNT + HYOGO_COUNT))"

# List example prediction files
echo "📁 Example prediction files:"
find chm_outputs/google_embedding_scenario1_predictions/ -name '*_mlp_prediction.tif' 2>/dev/null | head -5

echo "📂 Output directories:"
echo "   Training results: chm_outputs/google_embedding_scenario1/"
echo "   Predictions: chm_outputs/google_embedding_scenario1_predictions/"

echo "✅ Google Embedding Scenario 1 complete pipeline SUCCESS!"
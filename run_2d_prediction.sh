#!/bin/bash

#SBATCH --job-name=2d_prediction
#SBATCH --output=logs/2d_prediction_%j.txt
#SBATCH --error=logs/2d_prediction_error_%j.txt
#SBATCH --time=0-2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G

# Create output directories if they don't exist
mkdir -p logs
mkdir -p predictions_2d

echo "🔮 Starting 2D Model Prediction Pipeline"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

# Activate environment
source chm_env/bin/activate

echo "📦 Environment activated"

# Check if models exist
if [ ! -f "results_2d/rf/multi_patch_rf_model.pkl" ]; then
    echo "❌ RF model not found. Please run training first."
    exit 1
fi

# Generate predictions for Random Forest
echo "🌲 Generating RF predictions..."
python train_predict_map.py \
    --patch-dir "chm_outputs/" \
    --model rf \
    --mode predict \
    --model-path "results_2d/rf/multi_patch_rf_model.pkl" \
    --output-dir "predictions_2d/rf" \
    --patch-pattern "*_bandNum31_*.tif" \
    --verbose

echo "📊 RF predictions completed"

# Generate predictions for MLP (if model exists)
if [ -f "results_2d/mlp/multi_patch_mlp_model.pkl" ]; then
    echo "🧠 Generating MLP predictions..."
    python train_predict_map.py \
        --patch-dir "chm_outputs/" \
        --model mlp \
        --mode predict \
        --model-path "results_2d/mlp/multi_patch_mlp_model.pkl" \
        --output-dir "predictions_2d/mlp" \
        --patch-pattern "*_bandNum31_*.tif" \
        --verbose
    echo "📊 MLP predictions completed"
fi

# Generate predictions for 2D U-Net (if model exists)
if [ -f "results_2d/2d_unet/multi_patch_2d_unet_model.pth" ]; then
    echo "🏗️ Generating 2D U-Net predictions..."
    python train_predict_map.py \
        --patch-dir "chm_outputs/" \
        --model 2d_unet \
        --mode predict \
        --model-path "results_2d/2d_unet/multi_patch_2d_unet_model.pth" \
        --output-dir "predictions_2d/2d_unet" \
        --patch-pattern "*_bandNum31_*.tif" \
        --verbose
    echo "📊 2D U-Net predictions completed"
fi

echo "🎉 All predictions completed!"
echo "End time: $(date)"

# Display results summary
echo "📈 Predictions Summary:"
find predictions_2d/ -name "*.tif" | wc -l | xargs echo "Total prediction TIF files:"
find predictions_2d/ -name "*.tif" | head -10
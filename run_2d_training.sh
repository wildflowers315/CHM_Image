#!/bin/bash

#SBATCH --job-name=2d_model_training
#SBATCH --output=logs/2d_training_%j.txt
#SBATCH --error=logs/2d_training_error_%j.txt
#SBATCH --time=0-4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G

# Create output directories if they don't exist
mkdir -p logs
mkdir -p results_2d

echo "🚀 Starting 2D Model Training Pipeline"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

# Activate environment
source chm_env/bin/activate

echo "📦 Environment activated"
python -c "import numpy, pandas, sklearn, rasterio; print('✅ Key packages available')"

# Train Random Forest model
echo "🌲 Training Random Forest model..."
python train_predict_map.py \
    --patch-dir "chm_outputs/" \
    --model rf \
    --output-dir "results_2d/rf" \
    --min-gedi-samples 10 \
    --verbose \
    --patch-pattern "*_bandNum31_*.tif" \
    --generate-prediction \
    --save-model

echo "📊 Random Forest training completed"

# Train MLP model
echo "🧠 Training MLP model..."
python train_predict_map.py \
    --patch-dir "chm_outputs/" \
    --model mlp \
    --output-dir "results_2d/mlp" \
    --min-gedi-samples 10 \
    --verbose \
    --patch-pattern "*_bandNum31_*.tif" \
    --epochs 100 \
    --learning-rate 0.001 \
    --hidden-layers "128,64,32" \
    --generate-prediction \
    --save-model

echo "📊 MLP training completed"

# Train 2D U-Net model
echo "🏗️ Training 2D U-Net model..."
python train_predict_map.py \
    --patch-dir "chm_outputs/" \
    --model 2d_unet \
    --output-dir "results_2d/2d_unet" \
    --min-gedi-samples 10 \
    --verbose \
    --patch-pattern "*_bandNum31_*.tif" \
    --epochs 50 \
    --learning-rate 0.001 \
    --base-channels 32 \
    --generate-prediction \
    --save-model

echo "📊 2D U-Net training completed"

echo "🎉 All 2D model training completed!"
echo "End time: $(date)"

# Display results summary
echo "📈 Results Summary:"
echo "Random Forest results: results_2d/rf/"
echo "MLP results: results_2d/mlp/"
echo "2D U-Net results: results_2d/2d_unet/"

# List generated files
echo "📁 Generated files:"
find results_2d/ -name "*.pkl" -o -name "*.pth" -o -name "*.json" -o -name "*.tif" | head -20
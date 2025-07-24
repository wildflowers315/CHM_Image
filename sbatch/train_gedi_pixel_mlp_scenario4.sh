#!/bin/bash

#SBATCH --job-name=gedi_pixel_s4
#SBATCH --output=logs/%j_gedi_pixel_scenario4.txt
#SBATCH --error=logs/%j_gedi_pixel_scenario4_error.txt
#SBATCH --time=0-1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --partition=gpu

# Create logs directory
mkdir -p logs

# Create output directory
mkdir -p chm_outputs/gedi_pixel_mlp_scenario4

# Activate environment
source chm_env/bin/activate

# GEDI Pixel MLP Scenario 4: No Filter Training
# Train MLP using GEDI CSV pixel data with Google Embedding bands only
echo "Starting GEDI Pixel MLP Scenario 4 training at $(date)"
echo "Input: Google Embedding (A00-A63), Target: GEDI rh"

python train_gedi_pixel_mlp_scenario4.py \
  --csv-dir chm_outputs/ \
  --output-dir chm_outputs/gedi_pixel_mlp_scenario4/ \
  --scenario scenario4 \
  --band-selection embedding \
  --epochs 60 \
  --batch-size 512 \
  --learning-rate 0.001 \
  --max-samples 63000

echo "GEDI Pixel MLP Scenario 4 training completed at $(date)"
echo "Expected model saved as: chm_outputs/gedi_pixel_mlp_scenario4/gedi_pixel_mlp_scenario4_embedding_best.pth"

# Verify model was saved
if [ -f "chm_outputs/gedi_pixel_mlp_scenario4/gedi_pixel_mlp_scenario4_embedding_best.pth" ]; then
    echo "✅ GEDI Pixel MLP model file created successfully"
    ls -lh chm_outputs/gedi_pixel_mlp_scenario4/gedi_pixel_mlp_scenario4_embedding_best.pth
    
    # Show training results
    if [ -f "chm_outputs/gedi_pixel_mlp_scenario4/gedi_pixel_mlp_scenario4_embedding_results.json" ]; then
        echo "📊 Training Results:"
        python -c "
import json
with open('chm_outputs/gedi_pixel_mlp_scenario4/gedi_pixel_mlp_scenario4_embedding_results.json', 'r') as f:
    results = json.load(f)
print(f'Best Validation R²: {results[\"best_val_r2\"]:.4f}')
print(f'Total Samples: {results[\"total_samples\"]:,}')
print(f'Input Features: {results[\"input_features\"]}')
"
    fi
else
    echo "❌ GEDI Pixel MLP model file not found!"
    echo "Checking output directory contents:"
    ls -la chm_outputs/gedi_pixel_mlp_scenario4/
    exit 1
fi
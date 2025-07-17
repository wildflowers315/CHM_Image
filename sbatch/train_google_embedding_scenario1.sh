#!/bin/bash

#SBATCH --job-name=google_embed_s1
#SBATCH --output=logs/%j_google_embedding_scenario1_small.txt
#SBATCH --error=logs/%j_google_embedding_scenario1_small_error.txt
#SBATCH --time=0-2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --partition=gpu_amd

# Create output directory
mkdir -p chm_outputs/google_embedding_scenario1_small

# Activate environment
source chm_env/bin/activate

# Google Embedding Scenario 1: Reference-Only Training
# Following the plan in docs/google_embedding_training_plan.md
echo "Starting Google Embedding Scenario 1 training at $(date)"
echo "Using existing train_production_mlp.py with --band-selection embedding"

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

  python train_production_mlp.py \
    --patch-dir chm_outputs/ \
    --patch-pattern "*05LE4*embedding*" \
    --supervision-mode reference \
    --band-selection embedding \
    --reference-tif downloads/dchm_05LE4.tif \
    --output-dir chm_outputs/google_embedding_scenario1_small/ \
    --epochs 60 \
    --batch-size 512 \
    --learning-rate 0.001 \
    --max-samples 1000 \
    --augment-factor 3

echo "Google Embedding Scenario 1 training completed at $(date)"
echo "Expected model saved as: chm_outputs/production_mlp_reference_embedding_1000_best.pth"

# Verify model was saved
if [ -f "chm_outputs/production_mlp_reference_embedding_best.pth" ]; then
    echo "✅ Model file created successfully"
    ls -lh chm_outputs/production_mlp_reference_embedding_best.pth
else
    echo "❌ Model file not found!"
    exit 1
fi
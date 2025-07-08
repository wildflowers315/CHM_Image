#!/bin/bash

#SBATCH --job-name=mlp_production_gpu
#SBATCH --output=logs/%j_mlp_production_gpu.txt
#SBATCH --error=logs/%j_mlp_production_gpu_error.txt
#SBATCH --time=0-2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --constraint='nvidia&A100'

# Create output directories
mkdir -p logs chm_outputs/production_mlp_results

# Activate environment
source chm_env/bin/activate

# Production MLP training with reference height supervision and GPU acceleration
echo "🚀 Starting MLP production training with reference height supervision (GPU)"
echo "📅 Start time: $(date)"
echo "💻 Job ID: $SLURM_JOB_ID"
echo "🖥️  Node: $(hostname)"
echo "🎮 GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"

python train_production_mlp.py \
  --patch-dir chm_outputs/ \
  --reference-tif downloads/dchm_05LE4.tif \
  --output-dir chm_outputs/production_mlp_results \
  --epochs 60 \
  --batch-size 512 \
  --learning-rate 0.001 \
  --max-samples 1000 \
  --augment-factor 3

echo "✅ MLP training completed at: $(date)"
echo "📊 Results saved in: chm_outputs/production_mlp_results"
echo "🎯 Job finished with exit code: $?"
#!/bin/bash

#SBATCH --job-name=s3_mlp_ensemble
#SBATCH --output=logs/%j_s3_mlp_ensemble.txt
#SBATCH --error=logs/%j_s3_mlp_ensemble_error.txt
#SBATCH --time=0-2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --partition=gpu

echo "🚀 Starting Scenario 3 MLP Ensemble Training"
echo "📅 Start time: $(date)"

# Activate environment
source chm_env/bin/activate

# Create output directory
mkdir -p chm_outputs/scenario3_tochigi_mlp_ensemble

# Train ensemble with adapted GEDI MLP + Reference MLP
echo "🔧 Training ensemble with adapted GEDI MLP + Reference MLP"
python train_ensemble_mlp.py \
  --gedi-model-path chm_outputs/scenario3_tochigi_mlp_adaptation/fine_tuned_gedi_mlp_best.pth \
  --reference-model-path chm_outputs/production_mlp_best.pth \
  --patch-dir chm_outputs/enhanced_patches/ \
  --include-pattern "*09gd4*" \
  --reference-height-path downloads/dchm_09gd4.tif \
  --output-dir chm_outputs/scenario3_tochigi_mlp_ensemble \
  --epochs 50 \
  --learning-rate 0.001 \
  --batch-size 1024

echo "✅ Scenario 3 MLP ensemble training completed"
echo "📊 Results saved to: chm_outputs/scenario3_tochigi_mlp_ensemble/"
echo "📅 End time: $(date)"
echo "🎯 Job finished with exit code: $?"
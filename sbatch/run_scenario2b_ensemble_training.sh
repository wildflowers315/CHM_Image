#!/bin/bash

#SBATCH --job-name=s2b_ensemble_training
#SBATCH --output=logs/%j_s2b_ensemble_training.txt
#SBATCH --error=logs/%j_s2b_ensemble_training_error.txt
#SBATCH --time=0-4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --partition=gpu

# Activate environment
source chm_env/bin/activate

# Run the training
python train_ensemble_mlp.py \
  --gedi-model-path chm_outputs/production_mlp_gedi_only_best.pth \
  --reference-model-path chm_outputs/production_mlp_best.pth \
  --patch-dir chm_outputs/enhanced_patches/ \
  --patch-pattern "*05LE4*" \
  --reference-height-path downloads/dchm_05LE4.tif \
  --output-dir chm_outputs/scenario2b_dual_mlp_ensemble \
  --epochs 100 \
  --learning-rate 0.001 \
  --gedi-model-type mlp


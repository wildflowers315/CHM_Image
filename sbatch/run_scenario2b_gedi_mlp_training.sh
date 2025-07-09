#!/bin/bash

#SBATCH --job-name=s2b_gedi_mlp
#SBATCH --output=logs/%j_s2b_gedi_mlp.txt
#SBATCH --error=logs/%j_s2b_gedi_mlp_error.txt
#SBATCH --time=0-4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --partition=gpu

# Activate environment
source chm_env/bin/activate

# Run the training
python train_production_mlp.py \
  --patch-dir chm_outputs/enhanced_patches/ \
  --reference-tif downloads/dchm_05LE4.tif \
  --output-dir chm_outputs/scenario2b_gedi_mlp \
  --supervision-mode gedi_only \
  --epochs 100 \
  --learning-rate 0.001 \
  --batch-size 32


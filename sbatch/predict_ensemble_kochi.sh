#!/bin/bash

#SBATCH --job-name=s2b_predict_kochi
#SBATCH --output=logs/%j_s2b_predict_kochi.txt
#SBATCH --error=logs/%j_s2b_predict_kochi_error.txt
#SBATCH --time=0-2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --partition=gpu

# Activate environment
source chm_env/bin/activate

# Run the prediction for Kochi
python predict_ensemble.py \
  --gedi-model chm_outputs/production_mlp_gedi_only_best.pth \
  --mlp-model chm_outputs/production_mlp_best.pth \
  --ensemble-model chm_outputs/scenario2b_dual_mlp_ensemble/ensemble_mlp_best.pth \
  --patch-dir chm_outputs/enhanced_patches/ \
  --output-dir chm_outputs/scenario2b_kochi_predictions \
  --region kochi \
  --gedi-model-type mlp


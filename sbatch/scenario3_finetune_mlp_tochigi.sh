#!/bin/bash

#SBATCH --job-name=s3_mlp_ft
#SBATCH --output=logs/%j_s3_mlp_ft.txt
#SBATCH --error=logs/%j_s3_mlp_ft_error.txt
#SBATCH --time=0-2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --partition=gpu

echo "🚀 Starting Scenario 3 MLP Fine-tuning for Tochigi"
echo "📅 Start time: $(date)"

# Activate environment
source chm_env/bin/activate

# Create output directory
mkdir -p chm_outputs/scenario3_tochigi_mlp_adaptation

# Fine-tune GEDI MLP on Tochigi GEDI pixels
echo "🔧 Fine-tuning GEDI MLP on Tochigi GEDI pixels"
python train_production_mlp_finetune.py \
  --patch-dir chm_outputs/enhanced_patches/ \
  --reference-tif downloads/dchm_09gd4.tif \
  --supervision-mode gedi_only \
  --pretrained-model-path chm_outputs/production_mlp_gedi_only_best.pth \
  --output-dir chm_outputs/scenario3_tochigi_mlp_adaptation \
  --epochs 50 \
  --learning-rate 0.0001 \
  --batch-size 1024 \
  --max-samples 50000

echo "✅ Scenario 3 MLP fine-tuning completed"
echo "📊 Results saved to: chm_outputs/scenario3_tochigi_mlp_adaptation/"
echo "📅 End time: $(date)"
echo "🎯 Job finished with exit code: $?"
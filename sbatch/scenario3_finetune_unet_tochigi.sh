#!/bin/bash

#SBATCH --job-name=s3_unet_ft
#SBATCH --output=logs/%j_s3_unet_ft.txt
#SBATCH --error=logs/%j_s3_unet_ft_error.txt
#SBATCH --time=0-4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --partition=gpu

echo "🚀 Starting Scenario 3 U-Net Fine-tuning for Tochigi"
echo "📅 Start time: $(date)"

# Activate environment
source chm_env/bin/activate

# Create output directory
mkdir -p chm_outputs/scenario3_tochigi_unet_adaptation

# Fine-tune shift-aware U-Net on Tochigi GEDI data
echo "🔧 Fine-tuning shift-aware U-Net on Tochigi GEDI data"
python train_shift_aware_unet_finetune.py \
  --patch-dir chm_outputs/enhanced_patches/ \
  --pretrained-model-path chm_outputs/scenario2_gedi_shift_aware/shift_aware_unet_r2.pth \
  --output-dir chm_outputs/scenario3_tochigi_unet_adaptation \
  --epochs 30 \
  --learning-rate 0.00005 \
  --batch-size 32 \
  --shift-radius 2 \
  --max-samples 5000

echo "✅ Scenario 3 U-Net fine-tuning completed"
echo "📊 Results saved to: chm_outputs/scenario3_tochigi_unet_adaptation/"
echo "📅 End time: $(date)"
echo "🎯 Job finished with exit code: $?"
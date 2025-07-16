#!/bin/bash

#SBATCH --job-name=s3_direct_ensemble
#SBATCH --output=logs/%j_s3_direct_ensemble.txt
#SBATCH --error=logs/%j_s3_direct_ensemble_error.txt
#SBATCH --time=0-2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --partition=gpu

echo "🚀 Starting Scenario 3 Direct Ensemble Training"
echo "📅 Start time: $(date)"

# Activate environment
source chm_env/bin/activate

# Create output directory
mkdir -p chm_outputs/scenario3_tochigi_ensemble_direct

# Run direct ensemble training
echo "🔧 Training adaptive ensemble for Tochigi region"
python train_scenario3_ensemble_direct.py

echo "✅ Scenario 3 direct ensemble training completed"
echo "📊 Results saved to: chm_outputs/scenario3_tochigi_ensemble_direct/"
echo "📅 End time: $(date)"
echo "🎯 Job finished with exit code: $?"
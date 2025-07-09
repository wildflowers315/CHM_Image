#!/bin/bash

#SBATCH --job-name=ensemble_pred
#SBATCH --output=chm_outputs/logs/%j_ensemble_pred.txt
#SBATCH --error=chm_outputs/logs/%j_ensemble_pred_error.txt
#SBATCH --time=0-3:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --partition=gpu

echo "🚀 Starting Ensemble Cross-Region Prediction - Scenario 2"
echo "📅 Start time: $(date)"
echo "🖥️  Node: $(hostname)"
echo "🔧 GPU: $CUDA_VISIBLE_DEVICES"

# Create log directory
mkdir -p chm_outputs/logs

# Activate environment
source chm_env/bin/activate

echo "🔮 Running ensemble predictions on Kochi and Tochigi..."

# Run ensemble prediction on both regions
python predict_ensemble.py \
    --ensemble-model chm_outputs/scenario2_ensemble/ensemble_mlp_best.pth \
    --gedi-model chm_outputs/scenario2_gedi_shift_aware/shift_aware_unet_r2.pth \
    --mlp-model chm_outputs/production_mlp_best.pth \
    --region both \
    --patch-dir chm_outputs/ \
    --output-dir chm_outputs/scenario2_cross_region_predictions \
    --device cuda

echo "✅ Ensemble prediction completed at $(date)"

# Check results
echo "📊 Results summary:"
for region in kochi tochigi; do
    if [ -d "chm_outputs/scenario2_cross_region_predictions/$region" ]; then
        pred_count=$(ls chm_outputs/scenario2_cross_region_predictions/$region/*.tif 2>/dev/null | wc -l)
        echo "  $region: $pred_count prediction files"
        
        if [ -f "chm_outputs/scenario2_cross_region_predictions/$region/${region}_prediction_info.json" ]; then
            echo "  Info file exists for $region"
        fi
    else
        echo "  $region: No prediction directory found"
    fi
done

echo "🎯 Next step: Evaluate predictions against reference data"
echo "   - Kochi: downloads/dchm_04hf3.tif"
echo "   - Tochigi: downloads/dchm_09gd4.tif"
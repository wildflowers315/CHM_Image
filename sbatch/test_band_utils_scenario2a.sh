#!/bin/bash

#SBATCH --job-name=test_band_utils_scenario2a
#SBATCH --output=chm_outputs/logs/%j_test_band_utils_scenario2a.txt
#SBATCH --error=chm_outputs/logs/%j_test_band_utils_scenario2a_error.txt
#SBATCH --time=0-2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G

echo "🚀 Testing Scenario 2A with Band Utilities"
echo "📅 Start time: $(date)"
echo "💻 Job ID: $SLURM_JOB_ID"
echo "🖥️  Node: $(hostname)"

# Create output directories
mkdir -p chm_outputs/logs chm_outputs/band_utils_test_scenario2a

# Activate environment
source chm_env/bin/activate

echo "🔧 Testing updated Scenario 2A ensemble with robust band utilities..."

# Check if required models exist
if [ ! -f "chm_outputs/scenario2_ensemble/ensemble_mlp_best.pth" ]; then
    echo "❌ Ensemble model not found: chm_outputs/scenario2_ensemble/ensemble_mlp_best.pth"
    echo "   Please run ensemble training first or use existing trained models"
    exit 1
fi

if [ ! -f "chm_outputs/scenario2_gedi_shift_aware/shift_aware_unet_r2.pth" ]; then
    echo "❌ GEDI U-Net model not found: chm_outputs/scenario2_gedi_shift_aware/shift_aware_unet_r2.pth"
    echo "   Please run GEDI U-Net training first or use existing trained models"
    exit 1
fi

echo "✅ Required models found, proceeding with ensemble predictions..."

# Test Scenario 2A ensemble predictions on both regions with band utilities
echo "📍 Step 1: Ensemble prediction on Kochi (04hf3) with band utilities"
python predict_ensemble.py \
  --ensemble-model chm_outputs/scenario2_ensemble/ensemble_mlp_best.pth \
  --gedi-model chm_outputs/scenario2_gedi_shift_aware/shift_aware_unet_r2.pth \
  --mlp-model chm_outputs/production_mlp_best.pth \
  --region kochi \
  --patch-dir chm_outputs/enhanced_patches \
  --output-dir chm_outputs/band_utils_test_scenario2a

echo "📍 Step 2: Ensemble prediction on Tochigi (09gd4) with band utilities"
python predict_ensemble.py \
  --ensemble-model chm_outputs/scenario2_ensemble/ensemble_mlp_best.pth \
  --gedi-model chm_outputs/scenario2_gedi_shift_aware/shift_aware_unet_r2.pth \
  --mlp-model chm_outputs/production_mlp_best.pth \
  --region tochigi \
  --patch-dir chm_outputs/enhanced_patches \
  --output-dir chm_outputs/band_utils_test_scenario2a

echo "✅ Scenario 2A ensemble predictions completed"

# Run evaluations using existing CRS-aware evaluation script
echo "📊 Step 3: Evaluating with CRS-aware evaluation"

# Evaluate Kochi
if [ -f "downloads/dchm_04hf3.tif" ]; then
    echo "🔍 Evaluating Kochi region"
    python evaluate_with_crs_transform.py \
        --pred-dir chm_outputs/band_utils_test_scenario2a/kochi \
        --ref-tif downloads/dchm_04hf3.tif \
        --region-name kochi \
        --output-dir chm_outputs/band_utils_test_scenario2a/evaluation
fi

# Evaluate Tochigi
if [ -f "downloads/dchm_09gd4.tif" ]; then
    echo "🔍 Evaluating Tochigi region"
    python evaluate_with_crs_transform.py \
        --pred-dir chm_outputs/band_utils_test_scenario2a/tochigi \
        --ref-tif downloads/dchm_09gd4.tif \
        --region-name tochigi \
        --output-dir chm_outputs/band_utils_test_scenario2a/evaluation
fi

echo "📋 Step 4: Generating comparison summary"
python -c "
import json
import glob
import os

print('\\n🎯 BAND UTILITIES TEST - SCENARIO 2A EVALUATION SUMMARY')
print('=' * 60)

eval_files = glob.glob('chm_outputs/band_utils_test_scenario2a/evaluation/*_crs_evaluation.json')
all_results = []

for eval_file in eval_files:
    try:
        with open(eval_file, 'r') as f:
            result = json.load(f)
            all_results.append(result)
    except Exception as e:
        print(f'Error loading {eval_file}: {e}')

if all_results:
    print(f'\\n📊 Results for {len(all_results)} regions:')
    print()
    print('Region      | R² Score | RMSE (m) | MAE (m) | Pixels      | Pred Mean | Ref Mean')
    print('-' * 80)
    
    # Compare with previous Scenario 2A results
    previous_results = {
        'kochi': {'r2': -8.58, 'rmse': 13.37, 'note': 'Previous ensemble result'},
        'tochigi': {'r2': -7.95, 'rmse': 16.56, 'note': 'Previous ensemble result'}
    }
    
    for result in all_results:
        region = result['region']
        r2 = result['r2_score']
        rmse = result['rmse']
        mae = result['mae']
        pixels = result['n_pixels']
        pred_mean = result['pred_mean']
        ref_mean = result['ref_mean']
        
        print(f'{region:10} | {r2:8.4f} | {rmse:8.2f} | {mae:7.2f} | {pixels:11,} | {pred_mean:9.1f} | {ref_mean:8.1f}')
        
        # Compare with previous
        if region in previous_results:
            prev = previous_results[region]
            r2_change = r2 - prev['r2']
            rmse_change = rmse - prev['rmse']
            prev_note = prev['note']
            print(f'           | Change: {r2_change:+.4f} | Change: {rmse_change:+.2f} | {prev_note}')
    
    print()
    print('🔍 ANALYSIS:')
    print('- Previous Scenario 2A: Failed with R² around -8 (200x worse than Scenario 1)')
    print('- If results are identical: Band utilities working correctly (no indexing issues)')
    print('- If results improved: Band utilities may have fixed critical indexing bugs')
    print('- Scenario 2A fundamentally failed due to architecture mismatch, not indexing')
    
    # Save results
    summary = {
        'test_type': 'band_utils_scenario2a',
        'timestamp': os.popen('date').read().strip(),
        'results': all_results,
        'previous_results': previous_results
    }
    
    with open('chm_outputs/band_utils_test_scenario2a/evaluation/band_utils_test_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print('💾 Test summary saved to: chm_outputs/band_utils_test_scenario2a/evaluation/band_utils_test_summary.json')
else:
    print('❌ No evaluation results found')
"

echo "✅ Band utilities test completed at: $(date)"
echo "📊 Results directory: chm_outputs/band_utils_test_scenario2a/"
echo "🎯 Job finished with exit code: $?"
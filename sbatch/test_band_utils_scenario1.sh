#!/bin/bash

#SBATCH --job-name=test_band_utils_scenario1
#SBATCH --output=chm_outputs/logs/%j_test_band_utils_scenario1.txt
#SBATCH --error=chm_outputs/logs/%j_test_band_utils_scenario1_error.txt
#SBATCH --time=0-2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G

echo "🚀 Testing Scenario 1 with Band Utilities"
echo "📅 Start time: $(date)"
echo "💻 Job ID: $SLURM_JOB_ID"
echo "🖥️  Node: $(hostname)"

# Create output directories
mkdir -p chm_outputs/logs chm_outputs/band_utils_test_scenario1

# Activate environment
source chm_env/bin/activate

echo "🔧 Testing updated Scenario 1 with robust band utilities..."

# Test Scenario 1 predictions on both regions with band utilities
echo "📍 Step 1: Predicting Kochi (04hf3) with band utilities"
python predict_mlp_cross_region.py \
  --model-path chm_outputs/production_mlp_best.pth \
  --patch-dir chm_outputs/enhanced_patches \
  --output-dir chm_outputs/band_utils_test_scenario1/04hf3_kochi \
  --patch-pattern "*04hf3*"

echo "📍 Step 2: Predicting Tochigi (09gd4) with band utilities"
python predict_mlp_cross_region.py \
  --model-path chm_outputs/production_mlp_best.pth \
  --patch-dir chm_outputs/enhanced_patches \
  --output-dir chm_outputs/band_utils_test_scenario1/09gd4_tochigi \
  --patch-pattern "*09gd4*"

echo "📍 Step 3: Predicting Hyogo (05LE4) validation with band utilities"
python predict_mlp_cross_region.py \
  --model-path chm_outputs/production_mlp_best.pth \
  --patch-dir chm_outputs/enhanced_patches \
  --output-dir chm_outputs/band_utils_test_scenario1/05LE4_hyogo \
  --patch-pattern "*05LE4*"

echo "✅ Scenario 1 predictions completed"

# Run evaluations using existing CRS-aware evaluation script
echo "📊 Step 4: Evaluating with CRS-aware evaluation"

# Evaluate Kochi
if [ -f "downloads/dchm_04hf3.tif" ]; then
    echo "🔍 Evaluating Kochi region"
    python evaluate_with_crs_transform.py \
        --pred-dir chm_outputs/band_utils_test_scenario1/04hf3_kochi \
        --ref-tif downloads/dchm_04hf3.tif \
        --region-name kochi \
        --output-dir chm_outputs/band_utils_test_scenario1/evaluation
fi

# Evaluate Tochigi
if [ -f "downloads/dchm_09gd4.tif" ]; then
    echo "🔍 Evaluating Tochigi region"
    python evaluate_with_crs_transform.py \
        --pred-dir chm_outputs/band_utils_test_scenario1/09gd4_tochigi \
        --ref-tif downloads/dchm_09gd4.tif \
        --region-name tochigi \
        --output-dir chm_outputs/band_utils_test_scenario1/evaluation
fi

# Evaluate Hyogo (training region)
if [ -f "downloads/dchm_05LE4.tif" ]; then
    echo "🔍 Evaluating Hyogo region (training validation)"
    python evaluate_with_crs_transform.py \
        --pred-dir chm_outputs/band_utils_test_scenario1/05LE4_hyogo \
        --ref-tif downloads/dchm_05LE4.tif \
        --region-name hyogo \
        --output-dir chm_outputs/band_utils_test_scenario1/evaluation
fi

echo "📋 Step 5: Generating comparison summary"
python -c "
import json
import glob
import os

print('\\n🎯 BAND UTILITIES TEST - SCENARIO 1 EVALUATION SUMMARY')
print('=' * 60)

eval_files = glob.glob('chm_outputs/band_utils_test_scenario1/evaluation/*_crs_evaluation.json')
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
    
    # Compare with previous results
    previous_results = {
        'kochi': {'r2': -52.13, 'rmse': 41.4, 'note': 'Before bias correction'},
        'tochigi': {'r2': -67.94, 'rmse': 61.7, 'note': 'Before bias correction'},
        'hyogo': {'r2': 0.5026, 'rmse': 16.9, 'note': 'Training region'}
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
    print('- If results are identical to previous: Band utilities working correctly')
    print('- If results changed: Band utilities may have fixed indexing issues')
    print('- Cross-region bias correction may still be needed')
    
    # Save results
    summary = {
        'test_type': 'band_utils_scenario1',
        'timestamp': os.popen('date').read().strip(),
        'results': all_results
    }
    
    with open('chm_outputs/band_utils_test_scenario1/evaluation/band_utils_test_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print('💾 Test summary saved to: chm_outputs/band_utils_test_scenario1/evaluation/band_utils_test_summary.json')
else:
    print('❌ No evaluation results found')
"

echo "✅ Band utilities test completed at: $(date)"
echo "📊 Results directory: chm_outputs/band_utils_test_scenario1/"
echo "🎯 Job finished with exit code: $?"
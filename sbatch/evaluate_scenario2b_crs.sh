#!/bin/bash

#SBATCH --job-name=s2b_crs_eval
#SBATCH --output=logs/%j_s2b_crs_eval.txt
#SBATCH --error=logs/%j_s2b_crs_eval_error.txt
#SBATCH --time=0-1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G

# Create output directories
mkdir -p chm_outputs/scenario2b_crs_evaluation

# Activate environment
source chm_env/bin/activate

echo "🚀 Starting Scenario 2B CRS-aware evaluation"
echo "📅 Start time: $(date)"

# Evaluate Kochi region
echo "🔍 Evaluating Kochi region"
python evaluate_with_crs_transform.py \
  --pred-dir chm_outputs/scenario2b_kochi_predictions/kochi \
  --ref-tif downloads/dchm_04hf3.tif \
  --region-name kochi \
  --output-dir chm_outputs/scenario2b_crs_evaluation

# Evaluate Tochigi region
echo "🔍 Evaluating Tochigi region"
python evaluate_with_crs_transform.py \
  --pred-dir chm_outputs/scenario2b_tochigi_predictions/tochigi \
  --ref-tif downloads/dchm_09gd4.tif \
  --region-name tochigi \
  --output-dir chm_outputs/scenario2b_crs_evaluation

echo "✅ Scenario 2B evaluation completed"
echo "📊 Results saved to: chm_outputs/scenario2b_crs_evaluation/"
echo "📅 End time: $(date)"

# Generate comparison summary
echo "📋 Generating comparison summary"
python -c "
import json
import glob
import os

print('\\n🎯 SCENARIO 2B EVALUATION SUMMARY')
print('=' * 50)

eval_files = glob.glob('chm_outputs/scenario2b_crs_evaluation/*_crs_evaluation.json')
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
    
    # Compare with Scenario 1 baseline
    scenario1_results = {
        'kochi': {'r2': -89.09, 'rmse': 38.26, 'note': 'Scenario 1 baseline'},
        'tochigi': {'r2': -99.95, 'rmse': 45.97, 'note': 'Scenario 1 baseline'}
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
        
        # Compare with Scenario 1
        if region in scenario1_results:
            s1 = scenario1_results[region]
            r2_improvement = r2 - s1['r2']
            rmse_improvement = s1['rmse'] - rmse
            print(f'           | vs S1: {r2_improvement:+.2f} | vs S1: {rmse_improvement:+.2f} | {s1[\"note\"]}')
    
    print()
    print('🔍 SCENARIO 2B ANALYSIS:')
    print('- Pixel-level GEDI MLP + Reference MLP ensemble approach')
    print('- Compare with Scenario 1 baseline performance')
    print('- Expected improvement over Scenario 2A (failed spatial U-Net)')
    
    # Save summary
    summary = {
        'scenario': '2B',
        'approach': 'Dual-MLP Ensemble (GEDI MLP + Reference MLP)',
        'results': all_results,
        'comparison_baseline': 'Scenario 1 (Reference-only MLP)',
        'evaluation_date': os.popen('date').read().strip()
    }
    
    with open('chm_outputs/scenario2b_crs_evaluation/scenario2b_evaluation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print('💾 Summary saved to: chm_outputs/scenario2b_crs_evaluation/scenario2b_evaluation_summary.json')
else:
    print('❌ No evaluation results found')
"

echo "🎯 Job finished with exit code: $?"
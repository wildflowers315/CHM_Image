#!/bin/bash

#SBATCH --job-name=mlp_bias_test
#SBATCH --output=logs/%j_bias_correction_test.txt
#SBATCH --error=logs/%j_bias_correction_test_error.txt
#SBATCH --time=0-0:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G

# Create output directories
mkdir -p logs chm_outputs/bias_correction_test

# Activate environment
source chm_env/bin/activate

echo "🔧 Testing bias correction hypothesis for MLP cross-region predictions"
echo "📅 Start time: $(date)"
echo "💻 Job ID: $SLURM_JOB_ID"
echo "🖥️  Node: $(hostname)"

# Test multiple correction factors
echo "📊 Testing correction factors: 2.0x, 2.5x, 3.0x"

echo "🔍 Testing Kochi (04hf3) with 2.5x correction"
python evaluate_with_bias_correction.py \
  --pred-dir chm_outputs/cross_region_predictions/04hf3_kochi \
  --ref-tif downloads/dchm_04hf3.tif \
  --region-name 04hf3_kochi \
  --output-dir chm_outputs/bias_correction_test \
  --correction-factor 2.5

echo "🔍 Testing Tochigi (09gd4) with 3.7x correction"
python evaluate_with_bias_correction.py \
  --pred-dir chm_outputs/cross_region_predictions/09gd4_tochigi \
  --ref-tif downloads/dchm_09gd4.tif \
  --region-name 09gd4_tochigi \
  --output-dir chm_outputs/bias_correction_test \
  --correction-factor 3.7

echo "🔍 Testing Tochigi (09gd4) with 2.5x correction for comparison"
python evaluate_with_bias_correction.py \
  --pred-dir chm_outputs/cross_region_predictions/09gd4_tochigi \
  --ref-tif downloads/dchm_09gd4.tif \
  --region-name 09gd4_tochigi_2p5x \
  --output-dir chm_outputs/bias_correction_test \
  --correction-factor 2.5

# Generate comprehensive bias correction summary
echo "📋 Generating bias correction summary"
python -c "
import json
import glob
import os
import numpy as np

print('\\n🎯 BIAS CORRECTION TEST RESULTS')
print('=' * 80)

# Load all bias correction results
eval_files = glob.glob('chm_outputs/bias_correction_test/*_bias_corrected_evaluation.json')
results = []

for eval_file in eval_files:
    try:
        with open(eval_file, 'r') as f:
            result = json.load(f)
            results.append(result)
    except Exception as e:
        print(f'Error loading {eval_file}: {e}')

if results:
    print(f'\\n📊 Bias Correction Results ({len(results)} tests):')
    print()
    print('Region         | Factor | Original R² | Corrected R² | Improvement | RMSE Reduction')
    print('-' * 85)
    
    for result in results:
        region = result['region']
        factor = result['correction_factor']
        r2_orig = result['original']['r2_score']
        r2_corr = result['corrected']['r2_score']
        r2_imp = result['improvement']['r2_improvement']
        rmse_imp = result['improvement']['rmse_improvement']
        
        print(f'{region:14} | {factor:5.1f}x | {r2_orig:10.4f} | {r2_corr:11.4f} | {r2_imp:10.4f} | {rmse_imp:12.2f}m')
    
    # Find best results
    best_r2 = max(r['corrected']['r2_score'] for r in results)
    best_result = next(r for r in results if r['corrected']['r2_score'] == best_r2)
    
    print()
    print(f'🏆 BEST RESULT:')
    print(f'   Region: {best_result[\"region\"]}')
    print(f'   Correction Factor: {best_result[\"correction_factor\"]:.1f}x')
    print(f'   Corrected R²: {best_result[\"corrected\"][\"r2_score\"]:.4f}')
    print(f'   R² Improvement: {best_result[\"improvement\"][\"r2_improvement\"]:+.4f}')
    print(f'   RMSE Reduction: {best_result[\"improvement\"][\"rmse_improvement\"]:+.2f}m')
    
    # Summary statistics
    all_improvements = [r['improvement']['r2_improvement'] for r in results]
    avg_improvement = np.mean(all_improvements)
    
    print(f'\\n📈 SUMMARY:')
    print(f'   Average R² Improvement: {avg_improvement:+.4f}')
    print(f'   Best R² Achievement: {best_r2:.4f}')
    
    if avg_improvement > 50:
        print('🎉 MASSIVE IMPROVEMENT! Bias correction highly effective')
        print('💡 Recommendation: Apply systematic bias correction')
    elif avg_improvement > 10:
        print('✅ SIGNIFICANT IMPROVEMENT! Bias correction effective')
        print('💡 Recommendation: Implement bias correction with optimal factors')
    elif avg_improvement > 1:
        print('⚠️  MODERATE IMPROVEMENT. Consider additional calibration')
    else:
        print('❌ LIMITED IMPROVEMENT. Investigate other bias sources')
    
    print(f'\\n🔧 NEXT STEPS:')
    if best_r2 > 0.4:
        print('1. ✅ Bias correction confirmed effective')
        print('2. 🔧 Implement region-specific correction factors')
        print('3. 📊 Re-evaluate all cross-region predictions')
        print('4. 🎯 Document systematic bias in model pipeline')
    else:
        print('1. 🔍 Investigate additional bias sources')
        print('2. 📊 Check training data unit consistency')
        print('3. 🔧 Consider model recalibration')
        print('4. 🎯 Implement multi-region training')
    
else:
    print('❌ No bias correction results found')
"

echo "✅ Bias correction testing completed at: $(date)"
echo "📊 Results: chm_outputs/bias_correction_test/"
echo "🎯 Job finished with exit code: $?"
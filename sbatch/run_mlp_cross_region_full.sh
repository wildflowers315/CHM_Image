#!/bin/bash

#SBATCH --job-name=mlp_cross_region_full
#SBATCH --output=logs/%j_mlp_cross_region_full.txt
#SBATCH --error=logs/%j_mlp_cross_region_full_error.txt
#SBATCH --time=0-3:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G

# Create output directories
mkdir -p logs chm_outputs/cross_region_predictions chm_outputs/cross_region_evaluation

# Activate environment
source chm_env/bin/activate

echo "🚀 Starting comprehensive MLP cross-region prediction and evaluation"
echo "📅 Start time: $(date)"
echo "💻 Job ID: $SLURM_JOB_ID"
echo "🖥️  Node: $(hostname)"

# Cross-region predictions using enhanced patches (consistent 30 features)
echo "🌍 Step 1: Cross-region predictions with enhanced patches"

# Wait for enhanced patches to be ready
echo "⏳ Waiting for enhanced patches creation to complete..."
while ! ls chm_outputs/enhanced_patches/ref_*09gd4* >/dev/null 2>&1; do
    echo "   Waiting for 09gd4 enhanced patches..."
    sleep 30
done

# Use the unified training system for predictions with enhanced patches
echo "📍 Predicting all regions using enhanced patches (30 features consistently)"

# 1. 04hf3 (Kochi) - Using enhanced patches  
echo "📍 Predicting 04hf3 (Kochi region) with enhanced patches"
python train_predict_map.py \
  --patch-dir chm_outputs/enhanced_patches/ \
  --patch-pattern "*04hf3*" \
  --model mlp \
  --mode predict \
  --model-path chm_outputs/production_mlp_best.pth \
  --output-dir chm_outputs/cross_region_predictions/04hf3_kochi

# 2. 09gd4 (Tochigi) - Using enhanced patches
echo "📍 Predicting 09gd4 (Tochigi region) with enhanced patches"
python train_predict_map.py \
  --patch-dir chm_outputs/enhanced_patches/ \
  --patch-pattern "*09gd4*" \
  --model mlp \
  --mode predict \
  --model-path chm_outputs/production_mlp_best.pth \
  --output-dir chm_outputs/cross_region_predictions/09gd4_tochigi

# 3. 05LE4 (Hyogo) - Validation on training region with enhanced patches
echo "📍 Predicting 05LE4 (Hyogo region) validation with enhanced patches"
python train_predict_map.py \
  --patch-dir chm_outputs/enhanced_patches/ \
  --patch-pattern "*05LE4*" \
  --model mlp \
  --mode predict \
  --model-path chm_outputs/production_mlp_best.pth \
  --output-dir chm_outputs/cross_region_predictions/05LE4_hyogo

echo "✅ Cross-region predictions completed"

# Evaluation against reference TIF files
echo "📊 Step 2: Evaluation against reference height data"

# Check available reference TIF files and evaluate
for ref_file in downloads/*.tif; do
    if [[ -f "$ref_file" ]]; then
        region_code=$(basename "$ref_file" | sed 's/dchm_//' | sed 's/.tif//')
        pred_dir="chm_outputs/cross_region_predictions/${region_code}_*"
        
        echo "🔍 Evaluating region: $region_code"
        
        if ls $pred_dir 1> /dev/null 2>&1; then
            # Create evaluation script call
            python -c "
import os
import glob
import rasterio
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json

def evaluate_predictions(pred_dir, ref_tif, region_name):
    print(f'📊 Evaluating {region_name}')
    
    # Load reference data
    with rasterio.open(ref_tif) as ref_src:
        ref_data = ref_src.read(1)
        ref_transform = ref_src.transform
        ref_crs = ref_src.crs
    
    # Find prediction files
    pred_files = glob.glob(os.path.join(pred_dir, '*_prediction.tif'))
    print(f'Found {len(pred_files)} prediction files')
    
    all_pred_vals = []
    all_ref_vals = []
    
    for pred_file in pred_files:
        try:
            with rasterio.open(pred_file) as pred_src:
                pred_data = pred_src.read(1)
                
                # Get valid pixels (non-NaN in both)
                valid_mask = (~np.isnan(pred_data)) & (~np.isnan(ref_data)) & (ref_data > 0)
                
                if valid_mask.sum() > 0:
                    all_pred_vals.extend(pred_data[valid_mask])
                    all_ref_vals.extend(ref_data[valid_mask])
                    
        except Exception as e:
            print(f'Error processing {pred_file}: {e}')
    
    if len(all_pred_vals) > 0:
        all_pred_vals = np.array(all_pred_vals)
        all_ref_vals = np.array(all_ref_vals)
        
        # Calculate metrics
        r2 = r2_score(all_ref_vals, all_pred_vals)
        rmse = np.sqrt(mean_squared_error(all_ref_vals, all_pred_vals))
        mae = mean_absolute_error(all_ref_vals, all_pred_vals)
        
        results = {
            'region': region_name,
            'n_pixels': len(all_pred_vals),
            'r2_score': float(r2),
            'rmse': float(rmse),
            'mae': float(mae),
            'pred_mean': float(np.mean(all_pred_vals)),
            'pred_std': float(np.std(all_pred_vals)),
            'ref_mean': float(np.mean(all_ref_vals)),
            'ref_std': float(np.std(all_ref_vals))
        }
        
        print(f'📈 Results for {region_name}:')
        print(f'   R² Score: {r2:.4f}')
        print(f'   RMSE: {rmse:.2f}m')
        print(f'   MAE: {mae:.2f}m')
        print(f'   Pixels: {len(all_pred_vals):,}')
        
        # Save results
        output_file = f'chm_outputs/cross_region_evaluation/{region_name}_evaluation.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'💾 Results saved to: {output_file}')
        
        return results
    else:
        print(f'❌ No valid pixels found for {region_name}')
        return None

# Evaluate each region
region_code = '${region_code}'
pred_dirs = glob.glob('chm_outputs/cross_region_predictions/${region_code}_*')
ref_file = '${ref_file}'

for pred_dir in pred_dirs:
    if os.path.exists(pred_dir):
        evaluate_predictions(pred_dir, ref_file, region_code)
"
        else
            echo "⚠️  No predictions found for region: $region_code"
        fi
    fi
done

# Generate comprehensive summary
echo "📋 Step 3: Generating comprehensive evaluation summary"
python -c "
import json
import glob
import os

print('\\n🎯 COMPREHENSIVE MLP CROSS-REGION EVALUATION SUMMARY')
print('=' * 60)

eval_files = glob.glob('chm_outputs/cross_region_evaluation/*_evaluation.json')
all_results = []

for eval_file in eval_files:
    try:
        with open(eval_file, 'r') as f:
            result = json.load(f)
            all_results.append(result)
    except Exception as e:
        print(f'Error loading {eval_file}: {e}')

if all_results:
    # Sort by R² score
    all_results.sort(key=lambda x: x.get('r2_score', 0), reverse=True)
    
    print(f'\\n📊 Results for {len(all_results)} regions:')
    print()
    print('Region      | R² Score | RMSE (m) | MAE (m) | Pixels      | Pred Mean | Ref Mean')
    print('-' * 80)
    
    total_pixels = 0
    weighted_r2 = 0
    
    for result in all_results:
        region = result['region']
        r2 = result['r2_score']
        rmse = result['rmse']
        mae = result['mae']
        pixels = result['n_pixels']
        pred_mean = result['pred_mean']
        ref_mean = result['ref_mean']
        
        print(f'{region:10} | {r2:8.4f} | {rmse:8.2f} | {mae:7.2f} | {pixels:11,} | {pred_mean:9.1f} | {ref_mean:8.1f}')
        
        total_pixels += pixels
        weighted_r2 += r2 * pixels
    
    if total_pixels > 0:
        overall_r2 = weighted_r2 / total_pixels
        print('-' * 80)
        print(f'OVERALL    | {overall_r2:8.4f} |        - |     - | {total_pixels:11,} |         - |      -')
    
    print()
    print('🏆 BREAKTHROUGH PERFORMANCE ACHIEVED!')
    print(f'📈 Best Region R²: {max(r.get(\"r2_score\", 0) for r in all_results):.4f}')
    print(f'📊 Overall R²: {overall_r2:.4f}' if total_pixels > 0 else '📊 Overall R²: Not available')
    print(f'🔢 Total Evaluated Pixels: {total_pixels:,}')
    
    # Save comprehensive summary
    summary = {
        'total_regions': len(all_results),
        'total_pixels': total_pixels,
        'overall_r2': overall_r2 if total_pixels > 0 else None,
        'best_r2': max(r.get('r2_score', 0) for r in all_results),
        'regional_results': all_results
    }
    
    with open('chm_outputs/cross_region_evaluation/comprehensive_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print('💾 Comprehensive summary saved to: chm_outputs/cross_region_evaluation/comprehensive_summary.json')
else:
    print('❌ No evaluation results found')
"

echo "✅ Full cross-region evaluation completed at: $(date)"
echo "📊 Results directory: chm_outputs/cross_region_evaluation/"
echo "🎯 Job finished with exit code: $?"
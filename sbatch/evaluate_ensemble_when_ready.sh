#!/bin/bash

#SBATCH --job-name=eval_ensemble
#SBATCH --output=chm_outputs/logs/%j_eval_ensemble.txt
#SBATCH --error=chm_outputs/logs/%j_eval_ensemble_error.txt
#SBATCH --time=0-1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --dependency=afterok:59162750

echo "🚀 Starting Ensemble Cross-Region Evaluation - Scenario 2"
echo "📅 Start time: $(date)"
echo "🖥️  Node: $(hostname)"

# Create log directory
mkdir -p chm_outputs/logs

# Activate environment
source chm_env/bin/activate

echo "📊 Evaluating ensemble predictions against Scenario 1 baseline..."

# Run evaluation on both regions
python evaluate_ensemble_cross_region.py \
    --pred-dir chm_outputs/scenario2_cross_region_predictions \
    --output-dir chm_outputs/scenario2_evaluation \
    --region both

echo "✅ Evaluation completed at $(date)"

# Display key results
echo ""
echo "📈 SCENARIO 2 RESULTS SUMMARY:"
echo "================================"

if [ -f "chm_outputs/scenario2_evaluation/combined_evaluation_results.json" ]; then
    echo "✅ Combined results file created"
    
    # Extract key metrics using Python
    python -c "
import json
try:
    with open('chm_outputs/scenario2_evaluation/combined_evaluation_results.json', 'r') as f:
        results = json.load(f)
    
    print('\\n📊 Cross-Region Performance vs Scenario 1 Baseline (R² < 0.0):')
    print('=' * 60)
    
    for region_name, region_data in results['regions'].items():
        r2 = region_data['overall_metrics']['r2']
        rmse = region_data['overall_metrics']['rmse']
        mae = region_data['overall_metrics']['mae']
        pixels = region_data['total_valid_pixels']
        
        improvement = 'IMPROVED ✅' if r2 > 0.0 else 'Still poor ❌'
        
        print(f'{region_name.upper():>8}: R² = {r2:>7.4f} ({improvement})')
        print(f'{"":>8}  RMSE = {rmse:>5.2f}m, MAE = {mae:>5.2f}m, Pixels = {pixels:,}')
        print()
        
    print('🎯 Baseline: Scenario 1 (Reference MLP only) had R² ≈ -0.04 on both regions')
    print('🔮 Scenario 2: Ensemble (GEDI + MLP) performance shown above')
    
except Exception as e:
    print(f'Error reading results: {e}')
"
    
else
    echo "❌ Combined results file not found"
fi

echo ""
echo "📁 Detailed results saved to: chm_outputs/scenario2_evaluation/"
echo "📊 Individual region plots and metrics available in subdirectories"
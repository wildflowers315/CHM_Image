#!/bin/bash

#SBATCH --job-name=eval_gedi_s5_ensemble
#SBATCH --output=logs/%j_evaluate_gedi_scenario5_ensemble.txt
#SBATCH --error=logs/%j_evaluate_gedi_scenario5_ensemble_error.txt
#SBATCH --time=0-1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=48G
#SBATCH --partition=main

# Create output directory
mkdir -p logs chm_outputs/gedi_scenario5_evaluation

# Activate environment
source chm_env/bin/activate

echo "🚀 Starting GEDI Scenario 5 Ensemble Evaluation at $(date)"
echo "📊 Comparing: GEDI Scenario 5 Ensemble (R² train=0.7762) vs Google Embedding Scenario 1 (R² train=0.8734)"

# Run evaluation comparing GEDI Scenario 5 vs Google Embedding Scenario 1
python evaluate_google_embedding_scenario1.py \
    --google-embedding-dir chm_outputs/gedi_scenario5_predictions \
    --original-mlp-dir chm_outputs/google_embedding_scenario1_predictions \
    --downloads-dir downloads \
    --output-dir chm_outputs/gedi_scenario5_evaluation \
    --no-bias-correction \
    --max-patches 63

echo "✅ GEDI Scenario 5 Ensemble vs Google Embedding Scenario 1 evaluation completed at $(date)"

# Show results summary
echo "📊 Results Summary:"
if [ -f "chm_outputs/gedi_scenario5_evaluation/scenario_comparison.csv" ]; then
    echo "Performance comparison table:"
    head -10 chm_outputs/gedi_scenario5_evaluation/scenario_comparison.csv
fi

echo "📁 Generated files:"
ls -la chm_outputs/gedi_scenario5_evaluation/

# Show specific evaluation metrics
echo "🎯 GEDI Scenario 5 Ensemble vs Google Embedding Performance:"
if [ -f "chm_outputs/gedi_scenario5_evaluation/performance_summary.json" ]; then
    python -c "
import json
try:
    with open('chm_outputs/gedi_scenario5_evaluation/performance_summary.json', 'r') as f:
        data = json.load(f)
    print('📊 Performance Metrics:')
    for scenario, metrics in data.items():
        if isinstance(metrics, dict):
            print(f'  {scenario}:')
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f'    {metric}: {value:.4f}')
except Exception as e:
    print(f'Could not load performance summary: {e}')
"
fi

echo "🔬 GEDI Scenario 5 Ensemble evaluation job finished successfully!"
echo "🎉 Ensemble combines: Google Embedding MLP (weight=0.479) + GEDI Pixel MLP (weight=0.046)"
echo "📈 Training Performance: Scenario 5 (R²=0.7762) vs Scenario 1 (R²=0.8734) vs Scenario 4 (R²=0.1284)"
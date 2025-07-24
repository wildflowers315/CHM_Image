#!/bin/bash

#SBATCH --job-name=eval_gedi_pixel_s4
#SBATCH --output=logs/%j_evaluate_gedi_pixel_scenario4.txt
#SBATCH --error=logs/%j_evaluate_gedi_pixel_scenario4_error.txt
#SBATCH --time=0-1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=48G
#SBATCH --partition=main

# Create output directory
mkdir -p logs chm_outputs/gedi_pixel_scenario4_evaluation

# Activate environment
source chm_env/bin/activate

echo "🚀 Starting GEDI Pixel Scenario 4 vs Google Embedding Scenario 1 Evaluation at $(date)"

# Run evaluation comparing GEDI Pixel Scenario 4 vs Google Embedding Scenario 1
python evaluate_google_embedding_scenario1.py \
    --google-embedding-dir chm_outputs/gedi_pixel_scenario4_predictions \
    --original-mlp-dir chm_outputs/google_embedding_scenario1_predictions \
    --downloads-dir downloads \
    --output-dir chm_outputs/gedi_pixel_scenario4_evaluation \
    --no-bias-correction \
    --max-patches 63

echo "✅ GEDI Pixel Scenario 4 vs Google Embedding Scenario 1 evaluation completed at $(date)"

# Show results summary
echo "📊 Results Summary:"
if [ -f "chm_outputs/gedi_pixel_scenario4_evaluation/scenario_comparison.csv" ]; then
    echo "Performance comparison table:"
    head -10 chm_outputs/gedi_pixel_scenario4_evaluation/scenario_comparison.csv
fi

echo "📁 Generated files:"
ls -la chm_outputs/gedi_pixel_scenario4_evaluation/

# Show specific evaluation metrics
echo "🎯 GEDI Pixel vs Google Embedding Performance:"
if [ -f "chm_outputs/gedi_pixel_scenario4_evaluation/performance_summary.json" ]; then
    python -c "
import json
try:
    with open('chm_outputs/gedi_pixel_scenario4_evaluation/performance_summary.json', 'r') as f:
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

echo "🔬 GEDI Pixel Scenario 4 evaluation job finished successfully!"
echo "📈 Compare: GEDI pixel-level (R² train=0.1284) vs Google Embedding patch-based (R² train=0.8734)"
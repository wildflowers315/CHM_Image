#!/bin/bash

#SBATCH --job-name=eval_google_embed_s1
#SBATCH --output=logs/%j_evaluate_google_embedding_scenario1.txt
#SBATCH --error=logs/%j_evaluate_google_embedding_scenario1_error.txt
#SBATCH --time=0-1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=48G
#SBATCH --partition=main

# Create output directory
mkdir -p logs chm_outputs/google_embedding_evaluation_nobiascorrection

# Activate environment
source chm_env/bin/activate

echo "🚀 Starting Google Embedding Scenario 1 Evaluation at $(date)"

# Run evaluation comparing Google Embedding vs Original 30-band MLP
python evaluate_google_embedding_scenario1.py \
    --google-embedding-dir chm_outputs/google_embedding_scenario1_predictions \
    --original-mlp-dir chm_outputs/cross_region_predictions \
    --downloads-dir downloads \
    --output-dir chm_outputs/google_embedding_evaluation_full \
    --no-bias-correction \
    --max-patches 63


echo "✅ Google Embedding Scenario 1 evaluation completed at $(date)"

# Show results summary
echo "📊 Results Summary:"
if [ -f "chm_outputs/google_embedding_evaluation/scenario_comparison.csv" ]; then
    echo "Performance comparison table:"
    head -10 chm_outputs/google_embedding_evaluation/scenario_comparison.csv
fi

echo "📁 Generated files:"
ls -la chm_outputs/google_embedding_evaluation/

echo "🎯 Evaluation job finished successfully!"
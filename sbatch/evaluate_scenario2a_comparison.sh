#!/bin/bash

#SBATCH --job-name=eval_scenario2a
#SBATCH --output=logs/%j_evaluate_scenario2a_comparison.txt
#SBATCH --error=logs/%j_evaluate_scenario2a_comparison_error.txt
#SBATCH --time=0-1:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=48G
#SBATCH --partition=main

# Create output directory
mkdir -p logs chm_outputs/scenario2a_evaluation

# Activate environment
source chm_env/bin/activate

echo "🚀 Starting Scenario 2A Evaluation: Original vs Google Embedding Ensemble Models at $(date)"
echo "📊 Comparing:"
echo "   • Original Ensemble: chm_outputs/scenario2_cross_region_predictions/"
echo "   • Google Embedding Ensemble: chm_outputs/google_embedding_scenario2a_predictions/"

# Run evaluation comparing Original vs Google Embedding Ensemble models
python evaluate_google_embedding_scenario1.py \
    --google-embedding-dir chm_outputs/google_embedding_scenario2a_predictions \
    --original-mlp-dir chm_outputs/scenario2_cross_region_predictions \
    --downloads-dir downloads \
    --output-dir chm_outputs/scenario2a_evaluation \
    --no-bias-correction \
    --max-patches 63

echo "✅ Scenario 2A evaluation completed at $(date)"

# Show results summary
echo "📊 Results Summary:"
if [ -f "chm_outputs/scenario2a_evaluation/scenario_comparison.csv" ]; then
    echo "Performance comparison table:"
    head -10 chm_outputs/scenario2a_evaluation/scenario_comparison.csv
fi

if [ -f "chm_outputs/scenario2a_evaluation/evaluation_report.txt" ]; then
    echo ""
    echo "📋 Comparison Report Summary:"
    tail -15 chm_outputs/scenario2a_evaluation/evaluation_report.txt
fi

echo "📁 Generated files:"
ls -la chm_outputs/scenario2a_evaluation/

# Show region-specific results
echo ""
echo "🌍 Region-specific Performance:"
for region in kochi hyogo tochigi; do
    if [ -d "chm_outputs/scenario2a_evaluation/$region" ]; then
        echo "📍 $region region results:"
        ls -la chm_outputs/scenario2a_evaluation/$region/ | head -5
    fi
done

echo ""
echo "🎯 Key Comparison Points:"
echo "   • Original Ensemble: GEDI U-Net + Reference MLP (30-band satellite data)"
echo "   • Google Embedding Ensemble: GEDI U-Net + Reference MLP (64-band Google Embedding)"
echo "   • Both models use ensemble weights learned during Scenario 2A training"
echo "   • Evaluation across all three regions: Kochi, Hyogo, Tochigi"

echo "🎉 Scenario 2A Evaluation Pipeline Completed!"
echo "📈 Check results in: chm_outputs/scenario2a_evaluation/"
echo "⏰ Completed at: $(date)"
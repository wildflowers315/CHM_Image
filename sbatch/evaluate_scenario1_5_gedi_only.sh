#!/bin/bash
#SBATCH --job-name=eval_scenario1_5
#SBATCH --output=logs/%j_evaluate_scenario1_5_gedi_only.txt
#SBATCH --error=logs/%j_evaluate_scenario1_5_gedi_only_error.txt
#SBATCH --time=0-1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=32G
#SBATCH --partition=main

# Create output directory
mkdir -p logs chm_outputs/scenario1_5_gedi_only_evaluation

# Activate environment
source chm_env/bin/activate

echo "🚀 Starting Scenario 1.5 GEDI-only Evaluation at $(date)"
echo "📊 Evaluating pure GEDI shift-aware U-Net performance"
echo "🎯 Model: Hyogo-trained GEDI (Google Embedding) - no ensemble, no reference MLP"

# Use existing evaluation script with GEDI-only predictions as "google-embedding-dir"
# Leave original-mlp-dir empty to evaluate only one scenario
python evaluate_google_embedding_scenario1.py \
    --google-embedding-dir chm_outputs/scenario1_5_gedi_only_predictions \
    --original-mlp-dir "" \
    --downloads-dir downloads \
    --output-dir chm_outputs/scenario1_5_gedi_only_evaluation \
    --no-bias-correction \
    --max-patches 63

echo "✅ Scenario 1.5 GEDI-only evaluation completed at $(date)"

# Show results summary
echo ""
echo "📊 Results Summary:"
if [ -f "chm_outputs/scenario1_5_gedi_only_evaluation/detailed_evaluation_results.json" ]; then
    echo "📋 Detailed results available in JSON format"
fi

echo "📁 Generated files:"
ls -la chm_outputs/scenario1_5_gedi_only_evaluation/

echo ""
echo "🎯 Key Evaluation Focus:"
echo "   • Pure GEDI U-Net Performance (no ensemble combination)"
echo "   • Cross-region generalization from Hyogo training"
echo "   • Comparison baseline for ensemble scenarios"
echo "   • Google Embedding (64-band) input features"

echo ""
echo "🌍 Expected Regional Performance:"
echo "   • Hyogo: Best performance (training region)"
echo "   • Tochigi/Kochi: Cross-region performance"
echo "   • Overall: Baseline for ensemble comparison"

echo "🎉 Scenario 1.5 GEDI-only Evaluation Pipeline Completed!"
echo "📈 Check results in: chm_outputs/scenario1_5_gedi_only_evaluation/"
echo "⏰ Completed at: $(date)"
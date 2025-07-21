#!/bin/bash
#SBATCH --job-name=eval_scenario3_comp
#SBATCH --output=logs/%j_evaluate_scenario3_comprehensive.txt
#SBATCH --error=logs/%j_evaluate_scenario3_comprehensive_error.txt
#SBATCH --time=0-2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=48G
#SBATCH --partition=main

# Create output directory
mkdir -p logs chm_outputs/scenario3_comprehensive_evaluation

# Activate environment
source chm_env/bin/activate

echo "🚀 Starting Scenario 3A vs 3B Evaluation at $(date)"
echo "📊 Comparing:"
echo "   • Scenario 3A: Tochigi from-scratch GEDI + Fixed Ensemble"  
echo "   • Scenario 3B: Tochigi fine-tuned GEDI + Fixed Ensemble"

# Evaluate Scenario 3A vs 3B (Fine-tuning vs From-scratch)
echo ""
echo "=== Evaluating Scenario 3A vs 3B ==="
python evaluate_google_embedding_scenario1.py \
    --google-embedding-dir chm_outputs/google_embedding_scenario3b_predictions \
    --original-mlp-dir chm_outputs/google_embedding_scenario3a_predictions \
    --downloads-dir downloads \
    --output-dir chm_outputs/scenario3_comprehensive_evaluation \
    --no-bias-correction \
    --max-patches 63

echo "✅ Comprehensive evaluation completed at $(date)"

# Show results summary
echo ""
echo "📊 Results Summary:"
echo "📁 Evaluation results saved in: chm_outputs/scenario3_comprehensive_evaluation/"

echo ""
echo "🎯 Key Comparison Focus:"
echo "   • Scenario 3A: GEDI trained from scratch on Tochigi"
echo "   • Scenario 3B: GEDI fine-tuned from Scenario 2A on Tochigi"
echo "   • Both use fixed ensemble MLP (no retraining)"
echo "   • Question: Is fine-tuning better than from-scratch for target region?"

echo ""
echo "📈 Expected Findings:"
echo "   • Training validation: 3B had better validation loss (8.3112 vs 8.4774)"
echo "   • Cross-region performance: Compare R², RMSE, correlation across regions"
echo "   • Tochigi target region: Should show best performance for both scenarios"

echo "🎉 Scenario 3A vs 3B Evaluation Pipeline Completed!"
echo "⏰ Completed at: $(date)"
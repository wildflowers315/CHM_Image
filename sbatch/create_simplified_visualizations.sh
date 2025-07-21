#!/bin/bash
#SBATCH --job-name=simple_vis
#SBATCH --output=logs/%j_simplified_visualizations.txt  
#SBATCH --error=logs/%j_simplified_visualizations_error.txt
#SBATCH --time=0-1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=main

# Create output directories
mkdir -p logs chm_outputs/simplified_prediction_visualizations

# Set SSL library path for Earth Engine
export LD_LIBRARY_PATH="$HOME/openssl/lib:$LD_LIBRARY_PATH"

# Activate environment
source chm_env/bin/activate

echo "🎨 Starting Simplified Prediction Visualization Pipeline at $(date)"
echo "📊 Creating row-layout visualizations: RGB + Reference + 5 Scenarios per region"
echo ""

# Check available prediction directories
echo "🔍 Checking available prediction data:"
echo "  Scenario 1 (Reference MLP):"
ls -la chm_outputs/google_embedding_scenario1_predictions/ 2>/dev/null || echo "    ❌ Not found"

echo "  Scenario 1.5 (GEDI-only):"
ls -la chm_outputs/scenario1_5_gedi_only_predictions/ 2>/dev/null || echo "    ❌ Not found"

echo "  Scenario 2A (Ensemble):"
ls -la chm_outputs/google_embedding_scenario2a_predictions/ 2>/dev/null || echo "    ❌ Not found"

echo "  Scenario 3A (From-scratch):"
ls -la chm_outputs/google_embedding_scenario3a_predictions/ 2>/dev/null || echo "    ❌ Not found"

echo "  Scenario 3B (Fine-tuned):"
ls -la chm_outputs/google_embedding_scenario3b_predictions/ 2>/dev/null || echo "    ❌ Not found"

echo ""
echo "📍 Reference data:"
ls -la downloads/dchm_*.tif 2>/dev/null || echo "    ❌ Reference files not found"

echo ""
echo "🌍 AOI boundary files:"
ls -la downloads/dchm_*.geojson 2>/dev/null || echo "    ❌ AOI files not found"

echo ""
echo "=" * 70

# Run visualization script with all scenarios
echo "🚀 Creating visualizations for all scenarios..."
# Test with fewer scenarios first to check memory usage
python create_simplified_prediction_visualizations.py \
    --scenarios scenario1 scenario1_5 scenario2a scenario3a scenario3b \
    --patch-index 12 \
    --output-dir chm_outputs/simplified_prediction_visualizations \
    --vis-scale 0.7

echo ""
echo "✅ Simplified visualization pipeline completed at $(date)"

# Show generated files
echo ""
echo "📊 Generated Files:"
ls -la chm_outputs/simplified_prediction_visualizations/

# Count successful visualizations
PNG_COUNT=$(find chm_outputs/simplified_prediction_visualizations -name "*.png" | wc -l)
RGB_COUNT=$(find chm_outputs/simplified_prediction_visualizations -name "*_rgb_composite.tif" | wc -l)

echo ""
echo "📋 Generation Summary:"
echo "  🖼️  PNG visualizations: $PNG_COUNT"
echo "  🌈 RGB composites: $RGB_COUNT"

if [ $PNG_COUNT -eq 3 ]; then
    echo "✅ SUCCESS: All 3 region visualizations generated!"
else
    echo "⚠️  WARNING: Expected 3 PNG files, got $PNG_COUNT"
fi

echo ""
echo "🎯 Usage Examples:"
echo "  # View specific scenarios only:"
echo "  python create_simplified_prediction_visualizations.py --scenarios scenario1 scenario2a scenario3b"
echo ""
echo "  # Use specific patch index (consistent across regions):"
echo "  python create_simplified_prediction_visualizations.py --patch-index 0"
echo ""
echo "  # Use custom random seed:"
echo "  python create_simplified_prediction_visualizations.py --random-seed 123"
echo ""
echo "  # Custom output directory:"
echo "  python create_simplified_prediction_visualizations.py --output-dir custom_vis/"

echo ""
echo "🎉 Simplified Prediction Visualization Pipeline Completed!"
echo "📁 Results available in: chm_outputs/simplified_prediction_visualizations/"
echo "⏰ Completed at: $(date)"
#!/bin/bash
#SBATCH --job-name=gedi_s4_vis
#SBATCH --output=logs/%j_gedi_scenario4_visualizations.txt  
#SBATCH --error=logs/%j_gedi_scenario4_visualizations_error.txt
#SBATCH --time=0-1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=main

# Create output directories
mkdir -p logs chm_outputs/gedi_scenario4_visualizations

# Set SSL library path for Earth Engine
export LD_LIBRARY_PATH="$HOME/openssl/lib:$LD_LIBRARY_PATH"

# Activate environment
source chm_env/bin/activate

echo "🎨 Starting GEDI Scenario 4 Visualization Pipeline at $(date)"
echo "📊 Creating visualizations comparing GEDI Pixel Scenario 4 with key scenarios"
echo ""

# Check available prediction directories
echo "🔍 Checking available prediction data:"
echo "  Scenario 1 (Google Embedding):"
ls -la chm_outputs/google_embedding_scenario1_predictions/ 2>/dev/null || echo "    ❌ Not found"

echo "  Scenario 4 (GEDI Pixel):"
ls -la chm_outputs/gedi_pixel_scenario4_predictions/ 2>/dev/null || echo "    ❌ Not found"

echo "  Scenario 2A (Ensemble):"
ls -la chm_outputs/google_embedding_scenario2a_predictions/ 2>/dev/null || echo "    ❌ Not found"

echo ""
echo "📍 Reference data:"
ls -la downloads/dchm_*.tif 2>/dev/null || echo "    ❌ Reference files not found"

echo ""
echo "🌍 AOI boundary files:"
ls -la downloads/dchm_*.geojson 2>/dev/null || echo "    ❌ AOI files not found"

echo ""
echo "=" * 70

# Run visualization script focusing on GEDI Scenario 4 comparison
echo "🚀 Creating GEDI Scenario 4 vs key scenarios visualization..."
python create_simplified_prediction_visualizations.py \
    --scenarios scenario1 scenario4 scenario2a \
    --patch-index 12 \
    --output-dir chm_outputs/gedi_scenario4_visualizations \
    --vis-scale 1

echo ""
echo "✅ GEDI Scenario 4 visualization pipeline completed at $(date)"

# Show generated files
echo ""
echo "📊 Generated Files:"
ls -la chm_outputs/gedi_scenario4_visualizations/

# Count successful visualizations
PNG_COUNT=$(find chm_outputs/gedi_scenario4_visualizations -name "*.png" | wc -l)
RGB_COUNT=$(find chm_outputs/gedi_scenario4_visualizations -name "*_rgb_composite.tif" | wc -l)

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
echo "🔬 GEDI Scenario 4 Performance Summary:"
echo "  📈 Training R²: 0.1284 (pixel-level GEDI prediction)"
echo "  📊 Comparison scenarios:"
echo "    - Scenario 1 (Google Embedding): R² = 0.8734"
echo "    - Scenario 2A (Ensemble): R² = 0.7844"
echo "    - Scenario 4 (GEDI Pixel): R² = 0.1284"

echo ""
echo "🎯 Key Insights Expected:"
echo "  • GEDI pixel-level approach shows modest performance"
echo "  • Google Embedding patch-based methods significantly outperform"
echo "  • Visualization reveals spatial patterns in prediction quality"

echo ""
echo "🎉 GEDI Scenario 4 Visualization Pipeline Completed!"
echo "📁 Results available in: chm_outputs/gedi_scenario4_visualizations/"
echo "⏰ Completed at: $(date)"
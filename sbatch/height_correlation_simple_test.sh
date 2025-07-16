#!/bin/bash

#SBATCH --job-name=height_correlation_simple_test
#SBATCH --output=logs/%j_height_simple_test.txt
#SBATCH --error=logs/%j_height_simple_test.txt
#SBATCH --time=0-1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=main

# Create output directory if it doesn't exist
mkdir -p logs

# Activate the Python environment
source chm_env/bin/activate

echo "Starting Simple Height Correlation Analysis Test"
echo "Job ID: $SLURM_JOB_ID"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Testing with one patch per region..."
echo "=" * 50

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:/home/WUR/ishik001/CHM_Image"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run the simple test
python analysis/height_correlation_analysis_simple.py

# Check if the analysis completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "=" * 50
    echo "Simple test completed successfully!"
    echo "Job finished at: $(date)"
    
    # List generated files
    echo ""
    echo "Generated files:"
    ls -la chm_outputs/plot_analysis_simple_test/
    
    echo ""
    echo "Generated plots:"
    find chm_outputs/plot_analysis_simple_test/ -name "*.png" | wc -l
    find chm_outputs/plot_analysis_simple_test/ -name "*.png"
    
    # Show results summary
    if [ -f "chm_outputs/plot_analysis_simple_test/height_correlation_simple_test.csv" ]; then
        echo ""
        echo "Results summary:"
        cat chm_outputs/plot_analysis_simple_test/height_correlation_simple_test.csv
    fi
    
else
    echo ""
    echo "ERROR: Simple test failed!"
    echo "Job failed at: $(date)"
    echo "Exit code: $?"
fi

echo ""
echo "Job completed at: $(date)"
echo "Total job duration: $SECONDS seconds"
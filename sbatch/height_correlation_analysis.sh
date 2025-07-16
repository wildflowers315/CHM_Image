#!/bin/bash

#SBATCH --job-name=height_correlation_analysis
#SBATCH --output=logs/%j_height_analysis_log.txt
#SBATCH --error=logs/%j_height_analysis_log.txt
#SBATCH --time=0-5:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=main

# Create output directory if it doesn't exist
mkdir -p logs

# Activate the Python environment
source chm_env/bin/activate

echo "Starting Height Correlation Analysis"
echo "Job ID: $SLURM_JOB_ID"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "=" * 50

# Set environment variables for better performance
export PYTHONPATH="${PYTHONPATH}:/home/WUR/ishik001/CHM_Image"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Environment setup:"
echo "  Python path: $PYTHONPATH"
echo "  OMP threads: $OMP_NUM_THREADS"
echo "  CPUs per task: $SLURM_CPUS_PER_TASK"
echo "  Memory per CPU: 4G"
echo ""

# Run the height correlation analysis
echo "Running height correlation analysis..."
python analysis/height_correlation_analysis.py

# Check if the analysis completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "=" * 50
    echo "Height correlation analysis completed successfully!"
    echo "Job finished at: $(date)"
    
    # List generated files
    echo ""
    echo "Generated files:"
    ls -la chm_outputs/plot_analysis/
    
    echo ""
    echo "Analysis summary:"
    if [ -f "chm_outputs/plot_analysis/height_correlation_summary.csv" ]; then
        echo "Results saved to: chm_outputs/plot_analysis/height_correlation_summary.csv"
        echo "Sample of results:"
        head -n 10 chm_outputs/plot_analysis/height_correlation_summary.csv
    else
        echo "Warning: Summary file not found"
    fi
    
    echo ""
    echo "Plots generated:"
    find chm_outputs/plot_analysis/ -name "*.png" | wc -l
    find chm_outputs/plot_analysis/ -name "*.png" | head -5
    
else
    echo ""
    echo "=" * 50
    echo "ERROR: Height correlation analysis failed!"
    echo "Job failed at: $(date)"
    echo "Exit code: $?"
    
    # Show last few lines of error log if available
    if [ -f "output/height_analysis_log/error_${SLURM_JOB_ID}.txt" ]; then
        echo ""
        echo "Last 10 lines of error log:"
        tail -n 10 "output/height_analysis_log/error_${SLURM_JOB_ID}.txt"
    fi
fi

echo ""
echo "Job completed at: $(date)"
echo "Total job duration: $SECONDS seconds"
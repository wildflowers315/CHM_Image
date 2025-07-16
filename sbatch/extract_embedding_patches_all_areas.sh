#!/bin/bash

#SBATCH --job-name=extract_embedding_patches
#SBATCH --output=output/embedding_log/output_%j.txt
#SBATCH --error=output/embedding_log/error_%j.txt
#SBATCH --time=0-12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=cpu

# Create output directory if it doesn't exist
mkdir -p output/embedding_log

# Activate the Python environment
source chm_env/bin/activate

# Set year for embedding extraction
YEAR=2022

# Define the three areas with their corresponding GeoJSON files
AREAS=(
    "dchm_04hf3"  # Hyogo
    "dchm_05LE4"  # Kochi  
    "dchm_09gd4"  # Tochigi
)

echo "Starting Google Embedding v1 patch extraction for all three areas..."
echo "Year: $YEAR"
echo "Areas: ${AREAS[*]}"

# Process each area
for area in "${AREAS[@]}"; do
    echo "========================================="
    echo "Processing area: $area"
    echo "========================================="
    
    # Set AOI file path
    AOI_FILE="downloads/${area}.geojson"
    
    # Check if AOI file exists
    if [[ ! -f "$AOI_FILE" ]]; then
        echo "ERROR: AOI file not found: $AOI_FILE"
        continue
    fi
    
    # Run the embedding patch extraction
    echo "Running embedding patch extraction for $area..."
    python chm_main.py \
        --aoi "$AOI_FILE" \
        --year $YEAR \
        --embedding-only \
        --use-patches \
        --export-patches \
        --scale 10 \
        --output-dir "outputs/embedding_patches" \
        --patch-size 2560 \
        --patch-overlap 10 \
        --mask-type none
    
    # Check if the command was successful
    if [[ $? -eq 0 ]]; then
        echo "SUCCESS: Embedding patch extraction completed for $area"
    else
        echo "ERROR: Embedding patch extraction failed for $area"
    fi
    
    echo "Completed processing for $area"
    echo ""
done

echo "All areas processed. Check Google Drive GEE_exports folder for results."
echo "Expected output files: *_embedding_bandNum64_scale10_patch*.tif"
#!/bin/bash

#SBATCH --job-name=extract_gedi_embedding
#SBATCH --output=logs/%j_pixel_gedi_embedding_log.txt
#SBATCH --error=logs/%j_pixel_gedi_embedding_log_error.txt
#SBATCH --time=0-3:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=main


# Activate the Python environment
source chm_env/bin/activate

# Set year for GEDI and embedding extraction
YEAR=2022

# Set buffer size (5km = 5000m)
BUFFER=1

# Define the three areas with their corresponding GeoJSON files
AREAS=(
    # "dchm_04hf3"  # Hyogo
    "dchm_05LE4"  # Kochi  
    # "dchm_09gd4"  # Tochigi
)

echo "Starting GEDI data extraction with Google Embedding v1 for all three areas..."
echo "Year: $YEAR"
echo "Buffer: ${BUFFER}m"
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
    
    # Run the GEDI embedding extraction
    echo "Running GEDI embedding extraction for $area..."
    python extract_gedi_with_embedding.py \
        --aoi "$AOI_FILE" \
        --year $YEAR \
        --buffer $BUFFER \
        --gedi-start-date "${YEAR}-01-01" \
        --gedi-end-date "${YEAR}-12-31" \
        --quantile rh98 \
        --scale 10 \
        --mask-type NDVI \
        --ndvi-threshold 0.3 \
        --window-size 3 \
        --select-band All
    
    # Check if the command was successful
    if [[ $? -eq 0 ]]; then
        echo "SUCCESS: GEDI embedding extraction completed for $area"
    else
        echo "ERROR: GEDI embedding extraction failed for $area"
    fi
    
    echo "Completed processing for $area"
    echo ""
done

echo "All areas processed. Check Google Drive GEE_exports folder for results."
echo "Expected output files: gedi_embedding_*_NDVI30_b*_${YEAR}.csv"
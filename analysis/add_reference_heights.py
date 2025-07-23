#!/usr/bin/env python3
"""
Add reference height values from TIF files to GEDI CSV data.

This script reads a CSV file with GEDI data (containing longitude/latitude),
extracts reference height values from the corresponding TIF file in downloads/,
and creates a new CSV with the reference heights added.

Usage:
    python analysis/add_reference_heights.py --csv path/to/gedi_data.csv
"""

import pandas as pd
import rasterio
from rasterio.warp import transform_bounds
import numpy as np
import os
import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.spatial_utils import sample_raster_at_coordinates

def extract_area_code(csv_filename: str) -> str:
    """Extract area code from CSV filename (e.g., dchm_04hf3 from filename)."""
    # Look for pattern like dchm_04hf3, dchm_05LE4, dchm_09gd4
    parts = csv_filename.lower().split('_')
    for i, part in enumerate(parts):
        if part.startswith('dchm') and i + 1 < len(parts):
            return f"{part}_{parts[i+1]}"
    
    # Fallback: look for known area codes
    area_codes = ['dchm_04hf3', 'dchm_05le4', 'dchm_09gd4']
    for code in area_codes:
        if code in csv_filename.lower():
            return code
    
    raise ValueError(f"Could not extract area code from filename: {csv_filename}")

# Removed duplicate function - using sample_raster_at_coordinates from utils.py

def add_reference_heights_to_csv(csv_path: str, downloads_dir: str = "downloads") -> str:
    """
    Add reference height values to GEDI CSV data.
    
    Args:
        csv_path: Path to the input CSV file
        downloads_dir: Directory containing reference TIF files
        
    Returns:
        Path to the output CSV file with reference heights added
    """
    print(f"Processing CSV: {csv_path}")
    
    # Read the CSV file
    try:
        df = pd.read_csv(csv_path)
        print(f"  Loaded {len(df)} rows from CSV")
    except Exception as e:
        raise ValueError(f"Could not read CSV file: {e}")
    
    # Check for required columns
    if 'longitude' not in df.columns or 'latitude' not in df.columns:
        raise ValueError("CSV must contain 'longitude' and 'latitude' columns")
    
    # Extract area code from filename
    csv_filename = os.path.basename(csv_path)
    try:
        area_code = extract_area_code(csv_filename)
        print(f"  Detected area code: {area_code}")
    except ValueError as e:
        raise ValueError(f"Could not determine area code: {e}")
    
    # Find corresponding TIF file
    tif_path = os.path.join(downloads_dir, f"{area_code}.tif")
    if not os.path.exists(tif_path):
        # Try alternative naming patterns
        alternative_paths = [
            os.path.join(downloads_dir, f"{area_code.upper()}.tif"),
            os.path.join(downloads_dir, f"{area_code.replace('_', '')}.tif"),
        ]
        
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                tif_path = alt_path
                break
        else:
            raise FileNotFoundError(f"Could not find reference TIF file: {tif_path}")
    
    print(f"  Using reference TIF: {tif_path}")
    
    # Sample reference heights
    print("  Sampling reference heights...")
    with rasterio.open(tif_path) as src:
        print(f"  Reference raster CRS: {src.crs}")
        print(f"  Reference raster bounds: {src.bounds}")
        print(f"  Reference raster shape: {src.shape}")
    
    reference_heights = sample_raster_at_coordinates(
        tif_path, 
        df['longitude'].values, 
        df['latitude'].values
    )
    
    # Add reference heights to dataframe
    df['reference_height'] = reference_heights
    
    # Print statistics
    valid_refs = ~np.isnan(reference_heights)
    print(f"  Successfully sampled {np.sum(valid_refs)} / {len(reference_heights)} points")
    if np.sum(valid_refs) > 0:
        print(f"  Reference height range: {np.nanmin(reference_heights):.2f} to {np.nanmax(reference_heights):.2f} m")
        print(f"  Reference height mean: {np.nanmean(reference_heights):.2f} m")
    
    # Create output filename
    csv_dir = os.path.dirname(csv_path)
    csv_name = os.path.splitext(csv_filename)[0]
    output_path = os.path.join(csv_dir, f"{csv_name}_with_reference.csv")
    
    # Save the enhanced CSV
    df.to_csv(output_path, index=False)
    print(f"  Saved enhanced CSV: {output_path}")
    
    return output_path

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Add reference heights from TIF to GEDI CSV data')
    
    parser.add_argument('--csv', type=str, required=True,
                       help='Path to input CSV file with GEDI data')
    parser.add_argument('--downloads-dir', type=str, default='downloads',
                       help='Directory containing reference TIF files (default: downloads)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output CSV path (default: auto-generated)')
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Validate input file
    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"Input CSV file not found: {args.csv}")
    
    if not os.path.exists(args.downloads_dir):
        raise FileNotFoundError(f"Downloads directory not found: {args.downloads_dir}")
    
    try:
        # Add reference heights
        output_path = add_reference_heights_to_csv(args.csv, args.downloads_dir)
        
        if args.output:
            # Move to user-specified output path
            import shutil
            shutil.move(output_path, args.output)
            output_path = args.output
            print(f"  Final output: {output_path}")
        
        print("\n" + "="*60)
        print("SUCCESS: Reference heights added successfully!")
        print("="*60)
        print(f"Input:  {args.csv}")
        print(f"Output: {output_path}")
        print("="*60)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
#!/usr/bin/env python3
"""
Extract GEDI data with Google Embedding bands as CSV.

This script extracts GEDI footprints with Google Embedding v1 satellite data
and exports the combined dataset as CSV via Google Earth Engine batch export.
Uses existing functions from chm_main.py to maintain consistency.

Usage:
    python extract_gedi_with_embedding.py --aoi path/to/aoi.geojson
"""

import ee
import os
import argparse

# Import existing functions from chm_main.py
from chm_main import (
    initialize_ee, load_aoi, get_google_embedding_data, 
    export_featurecollection_to_csv, create_forest_mask
)
from l2a_gedi_source import get_gedi_vector_data
from canopyht_source import get_canopyht_data

def add_glcm_texture(google_embedding: ee.Image, window_size: int = 3, select_band: str = 'savg') -> ee.Image:
    """Add GLCM texture features to Google embedding data."""
    try:
        print("Adding GLCM texture features...")
        
        # Debug: Check input image bands
        input_bands = google_embedding.bandNames().getInfo()
        print(f"  Input image has {len(input_bands)} bands: {input_bands[:5]}..." if len(input_bands) > 5 else f"  Input image bands: {input_bands}")
        
        if len(input_bands) == 0:
            print("  ERROR: Input image has 0 bands, skipping texture calculation")
            return google_embedding
        
        # Scale Google Embedding data for better GLCM calculation
        # Google Embedding values are [-1, 1], scale to [0, 255] for better texture analysis
        scaled_band = google_embedding.add(1).multiply(127.5).toInt()
        
        # Compute GLCM textures with neighborhood size of window_size
        glcm = scaled_band.glcmTexture(size=window_size)
        
        # Debug: Check GLCM output bands
        glcm_band_names = glcm.bandNames().getInfo()
        print(f"  GLCM generated {len(glcm_band_names)} bands")
        print(f"  Sample GLCM band names: {glcm_band_names[:10]}")
        
        if select_band == 'All':
            select_band_list = ['asm', 'corr', 'var', 'idm', 'savg', 'ent', 'dvar']
        else:
            select_band_list = [select_band]
        
        for i,select_band in enumerate(select_band_list):
            # Find homogeneity bands (IDM = Inverse Difference Moment)
            select_band_bands = [band for band in glcm_band_names if f'_{select_band}' in band]            
            print(f"  Found {len(select_band_bands)} {select_band} bands: {select_band_bands[:3]}..." if len(select_band_bands) > 3 else f"  {select_band} bands: {select_band_bands}")

            if len(select_band_bands) == 0:
                print(f"  No {select_band} bands found, skipping texture features")
                return google_embedding
                    
            # Select select_band bands
            glcm_filtered = glcm.select(select_band_bands).float()
            
            # Debug: Check filtered bands
            filtered_bands = glcm_filtered.bandNames().getInfo()
            print(f"  Filtered to {len(filtered_bands)} {select_band} bands")
            
            if len(filtered_bands) == 0:
                print("  ERROR: Filtered image has 0 bands, skipping texture calculation")
                return google_embedding
            
            # Calculate mean and median select_band for each pixel
            mean_select_band = glcm_filtered.reduce('mean').rename(f'mean_{select_band}')
            median_select_band = glcm_filtered.reduce('median').rename(f'median_{select_band}')
                    
            if i == 0:
                result = google_embedding.addBands(mean_select_band).addBands(median_select_band)
            else:
                result = result.addBands(mean_select_band).addBands(median_select_band)
            print(f"  Successfully added {select_band} GLCM texture features")
        print("  Successfully added GLCM texture features")
        return result
        
    except Exception as e:
        print(f"  ERROR in GLCM texture calculation: {e}")
        print("  Continuing without texture features...")
        return google_embedding

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Extract GEDI data with Google Embedding bands')
    
    # Required arguments
    parser.add_argument('--aoi', type=str, required=True, 
                       help='Path to AOI GeoJSON file')
    
    # Optional arguments with defaults optimized for GEDI extraction
    parser.add_argument('--year', type=int, default=2022,
                       help='Year for Google Embedding data (default: 2022)')
    parser.add_argument('--buffer', type=int, default=5000,
                       help='Buffer size in meters (default: 5000m = 5km)')
    parser.add_argument('--gedi-start-date', type=str, default='2022-01-01',
                       help='GEDI start date (default: 2022-01-01)')
    parser.add_argument('--gedi-end-date', type=str, default='2022-12-31',
                       help='GEDI end date (default: 2022-12-31)')
    parser.add_argument('--quantile', type=str, default='rh98',
                       help='GEDI height quantile (default: rh98)')
    parser.add_argument('--scale', type=int, default=10,
                       help='Sampling scale in meters (default: 10)')
    parser.add_argument('--mask-type', type=str, default='NDVI',
                       choices=['NDVI', 'DW', 'ALL', 'none'],
                       help='Type of forest mask to apply (default: NDVI)')
    parser.add_argument('--ndvi-threshold', type=float, default=0.3,
                       help='NDVI threshold for forest mask (default: 0.3)')
    parser.add_argument('--window-size', type=int, default=3,
                       help='Window size for GLCM texture calculation (default: 3)')
    parser.add_argument('--select-band', type=str, default='savg',
                       choices=['asm', 'contrast', 'corr', 'var', 'idm', 'savg', 'svar', 'sent', 'ent', 'dvar', 'All'],
                       help='Band to select for GLCM texture calculation (default: contrast)')
    
    return parser.parse_args()

def main():
    """Main function to extract GEDI data with Google Embedding bands."""
    args = parse_args()
    
    # Initialize Earth Engine
    initialize_ee()
    
    # Load AOI and apply buffer
    print(f"Loading AOI from: {args.aoi}")
    original_aoi = load_aoi(args.aoi)
    aoi_buffered = original_aoi.buffer(args.buffer)
    print(f"Applied {args.buffer}m buffer to AOI")
    
    # Get Google Embedding v1 data
    google_embedding = get_google_embedding_data(aoi_buffered, args.year)
    
    # Debug: Check Google Embedding bands
    try:
        embedding_bands = google_embedding.bandNames().getInfo()
        print(f"Google Embedding loaded with {len(embedding_bands)} bands")
    except Exception as e:
        print(f"ERROR checking Google Embedding bands: {e}")
        return
    
    # Add GLCM texture features
    google_embedding = add_glcm_texture(google_embedding, args.window_size, args.select_band)
    
    # Add canopy height data
    print("Loading canopy height data...")
    canopy_ht = get_canopyht_data(aoi_buffered)
    if canopy_ht is not None:
        canopy_ht = canopy_ht.float()
        google_embedding = google_embedding.addBands(canopy_ht)
        print("Added canopy height data")
        
        # Debug: Check bands after adding canopy height
        try:
            total_bands = google_embedding.bandNames().getInfo()
            print(f"Total bands after adding canopy height: {len(total_bands)}")
        except Exception as e:
            print(f"ERROR checking bands after canopy height: {e}")
    
    
    
    # Create and add forest mask as a band
    if args.mask_type != 'none':
        print(f"Creating forest mask ({args.mask_type}, threshold: {args.ndvi_threshold})...")
        try:
            forest_mask = create_forest_mask(
                args.mask_type,
                aoi_buffered,
                ee.Date(args.gedi_start_date),
                ee.Date(args.gedi_end_date),
                args.ndvi_threshold
            )
            if forest_mask is not None:
                forest_mask = forest_mask.float().rename('forest_mask')
                google_embedding = google_embedding.addBands(forest_mask)
                print("Added forest mask as a band")
            else:
                print("WARNING: Forest mask creation returned None, continuing without forest mask")
                
        except Exception as e:
            print(f"ERROR creating forest mask: {e}")
            print("Continuing without forest mask...")
            
        # Debug: Check bands after forest mask processing
        try:
            total_bands = google_embedding.bandNames().getInfo()
            print(f"Total bands after forest mask processing: {len(total_bands)}")
        except Exception as e:
            print(f"ERROR checking bands after forest mask: {e}")
    else:
        print("Skipping forest mask creation (mask_type = 'none')")
    
    # Get GEDI vector data
    print(f"Loading GEDI data from {args.gedi_start_date} to {args.gedi_end_date}...")
    gedi_points = get_gedi_vector_data(
        aoi_buffered, 
        args.gedi_start_date, 
        args.gedi_end_date, 
        args.quantile
    ).select([args.quantile], ['rh']) # Select only the height quantile band
    
    if gedi_points is None or gedi_points.size().getInfo() == 0:
        print("No GEDI data found. Exiting.")
        return
    
    # Debug: Check final Google Embedding bands before sampling
    try:
        final_bands = google_embedding.bandNames().getInfo()
        print(f"Final Google Embedding has {len(final_bands)} bands before sampling")
        if len(final_bands) == 0:
            print("ERROR: No bands available for sampling!")
            return
    except Exception as e:
        print(f"ERROR checking final bands: {e}")
        return
    
    # Sample Google Embedding data at GEDI points
    print("Sampling Google Embedding data at GEDI locations...")
    try:
        reference_data = google_embedding.sampleRegions(
            collection=gedi_points,
            scale=args.scale,
            projection='EPSG:4326',
            tileScale=1,
            geometries=True
        )
        print("Sampling completed successfully")
    except Exception as e:
        print(f"ERROR during sampling: {e}")
        return
    
    # Forest mask is now included as a band in the sampled data
    # Users can filter based on forest_mask values in post-processing if needed
    
    # # Get final point count
    # try:
    #     final_count = reference_data.size().getInfo()
    #     print(f"Final dataset contains {final_count} points")
    # except Exception as e:
    #     print(f"ERROR getting final count: {e}")
    #     # Try to debug the reference_data structure
    #     try:
    #         sample_feature = reference_data.first().getInfo()
    #         print(f"Sample feature properties: {list(sample_feature.get('properties', {}).keys())}")
    #     except Exception as debug_e:
    #         print(f"ERROR getting sample feature: {debug_e}")
    #     return
    
    # if final_count == 0:
    #     print("No valid points after filtering. Exiting.")
    #     return
    
    # Get number of bands for export name
    band_count = google_embedding.bandNames().size().getInfo()
    
    # Generate export name
    geojson_name = os.path.splitext(os.path.basename(args.aoi))[0]
    ndvi_threshold_percent = int(round(args.ndvi_threshold * 100, 0))
    mask_type_name = args.mask_type if args.mask_type != 'NDVI' else f'{args.mask_type}{ndvi_threshold_percent}'
    export_name = f'gedi_embedding_{geojson_name}_{mask_type_name}_b{band_count}_{args.year}_{args.buffer}m_scale{args.scale}m_w{args.window_size}_{args.select_band}'
    
    # Export training data as CSV
    print(f'Exporting GEDI data with Google Embedding bands as CSV: {export_name}')
    export_featurecollection_to_csv(reference_data, export_name)
    
    print("\n" + "="*60)
    print("EXPORT SUMMARY")
    print("="*60)
    print(f"AOI: {os.path.basename(args.aoi)}")
    print(f"Buffer: {args.buffer}m")
    print(f"Year: {args.year}")
    print(f"GEDI period: {args.gedi_start_date} to {args.gedi_end_date}")
    print(f"Forest mask: {args.mask_type} (threshold: {args.ndvi_threshold})")
    print(f"Total bands: {band_count}")
    # print(f"Final points: {final_count}")
    print(f"Export name: {export_name}")
    print("="*60)

if __name__ == "__main__":
    main()
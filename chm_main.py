import ee
import geemap
import numpy as np
import json
import os
from pathlib import Path
import pandas as pd
import datetime
import argparse
from typing import Union, List, Dict, Any
import time
from tqdm import tqdm
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS

# Import custom functions
from l2a_gedi_source import get_gedi_data
from sentinel1_source import get_sentinel1_data
from sentinel2_source import get_sentinel2_data
from for_forest_masking import apply_forest_mask, create_forest_mask
from alos2_source import get_alos2_data
from new_random_sampling import create_training_data, generate_sampling_sites
from canopyht_source import get_canopyht_data
from dem_source import get_dem_data

def load_aoi(aoi_path: str) -> ee.Geometry:
    """
    Load AOI from GeoJSON file. Handles both simple Polygon/MultiPolygon and FeatureCollection formats.
    
    Args:
        aoi_path: Path to GeoJSON file
    
    Returns:
        ee.Geometry: Earth Engine geometry object
    """
    if not os.path.exists(aoi_path):
        raise FileNotFoundError(f"AOI file not found: {aoi_path}")
    
    with open(aoi_path, 'r') as f:
        geojson_data = json.load(f)
    
    def create_geometry(geom_type: str, coords: List) -> ee.Geometry:
        """Helper function to create ee.Geometry objects."""
        if geom_type == 'Polygon':
            return ee.Geometry.Polygon(coords)
        elif geom_type == 'MultiPolygon':
            # MultiPolygon coordinates are nested one level deeper than Polygon
            return ee.Geometry.MultiPolygon(coords[0])
        else:
            raise ValueError(f"Unsupported geometry type: {geom_type}")
    
    # Handle FeatureCollection
    if geojson_data['type'] == 'FeatureCollection':
        if not geojson_data['features']:
            raise ValueError("Empty FeatureCollection")
        
        # Get the first feature's geometry
        geometry = geojson_data['features'][0]['geometry']
        return create_geometry(geometry['type'], geometry['coordinates'])
    
    # Handle direct Polygon/MultiPolygon
    elif geojson_data['type'] in ['Polygon', 'MultiPolygon']:
        return create_geometry(geojson_data['type'], geojson_data['coordinates'])
    else:
        raise ValueError(f"Unsupported GeoJSON type: {geojson_data['type']}")

def parse_args():
    parser = argparse.ArgumentParser(description='Canopy Height Mapping using Earth Engine')
    # Basic parameters
    parser.add_argument('--aoi', type=str, required=True, help='Path to AOI GeoJSON file')
    parser.add_argument('--year', type=int, required=True, help='Year for analysis')
    parser.add_argument('--start-date', type=str, default='01-01', help='Start date (MM-DD)')
    parser.add_argument('--end-date', type=str, default='12-31', help='End date (MM-DD)')
    parser.add_argument('--clouds-th', type=float, default=65, help='Cloud threshold')
    parser.add_argument('--scale', type=int, default=30, help='Output resolution in meters')
    parser.add_argument('--mask-type', type=str, default='NDVI',
                       choices=['DW', 'FNF', 'NDVI', 'WC', 'CHM', 'ALL', 'none'],
                       help='Type of forest mask to apply')
    # resample method
    parser.add_argument('--resample', type=str, default='bilinear', choices=['bilinear', 'bicubic'],
                       help='Resampling method for image export')
    # ndvi threshold
    parser.add_argument('--ndvi-threshold', type=float, default=0.3, help='NDVI threshold for forest mask')
    
    # GEDI parameters
    parser.add_argument('--gedi-start-date', type=str, help='GEDI start date (YYYY-MM-DD)')
    parser.add_argument('--gedi-end-date', type=str, help='GEDI end date (YYYY-MM-DD)')
    parser.add_argument('--quantile', type=str, default='098', help='GEDI height quantile')
    parser.add_argument('--gedi-type', type=str, default='singleGEDI', help='GEDI data type')
    # Add buffer for AOI with default value 1000m
    parser.add_argument('--buffer', type=int, default=1000, help='Buffer size in meters')

    # Model parameters
    parser.add_argument('--model', type=str, default='RF', choices=['RF', 'GBM', 'CART'],
                       help='Machine learning model type')
    parser.add_argument('--num-trees-rf', type=int, default=100,
                       help='Number of trees for Random Forest')
    parser.add_argument('--min-leaf-pop-rf', type=int, default=1,
                       help='Minimum leaf population for Random Forest')
    parser.add_argument('--bag-frac-rf', type=float, default=0.5,
                       help='Bagging fraction for Random Forest')
    parser.add_argument('--max-nodes-rf', type=int, default=None,
                       help='Maximum nodes for Random Forest')
    
    # Output parameters
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Output directory for CSV and TIF files')
    parser.add_argument('--export-training', action='store_true',
                       help='Export training data as CSV')
    # Export merged stack image as TIF
    parser.add_argument('--export-stack', action='store_true',
                       help='Export merged stack image as TIF')
    # Export forest mask as TIF
    parser.add_argument('--export-forest-mask', action='store_true',
                       help='Export forest mask as TIF')
    parser.add_argument('--export-predictions', action='store_true',
                       help='Export predictions as TIF')
    
    args = parser.parse_args()
    return args

def initialize_ee():
    """Initialize Earth Engine with project ID."""
    EE_PROJECT_ID = "my-project-423921"
    ee.Initialize(project=EE_PROJECT_ID)

def export_training_data(reference_data: ee.FeatureCollection, output_dir: str):
    """Export training data as CSV."""
    # Get feature properties as a list of dictionaries
    features = reference_data.getInfo()['features']
    
    # Extract properties and coordinates
    data = []
    for feature in features:
        properties = feature['properties']
        geometry = feature['geometry']['coordinates']
        properties['longitude'] = geometry[0]
        properties['latitude'] = geometry[1]
        data.append(properties)
    
    # Convert to DataFrame and save
    df = pd.DataFrame(data)
    band_length = len(df.columns) - 3  # Exclude 'rh', 'longitude' and 'latitude'
    df_size = len(df)
    output_path = os.path.join(output_dir, f"training_data_b{band_length}_{df_size}.csv")
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Training data exported to: {output_path}")
    
    return output_path

def export_featurecollection_to_csv(feature_collection, export_name):
    """Export a FeatureCollection to CSV via Earth Engine's batch export.
    
    Args:
        feature_collection: The ee.FeatureCollection to export
        export_name: Name for the exported file
    """
    # Add longitude and latitude columns
    feature_collection = feature_collection.map(lambda feature: 
        feature.set({
            'longitude': feature.geometry().coordinates().get(0),
            'latitude': feature.geometry().coordinates().get(1)
        })
    )
    
    # Get property names and convert to Python list
    property_names = feature_collection.first().propertyNames().getInfo()
    
    # Remove system:index from the list if present
    if 'system:index' in property_names:
        property_names.remove('system:index')
        
    # Set up export task
    export_task = ee.batch.Export.table.toDrive(
        collection=feature_collection,
        description=export_name,
        fileNamePrefix=export_name,
        folder='GEE_exports',  # Folder in your Google Drive
        fileFormat='CSV',
        selectors=property_names  # All property names
    )
    
    # Start the export
    export_task.start()
    
    print(f"Export started with task ID: {export_task.id}")
    print("The CSV file will be available in your Google Drive once the export completes.")

def export_tif_via_ee(image: ee.Image, aoi: ee.Geometry, prefix: str, scale: int, resample: str = 'bilinear'):  
    """Export predicted canopy height map as GeoTIFF using Earth Engine export."""
    # Rename the classification band for clarity
    # if 'classification' in image.bandNames().getInfo():
    #     image = image.select(['classification'], ['canopy_height'])
    band_count = image.bandNames().size().getInfo()
    
    pixel_area = ee.Image.pixelArea().divide(10000)  # Convert to hectares
    area_img = ee.Image(1).rename('area').multiply(pixel_area)
    
    # Calculate total area in hectares
    image_area_ha = area_img.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=aoi,
        scale=scale,
        maxPixels=1e10
    ).get('area')
    
    image_area_ha = int(round(image_area_ha.getInfo(), 0))
    
    # Generate a unique task ID (sanitize prefix and ensure valid characters)
    clean_prefix = ''.join(c for c in prefix if c.isalnum() or c in '_-')
    task_id = f"{clean_prefix}_b{band_count}_s{scale}_p{image_area_ha}"
    
    # image = image.resample(resample)  # or 'bicubic'
    
    # Set export parameters
    export_params = {
        'image': image,
        'description': task_id,
        'fileNamePrefix': task_id,
        'folder': 'GEE_exports',
        'scale': scale,
        'region': aoi,
        'fileFormat': 'GeoTIFF',
        'maxPixels': 1e10
    }
    
    # Start the export task
    task = ee.batch.Export.image.toDrive(**export_params)
    task.start()
    
    print(f"Export started with task ID: {task_id}")
    print("The file will be available in your Google Drive once the export completes.")
    
def main():
    """Main function to run the canopy height mapping process."""
    # Parse arguments
    args = parse_args()
    
    # Initialize Earth Engine
    initialize_ee()
    
    # Load AOI
    original_aoi = load_aoi(args.aoi)
    aoi_buffered  = original_aoi.buffer(args.buffer)
    
    # Set dates
    start_date = f"{args.year}-{args.start_date}"
    end_date = f"{args.year}-{args.end_date}"
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get S1, S2 satellite data
    print("Collecting satellite data...")
    s1 = get_sentinel1_data(aoi_buffered, args.year, args.start_date, args.end_date)
    s2 = get_sentinel2_data(aoi_buffered, args.year, args.start_date, args.end_date, args.clouds_th)
    # Import ALOS2 sar data
    alos2 = get_alos2_data(aoi_buffered, args.year, args.start_date, args.end_date,include_texture=False,
                speckle_filter=False)
    # Get terrain data
    dem_data = get_dem_data(aoi_buffered)
    
    # Canopy height data
    canopy_ht = get_canopyht_data(aoi_buffered)
    
    # Reproject datasets to the same projection
    s2_projection = s2.projection()
    s2 = s2.float()                                       # Convert to Float32
    dem_data = dem_data.float()  # Convert to Float32
    s1 = s1.float()              # Convert to Float32
    alos2 = alos2.float()        # Convert to Float32
    canopy_ht = canopy_ht.float()  # Convert to Float32

    # Merge datasets
    merged = s2.addBands(s1).addBands(alos2).addBands(dem_data).addBands(canopy_ht)
        
    # Get predictor names before any masking
    print("Getting band information...")
    predictor_names = merged.bandNames()
    n_predictors = predictor_names.size().getInfo()
    var_split_rf = int(np.sqrt(n_predictors).round())
        
    # Create and apply forest mask
    print(f"Creating and applying forest mask (type: {args.mask_type})...")
    buffered_forest_mask = create_forest_mask(args.mask_type, aoi_buffered,
                                   ee.Date(f"{args.year}-{args.start_date}"),
                                   ee.Date(f"{args.year}-{args.end_date}"),
                                   args.ndvi_threshold)

    forest_mask = buffered_forest_mask.clip(original_aoi)
    
    ndvi_threshold_percent = int(round(args.ndvi_threshold * 100,0))
    
    # Get GEDI data
    print("Loading GEDI data...")
    gedi = get_gedi_data(aoi_buffered, args.gedi_start_date, args.gedi_end_date, args.quantile)
    
    # forest_geometry = forest_mask.geometry()
    # Sample GEDI points
    gedi_points = gedi.sample(
        region=aoi_buffered,
        scale=args.scale,
        geometries=True,
        dropNulls=True,
        seed=42
    )
    
        # Sample points
    reference_data = merged.sampleRegions(
        collection=gedi_points,
        scale=args.scale,
        projection=s2_projection,
        tileScale=1,
        geometries=True
    )
    
    # Clip to original AOI
    merged = merged.clip(original_aoi)
    
    # Export training data if requested
    if args.export_training:
        print('Exporting training data and tif through Earth Engine...')
        training_prefix = f'training_data_{args.mask_type}{ndvi_threshold_percent}_b{n_predictors}'
        # export_training_data_via_ee(reference_data, training_prefix)
        export_featurecollection_to_csv(reference_data, training_prefix)
        print(f"Exporting training data as CSV: {training_prefix}.csv")
    
    if args.export_stack:
        # Export the complete data stack
        print("Exporting full data stack...")
        export_tif_via_ee(merged, original_aoi, 'stack', args.scale, args.resample)
        
    if args.export_forest_mask:
        # Export forest mask using export_tif_via_ee
        forest_mask_prefix = f'forestMask{args.mask_type}{ndvi_threshold_percent}'
        buffered_forest_mask_prefix = f'buffered_forestMask{args.mask_type}{ndvi_threshold_percent}'
        # forest_mask_path = os.path.join(args.output_dir, f'{forest_mask_filename}.tif')
        print(f"Exporting forest mask as {forest_mask_prefix}...")
        export_tif_via_ee(forest_mask, original_aoi, forest_mask_prefix, args.scale, args.resample)
        print(f"Exporting buffered forest mask as {buffered_forest_mask_prefix}...")
        export_tif_via_ee(buffered_forest_mask, aoi_buffered, buffered_forest_mask_prefix, args.scale, args.resample)

    # Export predictions if requested
    if args.export_predictions:
        # Train model
        print("Training model...")
        if args.model == "RF":
            classifier = ee.Classifier.smileRandomForest(
                numberOfTrees=args.num_trees_rf,
                variablesPerSplit=var_split_rf,
                minLeafPopulation=args.min_leaf_pop_rf,
                bagFraction=args.bag_frac_rf,
                maxNodes=args.max_nodes_rf
            ).setOutputMode("Regression") \
            .train(reference_data, "rh", predictor_names)
        
            # Generate predictions
        print("Generating predictions...")
        predictions = merged.classify(classifier)
        # prediction_path = os.path.join(args.output_dir, 'predictions.tif')
        print('Exporting via Earth Engine instead')
        export_tif_via_ee(predictions, original_aoi, 'predictionCHM', args.scale, args.resample)
    
    print("Processing complete.")

if __name__ == "__main__":
    main()
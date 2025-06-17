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
import math

# Import custom functions
from l2a_gedi_source import get_gedi_data
from sentinel1_source import get_sentinel1_data
from sentinel2_source import get_sentinel2_data
from for_forest_masking import apply_forest_mask, create_forest_mask
from alos2_source import get_alos2_data
from canopyht_source import get_canopyht_data
from dem_source import get_dem_data

# Import patch-related functions
from data.image_patches import (
    create_3d_patches, prepare_training_patches,
    create_patch_grid, Patch
)
from data.large_area import collect_area_patches, merge_patch_predictions
from config.resolution_config import get_patch_size, RESOLUTION_CONFIG

def get_local_projection(lon, lat):
    """Return a suitable projection string for the AOI center."""
    if -80 <= lat <= 84:
        utm_zone = int((lon + 180) / 6) + 1
        hemisphere = 'N' if lat >= 0 else 'S'
        if hemisphere == 'N':
            return f'EPSG:326{utm_zone:02d}'
        else:
            return f'EPSG:327{utm_zone:02d}'
    else:
        return 'EPSG:3857'  # Web Mercator

def create_patches_from_ee_image(image: ee.Image, aoi: ee.Geometry, patch_size: int, scale: int, overlap: float = 0.1) -> List[Dict]:
    """
    Create patches from Earth Engine image.
    
    Args:
        image: Earth Engine image to create patches from
        aoi: Area of interest geometry
        patch_size: Size of patches in meters
        scale: Resolution in meters
        overlap: Overlap between patches as a fraction (0.0 to 1.0)
        
    Returns:
        List of patch dictionaries with coordinates and data
    """
    image = image.toFloat()
    center = aoi.centroid().coordinates().getInfo()
    lon, lat = center
    proj = get_local_projection(lon, lat)
    # Try to project AOI, fallback to EPSG:3857, then EPSG:4326
    try:
        aoi_proj = aoi.transform(proj, 1)
        bounds = aoi_proj.bounds(1).getInfo()
    except Exception as e:
        print(f"Failed to transform AOI to {proj}: {e}. Trying EPSG:3857.")
        try:
            proj = 'EPSG:3857'
            aoi_proj = aoi.transform(proj, 1)
            bounds = aoi_proj.bounds(1).getInfo()
        except Exception as e2:
            print(f"Failed to transform AOI to EPSG:3857: {e2}. Using EPSG:4326.")
            proj = 'EPSG:4326'
            aoi_proj = aoi.transform(proj, 1)
            bounds = aoi_proj.bounds(1).getInfo()
    min_x, min_y = bounds['coordinates'][0][0]
    max_x, max_y = bounds['coordinates'][0][2]
    stride = int(patch_size * (1 - overlap))
    width = max_x - min_x
    height = max_y - min_y
    n_patches_x = max(1, int(np.ceil(width / stride)))
    n_patches_y = max(1, int(np.ceil(height / stride)))
    print(f"Creating {n_patches_x * n_patches_y} patches with size {patch_size}m and {overlap*100}% overlap")
    print(f"Using projection: {proj}")
    patches = []
    for i in range(n_patches_x):
        for j in range(n_patches_y):
            x = min_x + i * stride
            y = min_y + j * stride
            try:
                patch_geom_proj = ee.Geometry.Rectangle([
                    [x, y],
                    [min(x + patch_size, max_x), min(y + patch_size, max_y)]
                ], proj, False)
                patch_geom = patch_geom_proj.transform('EPSG:4326', 1)
            except Exception as e:
                print(f"Failed to create/transform patch geometry at ({x},{y}): {e}. Skipping patch.")
                continue
            is_extruded = (
                x + patch_size > max_x or
                y + patch_size > max_y
            )
            actual_width = min(patch_size, max_x - x)
            actual_height = min(patch_size, max_y - y)
            patch_data = image.clip(patch_geom).toFloat()
            patches.append({
                'geometry': patch_geom,
                'geometry_proj': patch_geom_proj,
                'x': x,
                'y': y,
                'width': patch_size if not is_extruded else actual_width,
                'height': patch_size if not is_extruded else actual_height,
                'data': patch_data,
                'is_extruded': is_extruded,
                'projection': proj
            })
    return patches

def export_patches_to_tif(patches: List[Dict], output_dir: str, prefix: str, scale: int):
    """
    Export patches to GeoTIFF files.
    
    Args:
        patches: List of patch dictionaries
        output_dir: Output directory
        prefix: Prefix for output files
        scale: Resolution in meters
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i, patch in enumerate(patches):
        # Generate patch filename
        patch_filename = f"{prefix}_patch_{i:04d}.tif"
        patch_path = os.path.join(output_dir, patch_filename)
        
        # Export patch
        export_tif_via_ee(
            image=patch['data'],
            aoi=patch['geometry'],
            prefix=patch_filename,
            scale=scale
        )

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
    parser.add_argument('--gedi-start-date', type=str, default='2020-01-01', help='GEDI start date (YYYY-MM-DD)')
    parser.add_argument('--gedi-end-date', type=str, default='2020-12-31', help='GEDI end date (YYYY-MM-DD)')
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
    
    # Patch processing parameters
    parser.add_argument('--use-patches', action='store_true',
                       help='Enable patch-based processing')
    parser.add_argument('--patch-size', type=int, default=None,
                       help='Size of patches in meters (default: calculated from scale)')
    parser.add_argument('--patch-overlap', type=float, default=0.0,
                       help='Overlap between patches (0.0 to 1.0)')
    parser.add_argument('--export-patches', action='store_true',
                       help='Export individual patches as TIF files')
    
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
    
def create_patch_geometries(aoi: ee.Geometry, patch_size: int, overlap: float = 0.1) -> list:
    """Split AOI into patch geometries in meters using local projection, then transform to WGS84. If AOI is smaller than patch_size, buffer it to reach minimum patch_size. If projection fails, calculate patch grid in degrees using an approximate conversion."""
    # Print AOI bounds in degrees (WGS84)
    try:
        wgs84_bounds = aoi.bounds().getInfo()
        wgs84_min_x, wgs84_min_y = wgs84_bounds['coordinates'][0][0]
        wgs84_max_x, wgs84_max_y = wgs84_bounds['coordinates'][0][2]
        print(f"[DEBUG] AOI bounds in WGS84: min_x={wgs84_min_x}, max_x={wgs84_max_x}, min_y={wgs84_min_y}, max_y={wgs84_max_y}")
    except Exception as e:
        print(f"[DEBUG] Could not get AOI bounds in WGS84: {e}")
    center = aoi.centroid().coordinates().getInfo()
    lon, lat = center
    # Choose projection
    if -80 <= lat <= 84:
        utm_zone = int((lon + 180) / 6) + 1
        hemisphere = 'N' if lat >= 0 else 'S'
        if hemisphere == 'N':
            proj = f'EPSG:326{utm_zone:02d}'
        else:
            proj = f'EPSG:327{utm_zone:02d}'
    else:
        proj = 'EPSG:3857'
    # Project AOI
    try:
        aoi_proj = aoi.transform(proj, 1)
        bounds = aoi_proj.bounds(1).getInfo()
        min_x, min_y = bounds['coordinates'][0][0]
        max_x, max_y = bounds['coordinates'][0][2]
        width = max_x - min_x
        height = max_y - min_y
        print(f"[DEBUG] AOI bounds in {proj}: min_x={min_x}, max_x={max_x}, min_y={min_y}, max_y={max_y}")
        print(f"[DEBUG] AOI width: {width} meters, height: {height} meters, patch_size: {patch_size} meters")
        if width < 1 or height < 1:
            print(f"[WARNING] Projected AOI width or height is less than 1 meter. Falling back to patch grid in degrees.")
            raise ValueError("Projected AOI too small, fallback to degrees.")
        # Buffer if AOI is smaller than patch_size
        buffer_x = max(0, (patch_size - width) / 2)
        buffer_y = max(0, (patch_size - height) / 2)
        if buffer_x > 0 or buffer_y > 0:
            buffer_amount = max(buffer_x, buffer_y)
            print(f"AOI is smaller than patch_size. Buffering AOI by {buffer_amount:.2f} meters (projected).")
            aoi_proj = aoi_proj.buffer(buffer_amount)
            bounds = aoi_proj.bounds(1).getInfo()
            min_x, min_y = bounds['coordinates'][0][0]
            max_x, max_y = bounds['coordinates'][0][2]
            width = max_x - min_x
            height = max_y - min_y
            print(f"[DEBUG] After buffering: min_x={min_x}, max_x={max_x}, min_y={min_y}, max_y={max_y}")
            print(f"[DEBUG] After buffering: width={width} meters, height={height} meters")
        stride = patch_size * (1 - overlap)
        n_patches_x = max(1, int(math.ceil(width / stride)))
        n_patches_y = max(1, int(math.ceil(height / stride)))
        patch_geoms = []
        for i in range(n_patches_x):
            for j in range(n_patches_y):
                x = min_x + i * stride
                y = min_y + j * stride
                patch_geom_proj = ee.Geometry.Rectangle([
                    [x, y],
                    [min(x + patch_size, max_x), min(y + patch_size, max_y)]
                ], proj, False)
                patch_geom = patch_geom_proj.transform('EPSG:4326', 1)
                patch_geoms.append(patch_geom)
        print(f"Created {len(patch_geoms)} patch geometries.")
        return patch_geoms
    except Exception as e:
        print(f"Projection failed or fallback triggered: {e}, using WGS84 and patch grid in degrees.")
        proj = 'EPSG:4326'
        # Approximate conversion: 1 degree ≈ 111320 meters at equator
        patch_size_deg = patch_size / 111320.0
        stride_deg = patch_size_deg * (1 - overlap)
        width = wgs84_max_x - wgs84_min_x
        height = wgs84_max_y - wgs84_min_y
        print(f"[DEBUG] AOI width in degrees: {width}, height in degrees: {height}, patch_size in degrees: {patch_size_deg}")
        buffer_x = max(0, (patch_size_deg - width) / 2)
        buffer_y = max(0, (patch_size_deg - height) / 2)
        if buffer_x > 0 or buffer_y > 0:
            buffer_amount_deg = max(buffer_x, buffer_y)
            print(f"AOI is smaller than patch_size. Buffering AOI by {buffer_amount_deg:.6f} degrees (approximate, fallback).")
            aoi = aoi.buffer(buffer_amount_deg)
            wgs84_bounds = aoi.bounds().getInfo()
            wgs84_min_x, wgs84_min_y = wgs84_bounds['coordinates'][0][0]
            wgs84_max_x, wgs84_max_y = wgs84_bounds['coordinates'][0][2]
            width = wgs84_max_x - wgs84_min_x
            height = wgs84_max_y - wgs84_min_y
            print(f"[DEBUG] After buffering: min_x={wgs84_min_x}, max_x={wgs84_max_x}, min_y={wgs84_min_y}, max_y={wgs84_max_y}")
            print(f"[DEBUG] After buffering: width={width} degrees, height={height} degrees")
        n_patches_x = max(1, int(math.ceil(width / stride_deg)))
        n_patches_y = max(1, int(math.ceil(height / stride_deg)))
        patch_geoms = []
        for i in range(n_patches_x):
            for j in range(n_patches_y):
                x = wgs84_min_x + i * stride_deg
                y = wgs84_min_y + j * stride_deg
                patch_geom = ee.Geometry.Rectangle([
                    [x, y],
                    [min(x + patch_size_deg, wgs84_max_x), min(y + patch_size_deg, wgs84_max_y)]
                ], proj, False)
                patch_geoms.append(patch_geom)
        print(f"Created {len(patch_geoms)} patch geometries (in degrees).")
        return patch_geoms

def main():
    """Main function to run the canopy height mapping process."""
    args = parse_args()
    initialize_ee()
    original_aoi = load_aoi(args.aoi)
    aoi_buffered = original_aoi.buffer(args.buffer).transform('EPSG:4326', 1) if args.buffer > 0 else original_aoi
    start_date = f"{args.year}-{args.start_date}"
    end_date = f"{args.year}-{args.end_date}"
    os.makedirs(args.output_dir, exist_ok=True)

    if args.use_patches:
        print(f"Creating patch geometries with size {args.patch_size}m and {args.patch_overlap*100}% overlap...")
        patch_geoms = create_patch_geometries(aoi_buffered, args.patch_size, args.patch_overlap)
        for i, patch_geom in enumerate(patch_geoms):
            patch_id = f"patch{i:04d}"
            print(f"Processing {patch_id}")
            # Collect satellite data for this patch
            s1 = get_sentinel1_data(patch_geom, args.year, args.start_date, args.end_date).toFloat()
            s2 = get_sentinel2_data(patch_geom, args.year, args.start_date, args.end_date, args.clouds_th).toFloat()
            alos2 = get_alos2_data(patch_geom, args.year, args.start_date, args.end_date, include_texture=False, speckle_filter=False).toFloat()
            # Only add ALOS2 bands that exist
            alos2_band_names = alos2.bandNames().getInfo()
            patch_data = s1.addBands(s2)
            if 'ALOS2_HH' in alos2_band_names:
                patch_data = patch_data.addBands(alos2.select('ALOS2_HH'))
            if 'ALOS2_HV' in alos2_band_names:
                patch_data = patch_data.addBands(alos2.select('ALOS2_HV'))
            dem_data = get_dem_data(patch_geom).toFloat()
            canopy_ht = get_canopyht_data(patch_geom).toFloat()
            forest_mask = create_forest_mask(
                args.mask_type,
                patch_geom,
                ee.Date(start_date),
                ee.Date(end_date),
                args.ndvi_threshold
            ).toFloat()
            patch_data = patch_data.addBands(dem_data).addBands(canopy_ht)
            # Add forest mask as a band instead of using updateMask
            patch_data = patch_data.addBands(forest_mask.rename('forest_mask'))
            # Get GEDI data
            print("Loading GEDI data...")
            gedi = get_gedi_data(patch_geom, args.gedi_start_date, args.gedi_end_date, args.quantile)
            
            # Check GEDI data availability
            print("\nChecking GEDI data availability...")
            # Get a sample of the data to check values
            gedi_sample = gedi.select('rh').sample(
                region=patch_geom,
                scale=args.scale,
                numPixels=1000,
                seed=42
            ).getInfo()
            
            # Count non-null values
            valid_values = [f['properties']['rh'] for f in gedi_sample['features'] if f['properties']['rh'] is not None]
            print(f"Number of valid GEDI measurements in sample: {len(valid_values)}")
            if valid_values:
                print(f"GEDI height range: {min(valid_values):.2f} to {max(valid_values):.2f} meters")
            else:
                print("WARNING: No valid GEDI measurements found in the area!")
                print("This could be due to:")
                print("1. No GEDI coverage in this area")
                print("2. Quality filters being too strict")
                print("3. Time period not having data")
            
            # Convert GEDI points to raster at specified scale and ensure Float32 type
            print("\nConverting GEDI points to raster...")
            gedi_raster = gedi.select('rh').toFloat()# .rename('gedi_rh')
            
            # Add GEDI raster to patch data
            patch_gedi = gedi_raster.clip(patch_geom)
            patch_data = patch_data.addBands(patch_gedi)
            
            band_count = patch_data.bandNames().length()
            # Get geojson file name (without extension)
            geojson_name = os.path.splitext(os.path.basename(args.aoi))[0]
            # Compose fileNamePrefix as requested
            fileNamePrefix = f"{geojson_name}_bandNum{{}}_scale{args.scale}_{patch_id}".format(band_count.getInfo() if hasattr(band_count, 'getInfo') else band_count)
            if args.export_patches:
                export_task = ee.batch.Export.image.toDrive(
                    image=patch_data,
                    description=f"{patch_id}_data",
                    fileNamePrefix=fileNamePrefix,
                    folder='GEE_exports',
                    scale=args.scale,
                    region=patch_geom,
                    fileFormat='GeoTIFF',
                    maxPixels=1e13,
                    crs='EPSG:4326'
                )
                export_task.start()
                print(f"Started export for {patch_id} with fileNamePrefix: {fileNamePrefix}")
                break  # Only export the first patch for now (testing)
    else:
        # Fallback: process whole AOI as before
        print("Processing whole AOI (not patch-based)...")
        s1 = get_sentinel1_data(aoi_buffered, args.year, args.start_date, args.end_date).toFloat()
        s2 = get_sentinel2_data(aoi_buffered, args.year, args.start_date, args.end_date, args.clouds_th).toFloat()
        alos2 = get_alos2_data(aoi_buffered, args.year, args.start_date, args.end_date, include_texture=False, speckle_filter=False).toFloat()
        # Only add ALOS2 bands that exist
        alos2_band_names = alos2.bandNames().getInfo()
        data = s1.addBands(s2).addBands(alos2)
        dem_data = get_dem_data(aoi_buffered).toFloat()
        canopy_ht = get_canopyht_data(aoi_buffered).toFloat()
        forest_mask = create_forest_mask(
            args.mask_type,
            aoi_buffered,
            ee.Date(start_date),
            ee.Date(end_date),
            args.ndvi_threshold
        ).toFloat()
        data = data.addBands(dem_data).addBands(canopy_ht)
        data = data.updateMask(forest_mask)
        # Get GEDI data
        print("Loading GEDI data...")
        gedi = get_gedi_data(aoi_buffered, args.gedi_start_date, args.gedi_end_date, args.quantile)
        
        # Check GEDI data availability
        print("\nChecking GEDI data availability...")
        # Get a sample of the data to check values
        gedi_sample = gedi.select('rh').sample(
            region=aoi_buffered,
            scale=args.scale,
            numPixels=1000,
            seed=42
        ).getInfo()
        
        # Count non-null values
        valid_values = [f['properties']['rh'] for f in gedi_sample['features'] if f['properties']['rh'] is not None]
        print(f"Number of valid GEDI measurements in sample: {len(valid_values)}")
        if valid_values:
            print(f"GEDI height range: {min(valid_values):.2f} to {max(valid_values):.2f} meters")
        else:
            print("WARNING: No valid GEDI measurements found in the area!")
            print("This could be due to:")
            print("1. No GEDI coverage in this area")
            print("2. Quality filters being too strict")
            print("3. Time period not having data")
        
        # Convert GEDI points to raster at specified scale and ensure Float32 type
        print("\nConverting GEDI points to raster...")
        gedi_raster = gedi.select('rh').toFloat()#.rename('gedi_rh')
                # Add GEDI raster to patch data
        patch_gedi = gedi_raster.clip(aoi_buffered)
        data = data.addBands(patch_gedi)
        # Export whole AOI if requested
        if args.export_patches:
            export_task = ee.batch.Export.image.toDrive(
                image=data,
                description="aoi_data",
                fileNamePrefix=os.path.join(args.output_dir, 'aoi'),
                folder='GEE_exports',
                scale=args.scale,
                region=aoi_buffered,
                fileFormat='GeoTIFF',
                maxPixels=1e13,
                crs='EPSG:4326'
            )
            export_task.start()
            print("Started export for AOI")

if __name__ == "__main__":
    main()
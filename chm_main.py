import ee
# import geemap  # Only import when needed to avoid dependency issues
import numpy as np
import json
import os
from pathlib import Path
# import pandas as pd  # Only import when needed
import datetime
import argparse
from typing import Union, List, Dict, Any
import time
# from tqdm import tqdm  # Only import when needed
# import rasterio  # Only import when needed
# from rasterio.transform import from_origin
# from rasterio.crs import CRS
import math

# Import custom functions
from l2a_gedi_source import get_gedi_data, export_gedi_points, get_gedi_vector_data, export_gedi_vector_points
from sentinel1_source import get_sentinel1_data
from sentinel2_source import get_sentinel2_data
from for_forest_masking import apply_forest_mask, create_forest_mask
from alos2_source import get_alos2_data
from canopyht_source import get_canopyht_data
from dem_source import get_dem_data

# Import patch-related functions (commented out for minimal testing)
# from data.image_patches import (
#     create_3d_patches, prepare_training_patches,
#     create_patch_grid, Patch
# )
# from data.large_area import collect_area_patches, merge_patch_predictions
# from config.resolution_config import get_patch_size, RESOLUTION_CONFIG

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
    # Temporal compositing parameters (Paul's 2025 methodology)
    parser.add_argument('--temporal-mode', action='store_true',
                       help='Enable Paul\'s 2025 temporal compositing (12-monthly data for S1/S2/ALOS2)')
    parser.add_argument('--monthly-composite', type=str, default='median', choices=['median', 'mean'],
                       help='Monthly composite method for temporal mode')
    # resample method
    parser.add_argument('--resample', type=str, default='bilinear', choices=['bilinear', 'bicubic'],
                       help='Resampling method for image export')
    # ndvi threshold
    parser.add_argument('--ndvi-threshold', type=float, default=0.3, help='NDVI threshold for forest mask')
    
    # GEDI parameters
    parser.add_argument('--gedi-start-date', type=str, default='2020-01-01', help='GEDI start date (YYYY-MM-DD)')
    parser.add_argument('--gedi-end-date', type=str, default='2020-12-31', help='GEDI end date (YYYY-MM-DD)')
    parser.add_argument('--quantile', type=str, default='rh98', help='GEDI height quantile')
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
    parser.add_argument('--patch-overlap', type=int, default=10,
                       help='Overlap between patches in pixels (default: 10)')
    parser.add_argument('--export-patches', action='store_true',
                       help='Export individual patches as TIF files')
    
    args = parser.parse_args()
    return args

def initialize_ee():
    """Initialize Earth Engine with project ID."""
    EE_PROJECT_ID = "my-project-423921"
    
    try:
        # Try to initialize with existing credentials
        ee.Initialize(project=EE_PROJECT_ID)
        print("✅ Earth Engine initialized successfully")
    except ee.EEException as e:
        if "Please authorize access" in str(e):
            print("🔐 Earth Engine authentication required...")
            print("Please run one of the following:")
            print("  1. Command line: earthengine authenticate")
            print("  2. Python: ee.Authenticate()")
            print("\nTrying automatic authentication...")
            try:
                ee.Authenticate()
                ee.Initialize(project=EE_PROJECT_ID)
                print("✅ Earth Engine authenticated and initialized successfully")
            except Exception as auth_error:
                print(f"❌ Authentication failed: {auth_error}")
                print("\nPlease run manually: earthengine authenticate")
                raise
        else:
            print(f"❌ Earth Engine initialization failed: {e}")
            raise

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
    
def create_pixel_aligned_patches(aoi: ee.Geometry, patch_pixels: int, scale: int, overlap_pixels: int = 10) -> list:
    """
    Create patches with exact pixel dimensions aligned to pixel grid, working in WGS84.
    
    Args:
        aoi: Area of interest geometry
        patch_pixels: Patch size in pixels (e.g., 256)
        scale: Resolution in meters (e.g., 10)
        overlap_pixels: Overlap between patches in pixels (e.g., 10)
    
    Returns:
        List of patch dictionaries with exact pixel dimensions
    """
    # Convert patch size to meters
    patch_size_meters = patch_pixels * scale
    
    print(f"[DEBUG] Creating pixel-aligned patches: {patch_pixels}×{patch_pixels} pixels ({patch_size_meters}×{patch_size_meters}m) at {scale}m resolution")
    
    try:
        # Work directly with WGS84 bounds to avoid projection issues
        wgs84_bounds = aoi.bounds().getInfo()
        wgs84_coords = wgs84_bounds['coordinates'][0]
        wgs84_lons = [coord[0] for coord in wgs84_coords]
        wgs84_lats = [coord[1] for coord in wgs84_coords]
        wgs84_min_lon, wgs84_max_lon = min(wgs84_lons), max(wgs84_lons)
        wgs84_min_lat, wgs84_max_lat = min(wgs84_lats), max(wgs84_lats)
        
        print(f"[DEBUG] AOI bounds in WGS84: lon {wgs84_min_lon:.6f} to {wgs84_max_lon:.6f}, lat {wgs84_min_lat:.6f} to {wgs84_max_lat:.6f}")
        
        # Convert WGS84 degrees to approximate meters for patch grid calculation
        # For Japan region: 1 degree lon ≈ 91200m, 1 degree lat ≈ 111320m
        lat_center = (wgs84_min_lat + wgs84_max_lat) / 2
        lon_scale = 111320 * math.cos(math.radians(lat_center))  
        lat_scale = 111320
        
        # Convert to approximate meters for grid calculation
        min_x_m = wgs84_min_lon * lon_scale
        max_x_m = wgs84_max_lon * lon_scale  
        min_y_m = wgs84_min_lat * lat_scale
        max_y_m = wgs84_max_lat * lat_scale
        
        print(f"[DEBUG] Approximated bounds in meters: x={min_x_m:.2f} to {max_x_m:.2f}, y={min_y_m:.2f} to {max_y_m:.2f}")
        print(f"[DEBUG] Area dimensions: {max_x_m-min_x_m:.2f}m × {max_y_m-min_y_m:.2f}m")
        
        # Calculate stride (spacing between patch origins)
        overlap_meters = overlap_pixels * scale
        stride = patch_size_meters - overlap_meters
        
        # Calculate number of patches needed to cover the area
        area_width = max_x_m - min_x_m
        area_height = max_y_m - min_y_m
        n_patches_x = max(1, math.ceil(area_width / stride))
        n_patches_y = max(1, math.ceil(area_height / stride))
        
        print(f"[DEBUG] Grid coverage: {n_patches_x}×{n_patches_y} patches with {stride:.2f}m stride ({overlap_pixels} pixel overlap)")
        
        patches = []
        patch_count = 0
        
        for i in range(n_patches_x):
            for j in range(n_patches_y):
                # Calculate patch center in meters
                patch_center_x_m = min_x_m + (i + 0.5) * stride
                patch_center_y_m = min_y_m + (j + 0.5) * stride
                
                # Convert back to WGS84 degrees for patch center
                patch_center_lon = patch_center_x_m / lon_scale
                patch_center_lat = patch_center_y_m / lat_scale
                
                # Calculate patch half-size in degrees
                half_size_lon = (patch_size_meters / 2) / lon_scale
                half_size_lat = (patch_size_meters / 2) / lat_scale
                
                # Create patch rectangle in WGS84
                patch_min_lon = patch_center_lon - half_size_lon
                patch_max_lon = patch_center_lon + half_size_lon
                patch_min_lat = patch_center_lat - half_size_lat
                patch_max_lat = patch_center_lat + half_size_lat
                
                # Create patch geometry
                patch_geom = ee.Geometry.Rectangle([
                    [patch_min_lon, patch_min_lat],
                    [patch_max_lon, patch_max_lat]
                ], 'EPSG:4326', False)
                
                # Check if patch intersects with original AOI
                try:
                    intersects = aoi.intersects(patch_geom, ee.ErrorMargin(1)).getInfo()
                    if intersects:
                        patches.append({
                            'geometry': patch_geom,
                            'geometry_proj': patch_geom,  # Same as geometry since we're working in WGS84
                            'x': patch_center_x_m,
                            'y': patch_center_y_m,
                            'width': patch_size_meters,
                            'height': patch_size_meters,
                            'pixels_x': patch_pixels,
                            'pixels_y': patch_pixels,
                            'projection': 'EPSG:4326',
                            'extends_beyond_aoi': True,
                            'patch_id': patch_count
                        })
                        patch_count += 1
                        print(f"[DEBUG] Created patch {patch_count}: center ({patch_center_lon:.4f}, {patch_center_lat:.4f})")
                except Exception as e:
                    print(f"[WARNING] Failed to check intersection for patch at ({patch_center_lon:.4f}, {patch_center_lat:.4f}): {e}")
                    continue
        
        print(f"[SUCCESS] Created {len(patches)} pixel-aligned patches ({patch_pixels}×{patch_pixels} pixels each)")
        return patches
        
    except Exception as e:
        print(f"[ERROR] Pixel-aligned patch creation failed: {e}")
        print("[FALLBACK] Using original patch creation method...")
        return create_patch_geometries_fallback(aoi, patch_size_meters, overlap_pixels)

def create_patch_geometries_fallback(aoi: ee.Geometry, patch_size: int, overlap_pixels: int = 10) -> list:
    """Fallback to original patch creation method if pixel-aligned method fails.
    
    Args:
        aoi: Area of interest geometry
        patch_size: Size of patches in meters
        overlap_pixels: Overlap between patches in pixels
    
    Returns:
        List of patch geometries
    """
    print("[FALLBACK] Using coordinate-based patch creation...")
    center = aoi.centroid().coordinates().getInfo()
    lon, lat = center
    proj = get_local_projection(lon, lat)
    
    try:
        aoi_proj = aoi.transform(proj, 1)
        bounds = aoi_proj.bounds(1).getInfo()
        min_x, min_y = bounds['coordinates'][0][0]
        max_x, max_y = bounds['coordinates'][0][2]
        width = max_x - min_x
        height = max_y - min_y
        
        # Calculate stride from pixel overlap
        # Note: For fallback, we need to estimate scale from the original function call
        # This is a limitation of the fallback approach
        overlap_meters = overlap_pixels * 10  # Assume 10m scale for fallback
        stride = patch_size - overlap_meters
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
        
        print(f"[FALLBACK] Created {len(patch_geoms)} patches using original method")
        return patch_geoms
        
    except Exception as e2:
        print(f"[ERROR] Fallback method also failed: {e2}")
        raise RuntimeError("Both pixel-aligned and fallback patch creation methods failed")

def get_temporal_sentinel1_data(aoi: ee.Geometry, year: int, composite_method: str = 'median') -> ee.Image:
    """
    Get monthly Sentinel-1 composites for Paul's 2025 temporal modeling.
    
    Args:
        aoi: Area of interest
        year: Year for analysis
        composite_method: 'median' or 'mean' for monthly composites
        
    Returns:
        ee.Image: 24-band image (12 months × 2 polarizations)
    """
    print(f"Collecting monthly Sentinel-1 data for {year}...")
    
    monthly_bands = []
    for month in range(1, 13):
        # Get month date range
        start_date = ee.Date.fromYMD(year, month, 1)
        end_date = start_date.advance(1, 'month')
        
        # Filter Sentinel-1 data for this month
        s1_monthly = ee.ImageCollection('COPERNICUS/S1_GRD') \
            .filterDate(start_date, end_date) \
            .filterBounds(aoi) \
            .filter(ee.Filter.eq('instrumentMode', 'IW')) \
            .filter(ee.Filter.eq('resolution_meters', 10)) \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
            .select(['VV', 'VH'])
        
        # Create monthly composite
        if composite_method == 'median':
            monthly_composite = s1_monthly.median()
        else:
            monthly_composite = s1_monthly.mean()
        
        # Rename bands with month suffix
        monthly_composite = monthly_composite.select(
            ['VV', 'VH'],
            [f'S1_VV_M{month:02d}', f'S1_VH_M{month:02d}']
        )
        
        monthly_bands.append(monthly_composite)
        print(f"  Month {month:02d}: S1_VV_M{month:02d}, S1_VH_M{month:02d}")
    
    # Combine all monthly composites
    temporal_s1 = monthly_bands[0]
    for monthly_img in monthly_bands[1:]:
        temporal_s1 = temporal_s1.addBands(monthly_img)
    
    temporal_s1 = temporal_s1.clip(aoi)
    print(f"Created temporal S1 with {temporal_s1.bandNames().size().getInfo()} bands")
    return temporal_s1

def get_temporal_sentinel2_data(aoi: ee.Geometry, year: int, clouds_th: float, composite_method: str = 'median') -> ee.Image:
    """
    Get monthly Sentinel-2 composites for Paul's 2025 temporal modeling.
    
    Args:
        aoi: Area of interest
        year: Year for analysis  
        clouds_th: Cloud threshold
        composite_method: 'median' or 'mean' for monthly composites
        
    Returns:
        ee.Image: 132-band image (12 months × 11 bands: 10 spectral + NDVI)
    """
    print(f"Collecting monthly Sentinel-2 data for {year}...")
    
    monthly_bands = []
    for month in range(1, 13):
        # Get month date range
        start_date = ee.Date.fromYMD(year, month, 1)
        end_date = start_date.advance(1, 'month')
        
        # Import Sentinel-2 data for this month
        s2_monthly = ee.ImageCollection('COPERNICUS/S2_HARMONIZED') \
            .filterDate(start_date, end_date) \
            .filterBounds(aoi)
        
        # Get cloud probability data
        s2_cloud_monthly = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY') \
            .filterDate(start_date, end_date) \
            .filterBounds(aoi)
        
        # Join cloud data
        s2_monthly = ee.ImageCollection(s2_monthly) \
            .map(lambda img: img.addBands(s2_cloud_monthly.filter(ee.Filter.equals('system:index', img.get('system:index'))).first()))
        
        # Cloud masking function
        def maskClouds(img):
            clouds = ee.Image(img).select('probability')
            isNotCloud = clouds.lt(clouds_th)
            return img.mask(isNotCloud)
        
        def maskEdges(s2_img):
            return s2_img.updateMask(
                s2_img.select('B8A').mask().updateMask(s2_img.select('B9').mask())
            )
        
        # Apply masking and select bands
        s2_monthly = s2_monthly.map(maskEdges).map(maskClouds)
        s2_monthly = s2_monthly.select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'])
        
        # Add NDVI
        def add_ndvi(img):
            ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI')
            return img.addBands([ndvi])
        
        s2_monthly = s2_monthly.map(add_ndvi)
        
        # Create monthly composite
        if composite_method == 'median':
            monthly_composite = s2_monthly.median()
        else:
            monthly_composite = s2_monthly.mean()
        
        # Rename bands with month suffix
        original_bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12', 'NDVI']
        new_bands = [f'{band}_M{month:02d}' for band in original_bands]
        monthly_composite = monthly_composite.select(original_bands, new_bands)
        
        monthly_bands.append(monthly_composite)
        print(f"  Month {month:02d}: 11 bands (B2-B12, NDVI)")
    
    # Combine all monthly composites
    temporal_s2 = monthly_bands[0]
    for monthly_img in monthly_bands[1:]:
        temporal_s2 = temporal_s2.addBands(monthly_img)
    
    temporal_s2 = temporal_s2.clip(aoi)
    print(f"Created temporal S2 with {temporal_s2.bandNames().size().getInfo()} bands")
    return temporal_s2

def get_temporal_alos2_data(aoi: ee.Geometry, year: int, composite_method: str = 'median') -> ee.Image:
    """
    Get monthly ALOS-2 composites for Paul's 2025 temporal modeling.
    
    Args:
        aoi: Area of interest
        year: Year for analysis
        composite_method: 'median' or 'mean' for monthly composites
        
    Returns:
        ee.Image: 24-band image (12 months × 2 polarizations)
    """
    print(f"Collecting monthly ALOS-2 data for {year}...")
    
    monthly_bands = []
    for month in range(1, 13):
        # Get month date range  
        start_date = ee.Date.fromYMD(year, month, 1)
        end_date = start_date.advance(1, 'month')
        
        # Filter ALOS-2 data for this month
        alos2_monthly = ee.ImageCollection("JAXA/ALOS/PALSAR-2/Level2_2/ScanSAR") \
            .filterDate(start_date, end_date) \
            .filterBounds(aoi)
        
        # Check if any data exists for this month
        data_count = alos2_monthly.size()
        
        # Create monthly composite or use empty bands if no data
        monthly_composite = ee.Algorithms.If(
            data_count.gt(0),
            ee.Algorithms.If(
                ee.String(composite_method).equals('median'),
                alos2_monthly.median(),
                alos2_monthly.mean()
            ),
            # Create empty image with HH, HV bands if no data available
            ee.Image.constant([0, 0]).rename(['HH', 'HV']).toFloat()
        )
        monthly_composite = ee.Image(monthly_composite)
        
        # Handle missing bands gracefully and rename
        band_names = monthly_composite.bandNames()
        
        # Check for HH and HV bands, create if missing
        hh_band = ee.Algorithms.If(
            band_names.contains('HH'),
            monthly_composite.select('HH'),
            ee.Image.constant(0).rename('HH').toFloat()
        )
        hv_band = ee.Algorithms.If(
            band_names.contains('HV'), 
            monthly_composite.select('HV'),
            ee.Image.constant(0).rename('HV').toFloat()
        )
        
        # Combine and rename with month suffix
        monthly_composite = ee.Image.cat([hh_band, hv_band]).rename([
            f'ALOS2_HH_M{month:02d}', 
            f'ALOS2_HV_M{month:02d}'
        ])
        
        monthly_bands.append(monthly_composite)
        print(f"  Month {month:02d}: ALOS2_HH_M{month:02d}, ALOS2_HV_M{month:02d}")
    
    # Combine all monthly composites
    temporal_alos2 = monthly_bands[0]
    for monthly_img in monthly_bands[1:]:
        temporal_alos2 = temporal_alos2.addBands(monthly_img)
    
    temporal_alos2 = temporal_alos2.clip(aoi)
    print(f"Created temporal ALOS2 with {temporal_alos2.bandNames().size().getInfo()} bands")
    return temporal_alos2

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
        patch_pixels = args.patch_size // args.scale if args.patch_size else 256
        print(f"Creating pixel-aligned patches: {patch_pixels}×{patch_pixels} pixels ({patch_pixels*args.scale}×{patch_pixels*args.scale}m) with {args.patch_overlap} pixel overlap...")
        patches = create_pixel_aligned_patches(aoi_buffered, patch_pixels, args.scale, args.patch_overlap)
        break_num = 0
        for i, patch in enumerate(patches):
            patch_geom = patch['geometry']
            patch_id = f"patch{i:04d}"
            print(f"Processing {patch_id} ({patch['pixels_x']}×{patch['pixels_y']} pixels)")
            
            # Collect satellite data for this patch
            if args.temporal_mode:
                print("Using Paul's 2025 temporal compositing...")
                print("Expected band counts: S1=24, S2=132, ALOS2=24, total temporal=180+other bands")
                s1 = get_temporal_sentinel1_data(patch_geom, args.year, args.monthly_composite)
                s2 = get_temporal_sentinel2_data(patch_geom, args.year, args.clouds_th, args.monthly_composite)
                alos2 = get_temporal_alos2_data(patch_geom, args.year, args.monthly_composite)
            else:
                print("Using original yearly median compositing...")
                s1 = get_sentinel1_data(patch_geom, args.year, args.start_date, args.end_date)
                s2 = get_sentinel2_data(patch_geom, args.year, args.start_date, args.end_date, args.clouds_th)
                alos2 = get_alos2_data(patch_geom, args.year, args.start_date, args.end_date, include_texture=False, speckle_filter=False)
            
            # Ensure all satellite data is converted to Float32
            if s1 is not None:
                s1 = s1.toFloat()
            if s2 is not None:
                s2 = s2.toFloat()
            if alos2 is not None:
                alos2 = alos2.toFloat()
            
            # Build patch data step by step
            patch_data = None
            if s1 is not None:
                patch_data = s1
            if s2 is not None:
                patch_data = patch_data.addBands(s2) if patch_data else s2
            
            # Add ALOS2 bands (handling temporal vs non-temporal mode)
            if alos2 is not None:
                alos2_band_names = alos2.bandNames().getInfo()
                if args.temporal_mode:
                    # In temporal mode, add all ALOS2 monthly bands
                    patch_data = patch_data.addBands(alos2) if patch_data else alos2
                else:
                    # In non-temporal mode, add only existing ALOS2 bands
                    if 'ALOS2_HH' in alos2_band_names:
                        hh_band = alos2.select('ALOS2_HH')
                        patch_data = patch_data.addBands(hh_band) if patch_data else hh_band
                    if 'ALOS2_HV' in alos2_band_names:
                        hv_band = alos2.select('ALOS2_HV')
                        patch_data = patch_data.addBands(hv_band) if patch_data else hv_band
            
            # Add DEM and canopy height data
            dem_data = get_dem_data(patch_geom)
            canopy_ht = get_canopyht_data(patch_geom)
            if dem_data is not None:
                dem_data = dem_data.toFloat()
                patch_data = patch_data.addBands(dem_data) if patch_data else dem_data
            if canopy_ht is not None:
                canopy_ht = canopy_ht.toFloat()
                patch_data = patch_data.addBands(canopy_ht) if patch_data else canopy_ht
            
            # Add forest mask
            forest_mask = create_forest_mask(
                args.mask_type,
                patch_geom,
                ee.Date(start_date),
                ee.Date(end_date),
                args.ndvi_threshold
            )
            if forest_mask is not None:
                forest_mask = forest_mask.toFloat()
                forest_mask = forest_mask.rename('forest_mask')
                patch_data = patch_data.addBands(forest_mask) if patch_data else forest_mask
            
            # Get GEDI vector data
            print("Loading GEDI vector data...")
            gedi_fc = get_gedi_vector_data(patch_geom, args.gedi_start_date, args.gedi_end_date, args.quantile)
            
            # Convert GEDI vector data to raster for the patch data
            if gedi_fc is not None:
                # Convert to raster using reduceToImage
                gedi_raster = gedi_fc.select(args.quantile).reduceToImage(
                    properties=[args.quantile],
                    reducer=ee.Reducer.first()
                ).rename('rh').toFloat()
                
                # Add GEDI raster to patch data
                patch_gedi = gedi_raster.clip(patch_geom)
                patch_data = patch_data.addBands(patch_gedi) if patch_data else patch_gedi
            
            # Ensure all bands are Float32
            if patch_data is not None:
                patch_data = patch_data.toFloat()
                
                band_count = patch_data.bandNames().length()
                # Get geojson file name (without extension)
                geojson_name = os.path.splitext(os.path.basename(args.aoi))[0]
                # Compose fileNamePrefix as requested, including temporal indicator
                temporal_suffix = "_temporal" if args.temporal_mode else ""
                fileNamePrefix = f"{geojson_name}{temporal_suffix}_bandNum{{}}_scale{args.scale}_{patch_id}".format(band_count.getInfo() if hasattr(band_count, 'getInfo') else band_count)
                
                if args.export_patches:
                    # Note: scale is implicitly determined by region size / dimensions
                    # 2560m region ÷ 256 pixels = 10m/pixel resolution
                    export_task = ee.batch.Export.image.toDrive(
                        image=patch_data,
                        description=f"{patch_id}_data",
                        fileNamePrefix=fileNamePrefix,
                        folder='GEE_exports',
                        region=patch_geom,
                        fileFormat='GeoTIFF',
                        maxPixels=1e13,
                        crs='EPSG:4326',
                        dimensions=f"{patch['pixels_x']}x{patch['pixels_y']}"
                    )
                    export_task.start()
                    print(f"Started export for {patch_id} with fileNamePrefix: {fileNamePrefix}")
                    print(f"  Patch dimensions: {patch['pixels_x']}×{patch['pixels_y']} pixels at {args.scale}m resolution")
                    # break_num += 1
                    # if break_num > 1:
                        # break # during test, apply break
            else:
                print(f"[WARNING] No data available for {patch_id}, skipping export")
    else:
        # Fallback: process whole AOI as before
        print("Processing whole AOI (not patch-based)...")
        
        if args.temporal_mode:
            print("Using Paul's 2025 temporal compositing...")
            print("Expected band counts: S1=24, S2=132, ALOS2=24, total temporal=180+other bands")
            s1 = get_temporal_sentinel1_data(aoi_buffered, args.year, args.monthly_composite).toFloat()
            s2 = get_temporal_sentinel2_data(aoi_buffered, args.year, args.clouds_th, args.monthly_composite).toFloat()
            alos2 = get_temporal_alos2_data(aoi_buffered, args.year, args.monthly_composite).toFloat()
            # In temporal mode, add all bands
            data = s1.addBands(s2).addBands(alos2)
        else:
            print("Using original yearly median compositing...")
            s1 = get_sentinel1_data(aoi_buffered, args.year, args.start_date, args.end_date).toFloat()
            s2 = get_sentinel2_data(aoi_buffered, args.year, args.start_date, args.end_date, args.clouds_th).toFloat()
            alos2 = get_alos2_data(aoi_buffered, args.year, args.start_date, args.end_date, include_texture=False, speckle_filter=False).toFloat()
            # Only add ALOS2 bands that exist
            alos2_band_names = alos2.bandNames().getInfo()
            data = s1.addBands(s2).addBands(alos2)
        dem_data = get_dem_data(aoi_buffered)
        canopy_ht = get_canopyht_data(aoi_buffered)
        forest_mask = create_forest_mask(
            args.mask_type,
            aoi_buffered,
            ee.Date(start_date),
            ee.Date(end_date),
            args.ndvi_threshold
        )
        
        # Convert to Float32 if not None
        if dem_data is not None:
            dem_data = dem_data.toFloat()
        if canopy_ht is not None:
            canopy_ht = canopy_ht.toFloat()
        if forest_mask is not None:
            forest_mask = forest_mask.toFloat()
        # Add additional data bands
        if dem_data is not None:
            data = data.addBands(dem_data)
        if canopy_ht is not None:
            data = data.addBands(canopy_ht)
        if forest_mask is not None:
            data = data.updateMask(forest_mask)
        
        # Get GEDI vector data
        print("Loading GEDI vector data...")
        gedi_fc = get_gedi_vector_data(aoi_buffered, args.gedi_start_date, args.gedi_end_date, args.quantile)
        
        # Export GEDI vector points as CSV and GeoJSON
        geojson_name = os.path.splitext(os.path.basename(args.aoi))[0]
        prefix = f"{geojson_name}_aoi"
        export_gedi_vector_points(gedi_fc, prefix)
        
        # Convert GEDI vector data to raster for the main data
        if gedi_fc is not None:
            # Convert to raster using reduceToImage
            gedi_raster = gedi_fc.select(args.quantile).reduceToImage(
                properties=[args.quantile],
                reducer=ee.Reducer.first()
            ).rename('rh').toFloat()
            
            # Add GEDI raster to data
            patch_gedi = gedi_raster.clip(aoi_buffered)
            data = data.addBands(patch_gedi)
        
        # Ensure all bands are Float32
        data = data.toFloat()
        
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
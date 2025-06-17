import os
import rasterio
from rasterio import features
import geopandas as gpd
import numpy as np
from shapely.geometry import shape, box

def get_latest_file(dir_path: str, pattern: str, required: bool = True) -> str:
    files = [f for f in os.listdir(dir_path) if f.startswith(pattern)]
    if not files:
        if required:
            raise FileNotFoundError(f"No files matching pattern '{pattern}' found in {dir_path}")
        return None
    return os.path.join(dir_path, max(files, key=lambda x: os.path.getmtime(os.path.join(dir_path, x))))

def geotiff_to_geojson(geotiff_path: str) -> str:
    """
    Convert a GeoTIFF file's extent to a GeoJSON polygon in WGS84.
    
    Args:
        geotiff_path (str): Path to the input GeoTIFF file
        
    Returns:
        str: Path to the created GeoJSON file
    """
    # Generate output path by replacing extension
    geojson_path = os.path.splitext(geotiff_path)[0] + '.geojson'
    
    # Read the GeoTIFF
    with rasterio.open(geotiff_path) as src:
        # Get the bounds of the GeoTIFF
        bounds = src.bounds
        
        # Create a rectangle polygon from the bounds
        rectangle = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
        
        # Create GeoDataFrame with the rectangle
        gdf = gpd.GeoDataFrame(geometry=[rectangle], crs=src.crs)
        
        # Reproject to WGS84
        gdf = gdf.to_crs(epsg=4326)
        
        # Save to GeoJSON
        gdf.to_file(geojson_path, driver='GeoJSON')
        
    return geojson_path

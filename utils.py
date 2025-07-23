import os
import rasterio
from rasterio import features
import geopandas as gpd
import numpy as np
from shapely.geometry import shape, box
import pandas as pd

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

def sample_raster_at_coordinates(raster_path: str, lons: np.ndarray, lats: np.ndarray, 
                                band: int = 1) -> np.ndarray:
    """
    Sample raster values at given longitude/latitude coordinates using vectorized operations.
    
    Args:
        raster_path: Path to raster file
        lons: Array of longitude values
        lats: Array of latitude values  
        band: Raster band to sample (default: 1)
        
    Returns:
        Array of sampled values (NaN for points outside raster or nodata)
    """
    with rasterio.open(raster_path) as src:
        print(f"  Raster CRS: {src.crs}, Shape: {src.shape}, Bounds: {src.bounds}")
        
        # Convert coordinates to raster CRS if needed
        if src.crs.to_epsg() != 4326:
            from rasterio.warp import transform
            xs, ys = transform('EPSG:4326', src.crs, lons, lats)
        else:
            xs, ys = lons, lats
        
        # Convert to numpy arrays for vectorized operations
        xs = np.array(xs)
        ys = np.array(ys)
        
        # Convert coordinates to pixel indices using vectorized operations
        # Use rasterio's index function on arrays
        rows, cols = src.index(xs, ys)
        rows = np.array(rows)
        cols = np.array(cols)
        
        # Read the entire raster band once
        print(f"  Reading raster data...")
        raster_data = src.read(band)
        
        # Initialize output array with NaN
        sampled_values = np.full(len(xs), np.nan, dtype=np.float64)
        
        # Find valid pixel coordinates (within bounds)
        valid_mask = (
            (rows >= 0) & (rows < src.height) & 
            (cols >= 0) & (cols < src.width)
        )
        
        if np.any(valid_mask):
            # Sample values for valid coordinates
            valid_rows = rows[valid_mask]
            valid_cols = cols[valid_mask]
            values = raster_data[valid_rows, valid_cols]
            
            # Handle nodata values
            if src.nodata is not None:
                nodata_mask = values != src.nodata
                values = np.where(nodata_mask, values, np.nan)
            
            # Assign sampled values back to the result array
            sampled_values[valid_mask] = values.astype(np.float64)
        
        valid_count = np.sum(~np.isnan(sampled_values))
        print(f"  Sampled {valid_count}/{len(sampled_values)} points successfully")
        
        return sampled_values

def add_reference_to_gedi_csv(csv_path: str, reference_tif: str = None, 
                             output_path: str = None) -> str:
    """
    Add reference height values to GEDI CSV data.
    
    Args:
        csv_path: Path to GEDI CSV file
        reference_tif: Path to reference TIF file (auto-detected if None)
        output_path: Output CSV path (auto-generated if None)
        
    Returns:
        Path to output CSV file
    """
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Auto-detect reference TIF if not provided
    if reference_tif is None:
        csv_name = os.path.basename(csv_path).lower()
        area_codes = ['dchm_04hf3', 'dchm_05le4', 'dchm_09gd4']
        
        for code in area_codes:
            if code in csv_name:
                reference_tif = f"downloads/{code}.tif"
                break
        
        if reference_tif is None:
            raise ValueError("Could not auto-detect reference TIF file")
    
    # Sample reference heights
    reference_heights = sample_raster_at_coordinates(
        reference_tif, 
        df['longitude'].values, 
        df['latitude'].values
    )
    
    # Add to dataframe
    df['reference_height'] = reference_heights
    
    # Generate output path if not provided
    if output_path is None:
        base_name = os.path.splitext(csv_path)[0]
        output_path = f"{base_name}_with_reference.csv"
    
    # Save
    df.to_csv(output_path, index=False)
    
    return output_path

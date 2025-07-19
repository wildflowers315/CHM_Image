import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import Point, box
import geopandas as gpd
import os
from tqdm import tqdm
import glob
from typing import Tuple, Optional, List

# Assuming load_patch_data is available from data.patch_loader
from data.patch_loader import load_patch_data

def load_patches_from_directory(patches_dir: str, pattern: str = "*.tif") -> List[Tuple[np.ndarray, np.ndarray, str]]:
    """
    Load all patch files from directory.
    
    Args:
        patches_dir: Directory containing patch files
        pattern: File pattern to match
        
    Returns:
        List of (features, gedi_target, patch_name) tuples
    """
    patch_files = glob.glob(os.path.join(patches_dir, pattern))
    patch_data = []
    
    for patch_file in tqdm(patch_files, desc="Loading patches"):
        try:
            features, gedi_target, _ = load_patch_data(patch_file)
            patch_name = os.path.basename(patch_file)
            patch_data.append((features, gedi_target, patch_name))
        except Exception as e:
            print(f"Error loading patch {patch_file}: {e}")
            continue
    
    return patch_data

def load_training_data(csv_path: str, mask_path: Optional[str] = None,
                      feature_names: Optional[list] = None, ch_col: str = 'rh') -> Tuple[np.ndarray, np.ndarray]:
    """
    Load training data from CSV file and optionally mask with forest mask.
    
    Args:
        csv_path: Path to training data CSV
        mask_path: Optional path to forest mask TIF
        
    Returns:
        X: Feature matrix
        y: Target variable (rh)
    """
    # Read training data
    df = pd.read_csv(csv_path)
    
    # Create GeoDataFrame from points
    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df['longitude'], df['latitude'])],
        crs="EPSG:4326"
    )
    
    if mask_path:
        with rasterio.open(mask_path) as mask_src:
            # Check CRS
            mask_crs = mask_src.crs
            if mask_crs != gdf.crs:
                gdf = gdf.to_crs(mask_crs)
            
            # Get bounds of mask
            mask_bounds = box(*mask_src.bounds)
            
            # First filter points by mask bounds
            gdf_masked = gdf[gdf.geometry.within(mask_bounds)]
            
            if len(gdf_masked) == 0:
                raise ValueError("No training points fall within the mask bounds")
            else:
                gdf = gdf_masked
            
            # Convert points to pixel coordinates
            pts_pixels = []
            valid_indices = []
            for idx, point in enumerate(gdf.geometry):
                row, col = rasterio.transform.rowcol(mask_src.transform, 
                                                   point.x, 
                                                   point.y)
                if (0 <= row < mask_src.height and 
                    0 <= col < mask_src.width):
                    pts_pixels.append((row, col))
                    valid_indices.append(idx)
            
            if not pts_pixels:
                raise ValueError("No training points could be mapped to valid pixels")
            
            # Read forest mask values at pixel locations
            mask_values = [mask_src.read(1)[r, c] for r, c in pts_pixels]
            
            # Filter points by mask values
            mask_indices = [i for i, v in enumerate(mask_values) if v == 1]
            if not mask_indices:
                raise ValueError("No training points fall within the forest mask")
            
            final_indices = [valid_indices[i] for i in mask_indices]
            gdf = gdf.iloc[final_indices]
    
    # Convert back to original CRS if needed
    if mask_path and mask_crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    
    # Separate features and target
    df = pd.DataFrame(gdf.drop(columns='geometry'))
    y = df[ch_col].values
    
    # Get feature columns in same order as feature_names
    if feature_names is not None:
        missing_features = set(feature_names) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing features in training data: {missing_features}")
        X = df[feature_names].values
    else:
        X = df.drop([ch_col, 'longitude', 'latitude'], axis=1, errors='ignore').values
    
    return X, y

def load_prediction_data(stack_path: str, mask_path: Optional[str] = None, feature_names: Optional[list] = None) -> Tuple[np.ndarray, rasterio.DatasetReader]:
    """
    Load prediction data from stack TIF and optionally apply forest mask.
    
    Args:
        stack_path: Path to stack TIF file
        mask_path: Optional path to forest mask TIF
        feature_names: Optional list of feature names for filtering bands
        
    Returns:
        X: Feature matrix for prediction
        src: Rasterio dataset for writing results
    """
    if feature_names is None:
        raise ValueError("feature_names must be provided to ensure consistent features between training and prediction")
    # Read stack file
    with rasterio.open(stack_path) as src:
        stack = src.read()
        stack_crs = src.crs
        
        # Get band descriptions if available
        band_descriptions = src.descriptions
        
        # Filter bands based on feature names if provided
        # Create a mapping of band descriptions to indices
        band_indices = []
        for i, desc in enumerate(band_descriptions):
            if desc in feature_names:
                band_indices.append(i)
        
        if len(band_indices) != len(feature_names):
            missing_features = set(feature_names) - set(band_descriptions)
            raise ValueError(f"Could not find all feature names in stack bands. Missing features: {missing_features}")
        
        # Select only the bands that match feature names
        stack = stack[band_indices]
        
        # Reshape stack to 2D array (bands x pixels)
        n_bands, height, width = stack.shape
        X = stack.reshape(n_bands, -1).T
        
        # Apply mask if provided
        if mask_path:
            with rasterio.open(mask_path) as mask_src:
                # Check CRS
                if mask_src.crs != stack_crs:
                    raise ValueError(f"CRS mismatch: stack {stack_crs} != mask {mask_src.crs}")
                
                # Check dimensions
                if mask_src.shape != (height, width):
                    raise ValueError(f"Shape mismatch: stack {(height, width)} != mask {mask_src.shape}")
                
                mask = mask_src.read(1)
                mask = mask.reshape(-1)
                X = X[mask == 1]
        
        src_copy = rasterio.open(stack_path)
        return X, src_copy

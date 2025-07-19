import numpy as np
import rasterio
from typing import Optional

def save_predictions(predictions: np.ndarray, src: rasterio.DatasetReader, output_path: str,
                    mask_path: Optional[str] = None) -> None:
    """
    Save predictions to a GeoTIFF file.
    
    Args:
        predictions: Model predictions
        src: Source rasterio dataset for metadata
        output_path: Path to save predictions
        mask_path: Optional path to forest mask TIF
    """
    # Create output profile
    profile = src.profile.copy()
    profile.update(count=1, dtype='float32')
    
    # Initialize prediction array
    height, width = src.height, src.width
    pred_array = np.zeros((height, width), dtype='float32')
    
    if mask_path:
        # Apply predictions only to masked areas
        with rasterio.open(mask_path) as mask_src:
            # Check CRS
            if mask_src.crs != src.crs:
                raise ValueError(f"CRS mismatch: source {src.crs} != mask {mask_src.crs}")
            
            mask = mask_src.read(1)
            mask_idx = np.where(mask.reshape(-1) == 1)[0]
            pred_array.reshape(-1)[mask_idx] = predictions
    else:
        # Apply predictions to all pixels
        pred_array = predictions.reshape(height, width)
    
    try:
        # Save predictions
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(pred_array, 1)
    finally:
        src.close()

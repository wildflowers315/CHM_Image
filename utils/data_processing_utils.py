import numpy as np
import torch
from typing import Tuple, List

def apply_band_normalization(features: np.ndarray, band_descriptions: list, feature_indices: list) -> np.ndarray:
    """Apply band-specific normalization with temporal support."""
    
    for i, idx in enumerate(feature_indices):
        desc = band_descriptions[idx]
        if not desc:
            continue
            
        # Handle temporal bands (with _M## suffix)
        base_desc = desc.split('_M')[0] if '_M' in desc else desc
        
        # Apply normalization based on base description
        if base_desc.startswith('S1_'):
            # Sentinel-1 normalization: (val + 25) / 25
            features[i] = (features[i] + 25) / 25
        elif base_desc in ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']:
            # Sentinel-2 reflectance: val / 10000, clip to [0,1]
            features[i] = np.clip(features[i] / 10000.0, 0, 1)
        elif base_desc == 'NDVI':
            # NDVI: clip to [-1, 1]
            features[i] = np.clip(features[i], -1, 1)
        elif 'elevation' in desc.lower():
            # Assuming normalize_srtm_elevation is available or imported
            # from data.normalization import normalize_srtm_elevation
            # features[i] = normalize_srtm_elevation(features[i])
            pass # Placeholder, need to import or define normalization functions
        elif 'slope' in desc.lower():
            # Assuming normalize_srtm_slope is available or imported
            # from data.normalization import normalize_srtm_slope
            # features[i] = normalize_srtm_slope(features[i])
            pass # Placeholder
        elif 'aspect' in desc.lower():
            # Assuming normalize_srtm_aspect is available or imported
            # from data.normalization import normalize_srtm_aspect
            # features[i] = normalize_srtm_aspect(features[i])
            pass # Placeholder
        elif base_desc.startswith('ALOS2_'):
            # ALOS2: keep as-is or apply light normalization if needed
            features[i] = features[i]  # No normalization for now
        elif base_desc.startswith('ch_'):
            # Assuming normalize_canopy_height is available or imported
            # from data.normalization import normalize_canopy_height
            # features[i] = normalize_canopy_height(features[i])
            pass # Placeholder
        
        # Replace any remaining NaN/inf values
        features[i] = np.nan_to_num(features[i], nan=0.0, posinf=0.0, neginf=0.0)
    
    return features

def extract_sparse_gedi_pixels(features: np.ndarray, gedi_target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract feature vectors only for pixels with valid GEDI data.
    
    Args:
        features: Feature array [bands, height, width]
        gedi_target: GEDI target array [height, width]
        
    Returns:
        X: Feature matrix [n_valid_pixels, n_bands]
        y: Target vector [n_valid_pixels]
    """
    # Find valid GEDI pixels (not NaN and > 0)
    valid_mask = ~np.isnan(gedi_target) & (gedi_target > 0)
    valid_indices = np.where(valid_mask)
    
    if len(valid_indices[0]) == 0:
        raise ValueError("No valid GEDI pixels found in patch")
    
    # Extract features for valid pixels only
    X = features[:, valid_indices[0], valid_indices[1]].T  # Shape: (n_pixels, n_bands)
    y = gedi_target[valid_indices]  # Shape: (n_pixels,)
    
    print(f"Extracted {len(y)} valid GEDI pixels from {gedi_target.size} total pixels ({len(y)/gedi_target.size*100:.2f}%)")
    
    return X, y

def detect_temporal_mode(band_descriptions: list) -> bool:
    """
    Detect if patch data is temporal based on band naming convention.
    
    Args:
        band_descriptions: List of band descriptions
        
    Returns:
        True if temporal data detected, False otherwise
    """
    temporal_indicators = ['_M01', '_M02', '_M03', '_M04', '_M05', '_M06',
                          '_M07', '_M08', '_M09', '_M10', '_M11', '_M12']
    
    for desc in band_descriptions:
        if desc and any(indicator in desc for indicator in temporal_indicators):
            return True
    
    return False

def separate_temporal_nontemporal_bands(features: np.ndarray, band_descriptions: list) -> Tuple[np.ndarray, np.ndarray]:
    """
    Separate temporal and non-temporal bands from feature array.
    
    Args:
        features: Feature array [bands, height, width]
        band_descriptions: List of band descriptions
        
    Returns:
        temporal_features: Temporal bands [temporal_bands, height, width]
        nontemporal_features: Non-temporal bands [nontemporal_bands, height, width]
    """
    temporal_indices = []
    nontemporal_indices = []
    
    for i, desc in enumerate(band_descriptions):
        if desc and desc not in ['rh', 'forest_mask']:
            # Check if band has monthly suffix (_M01 to _M12)
            if any(f'_M{m:02d}' in desc for m in range(1, 13)):
                temporal_indices.append(i)
            else:
                nontemporal_indices.append(i)
    
    temporal_features = features[temporal_indices] if temporal_indices else np.empty((0, features.shape[1], features.shape[2]))
    nontemporal_features = features[nontemporal_indices] if nontemporal_indices else np.empty((0, features.shape[1], features.shape[2]))
    
    print(f"Separated bands: {len(temporal_indices)} temporal, {len(nontemporal_indices)} non-temporal")
    
    return temporal_features, nontemporal_features

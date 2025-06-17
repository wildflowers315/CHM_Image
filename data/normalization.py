"""
Module for normalizing different types of satellite data.
"""
import numpy as np
from typing import Union

def normalize_sentinel1(data: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """
    Normalize Sentinel-1 backscatter values.
    Formula: (value + 25) / 25
    
    Args:
        data: Input data in dB scale
        
    Returns:
        Normalized data
    """
    return (data + 25) / 25

def normalize_sentinel2(data: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """
    Normalize Sentinel-2 reflectance values.
    Formula: value / 10000
    
    Args:
        data: Input data (0-10000)
        
    Returns:
        Normalized data (0-1)
    """
    return data / 10000

def normalize_srtm_elevation(data: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """
    Normalize SRTM elevation values.
    Formula: value / 2000
    
    Args:
        data: Input elevation in meters
        
    Returns:
        Normalized elevation
    """
    return data / 2000

def normalize_srtm_slope(data: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """
    Normalize SRTM slope values.
    Formula: value / 50
    
    Args:
        data: Input slope in degrees
        
    Returns:
        Normalized slope
    """
    return data / 50

def normalize_srtm_aspect(data: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """
    Normalize SRTM aspect values.
    Formula: (value - 180) / 180
    
    Args:
        data: Input aspect in degrees (0-360)
        
    Returns:
        Normalized aspect (-1 to 1)
    """
    return (data - 180) / 180

def normalize_alos2_dn(data: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """
    Normalize ALOS-2 digital numbers.
    Formula: 10 * log10(DN^2) - 83.0
    
    Args:
        data: Input digital numbers
        
    Returns:
        Normalized backscatter in dB
    """
    return 10 * np.log10(data ** 2) - 83.0

def normalize_canopy_height(data: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """
    Normalize canopy height values.
    Formula: value / 50
    
    Args:
        data: Input height in meters
        
    Returns:
        Normalized height
    """
    return data / 50

def normalize_ndvi(data: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """
    Normalize NDVI values.
    NDVI is already normalized (-1 to 1), so just return as is.
    
    Args:
        data: Input NDVI values
        
    Returns:
        Same NDVI values
    """
    return data 
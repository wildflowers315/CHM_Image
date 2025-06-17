"""
Configuration module for resolution and patch size settings.
"""
from typing import Dict, Any

# Resolution configuration
RESOLUTION_CONFIG = {
    # Base resolution (10m)
    'base': {
        'scale': 10,
        'patch_size': 2560,  # 2.56km patches
        'pixel_count': 256   # 256x256 pixels
    },
    # Medium resolution (20m)
    'medium': {
        'scale': 20,
        'patch_size': 2560,  # 2.56km patches
        'pixel_count': 128   # 128x128 pixels
    },
    # Low resolution (30m)
    'low': {
        'scale': 30,
        'patch_size': 2560,  # 2.56km patches
        'pixel_count': 85    # ~85x85 pixels
    }
}

def validate_scale(scale: int) -> None:
    """
    Validate that the given scale is supported.
    
    Args:
        scale: Resolution in meters
        
    Raises:
        ValueError: If scale is not supported
    """
    valid_scales = [cfg['scale'] for cfg in RESOLUTION_CONFIG.values()]
    if scale not in valid_scales:
        raise ValueError(f"Unsupported scale: {scale}. Must be one of {valid_scales}")

def get_patch_size(scale: int) -> int:
    """
    Get patch size in meters for a given scale.
    
    Args:
        scale: Resolution in meters
        
    Returns:
        Patch size in meters
    """
    validate_scale(scale)
    for cfg in RESOLUTION_CONFIG.values():
        if cfg['scale'] == scale:
            return cfg['patch_size']
    raise ValueError(f"No patch size configured for scale {scale}")

def get_pixel_count(scale: int) -> int:
    """
    Get number of pixels per patch side for a given scale.
    
    Args:
        scale: Resolution in meters
        
    Returns:
        Number of pixels per patch side
    """
    validate_scale(scale)
    for cfg in RESOLUTION_CONFIG.values():
        if cfg['scale'] == scale:
            return cfg['pixel_count']
    raise ValueError(f"No pixel count configured for scale {scale}")

def get_config_for_scale(scale: int) -> Dict[str, Any]:
    """
    Get complete configuration for a given scale.
    
    Args:
        scale: Resolution in meters
        
    Returns:
        Configuration dictionary
    """
    validate_scale(scale)
    for cfg in RESOLUTION_CONFIG.values():
        if cfg['scale'] == scale:
            return cfg
    raise ValueError(f"No configuration found for scale {scale}") 
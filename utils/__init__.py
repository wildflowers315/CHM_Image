"""
Utility functions for CHM processing.
"""

from .spatial_utils import EnhancedSpatialMerger

# Import functions from utils.py for backward compatibility
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from utils import get_latest_file, geotiff_to_geojson
    __all__ = ['EnhancedSpatialMerger', 'get_latest_file', 'geotiff_to_geojson']
except ImportError:
    __all__ = ['EnhancedSpatialMerger']
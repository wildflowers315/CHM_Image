"""Utility modules for CHM processing."""

try:
    from .mosaic_utils import create_comprehensive_mosaic, find_all_patches
    MOSAIC_UTILS_AVAILABLE = True
except ImportError:
    MOSAIC_UTILS_AVAILABLE = False

__all__ = []

if MOSAIC_UTILS_AVAILABLE:
    __all__.extend(['create_comprehensive_mosaic', 'find_all_patches'])
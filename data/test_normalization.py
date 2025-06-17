import numpy as np
import pytest
from data.normalization import (
    normalize_sentinel1, normalize_sentinel2, normalize_srtm_elevation,
    normalize_srtm_slope, normalize_srtm_aspect, normalize_alos2_dn,
    normalize_canopy_height, normalize_ndvi
)

def test_sentinel1_normalization():
    # Test with typical range values
    assert np.isclose(normalize_sentinel1(-31), -0.24)  # Min value
    assert np.isclose(normalize_sentinel1(17), 1.68)    # Max value
    assert np.isclose(normalize_sentinel1(0), 1.0)      # Mid value

def test_sentinel2_normalization():
    # Test with typical range values
    assert np.isclose(normalize_sentinel2(10), 0.001)     # Min value
    assert np.isclose(normalize_sentinel2(15769), 1.5769) # Max value

def test_srtm_elevation_normalization():
    # Test with typical range values
    assert np.isclose(normalize_srtm_elevation(-27), -0.0135) # Min value
    assert np.isclose(normalize_srtm_elevation(331), 0.1655)  # Max value

def test_srtm_slope_normalization():
    # Test with typical range values
    assert np.isclose(normalize_srtm_slope(0.0), 0.0)   # Min value
    assert np.isclose(normalize_srtm_slope(39.3), 0.786) # Max value

def test_srtm_aspect_normalization():
    # Test with typical range values
    assert np.isclose(normalize_srtm_aspect(0), -1.0)   # Min value
    assert np.isclose(normalize_srtm_aspect(180), 0.0)  # Mid value
    assert np.isclose(normalize_srtm_aspect(360), 1.0)  # Max value

def test_alos2_dn_normalization():
    # Test with example DN values
    dn = 100
    expected = 10 * np.log10(dn ** 2) - 83.0
    assert np.isclose(normalize_alos2_dn(dn), expected)

def test_canopy_height_normalization():
    # Test with typical range values
    assert np.isclose(normalize_canopy_height(0), 0.0)   # Min value
    assert np.isclose(normalize_canopy_height(40), 0.8)  # Typical max value

def test_ndvi_normalization():
    # Test with typical range values
    assert np.isclose(normalize_ndvi(-1), -1.0)  # Min value
    assert np.isclose(normalize_ndvi(0), 0.0)    # Mid value
    assert np.isclose(normalize_ndvi(1), 1.0)    # Max value

if __name__ == "__main__":
    pytest.main([__file__]) 
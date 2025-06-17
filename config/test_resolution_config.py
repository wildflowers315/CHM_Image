import pytest
from config.resolution_config import (
    RESOLUTION_CONFIG,
    get_patch_size,
    get_pixel_count,
    validate_scale
)

def test_resolution_config_structure():
    """Test that the config has all required keys with correct types."""
    assert 'default_scale' in RESOLUTION_CONFIG
    assert isinstance(RESOLUTION_CONFIG['default_scale'], int)
    assert 'possible_scales' in RESOLUTION_CONFIG
    assert isinstance(RESOLUTION_CONFIG['possible_scales'], list)
    assert 'patch_sizes' in RESOLUTION_CONFIG
    assert isinstance(RESOLUTION_CONFIG['patch_sizes'], dict)
    assert 'pixel_counts' in RESOLUTION_CONFIG
    assert isinstance(RESOLUTION_CONFIG['pixel_counts'], dict)
    assert 'patch_overlap' in RESOLUTION_CONFIG
    assert isinstance(RESOLUTION_CONFIG['patch_overlap'], dict)

def test_get_patch_size():
    """Test patch size calculations for different scales."""
    # Test valid scales
    assert get_patch_size(10) == 2560  # 256 pixels × 10m
    assert get_patch_size(30) == 7680  # 256 pixels × 30m
    
    # Test invalid scale
    with pytest.raises(ValueError):
        get_patch_size(20)

def test_get_pixel_count():
    """Test pixel count retrieval for different scales."""
    # Test valid scales
    assert get_pixel_count(10) == 256
    assert get_pixel_count(30) == 256
    
    # Test invalid scale
    with pytest.raises(ValueError):
        get_pixel_count(20)

def test_validate_scale():
    """Test scale validation."""
    # Test valid scales
    validate_scale(10)  # Should not raise
    validate_scale(30)  # Should not raise
    
    # Test invalid scale
    with pytest.raises(ValueError):
        validate_scale(20)

def test_patch_size_pixel_count_relationship():
    """Test that patch sizes and pixel counts maintain the correct relationship."""
    for scale in RESOLUTION_CONFIG['possible_scales']:
        patch_size = get_patch_size(scale)
        pixel_count = get_pixel_count(scale)
        assert patch_size == scale * pixel_count  # e.g., 2560 = 10 * 256

if __name__ == "__main__":
    pytest.main([__file__]) 
import pytest
import numpy as np
from rasterio.windows import Window
from data.image_patches import Patch, create_patch_grid, normalize_patch_data

def test_patch_initialization():
    """Test patch initialization with valid parameters."""
    patch = Patch(x=0, y=0, width=2560, height=2560)
    assert patch.x == 0
    assert patch.y == 0
    assert patch.width == 2560
    assert patch.height == 2560
    assert patch.scale == 10  # default
    assert not patch.is_extruded
    assert patch.data is None
    assert patch.pixel_count == 256

def test_patch_invalid_scale():
    """Test patch initialization with invalid scale."""
    with pytest.raises(ValueError):
        Patch(x=0, y=0, width=2560, height=2560, scale=20)

def test_patch_window():
    """Test window calculation."""
    patch = Patch(x=2560, y=2560, width=2560, height=2560, scale=10)
    window = patch.get_window()
    assert isinstance(window, Window)
    assert window.col_off == 256  # 2560/10
    assert window.row_off == 256  # 2560/10
    assert window.width == 256    # 2560/10
    assert window.height == 256   # 2560/10

def test_create_patch_grid():
    """Test patch grid creation."""
    bounds = (0, 0, 5120, 5120)  # 5.12km x 5.12km area
    patches = create_patch_grid(bounds, patch_size=2560, scale=10)
    
    assert len(patches) == 4  # 2x2 grid
    assert all(isinstance(p, Patch) for p in patches)
    
    # Check first patch
    assert patches[0].x == 0
    assert patches[0].y == 0
    assert patches[0].width == 2560
    assert patches[0].height == 2560
    assert not patches[0].is_extruded

def test_create_patch_grid_with_overlap():
    """Test patch grid creation with overlap."""
    bounds = (0, 0, 5120, 5120)
    overlap = 256  # 256m overlap
    patch_size = 2560
    patches = create_patch_grid(bounds, patch_size=patch_size, scale=10, overlap=overlap)
    
    # With overlap, we expect more patches to cover the same area
    assert len(patches) == 9  # 3x3 grid with overlap
    
    # Sort patches by x, then y coordinates
    sorted_patches = sorted(patches, key=lambda p: (p.x, p.y))
    
    # Get first row of patches
    first_row = [p for p in sorted_patches if p.y == sorted_patches[0].y]
    
    # Check that patches overlap
    for i in range(len(first_row) - 1):
        patch0 = first_row[i]
        patch1 = first_row[i + 1]
        
        # Verify overlap
        assert patch1.x < patch0.x + patch0.width
        assert patch1.x + patch1.width > patch0.x
        
        # Verify stride (distance between patch starts)
        stride = patch1.x - patch0.x
        expected_stride = patch_size - overlap  # 2560 - 256 = 2304
        assert abs(stride - expected_stride) < 1  # Allow for small floating point differences
    
    # Verify total coverage
    max_x = max(p.x + p.width for p in patches)
    max_y = max(p.y + p.height for p in patches)
    assert max_x >= bounds[2]  # Should cover full width
    assert max_y >= bounds[3]  # Should cover full height

def test_create_patch_grid_non_divisible():
    """Test patch grid creation with non-divisible area."""
    bounds = (0, 0, 3000, 3000)  # Not divisible by 2560
    patches = create_patch_grid(bounds, patch_size=2560, scale=10)
    
    # Should still cover the entire area
    max_x = max(p.x + p.width for p in patches)
    max_y = max(p.y + p.height for p in patches)
    assert max_x >= bounds[2]
    assert max_y >= bounds[3]
    
    # Some patches should be marked as extruded
    assert any(p.is_extruded for p in patches)

def test_normalize_patch_data():
    """Test patch data normalization."""
    patch = Patch(x=0, y=0, width=2560, height=2560)
    # Create sample data for Sentinel-1 (VV, VH)
    patch.data = np.array([
        [[-31, -20, -10], [-5, 0, 5], [10, 15, 17]],  # VV band
        [[-31, -20, -10], [-5, 0, 5], [10, 15, 17]]   # VH band
    ])
    
    normalized = normalize_patch_data(patch, 'sentinel1')
    
    # Check normalization formula: (value + 25) / 25
    # Test a few key values
    assert np.isclose(normalized[0, 0, 0], -0.24)  # (-31 + 25) / 25
    assert np.isclose(normalized[0, 1, 1], 1.0)    # (0 + 25) / 25
    assert np.isclose(normalized[0, 2, 2], 1.68)   # (17 + 25) / 25
    
    # Test that both bands are normalized the same way
    np.testing.assert_array_almost_equal(normalized[0], normalized[1])

def test_normalize_patch_data_no_data():
    """Test normalization with no data loaded."""
    patch = Patch(x=0, y=0, width=2560, height=2560)
    with pytest.raises(ValueError):
        normalize_patch_data(patch, 'sentinel1')

def test_normalize_patch_data_unknown_type():
    """Test normalization with unknown data type."""
    patch = Patch(x=0, y=0, width=2560, height=2560)
    patch.data = np.zeros((1, 3, 3))
    with pytest.raises(ValueError):
        normalize_patch_data(patch, 'unknown_type')

if __name__ == "__main__":
    pytest.main([__file__]) 
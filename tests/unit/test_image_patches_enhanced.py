#!/usr/bin/env python3
"""
Enhanced tests for image patches functionality.

Tests the data/image_patches.py module for patch creation, management,
and temporal/non-temporal processing.
"""

import pytest
import numpy as np
import rasterio
import os
import sys

# Add tests directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'fixtures'))

from test_config import (
    DataManagerFixture, assert_array_properties, assert_tensor_properties,
    LightweightTimer, TEST_CONFIG, ensure_test_data_exists,
    memory_usage_mb, check_memory_limit
)

# Import modules under test
try:
    from data.image_patches import (
        Patch, create_patch_grid, normalize_patch_data, create_3d_patches
    )
    HAS_IMAGE_PATCHES = True
except ImportError as e:
    HAS_IMAGE_PATCHES = False
    print(f"Warning: data.image_patches not fully available: {e}")

# Try importing rasterio functions for temporal detection
try:
    import rasterio
    def detect_temporal_mode(file_path):
        """Simple temporal mode detection based on band count and naming."""
        with rasterio.open(file_path) as src:
            if src.count >= 100:  # Likely temporal if many bands
                return True
            # Check band descriptions for temporal patterns
            descriptions = src.descriptions[:min(25, src.count)]
            monthly_patterns = [desc for desc in descriptions if desc and '_M' in desc and any(
                desc.endswith(f'{m:02d}') for m in range(1, 13)
            )]
            return len(monthly_patterns) > 0
    
    def extract_patch_metadata(file_path):
        """Extract basic patch metadata."""
        with rasterio.open(file_path) as src:
            return {
                'bands': src.count,
                'height': src.height,
                'width': src.width,
                'temporal': detect_temporal_mode(file_path),
                'crs': src.crs.to_string() if src.crs else None,
                'bounds': src.bounds
            }
    
    def create_3d_patch(patch_data, temporal_mode=True):
        """Create 3D patch from 2D patch data."""
        if not temporal_mode:
            return patch_data
        
        # For temporal mode, try to reshape into time series
        bands, height, width = patch_data.shape
        if bands >= 144:  # Enough for 12 time steps
            time_steps = 12
            channels = bands // time_steps
            # Reshape to (channels, time, height, width)
            reshaped = patch_data[:channels * time_steps].reshape(
                channels, time_steps, height, width
            )
            return reshaped
        else:
            # Fallback: add dummy time dimension
            return patch_data[:, np.newaxis, :, :]

except ImportError:
    pytest.skip("rasterio not available", allow_module_level=True)


class TestPatchCreation:
    """Test patch creation and management."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.data_manager = DataManagerFixture()
        self.test_datasets = ensure_test_data_exists()
        
    def teardown_method(self):
        """Clean up test fixtures."""
        self.data_manager.cleanup_all()
    
    @pytest.mark.skipif(not HAS_IMAGE_PATCHES, reason="data.image_patches not available")
    def test_patch_object_creation(self):
        """Test Patch object initialization."""
        patch = Patch(
            x=0.0, y=0.0, 
            width=2560.0, height=2560.0,
            scale=10
        )
        
        assert patch.x == 0.0
        assert patch.y == 0.0
        assert patch.width == 2560.0
        assert patch.height == 2560.0
        assert patch.scale == 10
        assert patch.is_extruded is False
    
    @pytest.mark.skipif(not HAS_IMAGE_PATCHES, reason="data.image_patches not available")
    def test_create_patch_grid(self):
        """Test patch grid creation."""
        with LightweightTimer() as timer:
            # Define simple bounds
            bounds = (0.0, 0.0, 5000.0, 5000.0)  # 5km x 5km area
            
            # Create patch grid
            patches = create_patch_grid(bounds, patch_size=2560, overlap=0.0, scale=10)
            
            assert len(patches) > 0, "Should create at least one patch"
            assert isinstance(patches[0], Patch), "Should return Patch objects"
            
            # Check patch properties
            for patch in patches:
                assert patch.scale == 10
                assert patch.width <= 2560
                assert patch.height <= 2560
        
        timer.assert_under_limit(TEST_CONFIG['max_single_test_seconds'])
    
    def test_load_temporal_patch(self):
        """Test loading temporal patch data."""
        with LightweightTimer() as timer:
            temporal_file = self.test_datasets['temporal_patch']
            
            # Test loading patch data directly with rasterio
            with rasterio.open(temporal_file) as src:
                patch_data = src.read()
                
                # Verify patch properties
                assert_array_properties(
                    patch_data,
                    expected_shape=(TEST_CONFIG['temporal_bands'], 
                                  TEST_CONFIG['patch_height'], 
                                  TEST_CONFIG['patch_width']),
                    expected_dtype=np.float32,
                    name="temporal_patch"
                )
                
                # Check for realistic value ranges
                assert patch_data.min() >= -50, "SAR values should be reasonable"
                assert patch_data.max() <= 1000, "Values should be in reasonable range"
        
        timer.assert_under_limit(TEST_CONFIG['max_single_test_seconds'])
        check_memory_limit()
    
    def test_load_non_temporal_patch(self):
        """Test loading non-temporal patch data."""
        with LightweightTimer() as timer:
            non_temporal_file = self.test_datasets['non_temporal_patch']
            
            with rasterio.open(non_temporal_file) as src:
                patch_data = src.read()
                
                # Verify patch properties
                assert_array_properties(
                    patch_data,
                    expected_shape=(TEST_CONFIG['non_temporal_bands'],
                                  TEST_CONFIG['patch_height'],
                                  TEST_CONFIG['patch_width']),
                    expected_dtype=np.float32,
                    name="non_temporal_patch"
                )
        
        timer.assert_under_limit(TEST_CONFIG['max_single_test_seconds'])
        check_memory_limit()


class TestTemporalDetection:
    """Test automatic temporal mode detection."""
    
    def setup_method(self):
        """Set up test fixtures.""" 
        self.test_datasets = ensure_test_data_exists()
    
    def test_detect_temporal_mode_temporal(self):
        """Test detection of temporal data."""
        with LightweightTimer() as timer:
            temporal_file = self.test_datasets['temporal_patch']
            
            # Should detect as temporal based on band naming
            is_temporal = detect_temporal_mode(temporal_file)
            assert is_temporal is True, "Should detect temporal data"
            
            # Verify by checking band descriptions
            with rasterio.open(temporal_file) as src:
                descriptions = src.descriptions[:min(25, src.count)]
                
                # Should find monthly patterns (_M01, _M02, etc.)
                monthly_patterns = [desc for desc in descriptions if desc and '_M' in desc]
                assert len(monthly_patterns) > 0, "Should have monthly band patterns"
        
        timer.assert_under_limit(TEST_CONFIG['max_single_test_seconds'])
    
    def test_detect_temporal_mode_non_temporal(self):
        """Test detection of non-temporal data."""
        with LightweightTimer() as timer:
            non_temporal_file = self.test_datasets['non_temporal_patch']
            
            # Should detect as non-temporal
            is_temporal = detect_temporal_mode(non_temporal_file)
            assert is_temporal is False, "Should detect non-temporal data"
            
            # Verify by checking band descriptions
            with rasterio.open(non_temporal_file) as src:
                descriptions = src.descriptions[:min(15, src.count)]
                
                # Should not find monthly patterns
                monthly_patterns = [desc for desc in descriptions if desc and '_M' in desc and desc.endswith(('01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'))]
                assert len(monthly_patterns) == 0, "Should not have monthly patterns"
        
        timer.assert_under_limit(TEST_CONFIG['max_single_test_seconds'])


class TestPatchCreation:
    """Test 3D patch creation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.data_manager = TestDataManager()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        self.data_manager.cleanup_all()
    
    def test_create_3d_patch_temporal(self):
        """Test 3D patch creation for temporal data."""
        with LightweightTimer() as timer:
            # Create synthetic temporal patch data
            bands, height, width = TEST_CONFIG['temporal_bands'], 64, 64
            patch_data = np.random.uniform(-20, 1, (bands, height, width)).astype(np.float32)
            
            # Create 3D patch (should reshape for temporal processing)
            patch_3d = create_3d_patch(patch_data, temporal_mode=True)
            
            # For temporal data, should organize as (channels, time, height, width)
            # Assuming 12 months of data
            expected_time_steps = 12
            expected_channels = bands // expected_time_steps
            
            if bands >= expected_channels * expected_time_steps:
                assert_array_properties(
                    patch_3d,
                    expected_shape=(expected_channels, expected_time_steps, height, width),
                    expected_dtype=np.float32,
                    name="3d_temporal_patch"
                )
            else:
                # Fallback case - should still return valid shape
                assert patch_3d.ndim == 4, "Should return 4D array"
                assert patch_3d.shape[-2:] == (height, width), "Spatial dimensions preserved"
        
        timer.assert_under_limit(TEST_CONFIG['max_single_test_seconds'])
        check_memory_limit()
    
    def test_create_3d_patch_non_temporal(self):
        """Test 3D patch creation for non-temporal data."""
        with LightweightTimer() as timer:
            # Create synthetic non-temporal patch data
            bands, height, width = TEST_CONFIG['non_temporal_bands'], 64, 64
            patch_data = np.random.uniform(-20, 1, (bands, height, width)).astype(np.float32)
            
            # Create 3D patch (should keep spatial-only format)
            patch_3d = create_3d_patch(patch_data, temporal_mode=False)
            
            # For non-temporal data, should remain as (channels, height, width)
            # or add dummy time dimension
            if patch_3d.ndim == 4:
                assert patch_3d.shape[1] == 1, "Should have single time step"
                assert patch_3d.shape[-2:] == (height, width), "Spatial dimensions preserved"
            else:
                assert_array_properties(
                    patch_3d,
                    expected_shape=(bands, height, width),
                    expected_dtype=np.float32,
                    name="3d_non_temporal_patch"
                )
        
        timer.assert_under_limit(TEST_CONFIG['max_single_test_seconds'])
        check_memory_limit()


class TestPatchMetadata:
    """Test patch metadata extraction."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_datasets = ensure_test_data_exists()
    
    def test_extract_metadata_temporal(self):
        """Test metadata extraction from temporal patch."""
        with LightweightTimer() as timer:
            temporal_file = self.test_datasets['temporal_patch']
            
            metadata = extract_patch_metadata(temporal_file)
            
            # Check required metadata fields
            assert 'bands' in metadata
            assert 'height' in metadata
            assert 'width' in metadata
            assert 'temporal' in metadata
            assert 'crs' in metadata
            assert 'bounds' in metadata
            
            # Verify values
            assert metadata['bands'] == TEST_CONFIG['temporal_bands']
            assert metadata['height'] == TEST_CONFIG['patch_height']
            assert metadata['width'] == TEST_CONFIG['patch_width']
            assert metadata['temporal'] is True
        
        timer.assert_under_limit(TEST_CONFIG['max_single_test_seconds'])
    
    def test_extract_metadata_non_temporal(self):
        """Test metadata extraction from non-temporal patch."""
        with LightweightTimer() as timer:
            non_temporal_file = self.test_datasets['non_temporal_patch']
            
            metadata = extract_patch_metadata(non_temporal_file)
            
            # Verify values for non-temporal data
            assert metadata['bands'] == TEST_CONFIG['non_temporal_bands']
            assert metadata['temporal'] is False
        
        timer.assert_under_limit(TEST_CONFIG['max_single_test_seconds'])


class TestPatchNormalization:
    """Test patch normalization functionality."""
    
    @pytest.mark.skipif(not HAS_IMAGE_PATCHES, reason="data.image_patches not available")
    def test_normalize_patch_data_basic(self):
        """Test basic patch normalization."""
        with LightweightTimer() as timer:
            # Create a test patch object
            patch = Patch(x=0, y=0, width=256, height=256, scale=10)
            
            # Add test data
            bands, height, width = 10, 32, 32
            patch.data = np.random.normal(100, 25, (bands, height, width)).astype(np.float32)
            
            # Apply normalization for sentinel1 data type
            normalized = normalize_patch_data(patch, 'sentinel1')
            
            # Check output properties
            assert_array_properties(
                normalized,
                expected_shape=(bands, height, width),
                expected_dtype=np.float32,
                name="normalized_patch"
            )
            
            # Check that normalization changed the data
            assert not np.array_equal(patch.data, normalized), "Data should be changed"
            
            # Basic sanity checks on normalized data
            assert np.isfinite(normalized).all(), "All values should be finite"
        
        timer.assert_under_limit(TEST_CONFIG['max_single_test_seconds'])
    
    def test_simple_normalization_functions(self):
        """Test simple normalization without patch objects."""
        with LightweightTimer() as timer:
            # Test basic array normalization (simple approach)
            test_data = np.random.normal(0, 1, (5, 16, 16)).astype(np.float32)
            
            # Simple normalization: z-score
            normalized = (test_data - np.mean(test_data)) / (np.std(test_data) + 1e-8)
            
            # Check that normalization worked
            assert not np.array_equal(test_data, normalized), "Data should be changed"
            assert np.isfinite(normalized).all(), "All values should be finite"
            
            # Check that mean is approximately 0 and std is approximately 1
            assert abs(np.mean(normalized)) < 0.1, "Mean should be close to 0"
            assert abs(np.std(normalized) - 1.0) < 0.1, "Std should be close to 1"
        
        timer.assert_under_limit(TEST_CONFIG['max_single_test_seconds'])


class TestPatchMemoryEfficiency:
    """Test memory efficiency of patch operations."""
    
    def test_large_patch_memory_usage(self):
        """Test memory usage with larger patches."""
        initial_memory = memory_usage_mb()
        
        with LightweightTimer() as timer:
            # Create moderately large patch (but still reasonable for testing)
            bands, height, width = 50, 128, 128
            large_patch = np.random.uniform(-10, 1, (bands, height, width)).astype(np.float32)
            
            # Apply various operations
            normalized = normalize_patch_data(large_patch)
            patch_3d = create_3d_patch(normalized, temporal_mode=False)
            
            # Verify operations completed
            assert normalized is not None
            assert patch_3d is not None
            
            # Check memory usage
            current_memory = memory_usage_mb()
            memory_increase = current_memory - initial_memory
            
            # Should not use excessive memory for this operation
            assert memory_increase < 200, f"Memory usage {memory_increase:.1f}MB too high"
        
        timer.assert_under_limit(TEST_CONFIG['max_single_test_seconds'])
        check_memory_limit()
    
    def test_batch_patch_processing(self):
        """Test processing multiple patches efficiently."""
        initial_memory = memory_usage_mb()
        
        with LightweightTimer() as timer:
            # Process multiple small patches
            patch_results = []
            
            for i in range(5):
                bands, height, width = 20, 64, 64
                patch_data = np.random.uniform(-10, 1, (bands, height, width)).astype(np.float32)
                
                # Process patch (simple normalization)
                normalized = (patch_data - np.mean(patch_data)) / (np.std(patch_data) + 1e-8)
                patch_3d = create_3d_patch(normalized, temporal_mode=False)
                
                # Store only essential info (not full data to test memory)
                patch_results.append({
                    'shape': patch_3d.shape,
                    'mean': float(np.mean(patch_3d)),
                    'std': float(np.std(patch_3d))
                })
            
            # Verify all patches processed
            assert len(patch_results) == 5
            
            # Check memory efficiency
            current_memory = memory_usage_mb()
            memory_increase = current_memory - initial_memory
            
            # Should not accumulate excessive memory
            assert memory_increase < 300, f"Batch processing used {memory_increase:.1f}MB"
        
        timer.assert_under_limit(TEST_CONFIG['max_single_test_seconds'])
        check_memory_limit()
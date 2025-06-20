#!/usr/bin/env python3
"""
Enhanced tests for spatial utility functions.

Tests the enhanced spatial merger functionality that was refactored 
into utils/spatial_utils.py for proper geographic mosaicking.
"""

import pytest
import numpy as np
import rasterio
from rasterio.transform import from_bounds
import os
import tempfile
from pathlib import Path

# Add tests directory to path for imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'fixtures'))

from test_config import (
    DataManagerFixture, assert_array_properties, LightweightTimer,
    TEST_CONFIG, memory_usage_mb, check_memory_limit
)

# Import the module under test
try:
    from utils.spatial_utils import EnhancedSpatialMerger
except ImportError:
    pytest.skip("utils.spatial_utils not available", allow_module_level=True)


class TestEnhancedSpatialMerger:
    """Test enhanced spatial merger functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.data_manager = DataManagerFixture()
        self.merger = EnhancedSpatialMerger()
        
    def teardown_method(self):
        """Clean up test fixtures."""
        self.data_manager.cleanup_all()
    
    def test_initialization(self):
        """Test EnhancedSpatialMerger initialization."""
        # Test default initialization
        merger = EnhancedSpatialMerger()
        assert merger.merge_strategy == "first"
        
        # Test with different strategies
        strategies = ["first", "last", "min", "max", "average"]
        for strategy in strategies:
            merger = EnhancedSpatialMerger(merge_strategy=strategy)
            assert merger.merge_strategy == strategy
    
    def test_clean_prediction_data(self):
        """Test prediction data cleaning functionality."""
        # Create test data with various problematic values
        test_data = np.array([
            [1.0, 2.0, np.nan],
            [np.inf, -np.inf, 5.0],
            [0.0, -1.0, 10.0]
        ], dtype=np.float32)
        
        cleaned = self.merger.clean_prediction_data(test_data)
        
        # Check that NaN and inf values are replaced with 0
        assert not np.any(np.isnan(cleaned))
        assert not np.any(np.isinf(cleaned))
        assert cleaned[0, 0] == 1.0  # Valid values preserved
        assert cleaned[0, 2] == 0.0  # NaN replaced with 0
        assert cleaned[1, 0] == 0.0  # +inf replaced with 0
        assert cleaned[1, 1] == 0.0  # -inf replaced with 0
    
    def _create_single_prediction_file(self):
        """Helper method to create single prediction file for testing."""
        temp_dir = self.data_manager.create_temp_dir()
        
        # Create synthetic prediction data
        height, width = 128, 128
        prediction_data = np.random.uniform(0, 50, (height, width)).astype(np.float32)
        
        # Add some realistic patterns
        y, x = np.ogrid[:height, :width]
        center_y, center_x = height // 2, width // 2
        forest_pattern = 20 * np.exp(-((y - center_y)**2 + (x - center_x)**2) / (height * width / 8))
        prediction_data += forest_pattern
        
        # Create prediction file
        pred_file = os.path.join(temp_dir, "prediction_patch0001.tif")
        bounds = (-122.5, 37.5, -122.0, 38.0)  # San Francisco area
        transform = from_bounds(*bounds, width, height)
        
        with rasterio.open(
            pred_file, 'w',
            driver='GTiff',
            height=height, width=width, count=1,
            dtype=prediction_data.dtype,
            crs='EPSG:4326',
            transform=transform,
            nodata=0.0
        ) as dst:
            dst.write(prediction_data, 1)
        
        return pred_file, prediction_data
    
    def test_create_single_prediction_file(self):
        """Test creation of single prediction file for testing."""
        pred_file, prediction_data = self._create_single_prediction_file()
        
        # Verify file was created
        assert os.path.exists(pred_file), "Prediction file should be created"
        
        # Verify file properties
        with rasterio.open(pred_file) as src:
            assert src.count == 1, "Should have single band"
            assert src.height == 128, "Should have correct height"
            assert src.width == 128, "Should have correct width"
            
            # Verify data
            data = src.read(1)
            assert data.shape == prediction_data.shape, "Data shape should match"
    
    def test_merge_single_file(self):
        """Test merging with single prediction file."""
        with LightweightTimer() as timer:
            pred_file, original_data = self._create_single_prediction_file()
            
            # Test single file merging
            prediction_files = {"patch0001": pred_file}
            output_path = self.data_manager.create_temp_file(suffix='.tif')
            
            result_path = self.merger.merge_predictions_from_files(
                prediction_files, output_path
            )
            
            assert result_path == output_path
            assert os.path.exists(result_path)
            
            # Verify output data
            with rasterio.open(result_path) as src:
                result_data = src.read(1)
                assert_array_properties(
                    result_data, 
                    expected_shape=original_data.shape,
                    expected_dtype=np.float32,
                    name="merged_single_file"
                )
                
                # Data should be identical for single file
                np.testing.assert_array_almost_equal(result_data, original_data, decimal=5)
        
        timer.assert_under_limit(TEST_CONFIG['max_single_test_seconds'])
        check_memory_limit()
    
    def test_merge_multiple_files(self):
        """Test merging with multiple overlapping prediction files."""
        with LightweightTimer() as timer:
            temp_dir = self.data_manager.create_temp_dir()
            prediction_files = {}
            
            # Create multiple overlapping prediction files
            for i in range(3):
                height, width = 64, 64
                prediction_data = np.random.uniform(5, 25, (height, width)).astype(np.float32)
                
                # Different spatial bounds for each patch (overlapping)
                x_offset = i * 0.002  # Small overlap
                bounds = (-122.5 + x_offset, 37.5, -122.0 + x_offset, 38.0)
                transform = from_bounds(*bounds, width, height)
                
                pred_file = os.path.join(temp_dir, f"prediction_patch{i:04d}.tif")
                
                with rasterio.open(
                    pred_file, 'w',
                    driver='GTiff',
                    height=height, width=width, count=1,
                    dtype=prediction_data.dtype,
                    crs='EPSG:4326',
                    transform=transform,
                    nodata=0.0
                ) as dst:
                    dst.write(prediction_data, 1)
                
                prediction_files[f"patch{i:04d}"] = pred_file
            
            # Test merging with 'first' strategy
            output_path = self.data_manager.create_temp_file(suffix='.tif')
            result_path = self.merger.merge_predictions_from_files(
                prediction_files, output_path
            )
            
            assert os.path.exists(result_path)
            
            # Verify output
            with rasterio.open(result_path) as src:
                result_data = src.read(1)
                
                # Should have data from all patches
                valid_pixels = np.sum(result_data > 0)
                assert valid_pixels > 0, "Merged result should have valid data"
                
                # Check reasonable value range for forest heights
                valid_data = result_data[result_data > 0]
                if len(valid_data) > 0:
                    assert valid_data.min() >= 0, "Heights should be non-negative"
                    assert valid_data.max() <= 100, "Heights should be reasonable"
        
        timer.assert_under_limit(TEST_CONFIG['max_single_test_seconds'])
        check_memory_limit()
    
    def test_merge_strategy_average(self):
        """Test averaging merge strategy with overlapping data."""
        with LightweightTimer() as timer:
            temp_dir = self.data_manager.create_temp_dir()
            prediction_files = {}
            
            # Create two overlapping files with known values for testing averaging
            height, width = 32, 32
            
            # File 1: all 10s
            data1 = np.full((height, width), 10.0, dtype=np.float32)
            bounds = (-122.5, 37.5, -122.0, 38.0)
            
            file1 = os.path.join(temp_dir, "patch1.tif")
            with rasterio.open(
                file1, 'w',
                driver='GTiff', height=height, width=width, count=1,
                dtype=data1.dtype, crs='EPSG:4326',
                transform=from_bounds(*bounds, width, height),
                nodata=0.0
            ) as dst:
                dst.write(data1, 1)
            
            # File 2: all 20s (same bounds for complete overlap)
            data2 = np.full((height, width), 20.0, dtype=np.float32)
            file2 = os.path.join(temp_dir, "patch2.tif")
            with rasterio.open(
                file2, 'w',
                driver='GTiff', height=height, width=width, count=1,
                dtype=data2.dtype, crs='EPSG:4326',
                transform=from_bounds(*bounds, width, height),
                nodata=0.0
            ) as dst:
                dst.write(data2, 1)
            
            prediction_files = {"patch1": file1, "patch2": file2}
            
            # Test averaging strategy
            merger = EnhancedSpatialMerger(merge_strategy="average")
            output_path = self.data_manager.create_temp_file(suffix='.tif')
            
            result_path = merger.merge_predictions_from_files(
                prediction_files, output_path
            )
            
            # Verify averaging worked
            with rasterio.open(result_path) as src:
                result_data = src.read(1)
                
                # All overlapping pixels should average to 15.0
                valid_data = result_data[result_data > 0]
                if len(valid_data) > 0:
                    np.testing.assert_array_almost_equal(valid_data, 15.0, decimal=1)
        
        timer.assert_under_limit(TEST_CONFIG['max_single_test_seconds'])
        check_memory_limit()
    
    def test_handle_missing_files(self):
        """Test graceful handling of missing prediction files."""
        with LightweightTimer() as timer:
            # Create mix of existing and missing files
            existing_file, _ = self._create_single_prediction_file()
            missing_file = "/nonexistent/path/missing.tif"
            
            prediction_files = {
                "existing": existing_file,
                "missing": missing_file
            }
            
            output_path = self.data_manager.create_temp_file(suffix='.tif')
            
            # Should succeed with only the existing file
            result_path = self.merger.merge_predictions_from_files(
                prediction_files, output_path
            )
            
            assert os.path.exists(result_path)
            
            # Verify output contains data from existing file
            with rasterio.open(result_path) as src:
                result_data = src.read(1)
                valid_pixels = np.sum(result_data > 0)
                assert valid_pixels > 0, "Should contain data from existing file"
        
        timer.assert_under_limit(TEST_CONFIG['max_single_test_seconds'])
        check_memory_limit()
    
    def test_handle_empty_prediction_files(self):
        """Test handling of prediction files with no valid data."""
        with LightweightTimer() as timer:
            temp_dir = self.data_manager.create_temp_dir()
            
            # Create file with all zeros (no valid data)
            height, width = 32, 32
            empty_data = np.zeros((height, width), dtype=np.float32)
            
            empty_file = os.path.join(temp_dir, "empty_patch.tif")
            bounds = (-122.5, 37.5, -122.0, 38.0)
            
            with rasterio.open(
                empty_file, 'w',
                driver='GTiff', height=height, width=width, count=1,
                dtype=empty_data.dtype, crs='EPSG:4326',
                transform=from_bounds(*bounds, width, height),
                nodata=0.0
            ) as dst:
                dst.write(empty_data, 1)
            
            prediction_files = {"empty": empty_file}
            output_path = self.data_manager.create_temp_file(suffix='.tif')
            
            # Should handle empty files gracefully
            with pytest.raises(ValueError, match="No valid prediction files"):
                self.merger.merge_predictions_from_files(prediction_files, output_path)
        
        timer.assert_under_limit(TEST_CONFIG['max_single_test_seconds'])
        check_memory_limit()
    
    def test_memory_efficiency(self):
        """Test memory efficiency with multiple files."""
        initial_memory = memory_usage_mb()
        
        with LightweightTimer() as timer:
            temp_dir = self.data_manager.create_temp_dir()
            prediction_files = {}
            
            # Create several small prediction files
            for i in range(5):
                height, width = 64, 64
                prediction_data = np.random.uniform(0, 30, (height, width)).astype(np.float32)
                
                x_offset = i * 0.001
                bounds = (-122.5 + x_offset, 37.5, -122.0 + x_offset, 38.0)
                
                pred_file = os.path.join(temp_dir, f"patch{i:04d}.tif")
                with rasterio.open(
                    pred_file, 'w',
                    driver='GTiff', height=height, width=width, count=1,
                    dtype=prediction_data.dtype, crs='EPSG:4326',
                    transform=from_bounds(*bounds, width, height),
                    nodata=0.0
                ) as dst:
                    dst.write(prediction_data, 1)
                
                prediction_files[f"patch{i:04d}"] = pred_file
            
            # Test merging
            output_path = self.data_manager.create_temp_file(suffix='.tif')
            result_path = self.merger.merge_predictions_from_files(
                prediction_files, output_path
            )
            
            assert os.path.exists(result_path)
            
            # Check memory usage didn't spike dramatically
            final_memory = memory_usage_mb()
            memory_increase = final_memory - initial_memory
            
            # Should not use more than 500MB for this small test
            assert memory_increase < 500, f"Memory usage increased by {memory_increase:.1f}MB"
        
        timer.assert_under_limit(TEST_CONFIG['max_single_test_seconds'])
        check_memory_limit()
    
    def test_different_merge_strategies(self):
        """Test all supported merge strategies."""
        strategies = ["first", "last", "min", "max"]
        
        with LightweightTimer() as timer:
            for strategy in strategies:
                merger = EnhancedSpatialMerger(merge_strategy=strategy)
                
                # Create simple test case
                existing_file, _ = self._create_single_prediction_file()
                prediction_files = {"test": existing_file}
                output_path = self.data_manager.create_temp_file(suffix='.tif')
                
                # Should work with any strategy
                result_path = merger.merge_predictions_from_files(
                    prediction_files, output_path
                )
                
                assert os.path.exists(result_path), f"Strategy {strategy} failed"
        
        timer.assert_under_limit(TEST_CONFIG['max_single_test_seconds'])
    
    def test_output_metadata_preservation(self):
        """Test that output preserves important metadata."""
        with LightweightTimer() as timer:
            pred_file, original_data = self._create_single_prediction_file()
            prediction_files = {"test": pred_file}
            output_path = self.data_manager.create_temp_file(suffix='.tif')
            
            result_path = self.merger.merge_predictions_from_files(
                prediction_files, output_path
            )
            
            # Check output metadata
            with rasterio.open(result_path) as src:
                assert src.crs.to_string() == 'EPSG:4326'
                assert src.nodata == 0.0
                assert src.driver == 'GTiff'
                assert hasattr(src, 'transform')
                assert src.count == 1  # Single band output
        
        timer.assert_under_limit(TEST_CONFIG['max_single_test_seconds'])
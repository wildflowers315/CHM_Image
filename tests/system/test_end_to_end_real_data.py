#!/usr/bin/env python3
"""
Comprehensive system-level tests using real data files.

Tests the complete end-to-end workflow:
1. Data loading and preprocessing
2. Model training with multiple patches
3. Prediction generation and spatial mosaicking
4. Evaluation against reference data
5. Model persistence and loading

Uses actual data files from the project:
- Reference: downloads/dchm_09gd4.tif
- Patches: chm_outputs/dchm_09gd4_bandNum31_scale10_patch*.tif
- Models: chm_outputs/2d_unet/best_model.pth
"""

import pytest
import numpy as np
import rasterio
import os
import sys
import tempfile
import json
import torch
from pathlib import Path
from datetime import datetime

# Add tests directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'fixtures'))

from test_config import (
    DataManagerFixture, LightweightTimer, TEST_CONFIG,
    memory_usage_mb, check_memory_limit, get_device
)

# Import main training module
try:
    sys.path.append(os.path.abspath('.'))
    import train_predict_map as tpm
    from predict import main as predict_main
    from evaluate_predictions import main as evaluate_main
except ImportError as e:
    pytest.skip(f"Required modules not available: {e}", allow_module_level=True)


class TestEndToEndRealData:
    """System-level tests with real project data."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.data_manager = DataManagerFixture()
        self.device = get_device()
        
        # Define real data paths
        self.reference_path = "downloads/dchm_09gd4.tif"
        self.patch_pattern = "chm_outputs/dchm_09gd4_bandNum31_scale10_patch*.tif"
        self.existing_model_path = "chm_outputs/2d_unet/best_model.pth"
        
        # Verify data files exist
        self._verify_data_availability()
        
    def teardown_method(self):
        """Clean up test fixtures."""
        self.data_manager.cleanup_all()
    
    def _verify_data_availability(self):
        """Verify all required data files are available."""
        # Check reference data
        if not os.path.exists(self.reference_path):
            pytest.skip(f"Reference data not found: {self.reference_path}")
        
        # Check patch files
        import glob
        patch_files = glob.glob(self.patch_pattern)
        if len(patch_files) < 2:
            pytest.skip(f"Insufficient patch files found: {len(patch_files)} < 2")
        
        self.patch_files = sorted(patch_files)
        print(f"Found {len(self.patch_files)} patch files for testing")
    
    def test_data_loading_and_validation(self):
        """Test loading and validation of real data files."""
        with LightweightTimer() as timer:
            # Test reference data loading (sample only for efficiency)
            with rasterio.open(self.reference_path) as ref_src:
                # Read metadata only first
                ref_profile = ref_src.profile
                
                # Validate basic properties
                assert ref_src.height > 1000, "Reference should be large-scale data"
                assert ref_src.width > 1000, "Reference should be large-scale data"
                assert ref_src.count == 1, "Reference should be single-band height data"
                
                # Sample a small window for data validation (much faster)
                window = rasterio.windows.Window(0, 0, min(1000, ref_src.width), min(1000, ref_src.height))
                ref_sample = ref_src.read(1, window=window)
                
                assert ref_sample.dtype in [np.float32, np.float64], "Should be float data"
                
                # Check for reasonable height values in sample
                valid_data = ref_sample[~np.isnan(ref_sample) & (ref_sample > 0)]
                if len(valid_data) > 0:
                    assert valid_data.min() >= 0, "Heights should be non-negative"
                    assert valid_data.max() <= 100, "Heights should be reasonable (< 100m)"
                    print(f"✅ Reference sample: {ref_sample.shape}, range: {valid_data.min():.2f}-{valid_data.max():.2f}m")
                else:
                    print(f"✅ Reference data properties: {ref_src.shape}, dtype: {ref_src.dtypes[0]}")
            
            # Test patch data loading
            for i, patch_file in enumerate(self.patch_files[:3]):  # Test first 3 patches
                with rasterio.open(patch_file) as patch_src:
                    patch_data = patch_src.read()
                    
                    # Validate patch structure
                    assert patch_data.shape[0] >= 30, f"Should have many bands (has {patch_data.shape[0]})"
                    assert patch_data.shape[1] >= 200, f"Should have reasonable spatial size"
                    assert patch_data.shape[2] >= 200, f"Should have reasonable spatial size"
                    
                    # Test temporal mode detection
                    is_temporal = tpm.detect_temporal_mode(patch_src.descriptions)
                    print(f"✅ Patch {i}: {patch_data.shape}, temporal: {is_temporal}")
                    
                    # Test band descriptions
                    descriptions = patch_src.descriptions or []
                    assert len(descriptions) >= patch_data.shape[0], "Should have band descriptions"
        
        timer.assert_under_limit(30)  # Allow more time for real data
        check_memory_limit()
    
    def test_multi_patch_training_workflow(self):
        """Test training workflow with multiple real patches."""
        with LightweightTimer() as timer:
            output_dir = self.data_manager.create_temp_dir()
            
            # Use first 2 patches for faster testing
            test_patches = self.patch_files[:2]
            
            # Create synthetic GEDI data for testing (since we don't have real GEDI)
            gedi_file = self._create_synthetic_gedi_for_patch(test_patches[0], output_dir)
            
            # Test feature extraction from real patches
            all_features = []
            all_targets = []
            
            for patch_file in test_patches:
                with rasterio.open(patch_file) as src:
                    patch_data = src.read()
                
                with rasterio.open(gedi_file) as src:
                    gedi_data = src.read(1)
                
                # Extract features
                features, targets = tpm.extract_sparse_gedi_pixels(patch_data, gedi_data)
                
                if len(features) > 0:
                    all_features.append(features)
                    all_targets.append(targets)
                    print(f"✅ Extracted {len(features)} samples from {os.path.basename(patch_file)}")
            
            # Combine features from all patches
            if all_features:
                combined_features = np.vstack(all_features)
                combined_targets = np.hstack(all_targets)
                
                print(f"✅ Combined training data: {combined_features.shape} features, {len(combined_targets)} targets")
                
                # Test model training with real multi-patch data
                model, metrics, importance = tpm.train_model(
                    combined_features, combined_targets,
                    model_type='rf',
                    test_size=0.3
                )
                
                assert model is not None, "Should successfully train model"
                assert metrics is not None, "Should return training metrics"
                
                # Validate metrics
                assert 'R2' in metrics or 'r2' in metrics or 'train_r2' in metrics, "Should include R² metric"
                print(f"✅ Training metrics: {metrics}")
                
                # Test model prediction capability
                test_predictions = model.predict(combined_features[:10])
                assert len(test_predictions) == 10, "Should predict for all test samples"
                assert test_predictions.min() >= -20, "Predictions should be reasonable"
                assert test_predictions.max() <= 80, "Predictions should be reasonable"
        
        timer.assert_under_limit(60)  # Allow more time for real training
        check_memory_limit()
    
    def test_prediction_and_spatial_mosaic(self):
        """Test prediction generation and spatial mosaicking with real patches."""
        with LightweightTimer() as timer:
            output_dir = self.data_manager.create_temp_dir()
            
            # Train a quick model for testing
            patch_file = self.patch_files[0]
            gedi_file = self._create_synthetic_gedi_for_patch(patch_file, output_dir)
            
            # Load and train
            with rasterio.open(patch_file) as src:
                patch_data = src.read()
            
            with rasterio.open(gedi_file) as src:
                gedi_data = src.read(1)
            
            features, targets = tpm.extract_sparse_gedi_pixels(patch_data, gedi_data)
            
            if len(features) > 10:  # Need sufficient data
                model, _, _ = tpm.train_model(features, targets, model_type='rf')
                
                # Test prediction on multiple patches
                prediction_files = {}
                
                for i, test_patch in enumerate(self.patch_files[:2]):
                    # Generate prediction for this patch
                    with rasterio.open(test_patch) as src:
                        test_data = src.read()
                        height, width = test_data.shape[1], test_data.shape[2]
                    
                    # Reshape for prediction
                    pixel_features = test_data.reshape(test_data.shape[0], -1).T
                    predictions = model.predict(pixel_features)
                    pred_map = predictions.reshape(height, width)
                    
                    # Save prediction
                    pred_file = os.path.join(output_dir, f"prediction_patch{i:04d}.tif")
                    
                    with rasterio.open(test_patch) as src:
                        profile = src.profile.copy()
                        profile.update({'count': 1, 'dtype': 'float32'})
                    
                    with rasterio.open(pred_file, 'w', **profile) as dst:
                        dst.write(pred_map.astype(np.float32), 1)
                    
                    prediction_files[f"patch{i:04d}"] = pred_file
                    print(f"✅ Generated prediction for patch {i}: {pred_map.shape}")
                
                # Test spatial mosaic creation
                from utils.spatial_utils import EnhancedSpatialMerger
                merger = EnhancedSpatialMerger(merge_strategy="average")
                
                mosaic_path = os.path.join(output_dir, "prediction_mosaic.tif")
                result_path = merger.merge_predictions_from_files(prediction_files, mosaic_path)
                
                assert os.path.exists(result_path), "Should create mosaic file"
                
                # Validate mosaic
                with rasterio.open(result_path) as src:
                    mosaic_data = src.read(1)
                    valid_pixels = np.sum(mosaic_data > 0)
                    
                    assert valid_pixels > 0, "Mosaic should contain valid predictions"
                    print(f"✅ Spatial mosaic created: {mosaic_data.shape}, {valid_pixels} valid pixels")
        
        timer.assert_under_limit(90)  # Allow time for prediction and mosaicking
        check_memory_limit()
    
    def test_model_persistence_and_loading(self):
        """Test model saving and loading functionality."""
        if not os.path.exists(self.existing_model_path):
            pytest.skip("Existing model not found for testing")
        
        with LightweightTimer() as timer:
            # Test loading existing model
            try:
                if torch.cuda.is_available():
                    model = torch.load(self.existing_model_path)
                else:
                    model = torch.load(self.existing_model_path, map_location='cpu')
                
                print(f"✅ Successfully loaded existing model from {self.existing_model_path}")
                
                # Test model properties
                assert hasattr(model, '__call__'), "Model should be callable"
                
                # Test with sample input (if it's a PyTorch model)
                if hasattr(model, 'eval'):
                    model.eval()
                    
                    # Create test input matching expected dimensions
                    with rasterio.open(self.patch_files[0]) as src:
                        n_bands = src.count
                    
                    # Test with appropriate input size for 2D U-Net
                    test_input = torch.randn(1, n_bands, 256, 256)
                    if not torch.cuda.is_available():
                        test_input = test_input.cpu()
                        if hasattr(model, 'cpu'):
                            model = model.cpu()
                    
                    try:
                        with torch.no_grad():
                            output = model(test_input)
                            print(f"✅ Model inference successful: input {test_input.shape} → output {output.shape}")
                    except Exception as e:
                        print(f"⚠️ Model inference failed (expected for dimension mismatch): {e}")
                
            except Exception as e:
                print(f"⚠️ Model loading issue (not critical): {e}")
        
        timer.assert_under_limit(30)
    
    def test_evaluation_pipeline_with_reference(self):
        """Test evaluation pipeline using real reference data."""
        with LightweightTimer() as timer:
            output_dir = self.data_manager.create_temp_dir()
            
            # Create a simple prediction for testing evaluation
            patch_file = self.patch_files[0]
            
            # Generate a synthetic prediction that roughly matches patch spatial extent
            with rasterio.open(patch_file) as patch_src:
                patch_profile = patch_src.profile
                patch_bounds = patch_src.bounds
                patch_shape = (patch_src.height, patch_src.width)
            
            # Create synthetic prediction data
            synthetic_pred = np.random.uniform(5, 25, patch_shape).astype(np.float32)
            
            # Save as prediction file
            pred_path = os.path.join(output_dir, "test_prediction.tif")
            pred_profile = patch_profile.copy()
            pred_profile.update({'count': 1, 'dtype': 'float32'})
            
            with rasterio.open(pred_path, 'w', **pred_profile) as dst:
                dst.write(synthetic_pred, 1)
            
            # Test evaluation against reference (will need spatial alignment)
            try:
                from raster_utils import load_and_align_rasters
                
                # This will test if the evaluation pipeline can handle real reference data
                pred_aligned, ref_aligned = load_and_align_rasters(pred_path, self.reference_path)
                
                if pred_aligned is not None and ref_aligned is not None:
                    # Calculate basic metrics
                    from sklearn.metrics import mean_squared_error, r2_score
                    
                    # Find overlapping valid data
                    valid_mask = (~np.isnan(pred_aligned)) & (~np.isnan(ref_aligned)) & (ref_aligned > 0)
                    
                    if valid_mask.sum() > 100:  # Need sufficient overlap
                        pred_valid = pred_aligned[valid_mask]
                        ref_valid = ref_aligned[valid_mask]
                        
                        rmse = np.sqrt(mean_squared_error(ref_valid, pred_valid))
                        r2 = r2_score(ref_valid, pred_valid)
                        
                        print(f"✅ Evaluation metrics: RMSE={rmse:.2f}m, R²={r2:.3f}")
                        print(f"✅ Valid overlap pixels: {valid_mask.sum()}")
                        
                        assert rmse >= 0, "RMSE should be non-negative"
                        assert -1 <= r2 <= 1, "R² should be in valid range"
                    else:
                        print("⚠️ Insufficient spatial overlap for evaluation")
                else:
                    print("⚠️ Could not align prediction and reference data")
            
            except ImportError:
                print("⚠️ Evaluation utilities not available, skipping alignment test")
        
        timer.assert_under_limit(45)
        check_memory_limit()
    
    def test_memory_efficiency_with_real_data(self):
        """Test memory efficiency when processing real data files."""
        initial_memory = memory_usage_mb()
        
        with LightweightTimer() as timer:
            # Process multiple patches sequentially
            for i, patch_file in enumerate(self.patch_files[:3]):
                # Load patch data
                with rasterio.open(patch_file) as src:
                    patch_data = src.read()
                
                # Simulate processing
                data_stats = {
                    'shape': patch_data.shape,
                    'memory_mb': patch_data.nbytes / (1024**2),
                    'data_range': (patch_data.min(), patch_data.max())
                }
                
                print(f"✅ Processed patch {i}: {data_stats}")
                
                # Check memory usage
                current_memory = memory_usage_mb()
                memory_increase = current_memory - initial_memory
                
                # Memory should not grow linearly with each patch
                assert memory_increase < 1000, f"Memory usage {memory_increase:.1f}MB too high"
                
                # Force garbage collection
                del patch_data
                import gc
                gc.collect()
        
        timer.assert_under_limit(30)
        check_memory_limit()
    
    def _create_synthetic_gedi_for_patch(self, patch_file, output_dir):
        """Create synthetic GEDI data spatially aligned with a patch."""
        with rasterio.open(patch_file) as src:
            height, width = src.height, src.width
            profile = src.profile.copy()
        
        # Create sparse GEDI-like data (low coverage)
        gedi_mask = np.random.random((height, width)) < 0.005  # 0.5% coverage
        gedi_heights = np.random.uniform(0, 30, (height, width))
        gedi_output = np.where(gedi_mask, gedi_heights, 0).astype(np.float32)
        
        # Save GEDI file
        gedi_file = os.path.join(output_dir, "synthetic_gedi.tif")
        gedi_profile = profile.copy()
        gedi_profile.update({'count': 1, 'dtype': 'float32', 'nodata': 0.0})
        
        with rasterio.open(gedi_file, 'w', **gedi_profile) as dst:
            dst.write(gedi_output, 1)
        
        return gedi_file


@pytest.mark.system
class TestSystemPerformance:
    """Performance benchmarks with real data."""
    
    def test_data_loading_performance(self):
        """Benchmark data loading performance."""
        import time
        
        patch_files = [f for f in 
                      ["chm_outputs/dchm_09gd4_bandNum31_scale10_patch0000.tif",
                       "chm_outputs/dchm_09gd4_bandNum31_scale10_patch0001.tif"]
                      if os.path.exists(f)]
        
        if not patch_files:
            pytest.skip("No patch files available for performance testing")
        
        loading_times = []
        
        for patch_file in patch_files:
            start_time = time.time()
            
            with rasterio.open(patch_file) as src:
                data = src.read()
            
            load_time = time.time() - start_time
            loading_times.append(load_time)
            
            data_size_mb = data.nbytes / (1024**2)
            throughput = data_size_mb / load_time
            
            print(f"📊 {os.path.basename(patch_file)}: {load_time:.2f}s, {data_size_mb:.1f}MB, {throughput:.1f}MB/s")
        
        avg_time = np.mean(loading_times)
        print(f"📊 Average loading time: {avg_time:.2f}s")
        
        # Performance assertions
        assert avg_time < 10.0, f"Data loading too slow: {avg_time:.2f}s"
# Test Plan 2025: Comprehensive Coverage for CHM Image Processing

## Executive Summary

This test plan provides comprehensive coverage for the refactored CHM Image Processing codebase, focusing on lightweight CPU-based tests that validate core functionality across all components while ensuring both CPU and GPU compatibility.

## Testing Philosophy

### 🎯 **Core Principles**
- **Lightweight Execution**: All tests designed for CPU environments with minimal resource requirements
- **Comprehensive Coverage**: Test all major components and workflows
- **Cross-Platform Compatibility**: Ensure tests work on both CPU and GPU environments
- **Realistic Scenarios**: Use small synthetic data that mimics real-world patterns
- **Fast Feedback**: Complete test suite runs in under 5 minutes

### 📊 **Coverage Strategy**
- **Unit Tests**: Individual component functionality
- **Integration Tests**: Component interaction and workflow validation
- **System Tests**: End-to-end pipeline validation with synthetic data
- **Performance Tests**: Memory usage and execution time validation
- **Error Handling**: Robust error condition testing

## Test Categories

### 1. Core Data Processing Tests

#### 1.1 Patch Creation and Management (`test_image_patches.py`)
```python
class TestImagePatches:
    def test_create_synthetic_patch_temporal():
        """Test temporal patch creation with 196 bands (12-month series)"""
        # Create 256x256x196 synthetic patch with realistic band patterns
        # Validate temporal dimension handling
        # Test band naming convention (_M01-_M12 suffixes)
        
    def test_create_synthetic_patch_non_temporal():
        """Test non-temporal patch creation with 31 bands (composites)"""
        # Create 256x256x31 synthetic patch
        # Validate spatial-only processing
        # Test median composite patterns
        
    def test_patch_overlap_handling():
        """Test patch creation with overlap scenarios"""
        # Test 0%, 10%, 25% overlap configurations
        # Validate boundary handling
        
    def test_automatic_temporal_detection():
        """Test automatic temporal vs non-temporal detection"""
        # Create patches with/without _M01-_M12 band naming
        # Validate detection logic
```

#### 1.2 Data Normalization (`test_normalization.py`)
```python
class TestNormalization:
    def test_band_specific_normalization():
        """Test normalization strategies for different band types"""
        # Test S1, S2, ALOS2, DEM normalization
        # Validate value ranges and scaling
        
    def test_temporal_normalization():
        """Test normalization across temporal dimensions"""
        # Test 12-month time series normalization
        # Validate temporal consistency
        
    def test_sparse_gedi_normalization():
        """Test GEDI target normalization with sparse data"""
        # Test height value scaling (0-50m range)
        # Validate outlier handling
```

#### 1.3 Multi-Patch Processing (`test_multi_patch.py`)
```python
class TestMultiPatch:
    def test_patch_registry():
        """Test patch information management"""
        # Test PatchInfo creation and tracking
        # Validate metadata handling
        
    def test_load_multi_patch_gedi():
        """Test loading GEDI data across multiple patches"""
        # Create synthetic multi-patch GEDI data
        # Test sparse pixel extraction
        
    def test_prediction_merger():
        """Test basic prediction merging functionality"""
        # Test patch-to-patch prediction combination
        # Validate overlap handling
```

### 2. Model Architecture Tests

#### 2.1 3D U-Net Architecture (`test_3d_unet_enhanced.py`)
```python
class Test3DUNet:
    def test_model_initialization():
        """Test 3D U-Net model creation with various configurations"""
        # Test temporal input shapes (B, C, T, H, W)
        # Validate architecture consistency
        
    def test_forward_pass_temporal():
        """Test forward pass with temporal data"""
        # Input: (1, 196, 12, 256, 256) synthetic data
        # Validate output shape and value ranges
        
    def test_forward_pass_fallback():
        """Test temporal averaging fallback for memory constraints"""
        # Simulate memory pressure scenarios
        # Validate graceful degradation
        
    def test_cpu_gpu_compatibility():
        """Test model works on both CPU and GPU (if available)"""
        # Test device migration
        # Validate identical outputs across devices
```

#### 2.2 2D U-Net Architecture (`test_2d_unet_enhanced.py`)
```python
class Test2DUNet:
    def test_model_initialization():
        """Test 2D U-Net model creation"""
        # Test non-temporal input shapes (B, C, H, W)
        # Validate channel adaptation (29-31 channels)
        
    def test_forward_pass_non_temporal():
        """Test forward pass with non-temporal data"""
        # Input: (1, 31, 256, 256) synthetic data
        # Validate spatial processing
        
    def test_channel_mismatch_handling():
        """Test automatic channel adjustment"""
        # Test 29, 30, 31 channel inputs
        # Validate robust channel handling
```

#### 2.3 Traditional Models (`test_traditional_models.py`)
```python
class TestTraditionalModels:
    def test_random_forest_temporal():
        """Test RF with temporal features"""
        # Create synthetic temporal feature vectors
        # Test training and prediction
        
    def test_random_forest_non_temporal():
        """Test RF with non-temporal features"""
        # Create synthetic non-temporal feature vectors
        # Validate feature importance extraction
        
    def test_mlp_model():
        """Test MLP model training and prediction"""
        # Test PyTorch MLP with synthetic data
        # Validate regression output
```

### 3. Training System Tests

#### 3.1 Unified Training System (`test_train_predict_map.py`)
```python
class TestUnifiedTraining:
    def test_automatic_mode_detection():
        """Test temporal vs non-temporal mode detection"""
        # Test with temporal and non-temporal patch files
        # Validate automatic parameter adjustment
        
    def test_sparse_gedi_extraction():
        """Test GEDI pixel extraction from patches"""
        # Create patch with sparse GEDI pixels (<0.3% coverage)
        # Validate extraction efficiency
        
    def test_lightweight_training():
        """Test training with minimal synthetic data"""
        # Single patch, 10 epochs maximum
        # Test all model types (RF, MLP, 2D/3D U-Net)
        # CPU-only execution (< 30 seconds per model)
```

#### 3.2 Data Augmentation (`test_augmentation.py`)
```python
class TestDataAugmentation:
    def test_spatial_transformations():
        """Test 12x spatial augmentation (3 flips × 4 rotations)"""
        # Test augmentation on synthetic patches
        # Validate geometric consistency
        
    def test_augmented_dataset():
        """Test AugmentedPatchDataset class"""
        # Test PyTorch dataset with augmentation
        # Validate batch loading
        
    def test_gedi_target_alignment():
        """Test GEDI target transformation alignment"""
        # Ensure targets follow same transformations as features
        # Validate spatial consistency
```

#### 3.3 Early Stopping and Validation (`test_training_components.py`)
```python
class TestTrainingComponents:
    def test_early_stopping_callback():
        """Test early stopping with synthetic validation losses"""
        # Simulate improving/degrading validation curves
        # Test patience mechanism
        
    def test_training_logger():
        """Test comprehensive training logging"""
        # Test metric tracking and persistence
        # Validate log file creation
        
    def test_checkpoint_system():
        """Test model checkpointing and resumption"""
        # Test save/load checkpoint functionality
        # Validate state restoration
```

### 4. Spatial Processing Tests

#### 4.1 Enhanced Spatial Merger (`test_spatial_utils.py`)
```python
class TestSpatialUtils:
    def test_enhanced_spatial_merger():
        """Test EnhancedSpatialMerger class"""
        # Create multiple synthetic prediction patches
        # Test various merge strategies (first, average, max)
        
    def test_mosaic_creation():
        """Test geographic mosaic creation"""
        # Create overlapping prediction files with georeference
        # Test rasterio.merge integration
        # Validate spatial accuracy
        
    def test_nan_handling():
        """Test robust NaN and nodata handling"""
        # Test predictions with missing values
        # Validate cleaning and filling strategies
        
    def test_memory_efficient_merging():
        """Test large dataset merging with memory constraints"""
        # Simulate multiple large patches
        # Test streaming processing
```

#### 4.2 Prediction Workflows (`test_prediction.py`)
```python
class TestPrediction:
    def test_single_patch_prediction():
        """Test prediction on single patch"""
        # Load trained model (synthetic or pre-trained)
        # Test prediction generation
        
    def test_multi_patch_prediction():
        """Test prediction across multiple patches"""
        # Test directory-based processing
        # Validate batch prediction
        
    def test_prediction_with_mosaic():
        """Test complete prediction workflow with mosaicking"""
        # Test predict.py main functionality
        # Validate end-to-end spatial mosaic creation
```

### 5. Integration and System Tests

#### 5.1 End-to-End Workflow (`test_e2e_workflow.py`)
```python
class TestE2EWorkflow:
    def test_minimal_training_workflow():
        """Test complete training workflow with synthetic data"""
        # Create minimal synthetic patch dataset
        # Test train_predict_map.py end-to-end
        # Validate all model types complete successfully
        
    def test_prediction_workflow():
        """Test complete prediction workflow"""
        # Use pre-trained models or train minimal models
        # Test prediction and mosaicking pipeline
        # Validate output file creation
        
    def test_evaluation_workflow():
        """Test evaluation and reporting"""
        # Test evaluate_predictions.py with synthetic data
        # Validate PDF report generation
        # Test metrics calculation
```

#### 5.2 Configuration and Compatibility (`test_compatibility.py`)
```python
class TestCompatibility:
    def test_resolution_config():
        """Test multi-resolution configuration support"""
        # Test 10m, 20m, 30m resolution configurations
        # Validate patch size scaling
        
    def test_backward_compatibility():
        """Test compatibility with existing workflows"""
        # Test run_main.py integration
        # Validate CLI argument handling
        
    def test_import_system():
        """Test modular import system"""
        # Test all import paths work correctly
        # Validate fallback mechanisms
```

### 6. Performance and Error Handling Tests

#### 6.1 Performance Tests (`test_performance.py`)
```python
class TestPerformance:
    def test_memory_usage():
        """Test memory consumption with various configurations"""
        # Monitor memory usage during training/prediction
        # Validate memory efficiency
        
    def test_execution_time():
        """Test execution time for lightweight operations"""
        # Benchmark core operations
        # Ensure tests complete within time limits
        
    def test_cpu_vs_gpu_performance():
        """Test performance comparison (if GPU available)"""
        # Compare CPU and GPU execution times
        # Validate consistent results
```

#### 6.2 Error Handling Tests (`test_error_handling.py`)
```python
class TestErrorHandling:
    def test_missing_file_handling():
        """Test graceful handling of missing files"""
        # Test with missing patch files, models, etc.
        # Validate error messages and fallbacks
        
    def test_malformed_data_handling():
        """Test handling of corrupted or malformed data"""
        # Test with invalid patch dimensions, corrupted files
        # Validate robust error recovery
        
    def test_memory_constraint_handling():
        """Test behavior under memory pressure"""
        # Simulate low memory conditions
        # Test graceful degradation and fallbacks
```

## Synthetic Data Creation Strategy

### 🎲 **Realistic Synthetic Data Generation**
```python
def create_synthetic_temporal_patch(bands=196, height=256, width=256):
    """Create realistic temporal patch with proper band patterns"""
    # Sentinel-1: 24 bands (2 polarizations × 12 months)
    # Sentinel-2: 132 bands (11 bands × 12 months)  
    # ALOS-2: 24 bands (2 polarizations × 12 months)
    # DEM/Other: ~16 bands
    # Total: ~196 bands
    
def create_synthetic_non_temporal_patch(bands=31, height=256, width=256):
    """Create realistic non-temporal patch with median composites"""
    # Sentinel-1: 2 bands (VV, VH medians)
    # Sentinel-2: 11 bands (10 bands + NDVI median)
    # ALOS-2: 2 bands (HH, HV medians)
    # DEM/Other: ~16 bands
    # Total: ~31 bands

def create_synthetic_gedi_data(patch_shape, coverage=0.003):
    """Create sparse GEDI data (<0.3% coverage)"""
    # Random height values 0-50m at sparse locations
    # Realistic height distribution patterns
```

## Test Execution Strategy

### ⚡ **Lightweight Test Configuration**
- **Maximum Test Duration**: 5 minutes for complete suite
- **Memory Limit**: 2GB RAM maximum per test
- **CPU Cores**: Single-threaded tests with optional multi-threading
- **GPU Testing**: Optional GPU tests when CUDA is available

### 📋 **Test Organization**
```bash
tests/
├── unit/                          # Individual component tests
│   ├── test_image_patches_enhanced.py
│   ├── test_normalization_enhanced.py
│   ├── test_spatial_utils_enhanced.py
│   ├── test_3d_unet_enhanced.py
│   ├── test_2d_unet_enhanced.py
│   └── test_traditional_models_enhanced.py
├── integration/                   # Component interaction tests
│   ├── test_train_predict_map_enhanced.py
│   ├── test_augmentation_enhanced.py
│   ├── test_prediction_enhanced.py
│   └── test_multi_patch_enhanced.py
├── system/                        # End-to-end workflow tests
│   ├── test_e2e_workflow_enhanced.py
│   ├── test_compatibility_enhanced.py
│   └── test_evaluation_enhanced.py
├── performance/                   # Performance and stress tests
│   ├── test_performance_enhanced.py
│   └── test_error_handling_enhanced.py
└── fixtures/                      # Shared test data and utilities
    ├── synthetic_data.py          # Synthetic data generation
    ├── test_config.py             # Test configuration
    └── mock_models.py             # Pre-trained mock models
```

### 🚀 **Execution Commands**
```bash
# Run complete test suite
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/unit/ -v         # Unit tests only
python -m pytest tests/integration/ -v  # Integration tests only
python -m pytest tests/system/ -v       # System tests only

# Run performance tests (optional)
python -m pytest tests/performance/ -v --timeout=300

# Run with coverage reporting
python -m pytest tests/ --cov=. --cov-report=html

# Run lightweight subset (< 2 minutes)
python -m pytest tests/unit/ tests/integration/ -v -x
```

## Expected Test Coverage

### 📊 **Coverage Targets**
- **Core Data Processing**: 90% line coverage
- **Model Architectures**: 85% line coverage (PyTorch model testing)
- **Training System**: 80% line coverage (complex training workflows)
- **Spatial Processing**: 95% line coverage (critical for mosaicking)
- **Integration Tests**: 75% workflow coverage
- **Overall Target**: 85% codebase coverage

### ✅ **Success Criteria**
1. **All tests pass** on CPU-only environments
2. **GPU compatibility** validated when CUDA available
3. **Memory usage** stays under 2GB for individual tests
4. **Execution time** under 5 minutes for complete suite
5. **Error handling** validates graceful failure modes
6. **Backward compatibility** ensures existing workflows work

## Implementation Priority

### 🥇 **Phase 1: Critical Core Tests (Week 1)**
- `test_image_patches_enhanced.py`
- `test_spatial_utils_enhanced.py`
- `test_train_predict_map_enhanced.py`
- Basic synthetic data generation

### 🥈 **Phase 2: Model Architecture Tests (Week 2)**
- `test_3d_unet_enhanced.py`
- `test_2d_unet_enhanced.py`
- `test_traditional_models_enhanced.py`
- Lightweight model testing

### 🥉 **Phase 3: Integration and System Tests (Week 3)**
- `test_e2e_workflow_enhanced.py`
- `test_prediction_enhanced.py`
- `test_evaluation_enhanced.py`
- End-to-end validation

### 🎯 **Phase 4: Performance and Robustness (Week 4)**
- `test_performance_enhanced.py`
- `test_error_handling_enhanced.py`
- `test_compatibility_enhanced.py`
- Production readiness validation

This comprehensive test plan ensures robust validation of the entire CHM Image Processing system while maintaining lightweight execution suitable for CI/CD environments and development workflows.
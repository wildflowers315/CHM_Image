#!/usr/bin/env python3
"""
Test configuration and utilities for CHM Image Processing tests.

This module provides shared configuration, fixtures, and utilities
for lightweight CPU-based testing.
"""

import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import numpy as np

# Test configuration constants
TEST_CONFIG = {
    # Data dimensions
    'patch_height': 256,
    'patch_width': 256,
    'temporal_bands': 196,
    'non_temporal_bands': 31,
    
    # Training configuration (lightweight for testing)
    'max_epochs': 5,
    'batch_size': 2,
    'learning_rate': 0.001,
    'early_stopping_patience': 3,
    'validation_split': 0.8,
    
    # GEDI configuration
    'gedi_coverage': 0.003,  # 0.3% coverage
    'max_height': 50.0,
    'min_height': 0.0,
    
    # Performance limits
    'max_memory_gb': 2.0,
    'max_test_duration_seconds': 300,  # 5 minutes
    'max_single_test_seconds': 60,     # 1 minute per test
    
    # GPU testing
    'test_gpu_if_available': True,
    'force_cpu_testing': False,
    
    # File paths
    'temp_dir_prefix': 'chm_test_',
    'test_data_dir': 'tests/fixtures/data',
    'test_output_dir': 'tests/fixtures/output'
}

# Model configurations for lightweight testing
MODEL_CONFIGS = {
    'rf': {
        'n_estimators': 10,  # Reduced from production 500
        'max_depth': 5,      # Reduced from production None
        'min_samples_leaf': 2,
        'n_jobs': 1          # Single thread for testing
    },
    'mlp': {
        'hidden_layers': [32, 16],  # Reduced from production [128, 64, 32]
        'max_epochs': 5,
        'batch_size': 32,
        'learning_rate': 0.001
    },
    '2d_unet': {
        'base_channels': 16,    # Reduced from production 64
        'n_classes': 1,
        'in_channels': 31      # Non-temporal default
    },
    '3d_unet': {
        'in_channels': 196,    # Temporal default
        'out_channels': 1,
        'base_filters': 8,     # Reduced from production 32
        'num_levels': 3        # Reduced from production 4
    }
}

def get_device() -> torch.device:
    """Get appropriate device for testing (CPU or GPU)."""
    if TEST_CONFIG['force_cpu_testing']:
        return torch.device('cpu')
    
    if TEST_CONFIG['test_gpu_if_available'] and torch.cuda.is_available():
        return torch.device('cuda')
    
    return torch.device('cpu')

def create_temp_dir() -> str:
    """Create temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp(prefix=TEST_CONFIG['temp_dir_prefix'])
    return temp_dir

def cleanup_temp_dir(temp_dir: str) -> None:
    """Clean up temporary directory."""
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

class DataManagerFixture:
    """Manage test data creation and cleanup."""
    
    def __init__(self):
        self.temp_dirs = []
        self.temp_files = []
    
    def create_temp_dir(self) -> str:
        """Create and track temporary directory."""
        temp_dir = create_temp_dir()
        self.temp_dirs.append(temp_dir)
        return temp_dir
    
    def create_temp_file(self, suffix: str = '.tif', dir: Optional[str] = None) -> str:
        """Create and track temporary file."""
        if dir is None:
            dir = self.create_temp_dir()
        
        fd, temp_file = tempfile.mkstemp(suffix=suffix, dir=dir)
        os.close(fd)  # Close file descriptor
        self.temp_files.append(temp_file)
        return temp_file
    
    def cleanup_all(self) -> None:
        """Clean up all temporary files and directories."""
        # Clean up files first
        for temp_file in self.temp_files:
            if os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except OSError:
                    pass
        
        # Clean up directories
        for temp_dir in self.temp_dirs:
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except OSError:
                    pass
        
        self.temp_dirs.clear()
        self.temp_files.clear()

def assert_tensor_properties(tensor: torch.Tensor, 
                           expected_shape: tuple,
                           expected_dtype: torch.dtype = torch.float32,
                           value_range: Optional[tuple] = None,
                           name: str = "tensor") -> None:
    """Assert tensor has expected properties."""
    assert tensor.shape == expected_shape, f"{name} shape mismatch: got {tensor.shape}, expected {expected_shape}"
    assert tensor.dtype == expected_dtype, f"{name} dtype mismatch: got {tensor.dtype}, expected {expected_dtype}"
    
    if value_range is not None:
        min_val, max_val = value_range
        actual_min, actual_max = tensor.min().item(), tensor.max().item()
        assert min_val <= actual_min, f"{name} min value {actual_min} below expected range {min_val}"
        assert actual_max <= max_val, f"{name} max value {actual_max} above expected range {max_val}"

def assert_array_properties(array: np.ndarray,
                          expected_shape: tuple,
                          expected_dtype: np.dtype = np.float32,
                          value_range: Optional[tuple] = None,
                          name: str = "array") -> None:
    """Assert numpy array has expected properties."""
    assert array.shape == expected_shape, f"{name} shape mismatch: got {array.shape}, expected {expected_shape}"
    assert array.dtype == expected_dtype, f"{name} dtype mismatch: got {array.dtype}, expected {expected_dtype}"
    
    if value_range is not None:
        min_val, max_val = value_range
        actual_min, actual_max = array.min(), array.max()
        assert min_val <= actual_min, f"{name} min value {actual_min} below expected range {min_val}"
        assert actual_max <= max_val, f"{name} max value {actual_max} above expected range {max_val}"

def memory_usage_mb() -> float:
    """Get current memory usage in MB."""
    import psutil
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def check_memory_limit() -> None:
    """Check if memory usage exceeds test limits."""
    current_memory = memory_usage_mb()
    max_memory = TEST_CONFIG['max_memory_gb'] * 1024  # Convert to MB
    
    if current_memory > max_memory:
        raise MemoryError(f"Test memory usage {current_memory:.1f}MB exceeds limit {max_memory:.1f}MB")

class LightweightTimer:
    """Simple timer for performance testing."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        self.end_time = time.time()
    
    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None or self.end_time is None:
            raise ValueError("Timer not used properly (use as context manager)")
        return self.end_time - self.start_time
    
    def assert_under_limit(self, max_seconds: float) -> None:
        """Assert elapsed time is under limit."""
        elapsed = self.elapsed_seconds
        assert elapsed <= max_seconds, f"Test took {elapsed:.2f}s, exceeding limit {max_seconds:.2f}s"

def skip_if_no_gpu():
    """Decorator to skip test if GPU not available."""
    import pytest
    return pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="GPU not available"
    )

def skip_if_insufficient_memory(min_memory_gb: float = 2.0):
    """Decorator to skip test if insufficient memory."""
    import pytest
    import psutil
    
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    return pytest.mark.skipif(
        available_memory_gb < min_memory_gb,
        reason=f"Insufficient memory: {available_memory_gb:.1f}GB < {min_memory_gb}GB required"
    )

def create_minimal_cli_args(**kwargs) -> Dict[str, Any]:
    """Create minimal CLI arguments for testing."""
    default_args = {
        'patch_path': None,
        'patch_dir': None,
        'model': 'rf',
        'output_dir': 'test_output',
        'mode': 'train',
        'max_epochs': TEST_CONFIG['max_epochs'],
        'batch_size': TEST_CONFIG['batch_size'],
        'learning_rate': TEST_CONFIG['learning_rate'],
        'augment': False,
        'early_stopping_patience': TEST_CONFIG['early_stopping_patience'],
        'validation_split': TEST_CONFIG['validation_split'],
        'generate_prediction': False,
        'verbose': False
    }
    
    # Override with provided kwargs
    default_args.update(kwargs)
    return default_args

def ensure_test_data_exists() -> Dict[str, str]:
    """Ensure test data exists and return paths."""
    from synthetic_data import create_test_datasets
    
    test_data_dir = TEST_CONFIG['test_data_dir']
    
    # Check if test data already exists
    if (os.path.exists(os.path.join(test_data_dir, 'metadata.json')) and
        os.path.exists(os.path.join(test_data_dir, 'temporal_patch_196bands.tif')) and
        os.path.exists(os.path.join(test_data_dir, 'non_temporal_patch_31bands.tif'))):
        
        # Return existing paths
        return {
            'temporal_patch': os.path.join(test_data_dir, 'temporal_patch_196bands.tif'),
            'non_temporal_patch': os.path.join(test_data_dir, 'non_temporal_patch_31bands.tif'),
            'gedi_targets': os.path.join(test_data_dir, 'gedi_targets.tif'),
            'metadata': os.path.join(test_data_dir, 'metadata.json')
        }
    
    # Create test data
    print("🎲 Creating test datasets...")
    return create_test_datasets(test_data_dir)

# Export commonly used items
__all__ = [
    'TEST_CONFIG',
    'MODEL_CONFIGS', 
    'get_device',
    'create_temp_dir',
    'cleanup_temp_dir',
    'DataManagerFixture',
    'assert_tensor_properties',
    'assert_array_properties',
    'memory_usage_mb',
    'check_memory_limit',
    'LightweightTimer',
    'skip_if_no_gpu',
    'skip_if_insufficient_memory',
    'create_minimal_cli_args',
    'ensure_test_data_exists'
]